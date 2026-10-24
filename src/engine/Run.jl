
struct Run{T,G,L,D}
    r::NTuple{D,CuArray{T,1}}
    v::NTuple{D,CuArray{T,1}}
    f::NTuple{D,CuArray{T,1}}
    energy::CuArray{T,1}
    virial::CuArray{T,2}
    neighbors::NL.DeviceNeighborWorkspace{T,D}
    dt::T
    mass::T
    sigma::T
    epsilon::T
    ordering::Union{Nothing,OrderingWorkspace{T}}
    observables::ObservableWorkspace
    traversal::TraversalWorkspace{G,L}
    build_lanes::Int
    rebuild_policy::Symbol
    fusion::Union{Nothing,FusionWorkspace}
    sample_hook::Any # optional capture-safe callable run at the end of sampled steps
    model::Any       # PairModel (isbits) for the cooperative traversal
    model_data::Any  # tuple of coefficient arrays read by the model
    thermostat::Any  # nothing (NVE) or LangevinData (BAOAB/GSM)
    features::Tuple  # active EngineFeatures (host); see Features.jl
    fx::Tuple        # their device parts, passed to kernels as one bundle
    cluster::Any     # nothing (ELL pairs) or ClusterBackend (cluster pairs + mesh)
    barostat::Any    # DeviceCRescale of an ELL run (device pressure control), or nothing (cluster runs: cluster.barostat)
    mesh::Any        # ELL charged runs: (solver, q, Vglobal, order) of the long-range mesh, or nothing
    csvr_lazy_report::Bool # immutable choice shared by graph capture, recovery and host observations
end

function Run(r::NTuple{D,CuArray{T,1}},v::NTuple{D,CuArray{T,1}},box::NTuple{D,T};
             cutoff,skin,cap,dt,mass=1,sigma=1,epsilon=1,step=0,
             spatial_reorder::Bool=false,reorder_interval::Integer=500,
             row_threads::Integer=256,diagnostics::Bool=true,
             force_lanes::Integer=1,force_layout::Symbol=:ell,build_lanes::Integer=1,
             rebuild_policy::Symbol=:adaptive,fused::Bool=false,shared=nothing,
             sample_hook=nothing,model=nothing,model_data::Tuple=(),thermostat=nothing,features::Tuple=(),
             cluster=nothing,allpairs::Bool=false,barostat=nothing,mesh=nothing,cell_sub::Integer=1) where {T,D}
    validate_traversal(force_lanes,force_layout)
    D in (2,3) || throw(ArgumentError("The device engine supports 2D and 3D"))
    D==2 && (build_lanes==1 && !fused || throw(ArgumentError(
        "2D runs use the serial row builder and unfused steps (cooperative builder and fusion are 3D)")))
    validate_builder(build_lanes)
    validate_policy(rebuild_policy)
    fused && thermostat!==nothing && !(thermostat isa CSVRData && nbaths(thermostat)==1) &&
        throw(ArgumentError("Thermostatted runs other than single-bath CSVR are unfused"))
    # charged (mesh) runs fuse with per-particle masses: their prefix and gather-kick take the masses
    fused && !all(f->fusable(f) || (mesh!==nothing && f isa MassFeature),features) &&
        throw(ArgumentError("A feature of this run requires the unfused path"))
    fused && (rebuild_policy===:displacement && force_layout===:ell ||
        throw(ArgumentError("Control-kernel fusion requires the displacement policy and ELL layout")))
    cluster===nothing || (D==3 && !fused && !spatial_reorder ||
        throw(ArgumentError("Cluster-pair runs are 3D, unfused and unsorted (the list orders clusters itself)")))
    allpairs && (build_lanes==1 && !spatial_reorder ||
        throw(ArgumentError("All-pairs rows (small boxes) use the serial builder and no sorting")))
    N=length(r[1])
    all(length(x)==N for x in (r...,v...)) || throw(DimensionMismatch("Argon arrays differ in length"))
    dt,mass,sigma,epsilon=T.((dt,mass,sigma,epsilon))
    # ε = 0 is a non-interacting system (pure stochastic dynamics), as on the staged path.
    all(isfinite(x) && x>0 for x in (dt,mass,sigma)) && isfinite(epsilon) && epsilon>=0 ||
        throw(ArgumentError("Invalid Argon parameters"))
    # a device barostat: cells for a box up to 2% smaller (fractional device binning follows the box)
    nb=NL.DeviceNeighborWorkspace(T,N,box;cutoff,skin,cap,step,row_threads,diagnostics,allpairs,
        margin=barostat===nothing ? 1.0 : CP.GEOMETRY_SHRINK,sub=cell_sub)
    traversal=TraversalWorkspace(N,cap,Val(Int(force_lanes)),Val(force_layout))
    if shared===nothing # standalone run: own copies
        rr=map(copy,r); vv=map(copy,v); ff=ntuple(_->CUDA.zeros(T,N),D)
        energy=CUDA.zeros(T,N); virial=CUDA.zeros(T,N,_virial_width(D))
        ordering=spatial_reorder ? OrderingWorkspace(T,N,reorder_interval;dims=D) : nothing
    else # aliases another owner's arrays (SimulationState); sorting permutes them in place
        rr=r; vv=v; ff=shared.f; energy=shared.energy; virial=shared.virial
        all(length(x)==N for x in (ff...,energy)) && size(virial)==(N,_virial_width(D)) ||
            throw(DimensionMismatch("Shared force/energy/virial arrays must have N rows"))
        ordering=spatial_reorder ? OrderingWorkspace(T,N,reorder_interval;dims=D,tags=shared.tags,
            extra_real=shared.extra_real,extra_int=shared.extra_int,extra_mat=shared.extra_mat,
            slot_of=get(shared,:slot_of,nothing)) : nothing
    end
    run=Run{T,Int(force_lanes),force_layout,D}(rr,vv,ff,energy,virial,nb,dt,mass,sigma,epsilon,
        ordering,ObservableWorkspace(N),traversal,Int(build_lanes),rebuild_policy,
        fused ? FusionWorkspace(T,N) : nothing,sample_hook,
        model===nothing ? UniformLJ{T}(epsilon,sigma,T(cutoff)^2) : model,model_data,thermostat,features,
        device_bundle(features),cluster,barostat,mesh,_csvr_lazy_report(cluster,thermostat,features))
    if cluster===nothing
        decide_policy!(nb,run.r,Val(rebuild_policy))
        enqueue_rebuild!(run)
    else
        control_valid!(nb)   # cluster runs keep the ELL workspace as control block only
        CP.request_build!(cluster.eng.list)
    end
    _host_forces(run) || enqueue_force!(run)   # host-force runs: the state's owner supplies forces
    features_at_boundary!(run)    # feature checks (e.g. FENE domain) report their own errors first
    NL.device_neighbor_status(nb) # constructor boundary; throws on invalid input
    return run
end

_virial_width(D)=D==3 ? 6 : 3
_refs(nb,::Val{3})=(nb.rref_x,nb.rref_y,nb.rref_z)
_refs(nb,::Val{2})=(nb.rref_x,nb.rref_y)

# SimulationCore has a fused kick-drift kernel for 3D only; 2D uses the same
# arithmetic without z.
function _kick_drift2_kernel!(rx::CuDeviceVector{T},ry::CuDeviceVector{T},vx::CuDeviceVector{T},
                              vy::CuDeviceVector{T},fx::CuDeviceVector{T},fy::CuDeviceVector{T},
                              c::T,dt::T,box) where {T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(rx) && return nothing
    @inbounds begin
        ux=muladd(c,fx[i],vx[i]); uy=muladd(c,fy[i],vy[i])
        vx[i]=ux; vy[i]=uy
        x=muladd(dt,ux,rx[i]); y=muladd(dt,uy,ry[i])
        Lx,Ly=Definitions.box_lengths(box,Val(2))
        x-=(x>=T(0.5)*Lx)*Lx; x+=(x< -T(0.5)*Lx)*Lx
        y-=(y>=T(0.5)*Ly)*Ly; y+=(y< -T(0.5)*Ly)*Ly
        rx[i]=x; ry[i]=y
    end
    return nothing
end
# Generic kick-drift for features that change the NVE integration (e.g. unwrapped
# coordinates: unwrap += dt·v; per-particle masses: dt/(2m_i)). The tuned NVE
# kernels stay in use otherwise.
function _kick_drift_unwrap_kernel!(r::NTuple{D,<:CuDeviceVector{T}},v::NTuple{D,<:CuDeviceVector{T}},
                                    f::NTuple{D,<:CuDeviceVector{T}},c::T,dt::T,box,fx,mass::T) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        ci=kick_coef(fx,c,dt,mass,i)
        u=map((a,b)->muladd(ci,b,a),_get(v,i),_get(f,i))
        _put!(v,i,u)
        dp=map(a->dt*a,u)
        displace!(fx,i,dp)
        _put!(r,i,map((a,p,L)->_wrap(a+p,L),_get(r,i),dp,Definitions.box_lengths(box,Val(D))))
    end
    return nothing
end
function _kick_generic_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},c::T,dt::T,
                               fx,mass::T) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(v[1]) && return nothing
    @inbounds begin
        ci=kick_coef(fx,c,dt,mass,i)
        _put!(v,i,map((a,b)->muladd(ci,b,a),_get(v,i),_get(f,i)))
    end
    return nothing
end
_generic_nve(run)=any(generic_integration,run.features)
_generic_kick(run)=any(generic_kick,run.features)
_generic_pairs(run)=any(generic_pairs,run.features)
_host_forces(run)=any(host_forces,run.features)
_host_feature(run)=run.features[findfirst(host_forces,run.features)]
# The list feeds the pair forces; with host forces only features that read it keep it.
_list_needed(run)=!_host_forces(run) || any(needs_neighbors,run.features)
_kick_drift_kernel(::Val{3})=SC._nve_kick_drift3_kernel!
_kick_drift_kernel(::Val{2})=_kick_drift2_kernel!
_kick_kernel(::Val{3})=SC._nve_kick3_kernel!
_kick_kernel(::Val{2})=SC._nve_kick2_kernel!

# `always`: sort at this rebuild whatever the interval (the host-scheduled sort step)
function enqueue_rebuild!(run::Run{T,G,L,D};reorder::Bool=true,always::Bool=false) where {T,G,L,D}
    ws=run.neighbors; nb=ws.list
    NL._device_bin!(ws,run.r)
    reorder && enqueue_ordering!(run,run.ordering;always)   # evaluations outside step! never sort
    refs=_refs(nb,Val(D))
    if run.build_lanes==1
        NL._device_finish_rebuild!(ws,run.r,refs)
    else
        finish_rebuild_cooperative!(ws,run.r,refs,Val(run.build_lanes))
    end
    enqueue_packing!(run)
    features_after_rebuild!(run)   # e.g. collision contact flags for the new rows
    return nothing
end

"""Pair forces, then the after-force features in registry order (bonds, typed
bonded terms, freeze tether), as on the staged path."""
function enqueue_force!(run::Run;energy::Bool=false)
    _host_forces(run) && error("Forces of this run come from the host between graph segments (capture_split_steps)")
    if run.cluster!==nothing   # eager cluster pairs + mesh (inside steps: cluster_step_graph)
        cluster_forces!(run,run.cluster;energy)
        features_after_force!(run;energy)
        return nothing
    end
    _enqueue_pair_force!(run;energy)
    run.mesh===nothing || enqueue_ell_mesh!(run;energy)
    features_after_force!(run;energy)
    return nothing
end

"""Long-range mesh of an ELL charged run after its pair forces (the pair kernel overwrites
the force slot; the gather adds to it). The storage order is spatially sorted, so the mesh
traverses it directly."""
function enqueue_ell_mesh!(run;energy::Bool=false)
    m=run.mesh; geom=run.neighbors.geom
    ES.mesh_transform!(m.solver,run.r...,m.q,energy,m.order;geom)
    ES.mesh_gather!(m.solver,run.r...,m.q,run.f...,run.energy,m.Vglobal,energy,m.order;geom)
    return nothing
end

function _zero_pairs_kernel!(f::NTuple{D},energy,virial,::Val{E}) where {D,E}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(f[1]) && return nothing
    @inbounds begin
        ntuple(d->(f[d][i]=zero(eltype(f[d]))),Val(D))
        if E
            energy[i]=zero(eltype(energy))
            for c in 1:size(virial,2); virial[i,c]=zero(eltype(virial)); end
        end
    end
    return nothing
end

function _enqueue_pair_force!(run::Run{T,G,L,D};energy::Bool=false) where {T,G,L,D}
    if run.model isa NoPairs   # bonded-only systems: the after-force features add every force
        N=length(run.r[1])
        @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(run.neighbors.state,_zero_pairs_kernel!,
            run.f,run.energy,run.virial,Val(energy))
        return nothing
    end
    if D!=3 || G!=1 || L!==:ell || !(run.model isa UniformLJ) || _generic_pairs(run) || run.barostat!==nothing
        return enqueue_cooperative_force!(run,run.traversal,Val(energy))
    end
    ws=run.neighbors; nb=ws.list; box=ws.box; half=map(x->T(0.5)*x,box)
    N=length(run.r[1]); threads=N<50000 ? 64 : N<200000 ? 128 : 256
    if energy
        @cuda threads=threads blocks=cld(N,threads) NL._device_if_valid!(ws.state,
            NB._lj3_ell_kernel_virial!,run.r...,run.f...,run.energy,run.virial,
            nb.neighbors_index,nb.neighbors_flat,nb.counts,nb.cap,box...,half...,
            run.epsilon,run.sigma,nb.cutoff^2,run.model.eshift)
    else
        @cuda threads=threads blocks=cld(N,threads) NL._device_if_valid!(ws.state,
            NB._lj3_ell_noE_kernel!,run.r...,run.f...,nb.neighbors_index,
            nb.neighbors_flat,nb.counts,nb.cap,box...,half...,run.epsilon,run.sigma,nb.cutoff^2)
    end
    return nothing
end

function traversal_status(run::Run{T,G,L}) where {T,G,L}
    status=NL.device_neighbor_status(run.neighbors)
    packs=only(Array(run.traversal.packs))
    (L===:tiled ? packs==status.builds : packs==0) || error("Packing/build count mismatch")
    return (;lanes=G,layout=String(L),packs,packed_entries=run.traversal.packed===nothing ? 0 : length(run.traversal.packed))
end

function enqueue_drift!(run::Run{T,G,L,D};csvr::Bool=true) where {T,G,L,D}
    ws=run.neighbors; N=length(run.r[1]); c=(run.dt/T(2))/run.mass
    th=run.thermostat
    rw_fast_path(run) && (enqueue_rw_drift!(run;bath=csvr); return nothing)   # fused rigid-water drift
    th isa NHCData && _nhc_stage!(run,th) # NHC pre-stage (dt/2) before kick1
    if th isa BrownianData
        th.midpoint ? enqueue_bd_midpoint!(run,th) : enqueue_bd_step!(run,th)
        # Heun midpoint: features see the midpoint move (hold: positions only, as staged)
        features_after_position!(run;midpoint=th.midpoint)
        return nothing
    elseif _nve_drift(th) && _generic_nve(run)
        @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(ws.state,
            _kick_drift_unwrap_kernel!,run.r,run.v,run.f,c,run.dt,ws.geom,run.fx,run.mass)
    elseif _nve_drift(th)
        @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(ws.state,
            _kick_drift_kernel(Val(D)),run.r...,run.v...,run.f...,c,run.dt,ws.geom)
    else
        @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(ws.state,_drift_kernel(th),_drift_args(run,th)...)
    end
    features_after_position!(run) # e.g. frozen particles back to their anchors
    th isa CSVRData && csvr && enqueue_csvr_half!(run,th)   # CSVR on the half-step velocities
    return nothing
end

# `advance`: the kick's last kernel also advances the device step counter (`FOLD_TAIL`) where
# it can; returns whether it did (otherwise the caller launches the advance kernel).
function enqueue_kick!(run::Run{T,G,L,D};advance::Bool=false,report::Bool=true) where {T,G,L,D}
    ws=run.neighbors; N=length(run.r[1]); c=(run.dt/T(2))/run.mass
    th=run.thermostat
    w=advance ? NL._device_if_valid_advance! : NL._device_if_valid!
    rw_fast_path(run) && return enqueue_rw_kick!(run;advance)    # fused rigid-water kick
    if th===nothing && _generic_kick(run)
        @cuda threads=256 blocks=cld(N,256) w(ws.state,_kick_generic_kernel!,run.v,run.f,c,run.dt,run.fx,run.mass)
        return advance
    elseif th===nothing
        @cuda threads=256 blocks=cld(N,256) w(ws.state,_kick_kernel(Val(D)),run.v...,run.f...,c)
        return advance
    elseif th isa CSVRData
        return _csvr_kick!(run,th;advance,report)
    elseif th isa NHCData
        _nhc_kick!(run,th)
    elseif th isa BrownianData
        # overdamped: no velocity update after the force
    else
        @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(ws.state,_b2_kernel(th),_kick_args(run,th)...)
    end
    return false
end

_nve_drift(th)=th===nothing || th isa Union{CSVRData,NHCData}

function compile_integration!(run::Run{T,G,L,D}) where {T,G,L,D}
    ws=run.neighbors; c=(run.dt/T(2))/run.mass; th=run.thermostat
    _nve_drift(th) && _generic_nve(run) && (@cuda launch=false NL._device_if_valid!(ws.state,
        _kick_drift_unwrap_kernel!,run.r,run.v,run.f,c,run.dt,ws.geom,run.fx,run.mass))
    th===nothing && _generic_kick(run) && (@cuda launch=false NL._device_if_valid!(ws.state,
        _kick_generic_kernel!,run.v,run.f,c,run.dt,run.fx,run.mass))
    features_after_position!(run;launch=false); features_after_position!(run;midpoint=true,launch=false)
    features_after_force!(run;energy=true,launch=false); features_after_force!(run;energy=false,launch=false)
    features_after_kick!(run;energy=true,launch=false); features_after_kick!(run;energy=false,launch=false)
    features_after_rebuild!(run;launch=false); features_position_sample!(run;launch=false)
    @cuda launch=false NL._mark_section_kernel!(ws.state,Int64(1))   # section markers (capacity recovery)
    if th isa LangevinData && _split_baoa(run,th)
        @cuda launch=false NL._device_if_valid!(ws.state,_ba_kernel!,_ba_args(run,th)...)
        @cuda launch=false NL._device_if_valid!(ws.state,_oa_kernel!,_oa_args(run,th)...)
    end
    if th===nothing
        @cuda launch=false NL._device_if_valid!(ws.state,
            _kick_drift_kernel(Val(D)),run.r...,run.v...,run.f...,c,run.dt,ws.geom)
        @cuda launch=false NL._device_if_valid!(ws.state,_kick_kernel(Val(D)),run.v...,run.f...,c)
    elseif th isa BrownianData
        enqueue_bd_step!(run,th;launch=false)
        th.midpoint && enqueue_bd_midpoint!(run,th;launch=false)
    elseif th isa Union{CSVRData,NHCData}
        @cuda launch=false NL._device_if_valid!(ws.state,
            _kick_drift_kernel(Val(D)),run.r...,run.v...,run.f...,c,run.dt,ws.geom)
        th isa CSVRData ? _csvr_kick!(run,th;launch=false) : _nhc_kick!(run,th;launch=false)
        th isa CSVRData && enqueue_csvr_half!(run,th;launch=false)
    else
        @cuda launch=false NL._device_if_valid!(ws.state,_drift_kernel(th),_drift_args(run,th)...)
        @cuda launch=false NL._device_if_valid!(ws.state,_b2_kernel(th),_kick_args(run,th)...)
    end
    rw_fast_path(run) && (enqueue_rw_drift!(run;launch=false); enqueue_rw_kick!(run;launch=false);
        run.thermostat isa CSVRData && enqueue_rw_bath!(run;launch=false))
    return nothing
end

function enqueue_finish!(run::Run{T,G,L,D};energy::Bool=false) where {T,G,L,D}
    features_position_sample!(run) # position sample with a list valid at these coordinates
    enqueue_force!(run;energy)
    _finish_after_force!(run;energy)
    return nothing
end

function _finish_after_force!(run;energy::Bool=false)
    ws=run.neighbors
    energy && run.barostat!==nothing && enqueue_virial_sample!(run,_baro_ctx(run))   # device pressure control
    enqueue_kick!(run)
    features_after_kick!(run;energy)
    NL.advance_device_neighbor_step!(ws)
    energy && enqueue_observables!(run)
    energy && run.sample_hook!==nothing && run.sample_hook()
    return nothing
end

function enqueue_step!(run::Run;energy::Bool=false)
    if run.fusion!==nothing
        enqueue_fused_prefix!(run,UInt64(0),Val(false))
        enqueue_rebuild!(run)
        return fused_suffix!(run;energy)
    end
    mp=_ell_mesh_parallel(run)
    enqueue_drift!(run;csvr=!mp)
    decide_policy!(run.neighbors,run.r,Val(run.rebuild_policy))
    enqueue_rebuild!(run)
    mp && run.thermostat isa CSVRData && enqueue_csvr_half!(run,run.thermostat)
    enqueue_finish!(run;energy)
    return nothing
end

"""Capture one whole fixed-cell uniform-LJ NVE step, without moving particles.

The constructor warmed the list/force kernels. Compile the two integration
kernels and counter without launching them before entering capture. Force
buffers never swap, so odd/even graph replay uses the same physical state.
Retain `run` while using the graph; do not mutate its arrays externally.
With `energy=true`, the force stage writes per-particle potential/virial and
the completed step reduces global U/W and time-centered kinetic/thermodynamic
observables into persistent Float64 buffers (see `OBS_CONVENTION`).
Build separate regular and observable graphs, then select them at the desired
output cadence. Their device counters and physical buffers remain shared.
With `spatial_reorder=true`, sorting runs at rebuild boundaries after at least
`reorder_interval` device steps. Persistent scratch and copyback preserve
captured addresses; `run.ordering.tags` maps storage slots to original ids.
Per-particle data must be sorted with the particles (features register theirs);
bonded runs are not sorted (topology indices are storage slots).
"""
function capture_step(run::Run{T,G,L,D};conditional::Bool=true,energy::Bool=false,warm::Bool=true,
                      event::Bool=false,virial::Bool=false,fast::Bool=false,high::Bool=MESH_PRIORITY[]) where {T,G,L,D}
    _host_forces(run) && error("Runs with host forces use capture_split_steps")
    # pressure events of fused ELL runs are unfused steps (rare: every `interval` steps)
    (run.fusion===nothing || event) || return capture_fused_step(run;conditional,energy)
    ws=run.neighbors
    compile_integration!(run)
    @cuda launch=false NL._device_neighbor_advance_kernel!(ws.state)
    # Compile the observable specialization before capture. This refreshes
    # forces/observables at unchanged coordinates and does not advance a step.
    energy && warm && enqueue_force!(run;energy=true)
    energy && compile_observables!(run)
    CUDA.synchronize()
    graph=if run.cluster!==nothing
        event && enqueue_crescale!(run,run.cluster;launch=false)   # compile the event kernels before capture
        energy && enqueue_virial_sample!(run,run.cluster;launch=false)
        if virial   # compile the trace kernels (pairs, prune pass, scatter) before capture
            enqueue_virial_trace_sample!(run,run.cluster;launch=false)
            CP.enqueue_pairs!(run.cluster.eng,false,Val(false);trace=true)
            CP.enqueue_pairs!(run.cluster.eng,false,Val(true);trace=true)
            fill!(run.cluster.eng.vtrace,0.0)
            CUDA.synchronize()
        end
        cluster_step_graph(run,run.cluster;energy,event,virial,fast,high)
    elseif conditional && !event && _ell_mesh_parallel(run)
        energy && run.barostat!==nothing && enqueue_virial_sample!(run,_baro_ctx(run);launch=false)
        CUDA.synchronize()
        ell_mesh_graph(run;energy)
    elseif conditional
        event && enqueue_crescale!(run,_baro_ctx(run);launch=false)
        energy && run.barostat!==nothing && enqueue_virial_sample!(run,_baro_ctx(run);launch=false)
        CUDA.synchronize()
        sections,suffix=step_plan(run;event)
        length(sections)==1 ? NL.device_conditional_graph(sections[1]...,suffix(energy),ws.state) :
            chained_conditional_graph(sections,suffix(energy),ws.state)
    else
        (run.thermostat isa BrownianData && run.thermostat.midpoint ||
         run.thermostat isa LangevinData && _split_baoa(run,run.thermostat)) &&
            throw(ArgumentError("Euler–Heun runs need the conditional graph"))
        NL.capture_graph() do
            enqueue_step!(run;energy)
        end
    end
    return CUDA.instantiate(graph,(run.cluster!==nothing && high) ? GRAPH_FLAGS() : 0)
end

"""Sections of one captured step: `(prefix, rebuild)` pairs, each prefix ending with
the rebuild decision, and the suffix as a function of the energy flag. Capture
builds the step's graph from it, and capacity recovery builds `resume_graph` from
it, so both always agree. Steps with several list checks (BAOAB family sampling
after A1, Euler–Heun) mark their section on the device before rebuilding. Fused
runs set their own condition in the fused prefix kernel; their plan serves
recovery only."""
function step_plan(run;event::Bool=false)
    ws=run.neighbors; th=run.thermostat
    decide=() -> decide_policy!(ws,run.r,Val(run.rebuild_policy))
    marked(k)=() -> (NL.mark_section!(ws,k); enqueue_rebuild!(run))
    suffix(energy)=() -> enqueue_finish!(run;energy)
    # an ELL pressure event: the unfused step with the event after the move
    event && return [(() -> (enqueue_drift!(run); enqueue_crescale!(run,_baro_ctx(run)); decide()),
                      () -> enqueue_rebuild!(run))],suffix
    run.fusion===nothing ||
        return [(() -> nothing,() -> enqueue_rebuild!(run))],energy -> (() -> fused_suffix!(run;energy))
    if th isa LangevinData && _split_baoa(run,th)
        # B·A, check, sample, O·A, check, sample at the start of the suffix (staged collisions)
        ba=() -> (enqueue_ba!(run,th); features_after_position!(run); decide())
        oa=() -> (features_position_sample!(run); enqueue_oa!(run,th); features_after_position!(run); decide())
        return [(ba,marked(1)),(oa,marked(2))],suffix
    elseif th isa BrownianData && th.midpoint
        # Euler–Heun: midpoint move and check, midpoint force, final move and check
        final=() -> (enqueue_force!(run;energy=false); enqueue_bd_step!(run,th); features_after_position!(run); decide())
        return [(() -> (enqueue_drift!(run); decide()),marked(1)),(final,marked(2))],suffix
    end
    if _ell_mesh_parallel(run)   # same order as ell_mesh_graph (sequential: recovery, eager steps)
        msuffix(energy)=() -> (th isa CSVRData && enqueue_csvr_half!(run,th); enqueue_finish!(run;energy))
        return [(() -> (enqueue_drift!(run;csvr=false); decide()),() -> enqueue_rebuild!(run))],msuffix
    end
    return [(() -> (enqueue_drift!(run); decide()),() -> enqueue_rebuild!(run))],suffix
end

# ELL charged steps with parallel branches (15 Oct 2026): not for pressure events (their
# box change sits between the drift and the list check) nor host-force runs.
_ell_mesh_parallel(run)=run.mesh!==nothing && run.cluster===nothing && run.fusion===nothing &&
    !_host_forces(run) && !rw_fast_path(run)

"""Step graph of an ELL charged run: after the drift, the list check and the conditional
rebuild (which may sort the storage, so every branch follows it), the CSVR half-step update
(latency-bound: one thread draws the Bussi random numbers) runs parallel to the pair force
and the mesh transform (spread, FFT, solve, inverse FFT); the gather (adds the mesh force to
the pair force) and the kick join them."""
function ell_mesh_graph(run;energy::Bool=false)
    CC=NL.ConditionalCUDA; ws=run.neighbors; m=run.mesh; th=run.thermostat
    graph=CUDA.CuGraph()
    handle=Ref{CC.CUgraphConditionalHandle}()
    CC.cuGraphConditionalHandleCreate(handle,graph,graph.ctx,UInt32(0),CC.CU_GRAPH_COND_ASSIGN_DEFAULT)
    CUDA.@cuda launch=false NL._device_set_rebuild_condition_kernel!(handle[],ws.state)
    CUDA.synchronize()
    before=NL.capture_graph() do
        enqueue_drift!(run;csvr=false); decide_policy!(ws,run.r,Val(run.rebuild_policy))
        CUDA.@cuda threads=1 blocks=1 NL._device_set_rebuild_condition_kernel!(handle[],ws.state)
    end
    first=NL._add_child_graph!(graph,before,CC.CUgraphNode[])
    cond,body=NL._add_conditional_graph!(graph,handle[],first)
    NL._add_child_graph!(body,NL.capture_graph(()->enqueue_rebuild!(run)),CC.CUgraphNode[])
    deps=CC.CUgraphNode[cond]
    pairs=NL._add_child_graph!(graph,NL.capture_graph(()->(features_position_sample!(run);
        _enqueue_pair_force!(run;energy))),deps)
    # the mesh follows the pair force: run concurrently, both slow down (L2 traffic of the
    # spread atomics and the pair loads; salt 10k: pair 27 → 51 µs, spread 11 → 42 µs)
    mesh=NL._add_child_graph!(graph,NL.capture_graph(()->ES.mesh_transform!(m.solver,run.r...,m.q,energy,m.order;
        geom=ws.geom)),CC.CUgraphNode[pairs])
    joins=CC.CUgraphNode[mesh]
    th isa CSVRData && push!(joins,NL._add_child_graph!(graph,NL.capture_graph(()->enqueue_csvr_half!(run,th)),deps))
    NL._add_child_graph!(graph,NL.capture_graph(()->(ES.mesh_gather!(m.solver,run.r...,m.q,run.f...,run.energy,
        m.Vglobal,energy,m.order;geom=ws.geom); features_after_force!(run;energy); _finish_after_force!(run;energy))),joins)
    return graph
end

"""Graph that redoes the rebuild of section `k` (requested again on the device after
a capacity overflow) and runs the remaining sections and `suffix` (`nothing`: none)."""
resume_graph(run,sections,k::Integer,suffix)=CUDA.instantiate(chained_conditional_graph(
    [(() -> nothing,sections[k][2]),sections[k+1:end]...],suffix,run.neighbors.state))

"""Graph of a step with a device pressure event: the event after the move; no energies
(the virial it reads comes from the step before)."""
capture_event_graph(run)=capture_step(run;conditional=true,energy=false,warm=false,event=true)
"""Graph of a plain step between scheduled steps of a host-scheduled run (`HOST_SCHEDULE`): no list decision."""
capture_fast_graph(run;high::Bool=MESH_PRIORITY[])=capture_step(run;conditional=true,energy=false,warm=false,fast=true,high)
"""Graph of the step before an event when no energies are requested: pairs in virial-trace mode."""
capture_virial_graph(run)=capture_step(run;conditional=true,energy=false,warm=false,virial=true)

"""Regular and observable step graphs of `run` (split steps for host-force runs)."""
function capture_graphs(run;warm::Bool=true)
    _host_forces(run) && return capture_split_steps(run)
    if run.cluster!==nothing
        # The eager energy evaluation that compiles the sampled kernels may rebuild and
        # grow the cluster list; it runs before either graph captures list storage.
        warm && enqueue_force!(run;energy=true)
        return capture_step(run;conditional=true,warm=false),capture_step(run;conditional=true,energy=true,warm=false)
    end
    return capture_step(run;conditional=true),capture_step(run;conditional=true,energy=true)
end


"""One step of a run whose forces come from the host: graph segments that end
at the step's force evaluations. `step!` launches them in order and calls the
host force hook between consecutive segments; only the final evaluation of a
sampled step (`energy`) computes energies."""
struct SplitStep{G}
    segments::Vector{G}   # instantiated graphs (CUDA graph execs)
    energy::Bool
end

"""Capture the regular and the observable split step of a host-force run. The
segments mirror `capture_step`: the drift (and, when the list is kept, its check
and conditional rebuild) and the position sample end at the force evaluation;
the last segment is the after-force hooks, the kick, the step counter and,
sampled, the observables.
Euler–Heun has two force evaluations (midpoint, final), hence three segments.
Without a feature that reads the neighbor list, no list maintenance is captured:
the list built at construction stays valid and unused."""
function capture_split_steps(run)
    ws=run.neighbors
    compile_integration!(run)
    @cuda launch=false NL._device_neighbor_advance_kernel!(ws.state)
    compile_observables!(run)
    CUDA.synchronize()
    segs,finish,check=split_plan(run)
    graph(sections,suffix)=check ? chained_conditional_graph(sections,suffix,ws.state) :
        NL.capture_graph(() -> (foreach(sec -> sec[1](),sections); suffix===nothing || suffix()))
    pre=[CUDA.instantiate(graph(sections,suffix)) for (sections,suffix) in segs]
    return SplitStep([pre...,CUDA.instantiate(NL.capture_graph(finish(false)))],false),
           SplitStep([pre...,CUDA.instantiate(NL.capture_graph(finish(true)))],true)
end

"""Segments of a split step as `(sections, suffix)` pairs (host force calls fall
between consecutive segments), the finish closure (energy flag) and whether the
list is kept. Kept lists are checked after each move, and sections are numbered
across segments for capacity recovery. Without a list reader no list maintenance
is captured: the list built at construction stays valid and unused. After-force
hooks (the freeze tether, a user restraint on top of the host forces) open the
segment that follows each host force call."""
function split_plan(run)
    ws=run.neighbors; th=run.thermostat; check=_list_needed(run)
    decide=() -> decide_policy!(ws,run.r,Val(run.rebuild_policy))
    marked(k)=() -> (NL.mark_section!(ws,k); enqueue_rebuild!(run))
    sample=() -> features_position_sample!(run)
    sec(move,k)=check ? (() -> (move(); decide()),marked(k)) : (move,nothing)
    segs=if th isa BrownianData && th.midpoint
        # midpoint move | f(x_m) on the host | tether, final move | f(x) on the host
        [([sec(() -> enqueue_drift!(run),1)],nothing),
         ([sec(() -> (features_after_force!(run;energy=false); enqueue_bd_step!(run,th);
             features_after_position!(run)),2)],sample)]
    elseif th isa LangevinData && _split_baoa(run,th)
        # a feature samples after A1 (collisions, which keep the list)
        ba=() -> (enqueue_ba!(run,th); features_after_position!(run); decide())
        oa=() -> (features_position_sample!(run); enqueue_oa!(run,th); features_after_position!(run); decide())
        [([(ba,marked(1)),(oa,marked(2))],sample)]
    else
        [([sec(() -> enqueue_drift!(run),1)],sample)]
    end
    finish(energy)=() -> begin
        features_after_force!(run;energy)
        enqueue_kick!(run)
        NL.advance_device_neighbor_step!(ws)
        energy && enqueue_observables!(run)
    end
    return segs,finish,check
end

"""Segment and local section of the split-step section `k` (numbered across segments)."""
function split_section(segs,k::Integer)
    for (j,(sections,_)) in enumerate(segs)
        k<=length(sections) && return j,k
        k-=length(sections)
    end
    error("Section index beyond the split step")
end

"""Replay a batch; synchronize/error-check once at its boundary."""
function run_steps!(run::Run,graph,n::Integer)
    n>=0 || throw(ArgumentError("Step count must be nonnegative"))
    for _ in 1:n
        CUDA.launch(graph)
    end
    return NL.device_neighbor_status(run.neighbors)
end
