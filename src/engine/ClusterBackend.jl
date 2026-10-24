# Included in ParticleDynamics.DeviceEngine. Cluster-pair backend. A run with a
# `ClusterBackend` takes its short-range forces from `ClusterPairs` (Lennard-Jones and
# the real-space Coulomb term, exclusions and scaled pairs included) and, when the
# state carries an electrostatics solver, its long-range forces from the mesh. The
# engine's ELL workspace stays the run's control block (step counter, validity,
# latched errors, capacity sections); its list is never built.
#
# One step (`cluster_step_parts`), as flag-conditional graph sections:
#   drift → device decide │ [rebuild, error → control block] │ position sample, pack
#   │ [pruning pass] │ plain pass, scatter │ ‖ mesh transform (after the drift) │
#   gather, after-force features, kick, step counter, observables
# A cluster outer-list overflow stops the step like an ELL overflow: the control block
# latches DN_CAPACITY_ERROR with the required entries, later integration kernels do
# nothing, and the boundary grows the list and replays (`_recover_cluster!`).

const CP = ClusterPairs
const ES = Electrostatics

struct ClusterBackend
    eng::Any        # CP.ClusterPairEngine (shared with the state: `st.cluster`)
    solver::Any     # ES.ElectrostaticsSolver or nothing
    q::Any          # charges, or nothing
    typ::Any        # type ids
    order::Any      # mesh traversal order (copy of the cluster order; see cluster_step_parts)
    Vglobal::Any    # mesh global virial (st.virial_global), or nothing
    barostat::Any   # DeviceCRescale (device-resident pressure control), or nothing
end

ClusterBackend(eng,solver,q,typ,Vglobal;barostat=nothing)=ClusterBackend(eng,solver,q,typ,
    solver===nothing ? nothing : copy(CP.mesh_order(eng)),Vglobal,barostat)

function _cluster_error_kernel!(state,counts)
    @inbounds if counts[CP.CL_ERR]==CP.CL_ERR_CAPACITY
        state[NL.DN_ERROR]=NL.DN_CAPACITY_ERROR
        state[NL.DN_VALID]=0
        state[NL.DN_REQUIRED]=Int64(counts[CP.CL_REQ])
    elseif counts[CP.CL_ERR]!=Int32(0)
        state[NL.DN_ERROR]=NL.DN_COORDINATE_ERROR
        state[NL.DN_VALID]=0
    end
    return nothing
end
function _control_valid_kernel!(state)
    @inbounds begin state[NL.DN_ERROR]=0; state[NL.DN_VALID]=1; state[NL.DN_REQUEST]=0 end
    return nothing
end
"""Mark the control block valid (construction, recovery): cluster runs build no ELL list."""
control_valid!(ws)=(@cuda threads=1 blocks=1 _control_valid_kernel!(ws.state); nothing)

"""Short- and long-range forces at the current coordinates, eager (construction,
evaluation outside `step!`): the device decision is read back, the outer list grows
here if it overflowed. Ends with forces (energies, virials) in the run's buffers."""
function cluster_forces!(run,b::ClusterBackend;energy::Bool=false)
    eng=b.eng; L=eng.list; r=run.r
    CP.enqueue_decide!(eng,r...;fixed=false)   # outside the step schedule: exact triggers
    c=Array(L.counts)
    if c[CP.CL_BUILD]!=0
        CP.enqueue_build!(eng,r...)
        while CP.ensure_capacity!(L)
            CP.enqueue_build!(eng,r...)
        end
    end
    CP._pack!(eng,r...,b.q,b.typ)
    if c[CP.CL_PRUNE]!=0
        CP.enqueue_prune!(eng,r...,energy;mark=true)   # the next eager decision measures from here
    end
    # A distance-only prune updates the working list but leaves fc/Ec/Vc untouched.
    # As in the graph suffix, compute pairs on that list before scattering them.
    if c[CP.CL_PRUNE]==0 || L.prune_only
        CP.enqueue_pairs!(eng,energy,Val(false))
    end
    CP.enqueue_scatter!(eng,r...,b.q,run.f...,run.energy,run.virial,energy)
    if b.solver!==nothing
        order=CP.mesh_order(eng)
        ES.mesh_transform!(b.solver,r...,b.q,energy,order;geom=run.neighbors.geom)
        ES.mesh_gather!(b.solver,r...,b.q,run.f...,run.energy,b.Vglobal,energy,order;geom=run.neighbors.geom)
        copyto!(b.order,order)
    end
    return nothing
end

"""Graph parts of one step: `(sections, suffix, parallel, join, parallel_after)` for
`NL.flag_conditional_graph`. One force evaluation per step, or two for Euler–Heun
(midpoint force, then the final one). Each evaluation is a cluster rebuild section and
a prune section; with collision counting an ELL contact-row check precedes the
cluster decision after every physical move (two for the split BAOAB family, which
samples after A1). Rebuild sections are numbered in step order on the device. The
final evaluation's mesh runs as a parallel node after the last move; the midpoint mesh
runs in sequence. With `resume = k` the step restarts at rebuild section `k` (capacity
recovery: the moves before it already ran) — a cluster section from its decision, an
ELL section from its re-armed request."""
function cluster_step_parts(run,b::ClusterBackend;energy::Bool,resume::Integer=0,event::Bool=false,virial::Bool=false,
                            fast::Bool=false)
    eng=b.eng; L=eng.list; ws=run.neighbors; r=run.r; th=run.thermostat
    heun=th isa BrownianData && th.midpoint
    # Collision counting keeps ELL contact rows in the control block: an ELL check
    # after every physical move (and after A1 when the BAOAB family samples there).
    keep=any(needs_neighbors,run.features)
    split=keep && th isa LangevinData && _split_baoa(run,th)
    # host-scheduled fixed intervals: decisions follow the device step counter, which the host
    # mirrors, so between scheduled steps it launches `fast` graphs without any decision
    hs=host_scheduled(run)
    fold=FOLD_TAIL[]
    (fast && !hs) && throw(ArgumentError("fast cluster steps need host-scheduled fixed list intervals"))
    hstep=hs ? (ws.state,NL.DN_STEP) : nothing
    decide=() -> CP.enqueue_decide!(eng,r...;step=hstep)
    # fast path (W1 reduced the displacements): only the decision, except after a pressure
    # event (positions scaled after W1) and on recovery resumes, which re-measure
    dstep=rw_fast_path(run) && !event ? () -> CP.enqueue_decide_only!(eng;step=hstep) : decide
    fast && (dstep=() -> nothing)
    edecide=() -> decide_policy!(ws,r,Val(run.rebuild_policy))
    build(k)=() -> begin
        NL.mark_section!(ws,k)
        CP.enqueue_build!(eng,r...)
        @cuda threads=1 blocks=1 _cluster_error_kernel!(ws.state,L.counts)
        features_after_rebuild!(run)
    end
    ebuild(k)=() -> (NL.mark_section!(ws,k); enqueue_rebuild!(run;reorder=false))
    pack(sample::Bool)=() -> begin
        sample && features_position_sample!(run)   # physical positions only (not the Heun midpoint)
        CP._pack!(eng,r...,b.q,b.typ)
    end
    # virial: a barostat sampling step accumulates only the virial trace Σ r·f of the
    # pairs (one register per thread) instead of energies and per-particle virial tensors
    prune(e::Bool)=() -> CP.enqueue_prune!(eng,r...,e;trace=virial)
    # GATHER_EARLY: the final mesh gather writes the forces in the parallel node, beside the
    # pair pass; the pair scatter (in the join) adds to them
    early=GATHER_EARLY[] && b.solver!==nothing && ES.gather_sets(b.solver)
    scatter(e::Bool;add::Bool=false)=() -> CP.enqueue_scatter!(eng,r...,b.q,run.f...,run.energy,run.virial,e;trace=virial,add)
    pairs(e::Bool;scattered::Bool=true)=() -> begin
        CP.enqueue_pairs!(eng,e,Val(false);skip=(L.prune_only || fast) ? 0 : CP.CL_PRUNE,trace=virial)
        scattered && scatter(e)()
    end
    # The mesh traverses its own copy of the cluster order (a rebuild may rewrite the
    # order while the parallel mesh runs; any permutation gives the same result),
    # refreshed after the final gather.
    geom=run.neighbors.geom   # the device box
    # the mesh virial of a trace step: the solve's energy/virial reduction, stored as the global virial
    gather(e::Bool;set::Bool=false)=() -> (b.solver===nothing || begin
        ES.mesh_gather!(b.solver,r...,b.q,run.f...,run.energy,b.Vglobal,e,b.order;geom,set)
        (virial && !e) && ES._store_global_virial!(b.Vglobal,b.solver.ws.acc,b.solver.ubg)
    end; nothing)
    transform(e::Bool)=() -> ES.mesh_transform!(b.solver,r...,b.q,e || virial,b.order;geom)
    mesh(e::Bool;final::Bool=false)=b.solver===nothing ? nothing :
        (final && early) ? () -> (transform(e)(); gather(e;set=true)()) : transform(e)
    # Rebuild sections are numbered k = 1, 2, … in step order (marked on the device), so
    # capacity recovery resumes at the failed one: a cluster section from its decision,
    # an ELL section from its re-armed request.
    sections=Any[]; marks=Int[]; resumes=Any[]
    function rebuild!(prefix,kind)
        n=count(!iszero,marks)+1
        if kind===:cluster   # fast steps: the prefix alone (no decision, no conditional body)
            push!(sections,(prefix,L.counts,CP.CL_BUILD,fast ? nothing : build(n))); push!(resumes,decide)
        else
            push!(sections,(prefix,ws.state,NL.DN_REQUEST,ebuild(n))); push!(resumes,() -> nothing)
        end
        push!(marks,n)
    end
    pruning!(prefix,e)=(push!(sections,(prefix,L.counts,CP.CL_PRUNE,fast ? nothing : prune(e))); push!(marks,0); push!(resumes,nothing))
    # RW_BATH_SIDE: the rigid-water CSVR bath as a side branch beside the step's force work (not on
    # pressure-event steps, whose cell scaling follows the bath, nor in recovery graphs)
    sidebath=RW_BATH_SIDE[] && rw_fast_path(run) && th isa CSVRData && !event && !split && !keep && !heun && resume==0
    event && (split || keep) && throw(ArgumentError(
        "Device pressure control does not support collision counting or the split BAOA family"))
    if split
        # B·A, ELL check | sample, O·A, ELL check | cluster decision (staged collisions)
        rebuild!(() -> (enqueue_ba!(run,th); features_after_position!(run); edecide()),:ell)
        rebuild!(() -> (features_position_sample!(run); enqueue_oa!(run,th); features_after_position!(run); edecide()),:ell)
        rebuild!(decide,:cluster)
    elseif keep && !heun
        rebuild!(() -> (enqueue_drift!(run); edecide()),:ell)
        rebuild!(decide,:cluster)
    else
        # a pressure event scales the cell after the move and its constraints, before
        # the list decision (it requests a rebuild) and the forces of the step
        event && (heun && throw(ArgumentError("Device pressure control does not support Euler–Heun runs")))
        rebuild!(() -> (enqueue_drift!(run;csvr=!sidebath); event && enqueue_crescale!(run,b); dstep()),:cluster)   # Heun: the midpoint move
    end
    pruning!(pack(!heun),heun ? false : energy)
    if heun
        midpoint_force=() -> begin
            pairs(false)()
            m=mesh(false); m===nothing || m()
            gather(false)()
            features_after_force!(run;energy=false)
        end
        final_move=() -> (enqueue_bd_step!(run,th); features_after_position!(run))
        if keep
            rebuild!(() -> (midpoint_force(); final_move(); edecide()),:ell)
            rebuild!(decide,:cluster)
        else
            rebuild!(() -> (midpoint_force(); final_move(); decide()),:cluster)
        end
        pruning!(pack(true),energy)
    end
    after=length(sections)-1                     # the final evaluation's cluster rebuild section
    if resume>0
        i=findfirst(==(resume),marks)
        i===nothing && error("Device engine recovery: no rebuild section $resume in this step")
        sections=[(resumes[i],sections[i][2],sections[i][3],sections[i][4]),sections[i+1:end]...]
        after-=i-1
    end
    join=() -> begin
        early ? scatter(energy;add=true)() : gather(energy)()
        # the mesh's copy of the cluster order changes only at a build: not on fast steps (FOLD_TAIL)
        (b.solver===nothing || (fold && fast)) || CP.kcopy!(b.order,CP.mesh_order(eng))
        features_after_force!(run;energy)
        energy && enqueue_virial_sample!(run,b)   # device pressure control: virial for the next event
        virial && !energy && enqueue_virial_trace_sample!(run,b)
        # FOLD_TAIL: the kick's last kernel advances the step counter when no after-kick work follows
        folded=enqueue_kick!(run;advance=fold && quiet_after_kick(run,energy),
            report=energy || !csvr_lazy_report(run))
        features_after_kick!(run;energy)
        folded || NL.advance_device_neighbor_step!(ws)
        if energy
            enqueue_observables!(run)
            run.sample_hook!==nothing && run.sample_hook()
        end
    end
    side=sidebath ? (() -> enqueue_rw_bath!(run)) : nothing
    return sections,pairs(energy;scattered=!early),mesh(energy;final=true),join,after,side
end

"""Fewer tail kernels in cluster steps: the kick's last kernel advances the device step counter when
no after-kick work follows (`quiet_after_kick`), and fast steps skip the copy of the cluster order for
the mesh (unchanged without a build). Five paired runs with C1 lists: salt 10.6k NVE 1.014×, NVT10 1.012×."""
const FOLD_TAIL=Ref(true)

# Keep per-particle Ekin current, but reduce full-step bath totals only on sampled
# steps and host boundaries. Capture this choice in Run: later switch changes must
# not make boundary reporting disagree with the already captured step graphs.
# Pressure control, constraints and other feature hooks retain eager reporting.
const CSVR_LAZY_REPORT=Ref(true)
_csvr_lazy_report(cluster,thermostat,features)=CSVR_LAZY_REPORT[] && cluster!==nothing &&
    cluster.barostat===nothing && thermostat isa CSVRData &&
    all(f->f isa MassFeature || f isa UnwrapFeature,features)
csvr_lazy_report(run)=run.csvr_lazy_report

"""The mesh branch of cluster step graphs at high stream priority: its blocks go ahead of the pair
kernel's pending ones. `true` forces it in every step graph. Otherwise, with `MESH_PRIORITY_TUNE`,
host-scheduled runs with a mesh time both variants of the fast step graph on their first fast steps
(`MeshTune`) and keep the faster: it pays when the mesh chain is the critical path (salt 10.6k, five
paired runs: NVE 1.078×, NVT10 1.080×) and costs when the pair kernel is (10k water: 0.968×)."""
const MESH_PRIORITY=Ref(false)
const MESH_PRIORITY_TUNE=Ref(true)
# no after-kick kernels: no features, or only the rigid-water fast path's (none outside energy steps)
quiet_after_kick(run,energy::Bool)=all(f->f isa RigidWaterFeature && rw_fast_path(run) && !energy,run.features)

"""Run the mesh transform as a parallel graph node (default) or in sequence before the
pair pass (diagnostics: `MESH_PARALLEL[] = false`)."""
const MESH_PARALLEL=Ref(true)
"""Host-scheduled fixed list intervals: with `list_intervals` (nstlist a multiple of nstprune), the
rebuild/prune schedule follows the device step counter, and the host launches a `fast` graph without
list decisions or conditional nodes on the steps between scheduled ones. Five paired runs, C1 lists:
salt 10.6k NVE 1.10×, NVT10 1.10×, 10k water NPT 1.03× faster."""
const HOST_SCHEDULE=Ref(true)
host_scheduled(run)=HOST_SCHEDULE[] && run.cluster!==nothing && (L=run.cluster.eng.list; L.nstlist>0 && L.nstprune>0 &&
    L.nstlist%L.nstprune==0) && !any(needs_neighbors,run.features) && !(run.thermostat isa BrownianData && run.thermostat.midpoint)
"""Whether a launch at device step `device_step` of a host-scheduled run is a scheduled (decision) step."""
scheduled_step(run,device_step::Integer)=device_step%run.cluster.eng.list.nstprune==0
"""The final mesh gather runs in the parallel mesh node, beside the pair pass, writing the mesh
forces; the pair scatter in the join adds to them (`early` in `cluster_step_parts`; 16-lane gather
only, `ES.gather_sets`). `false` restores the gather after the scatter. 10k water NPT, five paired
runs: 1.028× faster."""
const GATHER_EARLY=Ref(true)
"""The rigid-water CSVR bath update as a side branch beside the list decision, mesh and pair
force (`enqueue_rw_bath!`): only the final kick reads its scale. `false` keeps it in the drift.
10k water NPT, five paired runs: 1.02× faster."""
const RW_BATH_SIDE=Ref(true)

function cluster_step_graph(run,b::ClusterBackend;energy::Bool,resume::Integer=0,event::Bool=false,virial::Bool=false,
                            fast::Bool=false,high::Bool=MESH_PRIORITY[])
    sections,suffix,parallel,join,after,side=cluster_step_parts(run,b;energy,resume,event,virial,fast)
    if parallel!==nothing && !MESH_PARALLEL[]
        mesh=parallel; pairs=suffix
        return NL.flag_conditional_graph(sections,() -> (mesh(); pairs());join,parallel_after=after,side)
    end
    return NL.flag_conditional_graph(sections,suffix;parallel,join,parallel_after=after,side,parallel_high=high)
end
