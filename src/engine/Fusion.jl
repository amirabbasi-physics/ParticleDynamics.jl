# Included in ParticleDynamics.DeviceEngine. Per-step control-kernel fusion for the
# displacement-only policy and ELL storage. A regular step becomes:
#   prefix: kick-drift + displacement + decision + graph condition (one kernel)
#   [conditional rebuild body, unchanged]
#   suffix: cooperative force + final kick + step counter (one kernel)
# Arithmetic, checks and decisions are identical to the unfused kernels.

struct FusionWorkspace{U}
    maxbits::CuArray{U,1} # running maximum of displacement² as float bits
    ticket::CuArray{Int32,1}
    ticket2::CuArray{Int32,1} # last-block ticket of the fused force/kick (CSVR K report)
    draws::CuArray{Float64,1} # CSVR draws of the step (r1, r2, step, dof), precomputed in the prefix
    kpart::CuArray{Float64,1} # CSVR K of the completed step, one partial per block of the fused kick
end
# kpart: one slot per block of the widest fused kick (≤ 32 lanes per particle, 128 threads).
# Per-block slots instead of same-address Float64 atomics and a last-block ticket: those
# serialize (salt 10k, 1331 blocks: +8 µs per step) and summed in arbitrary order.
_kpart(N)=CUDA.zeros(Float64,cld(32*max(N,1),128))
FusionWorkspace(::Type{Float32},N::Integer=1)=FusionWorkspace{UInt32}(CUDA.zeros(UInt32,1),CUDA.zeros(Int32,1),CUDA.zeros(Int32,1),CUDA.fill(-1.0,5),_kpart(N))
FusionWorkspace(::Type{Float64},N::Integer=1)=FusionWorkspace{UInt64}(CUDA.zeros(UInt64,1),CUDA.zeros(Int32,1),CUDA.zeros(Int32,1),CUDA.fill(-1.0,5),_kpart(N))

# Sum of the fused kick's block partials into the bath's reported K (one block).
function _fused_k_report_kernel!(kpart,nblocks::Int32,kinetic)
    x=0.0; k=Int32(threadIdx().x)
    while k<=nblocks
        x+=@inbounds kpart[k]; k+=Int32(blockDim().x)
    end
    sh=CUDA.CuStaticSharedArray(Float64,32)
    x=_fused_block_sum(x,sh)
    threadIdx().x==1 && nblocks>0 && (@inbounds kinetic[1]=eltype(kinetic)(x))
    return nothing
end
function enqueue_fused_k_report!(run,nblocks::Integer;launch::Bool=true)
    th=run.thermostat
    th isa CSVRData || return nothing
    k=@cuda launch=false NL._device_if_valid!(run.neighbors.state,_fused_k_report_kernel!,run.fusion.kpart,Int32(nblocks),th.kinetic)
    launch && k(run.neighbors.state,_fused_k_report_kernel!,run.fusion.kpart,Int32(nblocks),th.kinetic;threads=256,blocks=1)
    return nothing
end

const FUSED_PREFIX_THREADS=256 # NL._displacement_block_max assumes 256

@inline _float_bits(x::Float32)=reinterpret(UInt32,x)
@inline _float_bits(x::Float64)=reinterpret(UInt64,x)
@inline _bits_float(::Type{Float32},u)=reinterpret(Float32,u)
@inline _bits_float(::Type{Float64},u)=reinterpret(Float64,u)

@inline function _set_condition(handle::UInt64,value::Bool,::Val{SET}) where {SET}
    if SET
        ccall("extern cudaGraphSetConditional",llvmcall,Cvoid,(UInt64,UInt32),handle,UInt32(value))
    end
    return nothing
end

# Block sum of one Float64 per thread (256 threads): warp shuffles, then lane 0 of each warp.
@inline function _fused_block_sum(x::Float64,sh)
    for off in (16,8,4,2,1)
        x+=CUDA.shfl_down_sync(0xffffffff,x,off)
    end
    lane=(threadIdx().x-1)&31; w=(threadIdx().x-1)>>5
    lane==0 && (@inbounds sh[w+1]=x)
    sync_threads()
    if threadIdx().x==1
        s=0.0
        for k in 1:(blockDim().x+31)>>5; s+=@inbounds sh[k]; end
        @inbounds sh[1]=s
    end
    sync_threads()
    s=@inbounds sh[1]
    sync_threads()
    return s
end

# CSVR (single bath) in the fused prefix: the half-step K of v(t+h/2) reduced per block, and
# the Bussi update in the last block (`csvr` = (kacc, nacc, target, tau, kinetic, exchange,
# scale, dof, seed, dt, cdof, mass) or nothing).
function _fused_prefix_kernel!(state,handle::UInt64,rx,ry,rz,vx,vy,vz,fx,fy,fz,
                               refx,refy,refz,c::T,dt::T,box,
                               maxbits,ticket,threshold::T,::Val{SET},csvr=nothing,feats=()) where {T,SET}
    box=NL._boxlen(box,Val(3))   # host tuple or device geometry
    @inbounds state[NL.DN_ERROR]!=0 && return nothing # uniform: nothing runs after a latched error
    valid=NL._device_list_valid(state)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    value=zero(T); ek=0.0
    if i<=length(rx)
        @inbounds begin
            if valid # identical to SC._nve_kick_drift3_kernel!
                ci=kick_coef(feats,c,dt,c,i)   # c, or dt/(2m_i) with per-particle masses (charged runs)
                ux=muladd(ci,fx[i],vx[i]); uy=muladd(ci,fy[i],vy[i]); uz=muladd(ci,fz[i],vz[i])
                vx[i]=ux; vy[i]=uy; vz[i]=uz
                csvr===nothing || (ek=0.5*Float64(mass_of(feats,csvr[12],i))*Float64(muladd(ux,ux,muladd(uy,uy,uz*uz))))
                x=muladd(dt,ux,rx[i]); y=muladd(dt,uy,ry[i]); z=muladd(dt,uz,rz[i])
                Lx,Ly,Lz=box
                x-=(x>=T(0.5)*Lx)*Lx; x+=(x< -T(0.5)*Lx)*Lx
                y-=(y>=T(0.5)*Ly)*Ly; y+=(y< -T(0.5)*Ly)*Ly
                z-=(z>=T(0.5)*Lz)*Lz; z+=(z< -T(0.5)*Lz)*Lz
                rx[i]=x; ry[i]=y; rz[i]=z
            end
            # identical to NL._device_displacement_kernel! (z, y, x nesting)
            coords=(rx[i],ry[i],rz[i])
            for d in 3:-1:1
                x=coords[d]; h=T(0.5)*box[d]
                if !(isfinite(x) && -h<=x<=h)
                    value=T(Inf)
                elseif valid
                    ref=d==3 ? refz[i] : d==2 ? refy[i] : refx[i]
                    dx=NL.mic_fast(x-ref,h,box[d])
                    value=muladd(dx,dx,value)
                end
            end
        end
    end
    largest=NL._displacement_block_max(value) # nonnegative or +Inf: bit order = numeric order
    last=CUDA.CuStaticSharedArray(Bool,1)
    if csvr!==nothing
        sh=CUDA.CuStaticSharedArray(Float64,32)
        ks=_fused_block_sum(ek,sh)
        nb=min(Int(blockDim().x),length(rx)-Int(blockIdx().x-1)*Int(blockDim().x))
        if threadIdx().x==1
            CUDA.@atomic csvr[1][1]+=ks
            CUDA.@atomic csvr[2][1]+=Int32(nb)
        end
    end
    if threadIdx().x==1
        CUDA.atomic_max!(pointer(maxbits),_float_bits(largest))
        CUDA.threadfence()
        @inbounds last[1]=CUDA.atomic_add!(pointer(ticket),Int32(1))==Int32(gridDim().x-1)
    end
    sync_threads()
    if threadIdx().x==1 && @inbounds(last[1])
        largest=_bits_float(T,CUDA.atomic_xchg!(pointer(maxbits),zero(eltype(maxbits))))
        @inbounds begin
            ticket[1]=Int32(0)
            # identical to DeviceArgonPolicy._displacement_decide_kernel!
            state[NL.DN_CHECKS]+=1
            state[NL.DN_REQUEST]=0
            diagnostic=state[NL.DN_DIAGNOSTICS]!=0
            diagnostic && (state[NL.DN_REASON]=0)
            if !isfinite(largest)
                state[NL.DN_ERROR]=NL.DN_COORDINATE_ERROR
                state[NL.DN_VALID]=0
            elseif state[NL.DN_VALID]==0
                state[NL.DN_REQUEST]=1
                diagnostic && (state[NL.DN_REASON]=NL.DN_REASON_INVALID)
            elseif largest>threshold
                state[NL.DN_REQUEST]=1
                state[NL.DN_VALID]=0
                diagnostic && (state[NL.DN_REASON]=NL.DN_REASON_DISPLACEMENT)
            end
            _set_condition(handle,NL._device_rebuild_requested(state),Val(SET))
            if csvr!==nothing   # every block's K is in: the Bussi update (bath 1)
                kacc,nacc,target,tau,kinetic,exchange,scale,dof,seed,dtA,cdof,_,dr,iv=csvr
                _csvr_bath_update!(1,kacc,nacc,target,tau,kinetic,exchange,scale,dof,state,seed,dtA,3.0,cdof,dr,iv)
                @inbounds dr[5]=Float64(state[NL.DN_STEP]+1)   # the side kernel draws for the next step
            end
        end
    end
    return nothing
end

# Next step's Bussi draws, one thread, in a graph branch parallel to the rebuild and the force
# kernel (they need only the step, the seed and the dof), so the prefix's last block only does
# the few flops of the update.
function _csvr_draws_kernel!(dr,seed::UInt64,cdof,n::Int)
    @inbounds begin
        dr[5]<0 && return nothing
        step=UInt64(dr[5]); ndof=max(1,floor(Int,3.0*n-cdof[1]+0.5))
        r1,r2=_csvr_draws(seed,step,1,ndof)
        dr[1]=r1; dr[2]=r2; dr[3]=Float64(step); dr[4]=Float64(ndof)
    end
    return nothing
end
enqueue_csvr_draws!(run;launch::Bool=true)=(th=run.thermostat; o=run.fusion;
    k=@cuda launch=false _csvr_draws_kernel!(o.draws,th.seed,th.cdof,length(run.r[1]));
    launch && k(o.draws,th.seed,th.cdof,length(run.r[1]);threads=1,blocks=1); nothing)

# charged (mesh) runs: CSVR in its own branch (`fused_mesh_graph`), not in the prefix
_fused_csvr(run)=(run.thermostat isa CSVRData && run.mesh===nothing) ? (th=run.thermostat; T=eltype(run.r[1]);
    (th.kacc,th.nacc,th.target,th.tau,th.kinetic,th.exchange,th.scale,th.dof,th.seed,Float64(run.dt),th.cdof,run.mass,
     run.fusion.draws,th.interval)) : nothing

# `force`: a negative threshold, so every step requests a rebuild (the host-scheduled sort step)
function enqueue_fused_prefix!(run,handle::UInt64,::Val{SET};force::Bool=false) where {SET}
    ws=run.neighbors; nb=ws.list; T=eltype(run.r[1]); N=length(run.r[1])
    o=run.fusion::FusionWorkspace
    @cuda threads=FUSED_PREFIX_THREADS blocks=cld(N,FUSED_PREFIX_THREADS) _fused_prefix_kernel!(
        ws.state,handle,run.r...,run.v...,run.f...,nb.rref_x,nb.rref_y,nb.rref_z::CuArray{T,1},
        (run.dt/T(2))/run.mass,run.dt,ws.geom,o.maxbits,o.ticket,force ? -one(T) : T(0.25)*nb.skin*nb.skin,Val(SET),
        _fused_csvr(run),run.fx)
    return nothing
end

function compile_fused_prefix!(run,handle::UInt64,::Val{SET}) where {SET}
    ws=run.neighbors; nb=ws.list; T=eltype(run.r[1]); o=run.fusion::FusionWorkspace
    @cuda launch=false _fused_prefix_kernel!(ws.state,handle,run.r...,run.v...,run.f...,
        nb.rref_x,nb.rref_y,nb.rref_z::CuArray{T,1},(run.dt/T(2))/run.mass,run.dt,ws.geom,
        o.maxbits,o.ticket,T(0.25)*nb.skin*nb.skin,Val(SET),_fused_csvr(run),run.fx)
    return nothing
end

"""Cooperative force + final kick + step counter (one kernel)."""
# The fused kick's CSVR part: (scale, Ekin, block partials of K, nfull, kinetic, ticket, mass) or nothing.
_fused_kick_csvr(run)=run.thermostat isa CSVRData ? (th=run.thermostat;
    (th.scale,th.Ekin,run.fusion.kpart,th.nfull,th.kinetic,run.fusion.ticket2,run.mass)) : nothing

function enqueue_fused_force_kick!(run,o::TraversalWorkspace{G,:ell},::Val{E}) where {G,E}
    ws=run.neighbors; nb=ws.list; T=eltype(run.r[1]); N=length(run.r[1])
    kick=(run.v,(run.dt/T(2))/run.mass,ws.state,_fused_kick_csvr(run))
    force=E ? _pair_cooperative_observable_kernel_kick! : _pair_cooperative_noE_kernel_kick!
    @cuda threads=128 blocks=cld(N*G,128) NL._device_if_valid!(ws.state,force,kick,
        run.r,run.f,run.energy,run.virial,nb.neighbors_flat,nb.counts,o.slot_groups,
        ws.geom,run.model,run.model_data,run.fx,Val(G),Val(:ell))
    E && enqueue_fused_k_report!(run,cld(N*G,128))
    return nothing
end

# Fused charged step (ELL, 15 Oct 2026): the pair force (no kick), the mesh transform, then
# one kernel gathers the mesh force onto the pair force and kicks (`_gather_kick_kernel!`).
function _gather_kick_kernel!(f::NTuple{3},Epot,r::NTuple{3},q,order,grid,box,K1::Int32,K2::Int32,K3::Int32,
                              stencil,bg::T,::Val{E},kick,feats,dt::T) where {T,E}
    t,lane,i,qi,phi,gx,gy,gz,Lx,Ly,Lz=ES._gather_ad_lines_sums(r...,q,order,grid,box,K1,K2,K3,stencil)
    N=length(order)
    gx=ES._sum16(gx); gy=ES._sum16(gy); gz=ES._sum16(gz)
    E && (phi=ES._sum16(phi))
    active=lane==Int32(0) && t<=N
    if active
        @inbounds begin
            a=(f[1][i]-qi*gx*(T(K1)/Lx),f[2][i]-qi*gy*(T(K2)/Ly),f[3][i]-qi*gz*(T(K3)/Lz))
            _tstore!(f,i,a)
            E && (Epot[i]+=T(0.5)*qi*phi+bg*qi)
            v,c,_,cs=kick
            if cs===nothing
                _tkick!(v,i,kick_coef(feats,c,dt,c,i),a)
            else   # CSVR: v(t+h) = s·v(t+h/2) + h/2m·f, Ekin of the completed step (as _csvr_kick_measure_kernel!)
                u=_tskick!(v,i,thermo_kick_coef(feats,c,cs[7],i),T(cs[1][1]),a)
                cs[2][i]=T(0.5)*mass_of(feats,cs[7],i)*_r2(u)
            end
        end
    end
    _fused_kick_report!(kick,active,i,N,Val(ES.MESH_LANES),Val(E))
    return nothing
end

function _gather_kick_args(run,energy::Bool)
    m=run.mesh; s=m.solver; ws=s.ws; T=eltype(run.r[1])
    K1,K2,K3=Int32.(ws.mesh)
    kick=(run.v,(run.dt/T(2))/run.mass,run.neighbors.state,_fused_kick_csvr(run))
    return (run.f,run.energy,run.r,m.q,m.order,ws.grid_flat,run.neighbors.geom,K1,K2,K3,s.stencil,s.bg,
        Val(energy),kick,run.fx,run.dt)
end

function enqueue_gather_kick!(run;energy::Bool=false,launch::Bool=true)
    a=_gather_kick_args(run,energy); N=length(run.r[1]); state=run.neighbors.state
    k=@cuda launch=false NL._device_if_valid!(state,_gather_kick_kernel!,a...)
    launch || return nothing
    k(state,_gather_kick_kernel!,a...;threads=128,blocks=cld(N*ES.MESH_LANES,128))
    energy && enqueue_fused_k_report!(run,cld(N*ES.MESH_LANES,128))
    energy && ES._store_global_virial!(run.mesh.Vglobal,run.mesh.solver.ws.acc,run.mesh.solver.ubg)
    return nothing
end

const MESH_OVERLAP=Ref(true)   # charged fused steps: FFTs parallel to the pair force
const PACKED_XQ=Ref(true)      # charged fused steps: packed (x, y, z, q) for the pair kernel
# Charged fused steps of small systems: the FFT branch captured at high priority, so its blocks
# go ahead of the pair kernel's remaining ones instead of trailing its single wave
# (salt 10k, ms/step: ESP 0.0976 → 0.0915, PME 0.1085 → 0.1029; at 97k neutral to −2%: off).
const FFT_PRIORITY=Ref(true)
const FFT_PRIORITY_MAX_N=30_000
_fft_priority(run)=FFT_PRIORITY[] && length(run.r[1])<=FFT_PRIORITY_MAX_N
# Graph instantiation flags of step graphs: node priorities honored (CUDA ≥ 11.7)
GRAPH_FLAGS()=FFT_PRIORITY[] ? 8 : 0   # CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
_fused_mesh_forces!(run;energy::Bool=false)=(m=run.mesh; enqueue_cooperative_force!(run,run.traversal,Val(energy));
    ES.mesh_transform!(m.solver,run.r...,m.q,energy,m.order;geom=run.neighbors.geom); nothing)

# `csvr=false`: the CSVR half-step update runs elsewhere (a parallel branch of the step graph)
function fused_suffix!(run;energy::Bool=false,csvr::Bool=true)
    if run.mesh!==nothing
        _fused_mesh_forces!(run;energy)
        csvr && run.thermostat isa CSVRData && enqueue_csvr_half!(run,run.thermostat)
        enqueue_gather_kick!(run;energy)
    else
        enqueue_fused_force_kick!(run,run.traversal,Val(energy))
    end
    return _fused_after_kick!(run;energy)
end

fused_suffix_tail!(run;energy::Bool=false)=(enqueue_gather_kick!(run;energy); _fused_after_kick!(run;energy))

function _fused_after_kick!(run;energy::Bool=false)
    energy && run.barostat!==nothing && enqueue_virial_sample!(run,_baro_ctx(run))   # device pressure control
    energy && enqueue_observables!(run)
    energy && run.sample_hook!==nothing && run.sample_hook()
    return nothing
end

"""Host-scheduled sorting of fused ELL runs: the regular step graphs rebuild without sorting, so
nothing in them permutes storage and the mesh spread and the CSVR half-step run beside the list
check and rebuild instead of after them; every `reorder_interval` steps the host launches the sort
step (`StateEngine.sort`), whose rebuild is forced and sorts. Five paired runs, exact lists:
10.6k molten NaCl NVT 1.016×, Argon 10k NVE 1.024×."""
const HOST_SORT=Ref(true)
host_sorted(run)=HOST_SORT[] && run.fusion!==nothing && run.ordering!==nothing && !_host_forces(run)

"""Conditional graph whose prefix kernel sets the rebuild condition itself. `sort`: the
host-scheduled sort step (`HOST_SORT`), a forced rebuild that sorts."""
function fused_conditional_graph(run;energy::Bool=false,sort::Bool=false)
    CC=NL.ConditionalCUDA
    free=host_sorted(run) && !sort   # the rebuild does not sort: branches need not wait for it
    graph=CUDA.CuGraph()
    handle=Ref{CC.CUgraphConditionalHandle}()
    CC.cuGraphConditionalHandleCreate(handle,graph,graph.ctx,UInt32(0),CC.CU_GRAPH_COND_ASSIGN_DEFAULT)
    compile_fused_prefix!(run,UInt64(handle[]),Val(true))
    CUDA.synchronize()
    before=NL.capture_graph(()->enqueue_fused_prefix!(run,UInt64(handle[]),Val(true);force=sort))
    first=NL._add_child_graph!(graph,before,CC.CUgraphNode[])
    conditional,body=NL._add_conditional_graph!(graph,handle[],first)
    inside=NL.capture_graph(()->sort ? enqueue_rebuild!(run;always=true) : enqueue_rebuild!(run;reorder=!free))
    branch=free ? first : conditional   # what the mesh spread and the CSVR half-step follow
    NL._add_child_graph!(body,inside,CC.CUgraphNode[])
    if run.mesh!==nothing && run.thermostat isa CSVRData
        # charged: the CSVR update (after the rebuild, which may sort) parallel to the pair force
        # and the mesh transform; the gather-kick joins both
        th=run.thermostat
        m=run.mesh; geom=run.neighbors.geom
        if MESH_OVERLAP[]   # spread, then the pair force parallel to the transforms (small kernels)
            # the spread also writes the packed (x, y, z, q) the pair kernel loads (`PACKED_XQ`)
            xq=(PACKED_XQ[] && ES.MESH_LINES[] && length(run.r)==3) ? m.xq : nothing
            # (a grid clear parallel to the list check measured slower: 0.0915 → 0.0934 ms/step at 10k)
            spread=NL._add_child_graph!(graph,NL.capture_graph(()->ES.mesh_transform!(m.solver,run.r...,m.q,energy,m.sorder;
                geom,part=:spread,xq)),CC.CUgraphNode[branch])
            pfx=xq===nothing ? run.fx : (run.fx...,(xq=xq,))
            pairs=NL._add_child_graph!(graph,NL.capture_graph(()->enqueue_cooperative_force!(run,run.traversal,Val(energy);fx=pfx)),
                free ? CC.CUgraphNode[spread,conditional] : CC.CUgraphNode[spread])
            # high priority: the FFT blocks go ahead of the pair kernel's remaining blocks
            fft=NL._add_child_graph!(graph,NL.capture_graph(()->ES.mesh_transform!(m.solver,run.r...,m.q,energy,m.order;
                geom,part=:fft);high=_fft_priority(run)),CC.CUgraphNode[spread])
            forces=CC.CUgraphNode[pairs,fft]
        else
            forces=CC.CUgraphNode[NL._add_child_graph!(graph,NL.capture_graph(()->_fused_mesh_forces!(run;energy)),
                CC.CUgraphNode[conditional])]
        end
        bath=NL._add_child_graph!(graph,NL.capture_graph(()->enqueue_csvr_half!(run,th)),CC.CUgraphNode[branch])
        NL._add_child_graph!(graph,NL.capture_graph(()->fused_suffix_tail!(run;energy)),CC.CUgraphNode[forces...,bath])
        return graph
    end
    after=NL.capture_graph(()->fused_suffix!(run;energy))
    NL._add_child_graph!(graph,after,CC.CUgraphNode[conditional])
    if run.thermostat isa CSVRData   # next step's Bussi draws, parallel to the rebuild and the force kernel
        enqueue_csvr_draws!(run;launch=false); CUDA.synchronize()
        NL._add_child_graph!(graph,NL.capture_graph(()->enqueue_csvr_draws!(run)),CC.CUgraphNode[first])
    end
    return graph
end

"""The host-scheduled sort step of a fused ELL run (`HOST_SORT`)."""
capture_sort_graph(run)=capture_fused_step(run;conditional=true,energy=false,sort=true)

function compile_fused_suffix!(run;energy::Bool=false)
    enqueue_fused_k_report!(run,1;launch=false)
    if run.mesh!==nothing
        enqueue_gather_kick!(run;energy,launch=false)
        run.thermostat isa CSVRData && enqueue_csvr_half!(run,run.thermostat;launch=false)
        return nothing
    end
    o=run.traversal; ws=run.neighbors; nb=ws.list; T=eltype(run.r[1])
    G=typeof(o).parameters[1]
    kick=(run.v,(run.dt/T(2))/run.mass,ws.state,_fused_kick_csvr(run))
    force=energy ? _pair_cooperative_observable_kernel_kick! : _pair_cooperative_noE_kernel_kick!
    @cuda launch=false NL._device_if_valid!(ws.state,force,kick,run.r,run.f,run.energy,
        run.virial,nb.neighbors_flat,nb.counts,o.slot_groups,ws.geom,run.model,run.model_data,run.fx,
        Val(G),Val(:ell))
    energy && compile_observables!(run)
    return nothing
end

"""Capture a fused step (no execution). Conditional graphs set their own condition."""
function capture_fused_step(run;conditional::Bool=true,energy::Bool=false,sort::Bool=false)
    compile_fused_suffix!(run;energy)
    if conditional
        return CUDA.instantiate(fused_conditional_graph(run;energy,sort),GRAPH_FLAGS())
    end
    compile_fused_prefix!(run,UInt64(0),Val(false))
    CUDA.synchronize()
    graph=NL.capture_graph() do
        enqueue_fused_prefix!(run,UInt64(0),Val(false))
        enqueue_rebuild!(run)
        fused_suffix!(run;energy)
    end
    return CUDA.instantiate(graph)
end
