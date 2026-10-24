# Included in ParticleDynamics.DeviceEngine. Experimental uniform-LJ graph traversal only.
# G=1/:ell retains the original kernels and has no packing work.
struct TraversalWorkspace{G,L}
    packed::Union{Nothing,CuArray{Int32,1}}
    packs::CuArray{Int64,1}
    slot_groups::Int
end

function validate_traversal(lanes,layout)
    lanes in (1,2,4,8) || throw(ArgumentError("Force lanes must be 1, 2, 4 or 8"))
    layout in (:ell,:tiled) || throw(ArgumentError("Force layout must be :ell or :tiled"))
    return nothing
end

function TraversalWorkspace(N,cap,::Val{G},::Val{L}) where {G,L}
    validate_traversal(G,L)
    groups=cld(Int(cap),G)
    entries=Base.checked_mul(Base.checked_mul(cld(N,32÷G),groups),32)
    packed=L===:tiled ? CUDA.zeros(Int32,entries) : nothing
    return TraversalWorkspace{G,L}(packed,CUDA.zeros(Int64,1),groups)
end

# A warp addresses A=32/G atoms and G slots/atom at a fixed slot-group q.
# Adjacent lanes access adjacent Int32 entries. Capacity is fixed at capture.
@inline function _tiled_neighbor_index(i,t,slot_groups,::Val{G}) where {G}
    atom=i-1
    return ((atom÷(32÷G))*slot_groups+t÷G)*32+(atom%(32÷G))*G+t%G+1
end
@inline _traversal_neighbor_index(i,t,N,slot_groups,::Val{G},::Val{:tiled}) where {G} =
    _tiled_neighbor_index(i,t,slot_groups,Val(G))
@inline _traversal_neighbor_index(i,t,N,slot_groups,::Val{G},::Val{:ell}) where {G} = t*N+i

@inline _packing_requested(state)=NL._device_rebuild_requested(state) && NL._device_list_valid(state)

function _pack_tiled_neighbors_kernel!(state,packed,ell,counts,slot_groups,::Val{G}) where {G}
    _packing_requested(state) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    N=length(counts)
    if i<=N
        # Coalesced source rows; never read beyond populated ELL slots. Stale
        # packed padding is harmless: traversal reads only current counts[i].
        @inbounds for t in 0:Int(counts[i])-1
            packed[_tiled_neighbor_index(i,t,slot_groups,Val(G))]=ell[t*N+i]
        end
    end
    return nothing
end

function _packing_commit_kernel!(state,packs)
    if _packing_requested(state)
        @inbounds packs[1]+=1
    end
    return nothing
end

function enqueue_packing!(run)
    o=run.traversal
    enqueue_packing!(run,o)
end
enqueue_packing!(run,::TraversalWorkspace{G,:ell}) where {G}=nothing
function enqueue_packing!(run,o::TraversalWorkspace{G,:tiled}) where {G}
    ws=run.neighbors; nb=ws.list
    @cuda threads=128 blocks=cld(length(run.r[1]),128) _pack_tiled_neighbors_kernel!(
        ws.state,o.packed::CuArray{Int32,1},nb.neighbors_flat,nb.counts,o.slot_groups,Val(G))
    @cuda threads=1 blocks=1 _packing_commit_kernel!(ws.state,o.packs)
    return nothing
end

@inline function _cooperative_sum(value,::Val{G}) where {G}
    shift=G÷2
    while shift>0
        value+=CUDA.shfl_down_sync(UInt32(0xffffffff),value,shift,G)
        shift÷=2
    end
    return value
end

function _lj_cooperative_kernel!(rx,ry,rz,fx,fy,fz,energy,virial,indices,counts,
                                 slot_groups,box::NTuple{3,T},epsilon::T,sigma::T,
                                 cutoff2::T,::Val{G},::Val{L},::Val{E},kick) where {T,G,L,E}
    tid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    i=tid÷G+1; lane=tid%G; N=length(rx)
    ax=zero(T); ay=zero(T); az=zero(T); e=zero(T)
    wxx=zero(T); wyy=zero(T); wzz=zero(T); wxy=zero(T); wxz=zero(T); wyz=zero(T)
    if i<=N
        @inbounds begin
            xi=rx[i]; yi=ry[i]; zi=rz[i]
            slot=lane
            while slot<counts[i]
                j=indices[_traversal_neighbor_index(i,slot,N,slot_groups,Val(G),Val(L))]
                dx=NB.periodic_delta(xi,rx[j],T(0.5)*box[1],box[1])
                dy=NB.periodic_delta(yi,ry[j],T(0.5)*box[2],box[2])
                dz=NB.periodic_delta(zi,rz[j],T(0.5)*box[3],box[3])
                r2=muladd(dx,dx,muladd(dy,dy,dz*dz))
                if (r2>zero(T)) & (r2<cutoff2)
                    if E
                        x,y,z,u=NB.lj_pair_3d(dx,dy,dz,r2,epsilon,sigma)
                        e+=T(0.5)*u
                        xx,yy,zz,xy,xz,yz=NB._half_virial3(dx,dy,dz,x,y,z)
                        wxx+=xx; wyy+=yy; wzz+=zz; wxy+=xy; wxz+=xz; wyz+=yz
                    else
                        # Match the existing no-energy kernel arithmetic.
                        invr2=one(T)/r2; s2=(sigma*sigma)*invr2
                        s6=s2*s2*s2; s12=s6*s6
                        a=T(24)*epsilon*(T(2)*s12-s6)*invr2
                        x=a*dx; y=a*dy; z=a*dz
                    end
                    ax+=x; ay+=y; az+=z
                end
                slot+=G
            end
        end
    end
    # Every lane reaches every shuffle, including empty rows and all groups
    # beyond N. Blocks contain full warps and G divides the warp width.
    ax=_cooperative_sum(ax,Val(G)); ay=_cooperative_sum(ay,Val(G)); az=_cooperative_sum(az,Val(G))
    if E
        e=_cooperative_sum(e,Val(G))
        wxx=_cooperative_sum(wxx,Val(G)); wyy=_cooperative_sum(wyy,Val(G)); wzz=_cooperative_sum(wzz,Val(G))
        wxy=_cooperative_sum(wxy,Val(G)); wxz=_cooperative_sum(wxz,Val(G)); wyz=_cooperative_sum(wyz,Val(G))
    end
    if lane==0 && i<=N
        @inbounds begin
            fx[i]=ax; fy[i]=ay; fz[i]=az
            if E
                energy[i]=e
                virial[i,1]=wxx; virial[i,2]=wyy; virial[i,3]=wzz
                virial[i,4]=wxy; virial[i,5]=wxz; virial[i,6]=wyz
            end
            if kick!==nothing # fused final kick: identical to SC._nve_kick3_kernel!
                vx,vy,vz,c,_=kick
                vx[i]=muladd(c,ax,vx[i]); vy[i]=muladd(c,ay,vy[i]); vz[i]=muladd(c,az,vz[i])
            end
        end
    end
    if kick!==nothing && tid==0 # fused step counter; the wrapper guarantees a valid list
        @inbounds kick[5][NL.DN_STEP]+=1
    end
    return nothing
end

# Pass `kick` explicitly: a defaulted trailing argument adds a forwarding method
# whose inner call is not inlined, copying array descriptors to a per-thread
# stack frame (measured 7.8 -> 49 us at 10k; see the Argon log, 9 Oct 2026).
function _lj_cooperative_noE_kernel!(args...)
    @inline _lj_cooperative_kernel!(args...,Val(false),nothing)
    return nothing
end
function _lj_cooperative_observable_kernel!(args...)
    @inline _lj_cooperative_kernel!(args...,Val(true),nothing)
    return nothing
end
# Fused variants (kick, then the usual arguments); names keep the substrings the
# trace audit uses to count regular/observable force kernels.
function _lj_cooperative_noE_kernel_kick!(kick,args...)
    @inline _lj_cooperative_kernel!(args...,Val(false),kick)
    return nothing
end
function _lj_cooperative_observable_kernel_kick!(kick,args...)
    @inline _lj_cooperative_kernel!(args...,Val(true),kick)
    return nothing
end

@inline cooperative_threads(N,::Val{G}) where {G} = G==1 ? (N<50000 ? 64 : N<200000 ? 128 : 256) : 128

function enqueue_cooperative_force!(run,o::TraversalWorkspace{G,L},::Val{E};fx=run.fx) where {G,L,E}
    ws=run.neighbors; nb=ws.list
    indices=L===:tiled ? o.packed::CuArray{Int32,1} : nb.neighbors_flat
    threads=cooperative_threads(length(run.r[1]),Val(G))
    force=E ? _pair_cooperative_observable_kernel! : _pair_cooperative_noE_kernel!
    @cuda threads=threads blocks=cld(length(run.r[1])*G,threads) NL._device_if_valid!(
        ws.state,force,run.r,run.f,run.energy,run.virial,
        indices,nb.counts,o.slot_groups,ws.geom,run.model,run.model_data,fx,Val(G),Val(L))
    return nothing
end

"""Boundary readback only; validate healthy packing counts against builds."""
function traversal_status end # methods defined after Run in DeviceArgonNVE
