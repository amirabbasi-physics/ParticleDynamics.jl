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
