# Included in ParticleDynamics.DeviceEngine. Spatial sorting inside the rebuild branch:
# persistent scratch, stable addresses, identity tags, optional extra particle arrays.
struct OrderingWorkspace{T,R,I,M,S}
    scratch::S # NTuple{3D+1}: positions, velocities, forces, energy
    virial_scratch::CuArray{T,2}
    tags::CuArray{Int32,1}          # storage slot -> original particle id
    tag_scratch::CuArray{Int32,1}
    cell_scratch::CuArray{Int32,1}
    state::CuArray{Int64,1}        # requested, last sorted step, count
    interval::Int64
    # Further particle-indexed arrays owned elsewhere (e.g. a SimulationState),
    # permuted together with the run: real vectors, Int32 vectors, N×k matrices.
    extra_real::R; extra_real_scratch::R
    extra_int::I; extra_int_scratch::I
    extra_mat::M; extra_mat_scratch::M
    slot_of::Any                    # id → slot (sorted bonded states) or nothing
end

function OrderingWorkspace(::Type{T},N,interval::Integer;tags=nothing,dims::Integer=3,
                           extra_real::Tuple=(),extra_int::Tuple=(),extra_mat::Tuple=(),slot_of=nothing) where {T}
    0 < interval <= typemax(Int64) || throw(ArgumentError("Sorting interval must be positive"))
    all(length(a)==N for a in (extra_real...,extra_int...)) && all(size(a,1)==N for a in extra_mat) ||
        throw(DimensionMismatch("Extra particle arrays must have N rows"))
    if tags===nothing
        tags=CuArray{Int32}(undef,N)
        tags .= Int32(1):Int32(N)
    end
    length(tags)==N || throw(DimensionMismatch("Tag array must have N entries"))
    dims in (2,3) || throw(ArgumentError("Sorting supports 2D and 3D"))
    OrderingWorkspace(ntuple(_->CuArray{T}(undef,N),3dims+1),CuArray{T}(undef,N,dims==3 ? 6 : 3),
        tags,similar(tags),similar(tags),CuArray(Int64[0,-1,0]),Int64(interval),
        extra_real,map(similar,extra_real),extra_int,map(similar,extra_int),extra_mat,map(similar,extra_mat),slot_of)
end

function _ordering_decide_kernel!(state,order,interval)
    @inbounds order[1]=Int64(NL._device_rebuild_requested(state) &&
        (order[2]<0 || state[NL.DN_STEP]-order[2]>=interval))
    if order[1]!=0 && state[NL.DN_DIAGNOSTICS]!=0
        @inbounds state[NL.DN_REASON] |= NL.DN_REASON_SORT
    end
    return
end

@inline _ordering_requested(state,order)=NL._device_rebuild_requested(state) && (@inbounds order[1]!=0)

function _ordering_gather_kernel!(state,order,perm,cells,fields,scratch,
                                  virial,virial_scratch,tags,tag_scratch,cell_scratch)
    _ordering_requested(state,order) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=length(tags)
        @inbounds begin
            j=perm[i]
            for d in 1:length(fields); scratch[d][i]=fields[d][j]; end
            for d in 1:size(virial,2); virial_scratch[i,d]=virial[j,d]; end
            tag_scratch[i]=tags[j]
            cell_scratch[i]=cells[j]
        end
    end
    return
end

function _ordering_copyback_kernel!(state,order,perm,cells,fields,scratch,
                                    virial,virial_scratch,tags,tag_scratch,cell_scratch)
    _ordering_requested(state,order) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=length(tags)
        @inbounds begin
            for d in 1:length(fields); fields[d][i]=scratch[d][i]; end
            for d in 1:size(virial,2); virial[i,d]=virial_scratch[i,d]; end
            tags[i]=tag_scratch[i]
            cells[i]=cell_scratch[i]
            perm[i]=Int32(i) # storage now has the cell-sorted order
        end
    end
    return
end

# Extra arrays: gather before the base copyback (which resets `perm`).
function _ordering_extra_gather_kernel!(state,order,perm,arrays,scratch)
    _ordering_requested(state,order) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=length(perm)
        @inbounds begin
            j=perm[i]
            for a in 1:length(arrays)
                x=arrays[a]; y=scratch[a]
                for c in 1:size(x,2); y[i,c]=x[j,c]; end
            end
        end
    end
    return
end
function _ordering_extra_copyback_kernel!(state,order,arrays,scratch)
    _ordering_requested(state,order) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=size(arrays[1],1)
        @inbounds for a in 1:length(arrays)
            x=arrays[a]; y=scratch[a]
            for c in 1:size(x,2); x[i,c]=y[i,c]; end
        end
    end
    return
end

# Split extra fields into homogeneous (same array type) groups, host side at capture
# time (e.g. Int32 type ids and the UInt8 freeze mask). One group per type.
function _eltype_groups(arrays::Tuple,scratch::Tuple)
    isempty(arrays) && return ()
    types=unique(map(typeof,arrays))
    return map(Tuple(types)) do U
        k=findall(a->typeof(a)===U,collect(arrays))
        (Tuple(arrays[j] for j in k),Tuple(scratch[j] for j in k))
    end
end

function _enqueue_extra_ordering!(ws,ow,arrays::Tuple,scratch::Tuple,N)
    isempty(arrays) && return nothing
    @cuda threads=256 blocks=cld(N,256) _ordering_extra_gather_kernel!(ws.state,ow.state,ws.list.particle_ids_sorted,arrays,scratch)
    @cuda threads=256 blocks=cld(N,256) _ordering_extra_copyback_kernel!(ws.state,ow.state,arrays,scratch)
    return nothing
end

# Topology map after the copyback (tags are in the new order): particle id → slot.
function _ordering_slot_kernel!(state,order,tags,slot_of)
    _ordering_requested(state,order) || return
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i<=length(tags) && (@inbounds slot_of[tags[i]]=Int32(i))
    return
end

function _ordering_commit_kernel!(state,order)
    if _ordering_requested(state,order)
        @inbounds begin
            order[2]=state[NL.DN_STEP]
            order[3]+=1
            order[1]=0
        end
    end
    return
end

function enqueue_ordering!(run,ow::OrderingWorkspace;always::Bool=false)
    ws=run.neighbors; nb=ws.list
    fields=(run.r...,run.v...,run.f...,run.energy)
    args=(ws.state,ow.state,nb.particle_ids_sorted,nb.cell_of_particle,fields,
        ow.scratch,run.virial,ow.virial_scratch,ow.tags,ow.tag_scratch,ow.cell_scratch)
    @cuda threads=1 blocks=1 _ordering_decide_kernel!(ws.state,ow.state,always ? zero(ow.interval) : ow.interval)
    N=length(run.r[1])
    @cuda threads=256 blocks=cld(N,256) _ordering_gather_kernel!(args...)
    for (arrays,scratch) in ((ow.extra_real,ow.extra_real_scratch),(ow.extra_int,ow.extra_int_scratch),(ow.extra_mat,ow.extra_mat_scratch)),
        (a,s) in _eltype_groups(arrays,scratch) # the gather loop needs one element type per launch
        _enqueue_extra_ordering!(ws,ow,a,s,N)
    end
    @cuda threads=256 blocks=cld(N,256) _ordering_copyback_kernel!(args...)
    ow.slot_of===nothing || (@cuda threads=256 blocks=cld(N,256) _ordering_slot_kernel!(ws.state,ow.state,ow.tags,ow.slot_of))
    @cuda threads=1 blocks=1 _ordering_commit_kernel!(ws.state,ow.state)
    return nothing
end

enqueue_ordering!(run,::Nothing;always::Bool=false)=nothing

"""Explicit boundary readback; never called from graph replay."""
function ordering_status(run)
    run.ordering===nothing && return (count=0,last_step=-1)
    s=Array(run.ordering.state)
    return (count=s[3],last_step=s[2])
end
