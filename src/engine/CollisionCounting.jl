# Feature: collision counting (`Collisions.enable_collision_counting!`).
# Semantics (the former staged path, kept bit for bit; Collisions.jl kernels):
#   • a "position sample" happens after every physical position stage: once per
#     step (NVE, CSVR, NHC, VV, EM, Euler–Heun final), twice for BAOAB/BAOA/GSM
#     (after A1 and after A2); Heun midpoints are not samples;
#   • before a sample the neighbor list must cover the sample coordinates; on
#     every rebuild the per-entry contact flags are reinitialized from the last
#     sampled positions (`preserve_history`);
#   • a sample counts entries i<j whose separation crossed below the contact
#     cutoff (uniform `pair_lj.rcut` or per-type `rcut_pair`) into per-type-pair
#     bins, then snapshots the positions.
# The staged event/initialization kernels run unchanged on the engine's list
# (same ELL layout). Samples happen after the step's neighbor maintenance:
# `position_sample!` at the start of the step suffix (and, for BAOAB/BAOA/GSM,
# after the A1 neighbor check, see `samples_half_drift`). Rebuilds (which also
# sort) reinitialize the flags in `after_rebuild!`.

struct CollisionFeature{S,R<:Tuple} <: EngineFeature
    st::S             # the state: owns coll_prev/coll_counts/coll_bins/coll_ref_*
    ref::R            # snapshot of the last sampled positions (sorted with the particles)
end

_coll_ref(st)=st.rz===nothing ? (st.coll_ref_x,st.coll_ref_y) : (st.coll_ref_x,st.coll_ref_y,st.coll_ref_z)
function make_feature(::Type{CollisionFeature},st,spec)
    st.coll_enabled || return nothing
    st.coll_ref_x===nothing && Collisions._snapshot_collision_positions!(st) # first engine sample: current positions
    return CollisionFeature(st,_coll_ref(st))
end
feature_signature(::Type{CollisionFeature},st,spec)=st.coll_enabled ? (st.rcut_pair===nothing ? nothing : pointer(st.rcut_pair),
    st.coll_counts===nothing ? nothing : pointer(st.coll_counts),st.coll_bins===nothing ? nothing : pointer(st.coll_bins)) : nothing
sorted_real(f::CollisionFeature)=f.ref
samples_half_drift(::CollisionFeature)=true   # BAOAB/BAOA/GSM sample after A1 too (staged)
needs_neighbors(::CollisionFeature)=true      # events are counted over the engine's list

function _coll_args(run,f::CollisionFeature,::Val{:init})
    st=f.st; nb=run.neighbors.list; T=eltype(run.r[1]); box=run.neighbors.box
    base=(f.ref...,)
    if st.rcut_pair===nothing
        c2=T(st.pair_lj.rcut)*T(st.pair_lj.rcut)
        return (length(box)==3 ? Collisions._init_prev3! : Collisions._init_prev2!),
            (base...,nb.neighbors_index,nb.neighbors_flat,nb.counts,st.coll_prev,c2,box...)
    end
    return (length(box)==3 ? Collisions._init_prev3_pair! : Collisions._init_prev2_pair!),
        (base...,st.typeid,nb.neighbors_index,nb.neighbors_flat,nb.counts,st.coll_prev,st.rcut_pair,box...)
end
function _coll_args(run,f::CollisionFeature,::Val{:events})
    st=f.st; nb=run.neighbors.list; T=eltype(run.r[1]); box=run.neighbors.box
    if st.rcut_pair===nothing
        c2=T(st.pair_lj.rcut)*T(st.pair_lj.rcut)
        return (length(box)==3 ? Collisions._events3! : Collisions._events2!),
            (run.r...,st.typeid,nb.neighbors_index,nb.neighbors_flat,nb.counts,st.coll_prev,st.coll_bins,st.coll_counts,c2,box...)
    end
    return (length(box)==3 ? Collisions._events3_pair! : Collisions._events2_pair!),
        (run.r...,st.typeid,nb.neighbors_index,nb.neighbors_flat,nb.counts,st.coll_prev,st.coll_bins,st.coll_counts,
         st.rcut_pair,box...)
end

function _coll_launch!(run,kernel,args;launch::Bool)
    s=run.neighbors.state; N=length(run.r[1])
    k=@cuda launch=false NL._device_if_valid!(s,kernel,args...)
    launch && k(s,kernel,args...;threads=256,blocks=cld(N,256))
    return nothing
end

function _snapshot_kernel!(ref::NTuple{D},r::NTuple{D}) where {D}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i<=length(r[1]) && _put!(ref,i,_get(r,i))
    return nothing
end

"""Rebuild (and sort): contact flags of the new rows from the last sampled positions."""
function after_rebuild!(run,f::CollisionFeature;launch::Bool=true)
    st=f.st; n=length(run.neighbors.list.neighbors_flat)
    if st.coll_prev===nothing || length(st.coll_prev)!=n # sized outside capture (constructor rebuild)
        st.coll_prev=CUDA.zeros(UInt8,n)
    end
    kernel,args=_coll_args(run,f,Val(:init))
    _coll_launch!(run,kernel,args;launch)
    return nothing
end

"""Position sample: count contact entries since the last sample, then snapshot."""
function position_sample!(run,f::CollisionFeature;launch::Bool=true)
    kernel,args=_coll_args(run,f,Val(:events))
    _coll_launch!(run,kernel,args;launch)
    _coll_launch!(run,_snapshot_kernel!,(f.ref,run.r);launch)
    return nothing
end
