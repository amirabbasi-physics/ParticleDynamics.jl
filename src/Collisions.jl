module Collisions

using CUDA
using ..Definitions
using ..NeighborLists

export enable_collision_counting!, disable_collision_counting!,
       collisions_reset_counts!, collisions_read_counts!,
       set_collision_pair_cutoffs!,
       _collisions_reinit_on_rebuild!, _collisions_update_after_positions!

# Internal helpers -----------------------------------------------------------
@inline function _mic_fast(dx::T, halfL::T, L::T) where {T<:AbstractFloat}
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

@inline function _lut_index(ti::Int32, tj::Int32, lut::CuDeviceMatrix{Int32})
    return lut[ti, tj]
end

@inline function _bond_cache(i::Int32,
                             bindex::CuDeviceVector{Int32},
                             bflat::CuDeviceVector{Int32},
                             bcounts::CuDeviceVector{Int32})
    base = bindex[i]
    nb = bcounts[i]
    b1 = Int32(0)
    b2 = Int32(0)
    @inbounds begin
        if nb >= 1
            b1 = bflat[base + 1]
        end
        if nb >= 2
            b2 = bflat[base + 2]
        end
    end
    return base, nb, b1, b2
end

@inline function _is_bonded_cached(j::Int32,
                                   base::Int32, nb::Int32, b1::Int32, b2::Int32,
                                   bflat::CuDeviceVector{Int32})
    @inbounds begin
        if nb == 0
            return false
        elseif nb == 1
            return b1 == j
        elseif nb == 2
            return (b1 == j) | (b2 == j)
        end
        for t in 0:Int(nb-1)
            if bflat[base + t + 1] == j
                return true
            end
        end
    end
    return false
end

# Seed the contact state bitset (2D uniform cutoff) so the first counted step
# does not register already overlapping pairs.
function _init_prev2!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    cutoff2::T, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
        dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        contact_prev[base + t + 1] = (r2 > zero(T)) & (r2 < cutoff2)
    end
    return
end

# Same as `_init_prev2!` but uses per-type pair cutoffs.
function _init_prev2_pair!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8}, rcut2_pair::CuDeviceMatrix{T},
    Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        tj = typeid[j]
        rc = rcut2_pair[ti, tj]
        cutoff2 = rc*rc
        dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
        dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        contact_prev[base + t + 1] = (r2 > zero(T)) & (r2 < cutoff2)
    end
    return
end

# 3D initialization for uniform cutoff.
function _init_prev3!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    cutoff2::T, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
        dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
        dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        contact_prev[base + t + 1] = (r2 > zero(T)) & (r2 < cutoff2)
    end
    return
end

# 3D initialization with per-type cutoffs.
function _init_prev3_pair!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8}, rcut2_pair::CuDeviceMatrix{T},
    Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        tj = typeid[j]
        rc = rcut2_pair[ti, tj]
        cutoff2 = rc*rc
        dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
        dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
        dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        contact_prev[base + t + 1] = (r2 > zero(T)) & (r2 < cutoff2)
    end
    return
end

# Detect contact entry events in 2D (uniform cutoff). When `i<j` transitions
# from "separated" to "overlapping", increment the appropriate bin.
function _events2!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    cutoff2::T, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            r2 = muladd(dx, dx, dy*dy)
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                tj = typeid[j]
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)  # 1-based device indexing
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 2D contact events excluding directly bonded pairs.
function _events2_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    cutoff2::T, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat)
                continue
            end
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            r2 = muladd(dx, dx, dy*dy)
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                tj = typeid[j]
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 2D contact events with per-type cutoffs.
function _events2_pair!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    rcut2_pair::CuDeviceMatrix{T}, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            tj = typeid[j]
            rc = rcut2_pair[ti, tj]
            cutoff2 = rc*rc
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            r2 = muladd(dx, dx, dy*dy)
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 2D contact events with per-type cutoffs, excluding directly bonded pairs.
function _events2_pair_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    rcut2_pair::CuDeviceMatrix{T}, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    base  = neighbors_index[i]
    n     = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat)
                continue
            end
            tj = typeid[j]
            rc = rcut2_pair[ti, tj]
            cutoff2 = rc*rc
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            r2 = muladd(dx, dx, dy*dy)
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 3D contact events (uniform cutoff).
function _events3!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    cutoff2::T, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                tj = typeid[j]
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 3D contact events excluding directly bonded pairs.
function _events3_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    cutoff2::T, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat)
                continue
            end
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                tj = typeid[j]
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 3D contact events with per-type cutoffs.
function _events3_pair!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    rcut2_pair::CuDeviceMatrix{T}, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            tj = typeid[j]
            rc = rcut2_pair[ti, tj]
            cutoff2 = rc*rc
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# 3D contact events with per-type cutoffs, excluding directly bonded pairs.
function _events3_pair_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, typeid::CuDeviceVector{Int32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    contact_prev::CuDeviceVector{UInt8},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    bin_lut::CuDeviceMatrix{Int32},
    counts_bins::CuDeviceVector{Int64},
    rcut2_pair::CuDeviceMatrix{T}, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ti = typeid[i]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    base  = neighbors_index[i]
    n     = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(n-1)
        j = neighbors_flat[base + t + 1]
        if i < j
            if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat)
                continue
            end
            tj = typeid[j]
            rc = rcut2_pair[ti, tj]
            cutoff2 = rc*rc
            dx = _mic_fast(rx[i] - rx[j], halfLx, Lx)
            dy = _mic_fast(ry[i] - ry[j], halfLy, Ly)
            dz = _mic_fast(rz[i] - rz[j], halfLz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
            cur = (r2 > zero(T)) & (r2 < cutoff2)
            prev = contact_prev[base + t + 1] != 0
            if (!prev) & cur
                b = bin_lut[ti, tj]
                if b >= 0
                    CUDA.@atomic counts_bins[Int32(b+1)] += Int64(1)
                end
            end
            contact_prev[base + t + 1] = cur
        end
    end
    return
end

# Public API -----------------------------------------------------------------

"""
    enable_collision_counting!(st; ntypes=nothing, bins=:all_pairs)

Attach GPU buffers that count contact-entry events (pairs whose separation
crosses below the collision cutoff). `examples/TwoT_2D_LD_VV.jl` enables this
with `ntypes=2` to log `cold/cold`, `cold/hot`, and `hot/hot` encounters.

# Arguments
- `ntypes`: Number of particle types; defaults to `maximum(Array(st.typeid))`.
- `bins`: Currently only `:all_pairs`, which bins every unordered type pair.

# Examples
```julia
enable_collision_counting!(st; ntypes=2, bins=:all_pairs)
vv = velocityverlet(st; gamma=gamma, temperature=temperature, dt=dt)
for _ in 1:1_000_000
    step!(st, vv, dt; compute_energy=false)
end
counts = collisions_read_counts!(st) ./ (dt * 1_000_000)
```
"""
function enable_collision_counting!(st; ntypes::Union{Nothing,Int}=nothing, bins::Symbol=:all_pairs)
    T = eltype(st.rx)
    N = length(st.rx)
    # Infer number of types if not provided
    nt = if ntypes === nothing
        tid_host = Vector{Int32}(undef, length(st.typeid)); copyto!(tid_host, st.typeid); CUDA.synchronize(); maximum(tid_host)
    else
        ntypes
    end
    @assert bins == :all_pairs "Only bins=:all_pairs is implemented in this version"

    # Build bin LUT (nt x nt), -1 = ignore. Assign bins for unordered pairs (i<=j)
    lut_h = fill(Int32(-1), nt, nt)
    bin = Int32(0)
    for i in 1:nt
        for j in i:nt
            lut_h[i,j] = bin
            lut_h[j,i] = bin
            bin += 1
        end
    end
    nbins = Int(bin)
    st.coll_bins = CuArray(lut_h)
    st.coll_counts = CUDA.fill(Int64(0), nbins)

    # Allocate contact_prev with current neighbor capacity
    neighbors_flat = _collision_neighbor_matrix(st.nbh).neighbors_flat
    st.coll_prev = CUDA.fill(UInt8(0), length(neighbors_flat))
    st.coll_enabled = true

    # Initialize contact_prev with current contacts so that the first counted step
    # does not register pre-existing overlaps as entries.
    _collisions_reinit_on_rebuild!(st)
    return st
end

@inline _collision_neighbor_matrix(nb::NeighborLists.NeighborMatrix) = nb
@inline _collision_neighbor_matrix(nb::NeighborLists.StencilNeighborMatrix) = nb
function _collision_neighbor_matrix(nb)
    throw(ArgumentError("Collision counting requires a neighbor-list-backed simulation state; got $(typeof(nb))."))
end

"""
    disable_collision_counting!(st)

Tear down all collision-counting buffers.
"""
function disable_collision_counting!(st)
    st.coll_enabled = false
    st.coll_prev = nothing
    st.coll_counts = nothing
    st.coll_bins = nothing
    return st
end

"""
    collisions_reset_counts!(st)

Clear the histogram without reinitializing the contact state. Handy after a
warmup period (mirrors the procedure in `examples/TwoT_2D_LD_VV.jl`).
"""
function collisions_reset_counts!(st)
    if st.coll_enabled && st.coll_counts !== nothing
        fill!(st.coll_counts, 0)
    end
    return nothing
end

"""
    collisions_read_counts!(st) -> Vector{Int64}

Copy device counters to host. The vector ordering matches the bin assignment
printed by `enable_collision_counting!` (unordered type pairs).
"""
function collisions_read_counts!(st)
    if !st.coll_enabled || st.coll_counts === nothing
        return Int64[]
    end
    host = Vector{Int64}(undef, length(st.coll_counts))
    copyto!(host, st.coll_counts)
    CUDA.synchronize()
    return host
end

"""
    set_collision_pair_cutoffs!(st, rcut_pair)

Override the contact cutoff used for counting with an `ntypes×ntypes` matrix.
Each entry stores the distance (not squared). This mirrors the `RCUT_PAIR`
setup in `examples/3D_stencil_two_sizes.jl`.
"""
function set_collision_pair_cutoffs!(st, rcut_pair::AbstractMatrix{<:Real})
    T = eltype(st.rx)
    st.rcut_pair = CuArray(T.(rcut_pair))  # store distances; kernels square internally
    return st
end

# Hooks called from Simulation.step! -----------------------------------------

"""
Called when the neighbor list is rebuilt. Resizes and reinitializes
`coll_prev` to reflect current contacts, and avoids spurious entries.
"""
function _collisions_reinit_on_rebuild!(st)
    T = eltype(st.rx)
    if !st.coll_enabled; return; end
    nb = _collision_neighbor_matrix(st.nbh)
    # Ensure coll_prev matches neighbors_flat length
    if (st.coll_prev === nothing) || (length(st.coll_prev) != length(nb.neighbors_flat))
        st.coll_prev = CUDA.fill(UInt8(0), length(nb.neighbors_flat))
    else
        fill!(st.coll_prev, 0)
    end
    # Initialize to current contact state; prefer per-pair rcut if available
    N = length(st.rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    if st.rz === nothing
        Lx = st.box2[1]; Ly = st.box2[2]
        if st.rcut_pair === nothing
            cutoff2 = T(st.pair_lj.rcut) * T(st.pair_lj.rcut)
            k = CUDA.@cuda launch=false _init_prev2!(st.rx, st.ry, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, cutoff2, Lx, Ly)
            k(st.rx, st.ry, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, cutoff2, Lx, Ly; threads, blocks)
        else
            k = CUDA.@cuda launch=false _init_prev2_pair!(st.rx, st.ry, st.typeid, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, st.rcut_pair, Lx, Ly)
            k(st.rx, st.ry, st.typeid, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, st.rcut_pair, Lx, Ly; threads, blocks)
        end
    else
        Lx = st.box3[1]; Ly = st.box3[2]; Lz = st.box3[3]
        if st.rcut_pair === nothing
            cutoff2 = T(st.pair_lj.rcut) * T(st.pair_lj.rcut)
            k = CUDA.@cuda launch=false _init_prev3!(st.rx, st.ry, st.rz, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, cutoff2, Lx, Ly, Lz)
            k(st.rx, st.ry, st.rz, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, cutoff2, Lx, Ly, Lz; threads, blocks)
        else
            k = CUDA.@cuda launch=false _init_prev3_pair!(st.rx, st.ry, st.rz, st.typeid, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, st.rcut_pair, Lx, Ly, Lz)
            k(st.rx, st.ry, st.rz, st.typeid, nb.neighbors_index, nb.neighbors_flat, nb.counts, st.coll_prev, st.rcut_pair, Lx, Ly, Lz; threads, blocks)
        end
    end
    return nothing
end

"""
Called once per step after positions have been advanced and any neighbor rebuild
has been performed. Detects entry events and increments per-bin counters.
"""
function _collisions_update_after_positions!(st)
    T = eltype(st.rx)
    if !st.coll_enabled; return; end
    nb = _collision_neighbor_matrix(st.nbh)
    N = length(st.rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    if st.rz === nothing
        Lx = st.box2[1]; Ly = st.box2[2]
        has_bonds = st.bonds !== nothing
        if st.rcut_pair === nothing
            cutoff2 = T(st.pair_lj.rcut) * T(st.pair_lj.rcut)
            if has_bonds
                k = CUDA.@cuda launch=false _events2_excl!(st.rx, st.ry, st.typeid,
                                                           nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                           st.coll_prev,
                                                           st.bonds.index, st.bonds.flat, st.bonds.counts,
                                                           st.coll_bins, st.coll_counts,
                                                           cutoff2, Lx, Ly)
                k(st.rx, st.ry, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.bonds.index, st.bonds.flat, st.bonds.counts,
                  st.coll_bins, st.coll_counts,
                  cutoff2, Lx, Ly; threads, blocks)
            else
                k = CUDA.@cuda launch=false _events2!(st.rx, st.ry, st.typeid,
                                                      nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                      st.coll_prev,
                                                      st.coll_bins, st.coll_counts,
                                                      cutoff2, Lx, Ly)
                k(st.rx, st.ry, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.coll_bins, st.coll_counts,
                  cutoff2, Lx, Ly; threads, blocks)
            end
        else
            if has_bonds
                k = CUDA.@cuda launch=false _events2_pair_excl!(st.rx, st.ry, st.typeid,
                                                                nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                                st.coll_prev,
                                                                st.bonds.index, st.bonds.flat, st.bonds.counts,
                                                                st.coll_bins, st.coll_counts,
                                                                st.rcut_pair, Lx, Ly)
                k(st.rx, st.ry, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.bonds.index, st.bonds.flat, st.bonds.counts,
                  st.coll_bins, st.coll_counts,
                  st.rcut_pair, Lx, Ly; threads, blocks)
            else
                k = CUDA.@cuda launch=false _events2_pair!(st.rx, st.ry, st.typeid,
                                                           nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                           st.coll_prev,
                                                           st.coll_bins, st.coll_counts,
                                                           st.rcut_pair, Lx, Ly)
                k(st.rx, st.ry, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.coll_bins, st.coll_counts,
                  st.rcut_pair, Lx, Ly; threads, blocks)
            end
        end
    else
        Lx = st.box3[1]; Ly = st.box3[2]; Lz = st.box3[3]
        has_bonds = st.bonds !== nothing
        if st.rcut_pair === nothing
            cutoff2 = T(st.pair_lj.rcut) * T(st.pair_lj.rcut)
            if has_bonds
                k = CUDA.@cuda launch=false _events3_excl!(st.rx, st.ry, st.rz, st.typeid,
                                                           nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                           st.coll_prev,
                                                           st.bonds.index, st.bonds.flat, st.bonds.counts,
                                                           st.coll_bins, st.coll_counts,
                                                           cutoff2, Lx, Ly, Lz)
                k(st.rx, st.ry, st.rz, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.bonds.index, st.bonds.flat, st.bonds.counts,
                  st.coll_bins, st.coll_counts,
                  cutoff2, Lx, Ly, Lz; threads, blocks)
            else
                k = CUDA.@cuda launch=false _events3!(st.rx, st.ry, st.rz, st.typeid,
                                                      nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                      st.coll_prev,
                                                      st.coll_bins, st.coll_counts,
                                                      cutoff2, Lx, Ly, Lz)
                k(st.rx, st.ry, st.rz, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.coll_bins, st.coll_counts,
                  cutoff2, Lx, Ly, Lz; threads, blocks)
            end
        else
            if has_bonds
                k = CUDA.@cuda launch=false _events3_pair_excl!(st.rx, st.ry, st.rz, st.typeid,
                                                                nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                                st.coll_prev,
                                                                st.bonds.index, st.bonds.flat, st.bonds.counts,
                                                                st.coll_bins, st.coll_counts,
                                                                st.rcut_pair, Lx, Ly, Lz)
                k(st.rx, st.ry, st.rz, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.bonds.index, st.bonds.flat, st.bonds.counts,
                  st.coll_bins, st.coll_counts,
                  st.rcut_pair, Lx, Ly, Lz; threads, blocks)
            else
                k = CUDA.@cuda launch=false _events3_pair!(st.rx, st.ry, st.rz, st.typeid,
                                                           nb.neighbors_index, nb.neighbors_flat, nb.counts,
                                                           st.coll_prev,
                                                           st.coll_bins, st.coll_counts,
                                                           st.rcut_pair, Lx, Ly, Lz)
                k(st.rx, st.ry, st.rz, st.typeid,
                  nb.neighbors_index, nb.neighbors_flat, nb.counts,
                  st.coll_prev,
                  st.coll_bins, st.coll_counts,
                  st.rcut_pair, Lx, Ly, Lz; threads, blocks)
            end
        end
    end
    return nothing
end

end # module
