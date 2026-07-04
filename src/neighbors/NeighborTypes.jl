"""
    AbstractNeighborMatrix

Common supertype for all neighbor containers. Kernels expect the following
fields to be present:

- `neighbors_index`, `neighbors_flat`, `counts`: CSR layout describing, for
  every particle, the range of valid neighbor indices in `neighbors_flat`.
- `cap`: maximum number of stored neighbors per particle (sets memory layout).
- `skin`: additional radial buffer used by `update_needed!` to delay rebuilds.
- `N`, `D`: particle count and dimensionality.
"""
abstract type AbstractNeighborMatrix end

# ============================================================================
# Utility helpers
# ============================================================================

@inline function _launchdims(N::Int)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    return threads, blocks
end

# Neighbor rows are stored slot-major ("transposed ELL"): slot `t` (0-based)
# of particle `i` (1-based) lives at `t*N + i`, so warp lanes reading the same
# slot for adjacent particles access consecutive memory (coalesced).
@inline _ell_index(i::Integer, t::Integer, N::Integer) = Int64(t) * Int64(N) + Int64(i)

@inline function _choose_grid(box::Tuple, cell_size::T, D::Int) where {T<:AbstractFloat}
    if D == 2
        nx = max(1, Int32(floor(box[1] / cell_size)))
        ny = max(1, Int32(floor(box[2] / cell_size)))
        return (nx, ny, Int32(1))
    else
        nx = max(1, Int32(floor(box[1] / cell_size)))
        ny = max(1, Int32(floor(box[2] / cell_size)))
        nz = max(1, Int32(floor(box[3] / cell_size)))
        return (nx, ny, nz)
    end
end

@inline function mic_fast(dx::T, halfL::T, L::T) where {T<:AbstractFloat}
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# ============================================================================
# Data structures
# ============================================================================

"""
    NeighborMatrix{T}

Dense neighbor matrix that bins particles into a regular grid (`nx×ny×nz`)
with cell size `cutoff + skin`. Suitable for uniform-cutoff Lennard-Jones/WCA
simulations such as `examples/2D_example.jl` and the regression tests. The CSR
arrays (`neighbors_index`, `neighbors_flat`, `counts`) reference particle IDs
directly; MIC handling happens inside the force kernels.
"""
mutable struct NeighborMatrix{T<:AbstractFloat} <: AbstractNeighborMatrix
    neighbors_index::CuArray{Int32,1}
    neighbors_flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}

    cap::Int32
    cutoff::T
    skin::T
    cutoff2::T

    N::Int32
    D::Int32

    nx::Int32
    ny::Int32
    nz::Int32
    cell_size::T

    particle_ids_sorted::CuArray{Int32,1}
    cell_offsets::CuArray{Int32,1}
    cell_counts::CuArray{Int32,1}
    cell_of_particle::CuArray{Int32,1}

    rref_x::CuArray{T,1}
    rref_y::CuArray{T,1}
    rref_z::Union{CuArray{T,1},Nothing}
    dr2::CuArray{T,1}
    last_build_step::Int
    target_interval::Int
end

"""
    StencilNeighborMatrix{T}

Neighbor list variant where each particle carries its own interaction radius
(`rlist`, `rlist2`). Used by `examples/2D/3D_stencil_two_sizes*.jl` when mixing
different particle diameters. The per-particle cutoff is `σ_i + skin`, so the
`skin` argument still controls rebuild latency.
"""
mutable struct StencilNeighborMatrix{T<:AbstractFloat} <: AbstractNeighborMatrix
    neighbors_index::CuArray{Int32,1}
    neighbors_flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}

    cap::Int32
    skin::T

    N::Int32
    D::Int32

    nx::Int32
    ny::Int32
    nz::Int32
    cell_size::T

    particle_ids_sorted::CuArray{Int32,1}
    cell_offsets::CuArray{Int32,1}
    cell_counts::CuArray{Int32,1}
    cell_of_particle::CuArray{Int32,1}

    rlist::CuArray{T,1}
    rlist2::CuArray{T,1}

    rref_x::CuArray{T,1}
    rref_y::CuArray{T,1}
    rref_z::Union{CuArray{T,1},Nothing}
    dr2::CuArray{T,1}
    last_build_step::Int
    target_interval::Int
end

"""
    CellListNeighborMatrix{T}

Union of the neighbor containers backed by a cell-list grid (i.e. everything
except the all-pairs sentinel). Shared binning and rebuild-check code
dispatches on this alias.
"""
const CellListNeighborMatrix{T} = Union{NeighborMatrix{T},StencilNeighborMatrix{T}}
