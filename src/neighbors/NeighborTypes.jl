"""
    AbstractNeighborMatrix

Common supertype for neighbor containers. Cell-list containers provide:

- `neighbors_flat`, `counts`: slot-major ELL rows addressed by `_ell_index`.
- `neighbors_index`: legacy compatibility field, NOT ELL row offsets.
- `cap`: maximum number of stored neighbors per particle (sets memory layout).
- `skin`: additional radial buffer used by `update_needed!` to delay rebuilds.
- `N`, `D`: particle count and dimensionality.

The all-pairs sentinel has no stored rows. `valid` becomes true only after a
successful capacity-checked build. `required_capacity` records the largest
required row, including neighbors that could not fit in a failed build.
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

# At most n consecutive offsets: wrapping them visits each periodic cell once,
# including when the search radius spans the entire axis.
@inline function _stencil_offsets(radius::Int32, n::Int32)
    left = min(radius, n)
    return -left:min(radius, n - Int32(1) - left)
end

"""A neighbor build required more entries per particle than its fixed `cap`."""
struct NeighborCapacityError <: Exception
    capacity::Int
    required::Int
end

function Base.showerror(io::IO, err::NeighborCapacityError)
    print(io, "Neighbor capacity ", err.capacity, " is insufficient; at least ",
          err.required, " entries per particle are required. Rebuild with a larger cap. ",
          "The incomplete neighbor list cannot be used for forces or collisions.")
end

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
simulations and the regression tests. The ELL arrays (`neighbors_flat`,
`counts`) store particle IDs; MIC handling happens inside the force kernels.
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
    valid::Bool
    required_capacity::Int
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
    valid::Bool
    required_capacity::Int
end

"""
    CellListNeighborMatrix{T}

Union of the neighbor containers backed by a cell-list grid (i.e. everything
except the all-pairs sentinel). Shared binning and rebuild-check code
dispatches on this alias.
"""
const CellListNeighborMatrix{T} = Union{NeighborMatrix{T},StencilNeighborMatrix{T}}

@inline function require_valid_neighbors(nb::CellListNeighborMatrix)
    nb.valid || throw(ArgumentError("Neighbor list is unbuilt or its last rebuild failed; call update_neighbors_inplace! successfully before evaluating forces or collisions."))
    return nothing
end

function _check_neighbor_capacity!(nb::CellListNeighborMatrix)
    nb.required_capacity = Int(maximum(nb.counts))
    nb.required_capacity <= nb.cap ||
        throw(NeighborCapacityError(Int(nb.cap), nb.required_capacity))
    return nothing
end
