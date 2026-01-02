"""
Neighbor list builders and query utilities used by the force kernels.

`NeighborLists` implements three strategies:

- [`NeighborMatrix`](@ref) — dense cell lists with uniform cutoffs (default in
  `build_simulation`, parameter choices mirror `examples/2D_example.jl`).
- [`StencilNeighborMatrix`](@ref) — particle- or type-dependent cutoffs as used
  in `examples/3D_stencil_two_sizes*.jl`.
- [`AllPairsNeighborMatrix`](@ref) — sentinel representing O(N²) evaluation
  (`examples/2D_allpairs_quicktest.jl`).

All stores use a CSR-style `(neighbors_index, neighbors_flat, counts)` layout
so that kernels can iterate neighbors without branches.
"""
module NeighborLists

using CUDA

export AbstractNeighborMatrix,
       NeighborMatrix, StencilNeighborMatrix,
       build_neighbors_dense!, build_neighbors_stencil!,
       build_neighbors_stencil_by_types!,
       update_neighbors_inplace!, update_needed!,
       build_neighbors_allpairs!, AllPairsNeighborMatrix

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
    cell_ids_sorted::CuArray{Int32,1}
    cell_offsets::CuArray{Int32,1}
    cell_of_particle::CuArray{Int32,1}
    packed_keys::CuArray{UInt64,1}

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
    cell_ids_sorted::CuArray{Int32,1}
    cell_offsets::CuArray{Int32,1}
    cell_of_particle::CuArray{Int32,1}
    packed_keys::CuArray{UInt64,1}

    rlist::CuArray{T,1}
    rlist2::CuArray{T,1}

    rref_x::CuArray{T,1}
    rref_y::CuArray{T,1}
    rref_z::Union{CuArray{T,1},Nothing}
    dr2::CuArray{T,1}
    last_build_step::Int
    target_interval::Int
end

# ============================================================================
# All-pairs "neighbor list" (sentinel for O(N^2) interactions)
# ============================================================================

"""
AllPairsNeighborMatrix

Lightweight sentinel type indicating that all-to-all interactions should be
computed (no neighbor list), using periodic MIC and excluding i==j.

Fields are kept minimal but include `skin` so existing calls that read
`st.nbh.skin` continue to work without changes. `examples/2D_allpairs_quicktest.jl`
uses this type when validating the WCA path without the cell list overhead.
"""
struct AllPairsNeighborMatrix{T<:AbstractFloat} <: AbstractNeighborMatrix
    skin::T
    N::Int32
    D::Int32
end

"""
    build_neighbors_allpairs!(rx, ry[ , rz]; box, cutoff, cap, skin)

Construct an `AllPairsNeighborMatrix` sentinel. Arguments mirror the dense
builders to keep call-sites uniform.
"""
function build_neighbors_allpairs!(rx::CuArray{T,1}, ry::CuArray{T,1};
                                   box::NTuple{2,T}, cutoff::T, cap::Int32, skin::T) where {T<:AbstractFloat}
    N = Int32(length(rx))
    return AllPairsNeighborMatrix{T}(skin, N, Int32(2))
end

function build_neighbors_allpairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                   box::NTuple{3,T}, cutoff::T, cap::Int32, skin::T) where {T<:AbstractFloat}
    N = Int32(length(rx))
    return AllPairsNeighborMatrix{T}(skin, N, Int32(3))
end

# No-op update hooks so existing NL maintenance code paths remain valid
function update_neighbors_inplace!(nbh::AllPairsNeighborMatrix{T}, args...; kwargs...) where {T<:AbstractFloat}
    return nbh
end

function update_needed!(nbh::AllPairsNeighborMatrix{T}, args...; kwargs...) where {T<:AbstractFloat}
    return false
end

# ============================================================================
# Allocation helpers
# ============================================================================

function _alloc_neighbor_matrix(T::Type{<:AbstractFloat}, N::Int, D::Int,
                                box, cutoff::Real, skin::Real, cap::Int32)
    cutoffT = T(cutoff)
    skinT   = T(skin)
    cutoff2 = cutoffT * cutoffT
    cell_size = max(T(1e-6), cutoffT + skinT)
    nx, ny, nz = _choose_grid(box, cell_size, D)

    neighbors_index = CUDA.CuArray{Int32}(undef, N)
    neighbors_flat  = CUDA.fill(Int32(-1), N * Int(cap))
    counts          = CUDA.CuArray{Int32}(undef, N)
    fill!(counts, Int32(0))

    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rowstarts!(neighbors_index, cap)
    kset(neighbors_index, cap; threads, blocks)

    particle_ids_sorted = CUDA.CuArray{Int32}(undef, N)
    cell_ids_sorted     = CUDA.CuArray{Int32}(undef, N)
    ncell = Int(nx) * Int(ny) * Int(nz)
    cell_offsets        = CUDA.fill(Int32(1), ncell + 1)
    cell_of_particle    = CUDA.CuArray{Int32}(undef, N)
    packed_keys         = CUDA.CuArray{UInt64}(undef, N)

    rref_x = CUDA.CuArray{T}(undef, N)
    rref_y = CUDA.CuArray{T}(undef, N)
    rref_z = D == 3 ? CUDA.CuArray{T}(undef, N) : nothing
    dr2    = CUDA.CuArray{T}(undef, N)

    return NeighborMatrix{T}(
        neighbors_index, neighbors_flat, counts,
        cap, cutoffT, skinT, cutoff2,
        Int32(N), Int32(D),
        nx, ny, nz, cell_size,
        particle_ids_sorted, cell_ids_sorted, cell_offsets, cell_of_particle, packed_keys,
        rref_x, rref_y, rref_z, dr2, 0, 20
    )
end

function _alloc_stencil_matrix(T::Type{<:AbstractFloat}, N::Int, D::Int,
                               box, cell_size::Real, skin::Real, cap::Int32)
    cellT = T(cell_size)
    skinT = T(skin)
    nx, ny, nz = _choose_grid(box, cellT, D)

    neighbors_index = CUDA.CuArray{Int32}(undef, N)
    neighbors_flat  = CUDA.fill(Int32(-1), N * Int(cap))
    counts          = CUDA.CuArray{Int32}(undef, N)
    fill!(counts, Int32(0))

    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rowstarts!(neighbors_index, cap)
    kset(neighbors_index, cap; threads, blocks)

    particle_ids_sorted = CUDA.CuArray{Int32}(undef, N)
    cell_ids_sorted     = CUDA.CuArray{Int32}(undef, N)
    ncell = Int(nx) * Int(ny) * Int(nz)
    cell_offsets        = CUDA.fill(Int32(1), ncell + 1)
    cell_of_particle    = CUDA.CuArray{Int32}(undef, N)
    packed_keys         = CUDA.CuArray{UInt64}(undef, N)

    rlist  = CUDA.CuArray{T}(undef, N)
    rlist2 = CUDA.CuArray{T}(undef, N)

    rref_x = CUDA.CuArray{T}(undef, N)
    rref_y = CUDA.CuArray{T}(undef, N)
    rref_z = D == 3 ? CUDA.CuArray{T}(undef, N) : nothing
    dr2    = CUDA.CuArray{T}(undef, N)

    return StencilNeighborMatrix{T}(
        neighbors_index, neighbors_flat, counts,
        cap, skinT,
        Int32(N), Int32(D),
        nx, ny, nz, cellT,
        particle_ids_sorted, cell_ids_sorted, cell_offsets, cell_of_particle, packed_keys,
        rlist, rlist2,
        rref_x, rref_y, rref_z, dr2, 0, 20
    )
end

# ============================================================================
# Core kernels
# ============================================================================

# Initialize the CSR row start pointer for each particle: row i starts at i*cap.
function _kernel_set_rowstarts!(neighbors_index::CuDeviceVector{Int32}, cap::Int32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(neighbors_index); if i > N; return; end
    @inbounds neighbors_index[i] = (i-1) * cap
    return
end

# Pack `(cell_id, particle_id)` pairs so particles can be sorted by cell id.
function _kernel_compute_packed2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                  Lx::T, Ly::T, inv_cs::T,
                                  nx::Int32, ny::Int32,
                                  packed::CuDeviceVector{UInt64}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        x = rx[i] + T(0.5)*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + T(0.5)*Ly; y -= floor(y/Ly)*Ly
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        cid = Int32(cy*nx + cx)
        packed[i] = (UInt64(UInt32(cid)) << 32) | UInt64(UInt32(i-1))
    end
    return
end

# 3D variant of the packing kernel described above.
function _kernel_compute_packed3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                  Lx::T, Ly::T, Lz::T, inv_cs::T,
                                  nx::Int32, ny::Int32, nz::Int32,
                                  packed::CuDeviceVector{UInt64}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        x = rx[i] + T(0.5)*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + T(0.5)*Ly; y -= floor(y/Ly)*Ly
        z = rz[i] + T(0.5)*Lz; z -= floor(z/Lz)*Lz
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = cz >= nz ? (nz-1) : cz
        cid = Int32((cz*ny + cy)*nx + cx)
        packed[i] = (UInt64(UInt32(cid)) << 32) | UInt64(UInt32(i-1))
    end
    return
end

function _kernel_unpack_sorted!(packed::CuDeviceVector{UInt64},
                                cell_ids_sorted::CuDeviceVector{Int32},
                                particle_ids_sorted::CuDeviceVector{Int32},
                                cell_of_particle::CuDeviceVector{Int32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(packed); if i > N; return; end
    @inbounds begin
        pv = packed[i]
        cid = Int32(UInt32(pv >> 32))
        pid = Int32(UInt32(pv & 0xFFFF_FFFF)) + 1
        cell_ids_sorted[i]     = cid
        particle_ids_sorted[i] = pid
        cell_of_particle[pid]  = cid
    end
    return
end

@inline function _lb_search(arr::CuDeviceVector{Int32}, N::Int32, key::Int32)
    lo = Int32(1)
    hi = N + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        v = arr[mid]
        if v < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

function _kernel_cell_offsets!(cell_ids_sorted::CuDeviceVector{Int32},
                               cell_offsets::CuDeviceVector{Int32},
                               ncell::Int32)
    c = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if c < 1 || c > ncell + 1
        return
    end
    N = Int32(length(cell_ids_sorted))
    if c <= ncell
        @inbounds cell_offsets[c] = _lb_search(cell_ids_sorted, N, Int32(c-1))
    else
        @inbounds cell_offsets[c] = N + 1
    end
    return
end

# Scan the 3×3 neighborhood around the cell containing particle `i1`
# (with periodic wrapping) and append neighbors that satisfy the cutoff² test.
function _kernel_neighbors2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                             neighbors_index::CuDeviceVector{Int32},
                             neighbors_flat::CuDeviceVector{Int32},
                             counts::CuDeviceVector{Int32},
                             cell_offsets::CuDeviceVector{Int32},
                             particle_ids_sorted::CuDeviceVector{Int32},
                             cell_of_particle::CuDeviceVector{Int32},
                             Lx::T, Ly::T, halfLx::T, halfLy::T,
                             nx::Int32, ny::Int32,
                             cutoff2::T, cap::Int32) where {T<:AbstractFloat}
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        c0 = cell_of_particle[i1]
        cx = c0 % nx
        cy = c0 ÷ nx
        base  = neighbors_index[i1]
        found = Int32(0)
        for oy in Int32(-1):Int32(1)
            cy2 = cy + oy; cy2 -= (cy2 >= ny)*ny; cy2 += (cy2 < 0)*ny
            for ox in Int32(-1):Int32(1)
                cx2 = cx + ox; cx2 -= (cx2 >= nx)*nx; cx2 += (cx2 < 0)*nx
                c = cy2*nx + cx2
                s = cell_offsets[c+1]
                e = cell_offsets[c+2]
                for k in s:(e-1)
                    j = particle_ids_sorted[k]
                    if j != i1
                        dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                        dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                        r2 = muladd(dx, dx, dy*dy)
                        if r2 <= cutoff2 && found < cap
                            neighbors_flat[base + found + 1] = j
                            found += 1
                        end
                    end
                end
            end
        end
        counts[i1] = found
    end
    return
end

# 3D version of `_kernel_neighbors2!`, now looping over 27 neighboring cells.
function _kernel_neighbors3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                             neighbors_index::CuDeviceVector{Int32},
                             neighbors_flat::CuDeviceVector{Int32},
                             counts::CuDeviceVector{Int32},
                             cell_offsets::CuDeviceVector{Int32},
                             particle_ids_sorted::CuDeviceVector{Int32},
                             cell_of_particle::CuDeviceVector{Int32},
                             Lx::T, Ly::T, Lz::T,
                             halfLx::T, halfLy::T, halfLz::T,
                             nx::Int32, ny::Int32, nz::Int32,
                             rl2::T, cap::Int32) where {T<:AbstractFloat}
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        c0 = cell_of_particle[i1]
        cx = c0 % nx
        tmp = c0 ÷ nx
        cy = tmp % ny
        cz = tmp ÷ ny
        base  = neighbors_index[i1]
        found = Int32(0)
        for oz in Int32(-1):Int32(1)
            cz2 = cz + oz; cz2 -= (cz2 >= nz)*nz; cz2 += (cz2 < 0)*nz
            for oy in Int32(-1):Int32(1)
                cy2 = cy + oy; cy2 -= (cy2 >= ny)*ny; cy2 += (cy2 < 0)*ny
                for ox in Int32(-1):Int32(1)
                    cx2 = cx + ox; cx2 -= (cx2 >= nx)*nx; cx2 += (cx2 < 0)*nx
                    c = (cz2*ny + cy2)*nx + cx2
                    s = cell_offsets[c+1]
                    e = cell_offsets[c+2]
                    for k in s:(e-1)
                        j = particle_ids_sorted[k]
                        if j != i1
                            dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                            dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                            dz = mic_fast(rz[j] - rz[i1], halfLz, Lz)
                            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
                            if r2 <= rl2 && found < cap
                                neighbors_flat[base + found + 1] = j
                                found += 1
                            end
                        end
                    end
                end
            end
        end
        counts[i1] = found
    end
    return
end

# Store the coordinates used for the last successful rebuild.
function _kernel_copy_refs_2d!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                               rref_x::CuDeviceVector{T}, rref_y::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        rref_x[i] = rx[i]
        rref_y[i] = ry[i]
    end
    return
end

function _kernel_copy_refs_3d!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                               rref_x::CuDeviceVector{T}, rref_y::CuDeviceVector{T},
                               rref_z::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        rref_x[i] = rx[i]
        rref_y[i] = ry[i]
        rref_z[i] = rz[i]
    end
    return
end

# Accumulate the squared displacement from the reference coordinates.
function _kernel_accum_dr2_2d!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                               rref_x::CuDeviceVector{T}, rref_y::CuDeviceVector{T},
                               dr2::CuDeviceVector{T},
                               halfLx::T, halfLy::T, Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = mic_fast(rx[i] - rref_x[i], halfLx, Lx)
        dy = mic_fast(ry[i] - rref_y[i], halfLy, Ly)
        dr2[i] = muladd(dx, dx, dy*dy)
    end
    return
end

function _kernel_accum_dr2_3d!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                               rref_x::CuDeviceVector{T}, rref_y::CuDeviceVector{T}, rref_z::CuDeviceVector{T},
                               dr2::CuDeviceVector{T},
                               halfLx::T, halfLy::T, halfLz::T,
                               Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = mic_fast(rx[i] - rref_x[i], halfLx, Lx)
        dy = mic_fast(ry[i] - rref_y[i], halfLy, Ly)
        dz = mic_fast(rz[i] - rref_z[i], halfLz, Lz)
        dr2[i] = muladd(dx, dx, muladd(dy, dy, dz*dz))
    end
    return
end

# ============================================================================
# Binning helpers
# ============================================================================

function _bin_particles!(nbh::NeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1},
                         box::Tuple{T,T}) where {T<:AbstractFloat}
    N = Int(nbh.N)
    inv_cs = one(T) / nbh.cell_size
    threads, blocks = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed2!(rx, ry,
        T(box[1]), T(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys)
    kpack(rx, ry, T(box[1]), T(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys; threads, blocks)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted, nbh.cell_of_particle)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted, nbh.cell_of_particle; threads, blocks)

    ncell = Int(nbh.nx) * Int(nbh.ny)
    t2, b2 = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=t2, blocks=b2)
    return nothing
end

function _bin_particles!(nbh::NeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                         box::Tuple{T,T,T}) where {T<:AbstractFloat}
    N = Int(nbh.N)
    inv_cs = one(T) / nbh.cell_size
    threads, blocks = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed3!(rx, ry, rz,
        T(box[1]), T(box[2]), T(box[3]), inv_cs,
        nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys)
    kpack(rx, ry, rz, T(box[1]), T(box[2]), T(box[3]),
          inv_cs, nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys; threads, blocks)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted, nbh.cell_of_particle)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted, nbh.cell_of_particle; threads, blocks)

    ncell = Int(nbh.nx) * Int(nbh.ny) * Int(nbh.nz)
    t2, b2 = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=t2, blocks=b2)
    return nothing
end

# reuse for stencil matrices
_bin_particles!(nbh::StencilNeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, box::Tuple{T,T}) where {T<:AbstractFloat} =
    _bin_particles!(NeighborMatrix{T}(nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
                                      nbh.cap, zero(T), nbh.skin, zero(T),
                                      nbh.N, nbh.D, nbh.nx, nbh.ny, nbh.nz, nbh.cell_size,
                                      nbh.particle_ids_sorted, nbh.cell_ids_sorted, nbh.cell_offsets,
                                      nbh.cell_of_particle, nbh.packed_keys,
                                      nbh.rref_x, nbh.rref_y, nbh.rref_z, nbh.dr2,
                                      nbh.last_build_step, nbh.target_interval),
                   rx, ry, box)

_bin_particles!(nbh::StencilNeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}, box::Tuple{T,T,T}) where {T<:AbstractFloat} =
    _bin_particles!(NeighborMatrix{T}(nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
                                      nbh.cap, zero(T), nbh.skin, zero(T),
                                      nbh.N, nbh.D, nbh.nx, nbh.ny, nbh.nz, nbh.cell_size,
                                      nbh.particle_ids_sorted, nbh.cell_ids_sorted, nbh.cell_offsets,
                                      nbh.cell_of_particle, nbh.packed_keys,
                                      nbh.rref_x, nbh.rref_y, nbh.rref_z, nbh.dr2,
                                      nbh.last_build_step, nbh.target_interval),
                   rx, ry, rz, box)

# ============================================================================
# Public build routines
# ============================================================================

"""
    build_neighbors_dense!(rx, ry[, rz]; box, cutoff, cap, skin)

Construct a [`NeighborMatrix`](@ref) with uniform cutoff `cutoff` and
displacement buffer `skin`. The default path taken by `build_simulation`
(`examples/2D_example.jl` uses `cutoff = 2^(1/6)σ`, `skin = cutoff/2`).

# Arguments
- `rx, ry[, rz]`: Position components in `[-L/2, L/2)`.
- `box`: `(Lx, Ly[, Lz])` periodic extents.
- `cutoff`: Interaction cutoff; must be consistent with the potential (WCA or LJ).
- `cap`: Maximum stored neighbors per particle (e.g. `Int32(250)` in the WCA example).
- `skin`: Additional tolerance that delays rebuilds until particles move by
  `≈ skin/2`. Chosen from the validated examples/test (0.3–0.5 σ).
"""
function build_neighbors_dense!(rx::CuArray{T,1}, ry::CuArray{T,1};
                                box::Tuple{T,T}, cutoff::Real,
                                cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    nbh = _alloc_neighbor_matrix(T, N, 2, box, cutoff, skin, cap)
    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

function build_neighbors_dense!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                box::Tuple{T,T,T}, cutoff::Real,
                                cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    nbh = _alloc_neighbor_matrix(T, N, 3, box, cutoff, skin, cap)
    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

"""
    build_neighbors_stencil!(rx, ry[, rz]; box, rcut_particle, cap, skin)

Stencil neighbor list where each particle `i` owns a cutoff `rcut_particle[i]`.
`examples/3D_stencil_two_sizes.jl` uses this to mix `σ=5` and `σ=1` particles
by passing per-type cutoffs computed from the Lorentz mixing rule.
"""
function build_neighbors_stencil!(rx::CuArray{T,1}, ry::CuArray{T,1};
                                  box::Tuple{T,T},
                                  rcut_particle::AbstractVector{<:Real},
                                  cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(rcut_particle) == N
    rcut_host = T.(rcut_particle)
    min_rc = minimum(rcut_host)
    cell_size = max(T(1e-6), min_rc + skin)
    nbh = _alloc_stencil_matrix(T, N, 2, box, cell_size, skin, cap)
    rcut_d = CuArray(rcut_host)
    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, T(skin))
    kset(nbh.rlist, nbh.rlist2, rcut_d, T(skin); threads, blocks)
    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

function build_neighbors_stencil!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                  box::Tuple{T,T,T},
                                  rcut_particle::AbstractVector{<:Real},
                                  cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(rcut_particle) == N
    rcut_host = T.(rcut_particle)
    min_rc = minimum(rcut_host)
    cell_size = max(T(1e-6), min_rc + skin)
    nbh = _alloc_stencil_matrix(T, N, 3, box, cell_size, skin, cap)
    rcut_d = CuArray(rcut_host)
    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, T(skin))
    kset(nbh.rlist, nbh.rlist2, rcut_d, T(skin); threads, blocks)
    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

"""
    build_neighbors_stencil_by_types!(rx, ry[, rz]; box, typeid, rcut_pair, cap, skin)

Variant of [`build_neighbors_stencil!`](@ref) where the per-particle cutoff is
derived from a type–type lookup table `rcut_pair`. This mirrors the setup in
`examples/3D_stencil_two_sizes.jl`, which supplies the 2×2 `RCUT_PAIR` matrix
shown there.
"""
function build_neighbors_stencil_by_types!(rx::CuArray{T,1}, ry::CuArray{T,1};
                                           box::Tuple{T,T},
                                           typeid::CuArray{Int32,1},
                                           rcut_pair::AbstractMatrix{<:Real},
                                           cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    nt = size(rcut_pair, 1)
    @assert nt == size(rcut_pair, 2)
    rtype = [T(maximum(rcut_pair[t, :])) for t in 1:nt]
    cell_size = max(T(1e-6), minimum(rtype) + skin)
    nbh = _alloc_stencil_matrix(T, N, 2, box, cell_size, skin, cap)
    tid_h = Array(typeid)
    rcut_h = Vector{T}(undef, N)
    @inbounds for i in 1:N
        rcut_h[i] = rtype[Int(tid_h[i])]
    end
    rcut_d = CuArray(rcut_h)
    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, T(skin))
    kset(nbh.rlist, nbh.rlist2, rcut_d, T(skin); threads, blocks)
    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

function build_neighbors_stencil_by_types!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                           box::Tuple{T,T,T},
                                           typeid::CuArray{Int32,1},
                                           rcut_pair::AbstractMatrix{<:Real},
                                           cap::Int32, skin::Real) where {T<:AbstractFloat}
    N = length(rx)
    nt = size(rcut_pair, 1)
    @assert nt == size(rcut_pair, 2)
    rtype = [T(maximum(rcut_pair[t, :])) for t in 1:nt]
    cell_size = max(T(1e-6), minimum(rtype) + skin)
    nbh = _alloc_stencil_matrix(T, N, 3, box, cell_size, skin, cap)
    tid_h = Array(typeid)
    rcut_h = Vector{T}(undef, N)
    @inbounds for i in 1:N
        rcut_h[i] = rtype[Int(tid_h[i])]
    end
    rcut_d = CuArray(rcut_h)
    threads, blocks = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, T(skin))
    kset(nbh.rlist, nbh.rlist2, rcut_d, T(skin); threads, blocks)
    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

# ============================================================================
# R-list kernel for stencil
# ============================================================================

function _kernel_set_rlist!(rlist::CuDeviceVector{T}, rlist2::CuDeviceVector{T},
                            rcut::CuDeviceVector{T}, skin::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rlist); if i > N; return; end
    @inbounds begin
        rl = rcut[i] + skin
        rlist[i]  = rl
        rlist2[i] = rl * rl
    end
    return
end

# ============================================================================
# Update in place
# ============================================================================

"""
    update_neighbors_inplace!(nbh, rx, ry[, rz]; box, step=0)

Re-bin particles into cells, rebuild the CSR neighbor rows, and record the
reference coordinates used by [`update_needed!`](@ref). Called by `step!`
whenever the accumulated displacement exceeds `skin/2` or when the user forces
an update (e.g. after randomizing the configuration).
"""
function update_neighbors_inplace!(nbh::NeighborMatrix{T},
                                   rx::CuArray{T,1}, ry::CuArray{T,1};
                                   box::Tuple{T,T}, step::Int=0) where {T<:AbstractFloat}
    @assert nbh.D == 2
    _bin_particles!(nbh, rx, ry, box)
    fill!(nbh.counts, Int32(0))
    inv_cs = one(T) / nbh.cell_size
    halfLx = T(0.5)*box[1]; halfLy = T(0.5)*box[2]
    threads, blocks = _launchdims(Int(nbh.N))
    rl2 = (nbh.cutoff + nbh.skin) * (nbh.cutoff + nbh.skin)
    knei = CUDA.@cuda launch=false _kernel_neighbors2!(rx, ry,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted, nbh.cell_of_particle,
        box[1], box[2], halfLx, halfLy,
        nbh.nx, nbh.ny, rl2, nbh.cap)
    knei(rx, ry,
         nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted, nbh.cell_of_particle,
         box[1], box[2], halfLx, halfLy,
         nbh.nx, nbh.ny, rl2, nbh.cap; threads, blocks)

    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_2d!(rx, ry, nbh.rref_x, nbh.rref_y)
    kcopy(rx, ry, nbh.rref_x, nbh.rref_y; threads, blocks)
    nbh.last_build_step = step
    return nbh
end

function update_neighbors_inplace!(nbh::NeighborMatrix{T},
                                   rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                   box::Tuple{T,T,T}, step::Int=0) where {T<:AbstractFloat}
    @assert nbh.D == 3
    _bin_particles!(nbh, rx, ry, rz, box)
    fill!(nbh.counts, Int32(0))
    threads, blocks = _launchdims(Int(nbh.N))
    halfLx = T(0.5)*box[1]; halfLy = T(0.5)*box[2]; halfLz = T(0.5)*box[3]
    rl2 = (nbh.cutoff + nbh.skin) * (nbh.cutoff + nbh.skin)
    knei = CUDA.@cuda launch=false _kernel_neighbors3!(rx, ry, rz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted, nbh.cell_of_particle,
        box[1], box[2], box[3],
        halfLx, halfLy, halfLz,
        nbh.nx, nbh.ny, nbh.nz, rl2, nbh.cap)
    knei(rx, ry, rz,
         nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted, nbh.cell_of_particle,
         box[1], box[2], box[3],
         halfLx, halfLy, halfLz,
         nbh.nx, nbh.ny, nbh.nz, rl2, nbh.cap; threads, blocks)

    if nbh.rref_z === nothing
        nbh.rref_z = CUDA.CuArray{T}(undef, Int(nbh.N))
    end
    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_3d!(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1})
    kcopy(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1}; threads, blocks)
    nbh.last_build_step = step
    return nbh
end

function update_neighbors_inplace!(nbh::StencilNeighborMatrix{T},
                                   rx::CuArray{T,1}, ry::CuArray{T,1};
                                   box::Tuple{T,T}, step::Int=0) where {T<:AbstractFloat}
    @assert nbh.D == 2
    _bin_particles!(nbh, rx, ry, box)
    fill!(nbh.counts, Int32(0))
    threads, blocks = _launchdims(Int(nbh.N))
    halfLx = T(0.5)*box[1]; halfLy = T(0.5)*box[2]
    knei = CUDA.@cuda launch=false _kernel_neighbors_stencil2!(rx, ry,
        nbh.rlist, nbh.rlist2,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        box[1], box[2], halfLx, halfLy,
        nbh.nx, nbh.ny, nbh.cell_size, nbh.cap)
    knei(rx, ry,
         nbh.rlist, nbh.rlist2,
         nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted,
         box[1], box[2], halfLx, halfLy,
         nbh.nx, nbh.ny, nbh.cell_size, nbh.cap; threads, blocks)

    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_2d!(rx, ry, nbh.rref_x, nbh.rref_y)
    kcopy(rx, ry, nbh.rref_x, nbh.rref_y; threads, blocks)
    nbh.last_build_step = step
    return nbh
end

function update_neighbors_inplace!(nbh::StencilNeighborMatrix{T},
                                   rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                   box::Tuple{T,T,T}, step::Int=0) where {T<:AbstractFloat}
    @assert nbh.D == 3
    _bin_particles!(nbh, rx, ry, rz, box)
    fill!(nbh.counts, Int32(0))
    threads, blocks = _launchdims(Int(nbh.N))
    halfLx = T(0.5)*box[1]; halfLy = T(0.5)*box[2]; halfLz = T(0.5)*box[3]
    knei = CUDA.@cuda launch=false _kernel_neighbors_stencil3!(rx, ry, rz,
        nbh.rlist, nbh.rlist2,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        box[1], box[2], box[3], halfLx, halfLy, halfLz,
        nbh.nx, nbh.ny, nbh.nz, nbh.cell_size, nbh.cap)
    knei(rx, ry, rz,
         nbh.rlist, nbh.rlist2,
         nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted,
         box[1], box[2], box[3], halfLx, halfLy, halfLz,
         nbh.nx, nbh.ny, nbh.nz, nbh.cell_size, nbh.cap; threads, blocks)

    if nbh.rref_z === nothing
        nbh.rref_z = CUDA.CuArray{T}(undef, Int(nbh.N))
    end
    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_3d!(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1})
    kcopy(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1}; threads, blocks)
    nbh.last_build_step = step
    return nbh
end

# ============================================================================
# Stencil neighbor kernels
# ============================================================================

# Stencil neighbor search: expand the search radius based on each particle's rlist.
function _kernel_neighbors_stencil2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                     rlist::CuDeviceVector{T}, rlist2::CuDeviceVector{T},
                                     neighbors_index::CuDeviceVector{Int32},
                                     neighbors_flat::CuDeviceVector{Int32},
                                     counts::CuDeviceVector{Int32},
                                     cell_offsets::CuDeviceVector{Int32},
                                     particle_ids_sorted::CuDeviceVector{Int32},
                                     Lx::T, Ly::T, halfLx::T, halfLy::T,
                                     nx::Int32, ny::Int32, cell_size::T, cap::Int32) where {T<:AbstractFloat}
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        inv_cs = one(T) / cell_size
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        base  = neighbors_index[i1]
        found = Int32(0)
        rl    = rlist[i1]
        rl2   = rlist2[i1]
        Rmax  = Int32(ceil(rl * inv_cs))
        for oy in -Rmax:Rmax
            cy2 = cy + Int32(oy)
            cy2 -= (cy2 >= ny)*ny
            cy2 += (cy2 < 0)*ny
            for ox in -Rmax:Rmax
                cx2 = cx + Int32(ox)
                cx2 -= (cx2 >= nx)*nx
                cx2 += (cx2 < 0)*nx
                c = cy2*nx + cx2
                s = cell_offsets[c+1]
                e = cell_offsets[c+2]
                for k in s:(e-1)
                    j = particle_ids_sorted[k]
                    if j != i1
                        dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                        dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                        r2 = muladd(dx, dx, dy*dy)
                        if r2 <= rl2 && found < cap
                            neighbors_flat[base + found + 1] = j
                            found += 1
                        end
                    end
                end
            end
        end
        counts[i1] = found
    end
    return
end

# 3D stencil neighbor builder with per-particle cutoff radii.
function _kernel_neighbors_stencil3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                     rlist::CuDeviceVector{T}, rlist2::CuDeviceVector{T},
                                     neighbors_index::CuDeviceVector{Int32},
                                     neighbors_flat::CuDeviceVector{Int32},
                                     counts::CuDeviceVector{Int32},
                                     cell_offsets::CuDeviceVector{Int32},
                                     particle_ids_sorted::CuDeviceVector{Int32},
                                     Lx::T, Ly::T, Lz::T,
                                     halfLx::T, halfLy::T, halfLz::T,
                                     nx::Int32, ny::Int32, nz::Int32,
                                     cell_size::T, cap::Int32) where {T<:AbstractFloat}
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        z = rz[i1] + halfLz; z -= floor(z / Lz)*Lz
        inv_cs = one(T) / cell_size
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = cz >= nz ? (nz-1) : cz
        base  = neighbors_index[i1]
        found = Int32(0)
        rl    = rlist[i1]
        rl2   = rlist2[i1]
        Rmax  = Int32(ceil(rl * inv_cs))
        for oz in -Rmax:Rmax
            cz2 = cz + Int32(oz); cz2 -= (cz2 >= nz)*nz; cz2 += (cz2 < 0)*nz
            for oy in -Rmax:Rmax
                cy2 = cy + Int32(oy); cy2 -= (cy2 >= ny)*ny; cy2 += (cy2 < 0)*ny
                for ox in -Rmax:Rmax
                    cx2 = cx + Int32(ox); cx2 -= (cx2 >= nx)*nx; cx2 += (cx2 < 0)*nx
                    c = (cz2*ny + cy2)*nx + cx2
                    s = cell_offsets[c+1]
                    e = cell_offsets[c+2]
                    for k in s:(e-1)
                        j = particle_ids_sorted[k]
                        if j != i1
                            dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                            dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                            dz = mic_fast(rz[j] - rz[i1], halfLz, Lz)
                            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
                            if r2 <= rl2 && found < cap
                                neighbors_flat[base + found + 1] = j
                                found += 1
                            end
                        end
                    end
                end
            end
        end
        counts[i1] = found
    end
    return
end

# ============================================================================
# Update needed? logic
# ============================================================================

"""
    update_needed!(nbh, rx, ry[, rz]; skin, Lx, Ly[, Lz], step)

Check whether the maximum displacement since the last rebuild exceeds
`skin/2`, or whether the adaptive rebuild interval (`target_interval`) has
elapsed. `step!` calls this every `NL_CHECK_STRIDE` steps. The heuristic mirrors
the values tuned in the 2D/3D production scripts (skin between 0.3 and 0.5 σ).
"""
function update_needed!(nbh::NeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1};
                        skin::Real, Lx::T, Ly::T, step::Int) where {T<:AbstractFloat}
    threads, blocks = _launchdims(length(rx))
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _kernel_accum_dr2_2d!(rx, ry, nbh.rref_x, nbh.rref_y, nbh.dr2,
                                                      halfLx, halfLy, Lx, Ly)
    k(rx, ry, nbh.rref_x, nbh.rref_y, nbh.dr2,
      halfLx, halfLy, Lx, Ly; threads, blocks)

    max_dr2 = maximum(nbh.dr2)
    threshold = T(0.25) * T(skin) * T(skin)
    rebuild_needed = (max_dr2 > threshold) || ((step - nbh.last_build_step) >= nbh.target_interval)
    if rebuild_needed
        if max_dr2 > threshold
            nbh.target_interval = max(5, Int(round(T(0.9) * nbh.target_interval)))
        elseif max_dr2 < T(0.1) * threshold
            nbh.target_interval = min(100, Int(round(T(1.1) * nbh.target_interval)))
        end
    end
    return rebuild_needed
end

function update_needed!(nbh::NeighborMatrix{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                        skin::Real, Lx::T, Ly::T, Lz::T, step::Int) where {T<:AbstractFloat}
    threads, blocks = _launchdims(length(rx))
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _kernel_accum_dr2_3d!(rx, ry, rz,
                                                      nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1},
                                                      nbh.dr2,
                                                      halfLx, halfLy, halfLz,
                                                      Lx, Ly, Lz)
    k(rx, ry, rz,
      nbh.rref_x, nbh.rref_y, nbh.rref_z::CuArray{T,1},
      nbh.dr2,
      halfLx, halfLy, halfLz,
      Lx, Ly, Lz; threads, blocks)

    max_dr2 = maximum(nbh.dr2)
    threshold = T(0.25) * T(skin) * T(skin)
    rebuild_needed = (max_dr2 > threshold) || ((step - nbh.last_build_step) >= nbh.target_interval)
    if rebuild_needed
        if max_dr2 > threshold
            nbh.target_interval = max(5, Int(round(T(0.9) * nbh.target_interval)))
        elseif max_dr2 < T(0.1) * threshold
            nbh.target_interval = min(100, Int(round(T(1.1) * nbh.target_interval)))
        end
    end
    return rebuild_needed
end

function update_needed!(nbh::StencilNeighborMatrix{T}, args...; kwargs...) where {T<:AbstractFloat}
    return update_needed!(NeighborMatrix{T}(nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
                                            nbh.cap, zero(T), nbh.skin, zero(T),
                                            nbh.N, nbh.D, nbh.nx, nbh.ny, nbh.nz, nbh.cell_size,
                                            nbh.particle_ids_sorted, nbh.cell_ids_sorted,
                                            nbh.cell_offsets, nbh.cell_of_particle, nbh.packed_keys,
                                            nbh.rref_x, nbh.rref_y, nbh.rref_z, nbh.dr2,
                                            nbh.last_build_step, nbh.target_interval),
                               args...; kwargs...)
end

end # module NeighborLists
