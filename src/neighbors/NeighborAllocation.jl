# ============================================================================
# Allocation helpers
# ============================================================================

function _validate_neighbor_grid(box, cell_size, nx::Integer, ny::Integer, nz::Integer, D::Int, style::AbstractString)
    ok = D == 2 ? (nx >= 3 && ny >= 3) : (nx >= 3 && ny >= 3 && nz >= 3)
    ok && return nothing
    grid = D == 2 ? "($(Int(nx)), $(Int(ny)))" : "($(Int(nx)), $(Int(ny)), $(Int(nz)))"
    throw(ArgumentError("$(style) requires at least 3 cells per periodic dimension to avoid duplicate neighbor-cell visits. Got grid=$(grid) for box=$(box) and cell_size=$(cell_size). Increase the box size, reduce cutoff+skin, or use a different neighbor strategy."))
end

function _alloc_neighbor_matrix(T::Type{<:AbstractFloat}, N::Int, D::Int,
                                box, cutoff::Real, skin::Real, cap::Int32)
    cutoffT = T(cutoff)
    skinT   = T(skin)
    cutoff2 = cutoffT * cutoffT
    cell_size = max(T(1e-6), cutoffT + skinT)
    nx, ny, nz = _choose_grid(box, cell_size, D)
    _validate_neighbor_grid(box, cell_size, nx, ny, nz, D, "Dense cell-list neighbor builder")

    neighbors_index = CuArray(Int32[(i - 1) * Int(cap) for i in 1:N])
    neighbors_flat  = CUDA.fill(Int32(-1), N * Int(cap))
    counts          = CUDA.CuArray{Int32}(undef, N)
    fill!(counts, Int32(0))

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
    _validate_neighbor_grid(box, cellT, nx, ny, nz, D, "Stencil cell-list neighbor builder")

    neighbors_index = CuArray(Int32[(i - 1) * Int(cap) for i in 1:N])
    neighbors_flat  = CUDA.fill(Int32(-1), N * Int(cap))
    counts          = CUDA.CuArray{Int32}(undef, N)
    fill!(counts, Int32(0))

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
