module NeighborLists

using CUDA

export NeighborMatrix, build_neighbors_dense!, update_neighbors_inplace!, update_needed!,
       StencilNeighborMatrix, build_neighbors_stencil!

abstract type AbstractNeighborMatrix end

# ───────────────────────────────────────────────────────────────────────────────
# Data structure (CSR neighbors + sorted cell bins; no atomics; GPU-only updates)
# ───────────────────────────────────────────────────────────────────────────────
mutable struct NeighborMatrix <: AbstractNeighborMatrix
    # CSR storage
    neighbors_index::CuArray{Int32,1}     # length N: (i-1)*cap
    neighbors_flat::CuArray{Int32,1}      # length N*cap: 1-based particle ids
    counts::CuArray{Int32,1}              # length N

    # parameters
    cap::Int32
    cutoff::Float32
    skin::Float32
    cutoff2::Float32

    # system size
    N::Int32
    D::Int32

    # cell grid
    nx::Int32
    ny::Int32
    nz::Int32
    cell_size::Float32

    # sorted bin data (no-atomics path)
    particle_ids_sorted::CuArray{Int32,1} # length N, 1-based ids sorted by cell
    cell_ids_sorted::CuArray{Int32,1}     # length N, sorted cell ids [0..ncell-1]
    cell_offsets::CuArray{Int32,1}        # length ncell+1, starts into *_sorted

    # workspace (preallocated each build; reused every update)
    packed_keys::CuArray{UInt64,1}        # length N, (cid<<32)|(i-1)
    
    # adaptive update tracking
    rref_x::CuArray{Float32,1}            # positions at last neighbor-list build
    rref_y::CuArray{Float32,1}
    rref_z::Union{CuArray{Float32,1}, Nothing}
    dr2::CuArray{Float32,1}               # working buffer for squared displacements
    last_build_step::Int32                # step when last rebuilt
    target_interval::Int32                # desired #steps between rebuilds (heuristic)
end

# ───────────────────────────────────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────────────────────────────────
@inline function _choose_grid(box::Tuple, cell_size::Float32, D::Int)
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

@inline function _launchdims(N::Int)
    t = (N < 100_000) ? 128 : 256
    b = cld(N, t)
    return t, b
end

# ───────────────────────────────────────────────────────────────────────────────
# Adaptive neighbor list update kernels
# ───────────────────────────────────────────────────────────────────────────────

function _kernel_accum_dr2_2d!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    rref_x::CuDeviceVector{Float32}, rref_y::CuDeviceVector{Float32},
    dr2::CuDeviceVector{Float32},
    halfLx::Float32, halfLy::Float32, Lx::Float32, Ly::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = mic_fast(rx[i] - rref_x[i], halfLx, Lx)
        dy = mic_fast(ry[i] - rref_y[i], halfLy, Ly)
        dr2[i] = muladd(dx,dx,dy*dy)
    end
    return
end

function _kernel_accum_dr2_3d!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    rref_x::CuDeviceVector{Float32}, rref_y::CuDeviceVector{Float32}, rref_z::CuDeviceVector{Float32},
    dr2::CuDeviceVector{Float32},
    halfLx::Float32, halfLy::Float32, halfLz::Float32,
    Lx::Float32, Ly::Float32, Lz::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = mic_fast(rx[i] - rref_x[i], halfLx, Lx)
        dy = mic_fast(ry[i] - rref_y[i], halfLy, Ly)
        dz = mic_fast(rz[i] - rref_z[i], halfLz, Lz)
        dr2[i] = muladd(dx,dx, muladd(dy,dy, dz*dz))
    end
    return
end

# ============================================================================
# Stencil neighbor lists (variable per-particle list radii)
# ----------------------------------------------------------------------------
# Implements a per-particle "stencil" search over neighbor cells. For each
# particle i with list radius rlist[i] = r_cut(i) + skin, we search all cells
# whose minimum possible distance to i's cell is <= rlist[i]. This follows the
# core idea in Howard et al. (2016) and HOOMD-blue's Stencil neighbor list.
# We reuse the same binned/sorted cell data structure as the dense NL.
# ============================================================================

mutable struct StencilNeighborMatrix <: AbstractNeighborMatrix
    # CSR storage (same layout)
    neighbors_index::CuArray{Int32,1}
    neighbors_flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}

    # capacity & system
    cap::Int32
    skin::Float32

    # system size
    N::Int32
    D::Int32

    # cell grid
    nx::Int32
    ny::Int32
    nz::Int32
    cell_size::Float32

    # sorted bin data
    particle_ids_sorted::CuArray{Int32,1}
    cell_ids_sorted::CuArray{Int32,1}
    cell_offsets::CuArray{Int32,1}
    packed_keys::CuArray{UInt64,1}

    # per-particle list radii (rcut_i + skin) and r^2
    rlist::CuArray{Float32,1}
    rlist2::CuArray{Float32,1}

    # adaptive update tracking
    rref_x::CuArray{Float32,1}
    rref_y::CuArray{Float32,1}
    rref_z::Union{CuArray{Float32,1}, Nothing}
    dr2::CuArray{Float32,1}
    last_build_step::Int32
    target_interval::Int32
end

@inline function _alloc_stencil_matrix(N::Int, D::Int, box, cell_size::Float32, skin::Float32, cap::Int32)
    nx, ny, nz = _choose_grid(box, cell_size, D)

    neighbors_index = CUDA.CuArray{Int32}(undef, N)
    counts          = CUDA.fill(Int32(0), N)
    neighbors_flat  = CUDA.fill(Int32(-1), N*Int(cap))
    t,b = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rowstarts!(neighbors_index, Int32(cap))
    kset(neighbors_index, Int32(cap); threads=t, blocks=b)

    particle_ids_sorted = CUDA.CuArray{Int32}(undef, N)
    cell_ids_sorted     = CUDA.CuArray{Int32}(undef, N)
    ncell = Int(nx)*Int(ny)*Int(nz)
    cell_offsets        = CUDA.fill(Int32(1), ncell+1)
    packed_keys         = CUDA.CuArray{UInt64}(undef, N)

    rlist   = CUDA.CuArray{Float32}(undef, N)
    rlist2  = CUDA.CuArray{Float32}(undef, N)
    rref_x  = CUDA.CuArray{Float32}(undef, N)
    rref_y  = CUDA.CuArray{Float32}(undef, N)
    rref_z  = D == 3 ? CUDA.CuArray{Float32}(undef, N) : CUDA.CuArray{Float32}(undef, 0)
    dr2     = CUDA.CuArray{Float32}(undef, N)

    return StencilNeighborMatrix(
        neighbors_index, neighbors_flat, counts,
        cap, skin,
        Int32(N), Int32(D),
        nx, ny, nz, cell_size,
        particle_ids_sorted, cell_ids_sorted, cell_offsets, packed_keys,
        rlist, rlist2,
        rref_x, rref_y, rref_z, dr2, 0, 20
    )
end

# Compute rlist and rlist2 on device
function _kernel_set_rlist!(rlist::CuDeviceVector{Float32}, rlist2::CuDeviceVector{Float32},
                            rcut::CuDeviceVector{Float32}, skin::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rlist); if i > N; return; end
    @inbounds begin
        rl = rcut[i] + skin
        rlist[i]  = rl
        rlist2[i] = rl*rl
    end
    return
end

# Stencil neighbor builders -----------------------------------------------------

function _kernel_neighbors_stencil2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    rlist::CuDeviceVector{Float32}, rlist2::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    cell_offsets::CuDeviceVector{Int32},
    particle_ids_sorted::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    nx::Int32, ny::Int32, inv_cs::Float32, cell_size::Float32,
    cap::Int32)
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        # own cell index (0-based)
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy

        base  = neighbors_index[i1]
        found = Int32(0)
        rl    = rlist[i1]
        rl2   = rlist2[i1]
        Rmax  = Int32(ceil(rl * inv_cs))

        @inbounds for oy in -Rmax:Rmax
            # Min separation along y between the two cell rectangles
            dy_cells = abs(oy) - 1
            dy_cells = dy_cells < 0 ? 0 : dy_cells
            dmin_y = Float32(dy_cells) * cell_size

            cy2 = cy + Int32(oy)
            cy2 -= (cy2 >= ny) * ny
            cy2 += (cy2 < 0)   * ny

            for ox in -Rmax:Rmax
                dx_cells = abs(ox) - 1
                dx_cells = dx_cells < 0 ? 0 : dx_cells
                dmin_x = Float32(dx_cells) * cell_size

                # quick reject by cell-to-cell minimum distance
                dmin2 = muladd(dmin_x, dmin_x, dmin_y*dmin_y)
                if dmin2 > rl2
                    continue
                end

                cx2 = cx + Int32(ox)
                cx2 -= (cx2 >= nx) * nx
                cx2 += (cx2 < 0)   * nx

                c = cy2*nx + cx2
                s = cell_offsets[c+1]
                e = cell_offsets[c+2]
                for k in s:(e-1)
                    j = particle_ids_sorted[k]
                    if j != i1
                        dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                        dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                        r2 = muladd(dx, dx, dy*dy)
                        if r2 <= rl2
                            if found < cap
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

function _kernel_neighbors_stencil3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    rlist::CuDeviceVector{Float32}, rlist2::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    cell_offsets::CuDeviceVector{Int32},
    particle_ids_sorted::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    nx::Int32, ny::Int32, nz::Int32, inv_cs::Float32, cell_size::Float32,
    cap::Int32)
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        z = rz[i1] + halfLz; z -= floor(z / Lz)*Lz
        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = (cz >= nz) ? (nz-1) : cz

        base  = neighbors_index[i1]
        found = Int32(0)
        rl    = rlist[i1]
        rl2   = rlist2[i1]
        Rmax  = Int32(ceil(rl * inv_cs))

        for oz in -Rmax:Rmax
            dz_cells = abs(oz) - 1; dz_cells = dz_cells < 0 ? 0 : dz_cells
            dmin_z = Float32(dz_cells) * cell_size
            cz2 = cz + Int32(oz)
            cz2 -= (cz2 >= nz) * nz
            cz2 += (cz2 < 0)   * nz
            for oy in -Rmax:Rmax
                dy_cells = abs(oy) - 1; dy_cells = dy_cells < 0 ? 0 : dy_cells
                dmin_y = Float32(dy_cells) * cell_size
                cy2 = cy + Int32(oy)
                cy2 -= (cy2 >= ny) * ny
                cy2 += (cy2 < 0)   * ny
                for ox in -Rmax:Rmax
                    dx_cells = abs(ox) - 1; dx_cells = dx_cells < 0 ? 0 : dx_cells
                    dmin_x = Float32(dx_cells) * cell_size
                    dmin2 = muladd(dmin_x, dmin_x, muladd(dmin_y, dmin_y, dmin_z*dmin_z))
                    if dmin2 > rl2
                        continue
                    end
                    cx2 = cx + Int32(ox)
                    cx2 -= (cx2 >= nx) * nx
                    cx2 += (cx2 < 0)   * nx
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
                            if r2 <= rl2
                                if found < cap
                                    neighbors_flat[base + found + 1] = j
                                    found += 1
                                end
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

# Public API --------------------------------------------------------------

"""
    build_neighbors_stencil!(rx, ry; box, rcut_particle, skin, cap)

Construct a variable-radius stencil neighbor list in 2D. `rcut_particle`
is the per-particle cutoff (without skin). The internal list radius is
`rcut_particle + skin` and cell size is chosen as `minimum(rcut_particle)+skin`.
"""
function build_neighbors_stencil!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1};
                                  box::Tuple{Float32,Float32},
                                  rcut_particle::AbstractVector{<:Real},
                                  cap::Int32,
                                  skin::Float32)
    N = length(rx)
    @assert length(rcut_particle) == N "rcut_particle must have length N"
    # Host compute base cell size as min(rlist)
    rcut_host = Float32.(collect(rcut_particle))
    min_rc = minimum(rcut_host)
    cell_size = max(Float32(1e-6), min_rc + skin)
    nbh = _alloc_stencil_matrix(N, 2, box, cell_size, skin, cap)

    # set rlist arrays on device
    rcut_d = CuArray(rcut_host)
    t,b = _launchdims(N)
    ksetr = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, skin)
    ksetr(nbh.rlist, nbh.rlist2, rcut_d, skin; threads=t, blocks=b)

    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

"""
    build_neighbors_stencil!(rx, ry, rz; box, rcut_particle, skin, cap)

3D variant.
"""
function build_neighbors_stencil!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1};
                                  box::Tuple{Float32,Float32,Float32},
                                  rcut_particle::AbstractVector{<:Real},
                                  cap::Int32,
                                  skin::Float32)
    N = length(rx)
    @assert length(rcut_particle) == N "rcut_particle must have length N"
    rcut_host = Float32.(collect(rcut_particle))
    min_rc = minimum(rcut_host)
    cell_size = max(Float32(1e-6), min_rc + skin)
    nbh = _alloc_stencil_matrix(N, 3, box, cell_size, skin, cap)

    rcut_d = CuArray(rcut_host)
    t,b = _launchdims(N)
    ksetr = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, skin)
    ksetr(nbh.rlist, nbh.rlist2, rcut_d, skin; threads=t, blocks=b)

    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

"""
    build_neighbors_stencil!(rx, ry; box, typeid, rcut_pair, skin, cap)

Construct a stencil neighbor list by TYPE using a full pair cutoff matrix `rcut_pair`.
For each type `t`, the stencil radius is `max_j rcut_pair[t,j] + skin` as in Howard et al.
Cell size is chosen as `minimum_t(max_j rcut_pair[t,j]) + skin`.
"""
function build_neighbors_stencil_by_types!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1};
                                  box::Tuple{Float32,Float32},
                                  typeid::CuArray{Int32,1},
                                  rcut_pair::AbstractMatrix{<:Real},
                                  cap::Int32,
                                  skin::Float32)
    N = length(rx)
    @assert size(rcut_pair, 1) == size(rcut_pair, 2) "rcut_pair must be square (types×types)"
    T = size(rcut_pair, 1)
    # per-type max cutoff
    rtype = Vector{Float32}(undef, T)
    for t in 1:T
        rtype[t] = Float32(maximum(rcut_pair[t, :]))
    end
    cell_size = max(Float32(1e-6), minimum(rtype) + skin)
    nbh = _alloc_stencil_matrix(N, 2, box, cell_size, skin, cap)
    # per-particle base cutoffs from type
    tid_h = Array(typeid)
    rcut_h = Vector{Float32}(undef, N)
    @inbounds for i in 1:N
        t = Int(tid_h[i])
        rcut_h[i] = rtype[t]
    end
    rcut_d = CuArray(rcut_h)
    t,b = _launchdims(N)
    ksetr = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, skin)
    ksetr(nbh.rlist, nbh.rlist2, rcut_d, skin; threads=t, blocks=b)
    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

function build_neighbors_stencil_by_types!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1};
                                  box::Tuple{Float32,Float32,Float32},
                                  typeid::CuArray{Int32,1},
                                  rcut_pair::AbstractMatrix{<:Real},
                                  cap::Int32,
                                  skin::Float32)
    N = length(rx)
    @assert size(rcut_pair, 1) == size(rcut_pair, 2) "rcut_pair must be square (types×types)"
    T = size(rcut_pair, 1)
    rtype = Vector{Float32}(undef, T)
    for t in 1:T
        rtype[t] = Float32(maximum(rcut_pair[t, :]))
    end
    cell_size = max(Float32(1e-6), minimum(rtype) + skin)
    nbh = _alloc_stencil_matrix(N, 3, box, cell_size, skin, cap)
    tid_h = Array(typeid)
    rcut_h = Vector{Float32}(undef, N)
    @inbounds for i in 1:N
        rcut_h[i] = rtype[Int(tid_h[i])]
    end
    rcut_d = CuArray(rcut_h)
    t,b = _launchdims(N)
    ksetr = CUDA.@cuda launch=false _kernel_set_rlist!(nbh.rlist, nbh.rlist2, rcut_d, skin)
    ksetr(nbh.rlist, nbh.rlist2, rcut_d, skin; threads=t, blocks=b)
    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

# Updates ----------------------------------------------------------------

function update_neighbors_inplace!(nbh::StencilNeighborMatrix,
                                   rx::CuArray{Float32,1}, ry::CuArray{Float32,1};
                                   box::Tuple{Float32,Float32}, step::Int=0)
    @assert nbh.D == 2
    N = Int(nbh.N)
    _bin_particles_2d!(nbh, rx, ry, box)
    fill!(nbh.counts, Int32(0))
    inv_cs = 1f0 / nbh.cell_size
    halfLx = 0.5f0 * Float32(box[1])
    halfLy = 0.5f0 * Float32(box[2])
    t,b = _launchdims(N)
    k = CUDA.@cuda launch=false _kernel_neighbors_stencil2!(rx, ry, nbh.rlist, nbh.rlist2,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        Float32(box[1]), Float32(box[2]), halfLx, halfLy,
        nbh.nx, nbh.ny, inv_cs, nbh.cell_size, nbh.cap)
    k(rx, ry, nbh.rlist, nbh.rlist2,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      nbh.cell_offsets, nbh.particle_ids_sorted,
      Float32(box[1]), Float32(box[2]), halfLx, halfLy,
      nbh.nx, nbh.ny, inv_cs, nbh.cell_size, nbh.cap; threads=t, blocks=b)

    # Copy current positions as reference for adaptive updates
    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_2d!(rx, ry, nbh.rref_x, nbh.rref_y)
    kcopy(rx, ry, nbh.rref_x, nbh.rref_y; threads=t, blocks=b)
    nbh.last_build_step = step
    return nbh
end

function update_neighbors_inplace!(nbh::StencilNeighborMatrix,
                                   rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1};
                                   box::Tuple{Float32,Float32,Float32}, step::Int=0)
    @assert nbh.D == 3
    N = Int(nbh.N)
    _bin_particles_3d!(nbh, rx, ry, rz, box)
    fill!(nbh.counts, Int32(0))
    inv_cs = 1f0 / nbh.cell_size
    halfLx = 0.5f0 * Float32(box[1])
    halfLy = 0.5f0 * Float32(box[2])
    halfLz = 0.5f0 * Float32(box[3])
    t,b = _launchdims(N)
    k = CUDA.@cuda launch=false _kernel_neighbors_stencil3!(rx, ry, rz, nbh.rlist, nbh.rlist2,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        Float32(box[1]), Float32(box[2]), Float32(box[3]), halfLx, halfLy, halfLz,
        nbh.nx, nbh.ny, nbh.nz, inv_cs, nbh.cell_size, nbh.cap)
    k(rx, ry, rz, nbh.rlist, nbh.rlist2,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      nbh.cell_offsets, nbh.particle_ids_sorted,
      Float32(box[1]), Float32(box[2]), Float32(box[3]), halfLx, halfLy, halfLz,
      nbh.nx, nbh.ny, nbh.nz, inv_cs, nbh.cell_size, nbh.cap; threads=t, blocks=b)

    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_3d!(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z)
    kcopy(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z; threads=t, blocks=b)
    nbh.last_build_step = step
    return nbh
end

# Reuse the same adaptive displacement-based rebuild checks for stencil NL
function update_needed!(neigh::StencilNeighborMatrix, rx, ry; 
                        skin=1.5f0, Lx::Float32, Ly::Float32, step::Int)
    N = length(rx)
    halfLx, halfLy = 0.5f0*Lx, 0.5f0*Ly
    t, b = _launchdims(N)
    @cuda threads=t blocks=b _kernel_accum_dr2_2d!(rx, ry, neigh.rref_x, neigh.rref_y, neigh.dr2, halfLx, halfLy, Lx, Ly)
    max_dr2 = maximum(neigh.dr2)
    rebuild_threshold = 0.25f0 * skin * skin
    steps_since_build = step - neigh.last_build_step
    interval_trigger = steps_since_build >= neigh.target_interval
    rebuild_needed = (max_dr2 > rebuild_threshold) || interval_trigger
    if rebuild_needed
        if max_dr2 > rebuild_threshold
            neigh.target_interval = max(5, Int(round(0.9f0 * neigh.target_interval)))
        elseif interval_trigger && max_dr2 < 0.1f0 * rebuild_threshold
            neigh.target_interval = min(100, Int(round(1.1f0 * neigh.target_interval)))
        end
    end
    return rebuild_needed
end

function update_needed!(neigh::StencilNeighborMatrix, rx, ry, rz; 
                        skin=1.5f0, Lx::Float32, Ly::Float32, Lz::Float32, step::Int)
    N = length(rx)
    halfLx, halfLy, halfLz = 0.5f0*Lx, 0.5f0*Ly, 0.5f0*Lz
    t, b = _launchdims(N)
    @cuda threads=t blocks=b _kernel_accum_dr2_3d!(rx, ry, rz, neigh.rref_x, neigh.rref_y, neigh.rref_z, neigh.dr2, halfLx, halfLy, halfLz, Lx, Ly, Lz)
    max_dr2 = maximum(neigh.dr2)
    rebuild_threshold = 0.25f0 * skin * skin
    steps_since_build = step - neigh.last_build_step
    interval_trigger = steps_since_build >= neigh.target_interval
    rebuild_needed = (max_dr2 > rebuild_threshold) || interval_trigger
    if rebuild_needed
        if max_dr2 > rebuild_threshold
            neigh.target_interval = max(5, Int(round(0.9f0 * neigh.target_interval)))
        elseif interval_trigger && max_dr2 < 0.1f0 * rebuild_threshold
            neigh.target_interval = min(100, Int(round(1.1f0 * neigh.target_interval)))
        end
    end
    return rebuild_needed
end

function _kernel_copy_refs_2d!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
                               rref_x::CuDeviceVector{Float32}, rref_y::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        rref_x[i] = rx[i]
        rref_y[i] = ry[i]
    end
    return
end

function _kernel_copy_refs_3d!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
                               rref_x::CuDeviceVector{Float32}, rref_y::CuDeviceVector{Float32}, rref_z::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        rref_x[i] = rx[i]
        rref_y[i] = ry[i]
        rref_z[i] = rz[i]
    end
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Adaptive neighbor list update control
# ───────────────────────────────────────────────────────────────────────────────

"""
    update_needed!(neigh::NeighborMatrix, rx, ry, [rz]; 
                   skin=1.5f0, Lx, Ly, [Lz], step)

Determines if neighbor list rebuild is needed based on particle displacements.
Uses adaptive skin parameter tuning and overflow protection.
"""
function update_needed!(neigh::NeighborMatrix, rx, ry; 
                       skin=1.5f0, Lx::Float32, Ly::Float32, step::Int)
    N = length(rx)
    halfLx, halfLy = 0.5f0*Lx, 0.5f0*Ly
    t, b = _launchdims(N)
    
    # Calculate squared displacements since last rebuild
    @cuda threads=t blocks=b _kernel_accum_dr2_2d!(
        rx, ry, neigh.rref_x, neigh.rref_y, neigh.dr2, 
        halfLx, halfLy, Lx, Ly
    )
    
    # Check maximum displacement (GPU reduction)
    max_dr2 = maximum(neigh.dr2)
    
    # Rebuild criterion: max displacement > 0.5 * skin
    rebuild_threshold = 0.25f0 * skin * skin
    
    # Also check for potential overflow based on interval
    steps_since_build = step - neigh.last_build_step
    interval_trigger = steps_since_build >= neigh.target_interval
    
    rebuild_needed = (max_dr2 > rebuild_threshold) || interval_trigger
    
    # Adaptive skin tuning based on performance
    if rebuild_needed
        if max_dr2 > rebuild_threshold
            # Particles moved too much - reduce interval slightly
            neigh.target_interval = max(5, Int(round(0.9f0 * neigh.target_interval)))
        elseif interval_trigger && max_dr2 < 0.1f0 * rebuild_threshold
            # Interval expired but particles barely moved - increase interval
            neigh.target_interval = min(100, Int(round(1.1f0 * neigh.target_interval)))
        end
    end
    
    return rebuild_needed
end

function update_needed!(neigh::NeighborMatrix, rx, ry, rz; 
                       skin=1.5f0, Lx::Float32, Ly::Float32, Lz::Float32, step::Int)
    N = length(rx)
    halfLx, halfLy, halfLz = 0.5f0*Lx, 0.5f0*Ly, 0.5f0*Lz
    t, b = _launchdims(N)
    
    # Calculate squared displacements since last rebuild
    @cuda threads=t blocks=b _kernel_accum_dr2_3d!(
        rx, ry, rz, neigh.rref_x, neigh.rref_y, neigh.rref_z, neigh.dr2,
        halfLx, halfLy, halfLz, Lx, Ly, Lz
    )
    
    # Check maximum displacement (GPU reduction)
    max_dr2 = maximum(neigh.dr2)
    
    # Rebuild criterion: max displacement > 0.5 * skin
    rebuild_threshold = 0.25f0 * skin * skin
    
    # Also check for potential overflow based on interval
    steps_since_build = step - neigh.last_build_step
    interval_trigger = steps_since_build >= neigh.target_interval
    
    rebuild_needed = (max_dr2 > rebuild_threshold) || interval_trigger
    
    # Adaptive skin tuning based on performance
    if rebuild_needed
        if max_dr2 > rebuild_threshold
            # Particles moved too much - reduce interval slightly
            neigh.target_interval = max(5, Int(round(0.9f0 * neigh.target_interval)))
        elseif interval_trigger && max_dr2 < 0.1f0 * rebuild_threshold
            # Interval expired but particles barely moved - increase interval
            neigh.target_interval = min(100, Int(round(1.1f0 * neigh.target_interval)))
        end
    end
    
    return rebuild_needed
end

# neighbors_index[i] = (i-1)*cap
function _kernel_set_rowstarts!(neighbors_index::CuDeviceVector{Int32}, cap::Int32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(neighbors_index); if i > N; return; end
    @inbounds neighbors_index[i] = (i-1) * cap
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Key packing kernels (no divide; use reciprocal and clamp)
# We compute:
#   cx = floor((x+L/2)*invL * (L/cell) ) == floor((x+L/2) * inv_cell)
# then clamp to [0, n-1].
# ───────────────────────────────────────────────────────────────────────────────

# 2D: packed[i] = (UInt64(cid)<<32) | (i-1)
function _kernel_compute_packed2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    Lx::Float32, Ly::Float32, inv_cs::Float32, nx::Int32, ny::Int32,
    packed::CuDeviceVector{UInt64}
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        # shift to [0,L)
        x = rx[i] + 0.5f0*Lx
        y = ry[i] + 0.5f0*Ly
        # wrap without divide (fast path if already in [0,L); fallback is cheap)
        x -= floor(x/Lx)*Lx
        y -= floor(y/Ly)*Ly

        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy

        cid = Int32(cy*nx + cx)
        packed[i] = (UInt64(UInt32(cid)) << 32) | UInt64(UInt32(i-1))
    end
    return
end

# 3D
function _kernel_compute_packed3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    Lx::Float32, Ly::Float32, Lz::Float32, inv_cs::Float32, nx::Int32, ny::Int32, nz::Int32,
    packed::CuDeviceVector{UInt64}
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        x = rx[i] + 0.5f0*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + 0.5f0*Ly; y -= floor(y/Ly)*Ly
        z = rz[i] + 0.5f0*Lz; z -= floor(z/Lz)*Lz

        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = (cz >= nz) ? (nz-1) : cz

        cid = Int32((cz*ny + cy)*nx + cx)
        packed[i] = (UInt64(UInt32(cid)) << 32) | UInt64(UInt32(i-1))
    end
    return
end

# After sort: unpack to cell_ids_sorted & particle_ids_sorted
function _kernel_unpack_sorted!(
    packed::CuDeviceVector{UInt64},
    cell_ids_sorted::CuDeviceVector{Int32},
    particle_ids_sorted::CuDeviceVector{Int32}
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(packed); if i > N; return; end
    @inbounds begin
        pv = packed[i]
        cell_ids_sorted[i]     = Int32(UInt32(pv >> 32))
        particle_ids_sorted[i] = Int32(UInt32(pv & 0xFFFF_FFFF)) + 1
    end
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Build cell_offsets on GPU: lower_bound-style search per cell id
# Each thread handles one cell id and binary-searches the sorted key array.
# ncell ≪ N, so O(ncell log N) is cheap and avoids host copies/sync.
# ───────────────────────────────────────────────────────────────────────────────
@inline function _lb_search(a::CuDeviceVector{Int32}, N::Int32, key::Int32)::Int32
    lo = Int32(1)
    hi = N + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        v = a[mid]
        if v < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

function _kernel_cell_offsets!(
    cell_ids_sorted::CuDeviceVector{Int32},
    cell_offsets::CuDeviceVector{Int32},
    ncell::Int32
)
    c = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if c > ncell+1 || c < 1; return; end
    N = Int32(length(cell_ids_sorted))
    if c <= ncell
        @inbounds cell_offsets[c] = _lb_search(cell_ids_sorted, N, Int32(c-1))
    else
        @inbounds cell_offsets[c] = N + 1
    end
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# MIC without divide/round: clamp by half box
# Assumes positions are maintained in [-L/2, L/2); this holds for your code.
# ───────────────────────────────────────────────────────────────────────────────
@inline function mic_fast(dx::Float32, halfL::Float32, L::Float32)
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# ───────────────────────────────────────────────────────────────────────────────
# Neighbor builders (2D / 3D)
# Use inv_cs and mic_fast (no divides) on hot paths
# ───────────────────────────────────────────────────────────────────────────────

function _kernel_neighbors2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    cell_offsets::CuDeviceVector{Int32}, # len = ncell+1
    particle_ids_sorted::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    nx::Int32, ny::Int32, inv_cs::Float32,
    cutoff2::Float32, cap::Int32
)
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        # own cell (no divide)
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy

        base  = neighbors_index[i1]
        found = Int32(0)

        for oy in Int32(-1):Int32(1)
            cy2 = cy + oy; cy2 -= (cy2 >= ny)*ny; cy2 += (cy2 < 0)*ny
            for ox in Int32(-1):Int32(1)
                cx2 = cx + ox; cx2 -= (cx2 >= nx)*nx; cx2 += (cx2 < 0)*nx
                c   = cy2*nx + cx2   # 0-based
                s   = cell_offsets[c+1]
                e   = cell_offsets[c+2]
                for k in s:(e-1)
                    j = particle_ids_sorted[k]
                    if j != i1
                        dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                        dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                        r2 = muladd(dx, dx, dy*dy)
                        if r2 <= cutoff2
                            if found < cap
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

function _kernel_neighbors3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    cell_offsets::CuDeviceVector{Int32},
    particle_ids_sorted::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    nx::Int32, ny::Int32, nz::Int32, inv_cs::Float32,
    cutoff2::Float32, cap::Int32
)
    i1 = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i1 > N; return; end
    @inbounds begin
        x = rx[i1] + halfLx; x -= floor(x / Lx)*Lx
        y = ry[i1] + halfLy; y -= floor(y / Ly)*Ly
        z = rz[i1] + halfLz; z -= floor(z / Lz)*Lz
        cx = Int32(floor(x * inv_cs)); cx = (cx >= nx) ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = (cy >= ny) ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = (cz >= nz) ? (nz-1) : cz

        base  = neighbors_index[i1]
        found = Int32(0)

        for oz in Int32(-1):Int32(1)
            cz2 = cz + oz; cz2 -= (cz2 >= nz)*nz; cz2 += (cz2 < 0)*nz
            for oy in Int32(-1):Int32(1)
                cy2 = cy + oy; cy2 -= (cy2 >= ny)*ny; cy2 += (cy2 < 0)*ny
                for ox in Int32(-1):Int32(1)
                    cx2 = cx + ox; cx2 -= (cx2 >= nx)*nx; cx2 += (cx2 < 0)*nx
                    c   = (cz2*ny + cy2)*nx + cx2
                    s   = cell_offsets[c+1]
                    e   = cell_offsets[c+2]
                    for k in s:(e-1)
                        j = particle_ids_sorted[k]
                        if j != i1
                            dx = mic_fast(rx[j] - rx[i1], halfLx, Lx)
                            dy = mic_fast(ry[j] - ry[i1], halfLy, Ly)
                            dz = mic_fast(rz[j] - rz[i1], halfLz, Lz)
                            r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
                            if r2 <= cutoff2
                                if found < cap
                                    neighbors_flat[base + found + 1] = j
                                    found += 1
                                end
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

# ───────────────────────────────────────────────────────────────────────────────
# Allocation & construction
# ───────────────────────────────────────────────────────────────────────────────
function _alloc_neighbor_matrix(N::Int, D::Int, box, cutoff::Float32, skin::Float32, cap::Int32)
    cutoff2   = cutoff*cutoff
    cell_size = cutoff + skin
    nx, ny, nz = _choose_grid(box, cell_size, D)

    # CSR
    neighbors_index = CUDA.CuArray{Int32}(undef, N)
    counts          = CUDA.fill(Int32(0), N)
    neighbors_flat  = CUDA.fill(Int32(-1), N*Int(cap))
    t,b = _launchdims(N)
    kset = CUDA.@cuda launch=false _kernel_set_rowstarts!(neighbors_index, Int32(cap))
    kset(neighbors_index, Int32(cap); threads=t, blocks=b)

    # sorted bins + workspace
    particle_ids_sorted = CUDA.CuArray{Int32}(undef, N)
    cell_ids_sorted     = CUDA.CuArray{Int32}(undef, N)
    ncell = Int(nx)*Int(ny)*Int(nz)
    cell_offsets        = CUDA.fill(Int32(1), ncell+1)
    packed_keys         = CUDA.CuArray{UInt64}(undef, N)

    # Adaptive neighbor list fields
    rref_x = CUDA.CuArray{Float32}(undef, N)
    rref_y = CUDA.CuArray{Float32}(undef, N)
    rref_z = D == 3 ? CUDA.CuArray{Float32}(undef, N) : CUDA.CuArray{Float32}(undef, 0)
    dr2 = CUDA.CuArray{Float32}(undef, N)

    return NeighborMatrix(
        neighbors_index, neighbors_flat, counts,
        cap, cutoff, skin, cutoff2,
        Int32(N), Int32(D),
        nx, ny, nz, cell_size,
        particle_ids_sorted, cell_ids_sorted, cell_offsets,
        packed_keys,
        rref_x, rref_y, rref_z, dr2, 0, 20  # last_build_step=0, target_interval=20
    )
end

# ───────────────────────────────────────────────────────────────────────────────
# Binning without atomics: GPU key sort + GPU cell_offsets via lower_bound
# ───────────────────────────────────────────────────────────────────────────────
function _bin_particles_2d!(nbh::NeighborMatrix,
                            rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                            box::Tuple{Float32,Float32})
    N = Int(nbh.N)
    inv_cs = 1f0 / nbh.cell_size
    t,b = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed2!(rx, ry,
        Float32(box[1]), Float32(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys)
    kpack(rx, ry, Float32(box[1]), Float32(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys; threads=t, blocks=b)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted; threads=t, blocks=b)

    # cell_offsets on GPU (ncell+1 threads)
    ncell = Int(nbh.nx) * Int(nbh.ny)
    tt,bb = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=tt, blocks=bb)
    return nothing
end

function _bin_particles_3d!(nbh::NeighborMatrix,
                            rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                            box::Tuple{Float32,Float32,Float32})
    N = Int(nbh.N)
    inv_cs = 1f0 / nbh.cell_size
    t,b = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed3!(rx, ry, rz,
        Float32(box[1]), Float32(box[2]), Float32(box[3]), inv_cs, nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys)
    kpack(rx, ry, rz, Float32(box[1]), Float32(box[2]), Float32(box[3]),
          inv_cs, nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys; threads=t, blocks=b)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted; threads=t, blocks=b)

    ncell = Int(nbh.nx) * Int(nbh.ny) * Int(nbh.nz)
    tt,bb = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=tt, blocks=bb)
    return nothing
end

# Overloads for stencil matrices (reuse identical binning path)
function _bin_particles_2d!(nbh::StencilNeighborMatrix,
                            rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                            box::Tuple{Float32,Float32})
    N = Int(nbh.N)
    inv_cs = 1f0 / nbh.cell_size
    t,b = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed2!(rx, ry,
        Float32(box[1]), Float32(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys)
    kpack(rx, ry, Float32(box[1]), Float32(box[2]), inv_cs, nbh.nx, nbh.ny, nbh.packed_keys; threads=t, blocks=b)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted; threads=t, blocks=b)

    ncell = Int(nbh.nx) * Int(nbh.ny)
    tt,bb = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=tt, blocks=bb)
    return nothing
end

function _bin_particles_3d!(nbh::StencilNeighborMatrix,
                            rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                            box::Tuple{Float32,Float32,Float32})
    N = Int(nbh.N)
    inv_cs = 1f0 / nbh.cell_size
    t,b = _launchdims(N)
    kpack = CUDA.@cuda launch=false _kernel_compute_packed3!(rx, ry, rz,
        Float32(box[1]), Float32(box[2]), Float32(box[3]), inv_cs, nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys)
    kpack(rx, ry, rz, Float32(box[1]), Float32(box[2]), Float32(box[3]),
          inv_cs, nbh.nx, nbh.ny, nbh.nz, nbh.packed_keys; threads=t, blocks=b)

    CUDA.sort!(nbh.packed_keys)

    kunpack = CUDA.@cuda launch=false _kernel_unpack_sorted!(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted)
    kunpack(nbh.packed_keys, nbh.cell_ids_sorted, nbh.particle_ids_sorted; threads=t, blocks=b)

    ncell = Int(nbh.nx) * Int(nbh.ny) * Int(nbh.nz)
    tt,bb = _launchdims(ncell+1)
    koff = CUDA.@cuda launch=false _kernel_cell_offsets!(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell))
    koff(nbh.cell_ids_sorted, nbh.cell_offsets, Int32(ncell); threads=tt, blocks=bb)
    return nothing
end

# ───────────────────────────────────────────────────────────────────────────────
# Public: build & update (no allocations; no host sync)
# ───────────────────────────────────────────────────────────────────────────────
function build_neighbors_dense!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1};
                                box::Tuple{Float32,Float32},
                                cutoff::Float32,
                                cap::Int32,
                                skin::Float32)
    N = length(rx)
    nbh = _alloc_neighbor_matrix(N, 2, box, cutoff, skin, cap)
    update_neighbors_inplace!(nbh, rx, ry; box)
    return nbh
end

function build_neighbors_dense!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1};
                                box::Tuple{Float32,Float32,Float32},
                                cutoff::Float32,
                                cap::Int32,
                                skin::Float32)
    N = length(rx)
    nbh = _alloc_neighbor_matrix(N, 3, box, cutoff, skin, cap)
    update_neighbors_inplace!(nbh, rx, ry, rz; box)
    return nbh
end

function update_neighbors_inplace!(nbh::NeighborMatrix,
                                   rx::CuArray{Float32,1}, ry::CuArray{Float32,1};
                                   box::Tuple{Float32,Float32}, step::Int=0)
    @assert nbh.D == 2
    N = Int(nbh.N)
    _bin_particles_2d!(nbh, rx, ry, box)
    fill!(nbh.counts, Int32(0))
    inv_cs = 1f0 / nbh.cell_size
    halfLx = 0.5f0 * Float32(box[1])
    halfLy = 0.5f0 * Float32(box[2])
    t,b = _launchdims(N)
    knei = CUDA.@cuda launch=false _kernel_neighbors2!(rx, ry,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        Float32(box[1]), Float32(box[2]), halfLx, halfLy,
        nbh.nx, nbh.ny, inv_cs, nbh.cutoff2, nbh.cap)
    knei(rx, ry, nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted,
         Float32(box[1]), Float32(box[2]), halfLx, halfLy,
         nbh.nx, nbh.ny, inv_cs, nbh.cutoff2, nbh.cap; threads=t, blocks=b)
    
    # Copy current positions as reference for adaptive updates
    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_2d!(rx, ry, nbh.rref_x, nbh.rref_y)
    kcopy(rx, ry, nbh.rref_x, nbh.rref_y; threads=t, blocks=b)
    
    # Update build step for adaptive algorithm
    nbh.last_build_step = step
    
    return nbh
end

function update_neighbors_inplace!(nbh::NeighborMatrix,
                                   rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1};
                                   box::Tuple{Float32,Float32,Float32}, step::Int=0)
    @assert nbh.D == 3
    N = Int(nbh.N)
    _bin_particles_3d!(nbh, rx, ry, rz, box)
    fill!(nbh.counts, Int32(0))
    inv_cs = 1f0 / nbh.cell_size
    halfLx = 0.5f0 * Float32(box[1])
    halfLy = 0.5f0 * Float32(box[2])
    halfLz = 0.5f0 * Float32(box[3])
    t,b = _launchdims(N)
    knei = CUDA.@cuda launch=false _kernel_neighbors3!(rx, ry, rz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        nbh.cell_offsets, nbh.particle_ids_sorted,
        Float32(box[1]), Float32(box[2]), Float32(box[3]), halfLx, halfLy, halfLz,
        nbh.nx, nbh.ny, nbh.nz, inv_cs, nbh.cutoff2, nbh.cap)
    knei(rx, ry, rz, nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
         nbh.cell_offsets, nbh.particle_ids_sorted,
         Float32(box[1]), Float32(box[2]), Float32(box[3]), halfLx, halfLy, halfLz,
         nbh.nx, nbh.ny, nbh.nz, inv_cs, nbh.cutoff2, nbh.cap; threads=t, blocks=b)
    
    # Copy current positions as reference for adaptive updates
    kcopy = CUDA.@cuda launch=false _kernel_copy_refs_3d!(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z)
    kcopy(rx, ry, rz, nbh.rref_x, nbh.rref_y, nbh.rref_z; threads=t, blocks=b)
    
    # Update build step for adaptive algorithm
    nbh.last_build_step = step
    
    return nbh
end

# Note: The old interval-based helper function has been replaced
# with the adaptive displacement-based update_needed! functions above.
# Use: update_needed!(neigh, rx, ry; skin=1.5f0, Lx, Ly, step)

end # module
