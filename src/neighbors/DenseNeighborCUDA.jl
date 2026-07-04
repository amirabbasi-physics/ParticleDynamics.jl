# ============================================================================
# Core kernels
# ============================================================================
#
# Binning is a counting sort by cell id:
#   1. `_kernel_cell_ids{2,3}!`     — cell id per particle;
#   2. `_kernel_cell_histogram!`    — particles per cell (atomic);
#   3. prefix scan (host-driven)    — `cell_offsets` from the histogram;
#   4. `_kernel_scatter_by_cell!`   — cell-sorted particle ids.
# This is O(N) with two light passes and replaces the previous global
# `CUDA.sort!` of packed 64-bit keys plus per-cell binary searches, which
# dominated rebuild time at large N. The order of particles within one cell
# is nondeterministic (atomic cursor), which is physically irrelevant but
# changes floating-point summation order between runs.

function _kernel_cell_ids2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                            Lx::T, Ly::T, inv_cs::T,
                            nx::Int32, ny::Int32,
                            cell_of_particle::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        x = rx[i] + T(0.5)*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + T(0.5)*Ly; y -= floor(y/Ly)*Ly
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        cell_of_particle[i] = cy*nx + cx
    end
    return
end

function _kernel_cell_ids3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                            Lx::T, Ly::T, Lz::T, inv_cs::T,
                            nx::Int32, ny::Int32, nz::Int32,
                            cell_of_particle::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        x = rx[i] + T(0.5)*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + T(0.5)*Ly; y -= floor(y/Ly)*Ly
        z = rz[i] + T(0.5)*Lz; z -= floor(z/Lz)*Lz
        cx = Int32(floor(x * inv_cs)); cx = cx >= nx ? (nx-1) : cx
        cy = Int32(floor(y * inv_cs)); cy = cy >= ny ? (ny-1) : cy
        cz = Int32(floor(z * inv_cs)); cz = cz >= nz ? (nz-1) : cz
        cell_of_particle[i] = (cz*ny + cy)*nx + cx
    end
    return
end

function _kernel_cell_histogram!(cell_of_particle::CuDeviceVector{Int32},
                                 cell_counts::CuDeviceVector{Int32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(cell_of_particle); if i > N; return; end
    @inbounds begin
        c = cell_of_particle[i]
        CUDA.@atomic cell_counts[c+1] += Int32(1)
    end
    return
end

function _kernel_scatter_by_cell!(cell_of_particle::CuDeviceVector{Int32},
                                  cell_offsets::CuDeviceVector{Int32},
                                  cursors::CuDeviceVector{Int32},
                                  particle_ids_sorted::CuDeviceVector{Int32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(cell_of_particle); if i > N; return; end
    @inbounds begin
        c = cell_of_particle[i]
        slot = CUDA.@atomic cursors[c+1] += Int32(1)
        particle_ids_sorted[cell_offsets[c+1] + slot] = Int32(i)
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
                            neighbors_flat[_ell_index(i1, found, N)] = j
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
                                neighbors_flat[_ell_index(i1, found, N)] = j
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
