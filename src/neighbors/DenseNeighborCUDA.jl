# ============================================================================
# Core kernels
# ============================================================================

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
        base  = _csr_base(i1, cap)
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
        base  = _csr_base(i1, cap)
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
