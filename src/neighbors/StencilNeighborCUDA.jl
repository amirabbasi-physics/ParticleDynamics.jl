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
        base  = _csr_base(i1, cap)
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
        base  = _csr_base(i1, cap)
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
