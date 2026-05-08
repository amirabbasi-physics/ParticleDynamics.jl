# ============================================================================
# Update kernels
# ============================================================================

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
