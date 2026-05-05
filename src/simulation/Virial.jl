# =========================
#   Virial (GPU)
# =========================
@inline _virial_ncomponents(st::SimulationState) = st.rz === nothing ? 3 : 6

virial_components(st::SimulationState) =
    st.rz === nothing ? (:xx, :yy, :xy) : (:xx, :yy, :zz, :xy, :xz, :yz)

@inline function _virial_buffer(st::SimulationState, part::Symbol, accumulated::Bool)
    if accumulated
        part === :total || throw(ArgumentError("only the total virial tensor has an accumulated buffer"))
        return st.virial_tensor_accum
    end
    if part === :total
        return st.virial_tensor
    elseif part === :nonbonded
        return st.virial_nonbonded
    elseif part === :bonded
        return st.virial_bonded
    end
    throw(ArgumentError("unknown virial part $(part); use :total, :nonbonded, or :bonded"))
end

"""
    virial_tensor(st; part=:total, accumulated=false)

Return the particle-summed configurational virial tensor components as a
`NamedTuple`. The stored sign convention is the raw configurational virial
`W = Σ r_ij ⊗ F_ij`, so pressure/stress formulas may need an additional minus
sign depending on convention.

Component ordering follows:
- 2D: `(:xx, :yy, :xy)`
- 3D: `(:xx, :yy, :zz, :xy, :xz, :yz)`

The primary buffers remain GPU-resident in `SimulationState`; this helper
reduces the selected tensor to a small host-side summary for diagnostics.
Virial buffers are refreshed by force evaluations with `compute_energy=true`.
"""
function virial_tensor(st::SimulationState{T}; part::Symbol=:total, accumulated::Bool=false) where {T<:AbstractFloat}
    comps = vec(Array(CUDA.sum(_virial_buffer(st, part, accumulated); dims=1)))
    if st.rz === nothing
        return (xx=comps[1], yy=comps[2], xy=comps[3])
    end
    return (xx=comps[1], yy=comps[2], zz=comps[3], xy=comps[4], xz=comps[5], yz=comps[6])
end

function _combine_virial2_kernel!(virial::CuDeviceVector{T},
                                  total::CuDeviceMatrix{T},
                                  nonbonded::CuDeviceMatrix{T},
                                  bonded::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        vxx = nonbonded[i, 1] + bonded[i, 1]
        vyy = nonbonded[i, 2] + bonded[i, 2]
        vxy = nonbonded[i, 3] + bonded[i, 3]
        total[i, 1] = vxx
        total[i, 2] = vyy
        total[i, 3] = vxy
        virial[i] = vxx + vyy
    end
    return
end

function _combine_virial3_kernel!(virial::CuDeviceVector{T},
                                  total::CuDeviceMatrix{T},
                                  nonbonded::CuDeviceMatrix{T},
                                  bonded::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        vxx = nonbonded[i, 1] + bonded[i, 1]
        vyy = nonbonded[i, 2] + bonded[i, 2]
        vzz = nonbonded[i, 3] + bonded[i, 3]
        vxy = nonbonded[i, 4] + bonded[i, 4]
        vxz = nonbonded[i, 5] + bonded[i, 5]
        vyz = nonbonded[i, 6] + bonded[i, 6]
        total[i, 1] = vxx
        total[i, 2] = vyy
        total[i, 3] = vzz
        total[i, 4] = vxy
        total[i, 5] = vxz
        total[i, 6] = vyz
        virial[i] = vxx + vyy + vzz
    end
    return
end

function _combine_virial!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.virial)
    threads = min(256, N)
    blocks = cld(N, threads)
    if st.rz === nothing
        k = CUDA.@cuda launch=false _combine_virial2_kernel!(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded)
        k(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded; threads, blocks)
    else
        k = CUDA.@cuda launch=false _combine_virial3_kernel!(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded)
        k(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded; threads, blocks)
    end
    return nothing
end

function _accumulate_virial2!(virial_accum::CuDeviceVector{T},
                              virial::CuDeviceVector{T},
                              virial_tensor_accum::CuDeviceMatrix{T},
                              virial_tensor::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        virial_accum[i] += virial[i]
        virial_tensor_accum[i, 1] += virial_tensor[i, 1]
        virial_tensor_accum[i, 2] += virial_tensor[i, 2]
        virial_tensor_accum[i, 3] += virial_tensor[i, 3]
    end
    return
end

function _accumulate_virial3!(virial_accum::CuDeviceVector{T},
                              virial::CuDeviceVector{T},
                              virial_tensor_accum::CuDeviceMatrix{T},
                              virial_tensor::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        virial_accum[i] += virial[i]
        virial_tensor_accum[i, 1] += virial_tensor[i, 1]
        virial_tensor_accum[i, 2] += virial_tensor[i, 2]
        virial_tensor_accum[i, 3] += virial_tensor[i, 3]
        virial_tensor_accum[i, 4] += virial_tensor[i, 4]
        virial_tensor_accum[i, 5] += virial_tensor[i, 5]
        virial_tensor_accum[i, 6] += virial_tensor[i, 6]
    end
    return
end

"""
    accumulate_virial!(st)

Add the instantaneous virial trace and total virial tensor into their
per-interval accumulators. `st.virial` remains the scalar trace of the
configurational virial tensor for backward compatibility.
"""
function accumulate_virial!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.virial)
    threads = min(256, N)
    blocks  = cld(N, threads)
    if st.rz === nothing
        k = CUDA.@cuda launch=false _accumulate_virial2!(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor)
        k(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor; threads, blocks)
    else
        k = CUDA.@cuda launch=false _accumulate_virial3!(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor)
        k(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor; threads, blocks)
    end
    return nothing
end
