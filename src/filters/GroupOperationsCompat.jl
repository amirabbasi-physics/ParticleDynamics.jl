# -----------------------------------------------------------------------------
# Assign helpers
# -----------------------------------------------------------------------------

"""
    assign_scalar!(dest, st[, filter], value)

Fill elements of `dest` referenced by `filter` (or all particles) with `value`.
This underpins `set_friction!` and the temperature setup in
`examples/TwoT_2D_LD_VV.jl`.
"""
function assign_scalar!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, value::Real) where {T}
    N = length(idx)
    N == 0 && return dest
    val = T(value)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_scalar_kernel!(dest, idx, val)
    CUDA.@sync k(dest, idx, val; threads, blocks)
    return dest
end

function assign_scalar!(dest::CuArray{T,1}, sel::Selection, value::Real) where {T}
    ParticleGroups.apply_scalar!(dest, _to_particle_group(sel), value)
    return dest
end

function assign_scalar!(dest::CuArray{T,1}, st::SimulationState, f::Filter, value::Real) where {T}
    sel = selection(st, f)
    ParticleGroups.apply_scalar!(dest, _to_particle_group(sel), value)
    return sel.device
end

function assign_scalar!(dest::CuArray{T,1}, st::SimulationState; filter::Filter=All(), value::Real) where {T}
    return assign_scalar!(dest, st, filter, value)
end

function assign_scalar!(dest::AbstractVector{T}, idx::Vector{Int}, value::Real) where {T}
    val = T(value)
    @inbounds for j in idx
        dest[j] = val
    end
    return dest
end

function assign_scalar!(dest::AbstractVector{T}, st::SimulationState, f::Filter, value::Real) where {T}
    sel = selection(st, f)
    ParticleGroups.apply_scalar!(dest, _to_particle_group(sel), value)
    return sel.host
end

function assign_scalar!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T}
    return assign_scalar!(dest, st, filter, value)
end

"""
    assign_values!(dest, st[, filter], values)

Assign distinct values per particle according to `values`. Useful when setting
custom noise scales per lattice site.
"""
function assign_values!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, values::CuArray{T,1}) where {T}
    N = length(idx)
    @assert length(values) == N "values length $(length(values)) must match index length $(N)"
    N == 0 && return dest
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_vector_kernel!(dest, idx, values)
    CUDA.@sync k(dest, idx, values; threads, blocks)
    return dest
end

function assign_values!(dest::CuArray{T,1}, sel::Selection, values::AbstractVector{<:Real}) where {T}
    ParticleGroups.apply_values!(dest, _to_particle_group(sel), values)
    return dest
end

function assign_values!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, values::AbstractVector{<:Real}) where {T}
    vals_gpu = CuArray(T.(values))
    return assign_values!(dest, idx, vals_gpu)
end

function assign_values!(dest::CuArray{T,1}, st::SimulationState, f::Filter, values::AbstractVector{<:Real}) where {T}
    sel = selection(st, f)
    ParticleGroups.apply_values!(dest, _to_particle_group(sel), values)
    return sel.device
end

function assign_values!(dest::CuArray{T,1}, st::SimulationState; filter::Filter=All(), values::AbstractVector{<:Real}) where {T}
    return assign_values!(dest, st, filter, values)
end

function assign_values!(dest::AbstractVector{T}, idx::Vector{Int}, values::AbstractVector{<:Real}) where {T}
    @assert length(values) == length(idx)
    @inbounds for (k, j) in enumerate(idx)
        dest[j] = T(values[k])
    end
    return dest
end

function assign_values!(dest::AbstractVector{T}, st::SimulationState, f::Filter, values::AbstractVector{<:Real}) where {T}
    sel = selection(st, f)
    ParticleGroups.apply_values!(dest, _to_particle_group(sel), values)
    return sel.host
end

function assign_values!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), values::AbstractVector{<:Real}) where {T}
    return assign_values!(dest, st, filter, values)
end

# -----------------------------------------------------------------------------
# Gather / sum
# -----------------------------------------------------------------------------

"""
    gather(src, st[, filter]) -> Vector

Collect the values referenced by `filter` onto the host. Used in the tests to
verify that per-type energies match expectations.
"""
function gather(src::CuArray{T,1}, idx::CuArray{Int32,1}) where {T}
    N = length(idx)
    N == 0 && return T[]
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, idx)
    CUDA.@sync k(tmp, src, idx; threads, blocks)
    return Array(tmp)
end

function gather(src::CuArray{T,1}, sel::Selection) where {T}
    return ParticleGroups.gather(src, _to_particle_group(sel))
end

function gather(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T}
    return ParticleGroups.gather(src, _to_particle_group(selection(st, f)))
end

function gather(src::AbstractVector{T}, idx::Vector{Int}) where {T}
    return T.(src[idx])
end

function gather(src::AbstractVector{T}, sel::Selection) where {T}
    return ParticleGroups.gather(src, _to_particle_group(sel))
end

function gather(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T}
    return ParticleGroups.gather(src, _to_particle_group(selection(st, f)))
end

"""
    sum(src, st[, filter])

Sum the selected entries of `src`, returning a scalar on the host. Mirrors the
heat/energy aggregation in `examples/TwoT_2D_LD_VV.jl`.
"""
function sum(src::CuArray{T,1}, idx::CuArray{Int32,1}) where {T<:Real}
    N = length(idx)
    N == 0 && return zero(T)
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, idx)
    CUDA.@sync k(tmp, src, idx; threads, blocks)
    return Base.sum(tmp)
end

function sum(src::CuArray{T,1}, sel::Selection) where {T<:Real}
    return ParticleGroups.sum_values(src, _to_particle_group(sel))
end

function sum(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T<:Real}
    return ParticleGroups.sum_values(src, _to_particle_group(selection(st, f)))
end

function sum(src::AbstractVector{T}, idx::Vector{Int}) where {T<:Real}
    acc = zero(T)
    @inbounds for j in idx
        acc += src[j]
    end
    return acc
end

function sum(src::AbstractVector{T}, sel::Selection) where {T<:Real}
    return ParticleGroups.sum_values(src, _to_particle_group(sel))
end

function sum(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T<:Real}
    return ParticleGroups.sum_values(src, _to_particle_group(selection(st, f)))
end
