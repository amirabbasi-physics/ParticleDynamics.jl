module Filters

using Base: Set
using CUDA
using CUDA: CuArray, CuDeviceVector
using ..Simulation: SimulationState

export Filter, All, TypeIDs, Indices, Selection,
       resolve, resolve_gpu, selection, count,
       assign_scalar!, assign_values!, gather, sum,
       set_noise_scale!, set_langevin_temperature!

abstract type Filter end

struct All <: Filter end

struct TypeIDs{I<:Integer} <: Filter
    ids::Vector{I}
end
TypeIDs(ids::AbstractVector{<:Integer}) = TypeIDs{Int}(Int.(ids))
TypeIDs(id::Integer) = TypeIDs([Int(id)])

struct Indices{I<:Integer} <: Filter
    idx::Vector{I}
end
Indices(idx::AbstractVector{<:Integer}) = Indices{Int}(Int.(idx))
Indices(id::Integer) = Indices([Int(id)])

struct Selection
    host::Vector{Int}
    device::CuArray{Int32,1}
end

function Selection(host::Vector{Int})
    dev = CuArray(Int32.(host))
    return Selection(host, dev)
end

count(sel::Selection) = length(sel.host)

# -----------------------------------------------------------------------------
# Index resolution helpers
# -----------------------------------------------------------------------------

function resolve(::All, st::SimulationState)
    N = length(st.rx)
    return collect(1:N)
end

function resolve(f::Indices, st::SimulationState)
    idx = Int.(f.idx)
    _validate_indices(idx, length(st.rx))
    return idx
end

function resolve(f::TypeIDs, st::SimulationState)
    types = Array(st.typeid)
    wanted = Set(Int.(f.ids))
    idx = Int[]
    @inbounds for (i, t) in enumerate(types)
        if t in wanted
            push!(idx, i)
        end
    end
    return idx
end

resolve(st::SimulationState, f::Filter) = resolve(f, st)

function resolve_gpu(f::Filter, st::SimulationState)
    host = resolve(f, st)
    return CuArray(Int32.(host))
end

resolve_gpu(st::SimulationState, f::Filter) = resolve_gpu(f, st)

function selection(st::SimulationState, f::Filter)
    host = resolve(f, st)
    return Selection(host)
end

function _validate_indices(idx::Vector{Int}, N::Int)
    for (k, i) in enumerate(idx)
        @assert 1 <= i <= N "Index $(i) at position $(k) out of bounds 1:$(N)"
    end
    nothing
end

count(f::Filter, st::SimulationState) = length(resolve(f, st))
count(st::SimulationState, f::Filter) = count(f, st)

# -----------------------------------------------------------------------------
# GPU kernels
# -----------------------------------------------------------------------------

function _assign_scalar_kernel!(dest::CuDeviceVector{T}, idx::CuDeviceVector{Int32}, value::T) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    dest[j] = value
    return
end

function _assign_vector_kernel!(dest::CuDeviceVector{T}, idx::CuDeviceVector{Int32}, values::CuDeviceVector{T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    dest[j] = values[i]
    return
end

function _gather_kernel!(out::CuDeviceVector{T}, src::CuDeviceVector{T}, idx::CuDeviceVector{Int32}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    out[i] = src[j]
    return
end

# -----------------------------------------------------------------------------
# Assign helpers
# -----------------------------------------------------------------------------

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
    return assign_scalar!(dest, sel.device, value)
end

function assign_scalar!(dest::CuArray{T,1}, st::SimulationState, f::Filter, value::Real) where {T}
    idx = resolve_gpu(f, st)
    assign_scalar!(dest, idx, value)
    return idx
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
    idx = resolve(f, st)
    assign_scalar!(dest, idx, value)
    return idx
end

function assign_scalar!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T}
    return assign_scalar!(dest, st, filter, value)
end

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
    vals_gpu = CuArray(T.(values))
    return assign_values!(dest, sel.device, vals_gpu)
end

function assign_values!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, values::AbstractVector{<:Real}) where {T}
    vals_gpu = CuArray(T.(values))
    return assign_values!(dest, idx, vals_gpu)
end

function assign_values!(dest::CuArray{T,1}, st::SimulationState, f::Filter, values::AbstractVector{<:Real}) where {T}
    idx = resolve_gpu(f, st)
    assign_values!(dest, idx, values)
    return idx
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
    idx = resolve(f, st)
    assign_values!(dest, idx, values)
    return idx
end

function assign_values!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), values::AbstractVector{<:Real}) where {T}
    return assign_values!(dest, st, filter, values)
end

# -----------------------------------------------------------------------------
# Gather / sum
# -----------------------------------------------------------------------------

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
    return gather(src, sel.device)
end

function gather(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T}
    idx = resolve_gpu(f, st)
    return gather(src, idx)
end

function gather(src::AbstractVector{T}, idx::Vector{Int}) where {T}
    return T.(src[idx])
end

function gather(src::AbstractVector{T}, sel::Selection) where {T}
    return gather(src, sel.host)
end

function gather(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T}
    idx = resolve(f, st)
    return gather(src, idx)
end

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
    return sum(src, sel.device)
end

function sum(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T<:Real}
    idx = resolve_gpu(f, st)
    return sum(src, idx)
end

function sum(src::AbstractVector{T}, idx::Vector{Int}) where {T<:Real}
    acc = zero(T)
    @inbounds for j in idx
        acc += src[j]
    end
    return acc
end

function sum(src::AbstractVector{T}, sel::Selection) where {T<:Real}
    return sum(src, sel.host)
end

function sum(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T<:Real}
    idx = resolve(f, st)
    return sum(src, idx)
end

# -----------------------------------------------------------------------------
# Convenience APIs
# -----------------------------------------------------------------------------

function set_noise_scale!(st::SimulationState, value::Real; filter::Filter=All())
    sel = selection(st, filter)
    set_noise_scale!(st, value, sel)
    return sel
end

function set_noise_scale!(st::SimulationState, value::Real, sel::Selection)
    assign_scalar!(st.vv.noise_scale, sel.device, Float32(value))
    return sel
end

function set_noise_scale!(st::SimulationState, value::Real, idx::CuArray{Int32,1})
    assign_scalar!(st.vv.noise_scale, idx, Float32(value))
    return idx
end

function set_noise_scale!(st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_noise_scale!(st, val; filter=f)
    end
    return st
end

function set_noise_scale!(st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_noise_scale!(st, val; filter=f)
    end
    return st
end

function set_langevin_temperature!(st::SimulationState, dt::Real, temperature::Real; filter::Filter=All())
    sel = selection(st, filter)
    set_langevin_temperature!(st, dt, temperature, sel)
    return sel
end

function set_langevin_temperature!(st::SimulationState, dt::Real, temperature::Real, sel::Selection)
    γ = Float32(st.vv.gamma)
    Δt = Float32(dt)
    Tval = Float32(temperature)
    scale = sqrt(2f0 * γ * Tval * Δt)
    assign_scalar!(st.vv.noise_scale, sel.device, scale)
    return sel
end

function set_langevin_temperature!(st::SimulationState, dt::Real, temperature::Real, idx::CuArray{Int32,1})
    γ = Float32(st.vv.gamma)
    Δt = Float32(dt)
    Tval = Float32(temperature)
    scale = sqrt(2f0 * γ * Tval * Δt)
    assign_scalar!(st.vv.noise_scale, idx, scale)
    return idx
end

function set_langevin_temperature!(st::SimulationState, dt::Real, mapping::AbstractDict{<:Filter,<:Real})
    for (f, temp) in mapping
        set_langevin_temperature!(st, dt, temp; filter=f)
    end
    return st
end

function set_langevin_temperature!(st::SimulationState, dt::Real, pairs::Pair{<:Filter,<:Real}...)
    for (f, temp) in pairs
        set_langevin_temperature!(st, dt, temp; filter=f)
    end
    return st
end

end # module Filters
