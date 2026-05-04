"""
Particle selection and grouping subsystem for thermostats, observables, and diagnostics.

This module provides a unified interface for particle selections that can be used across:
- Thermostat applications (different baths or controls)
- Observable computations (subset-specific diagnostics)
- Writer/diagnostic scheduling
- Filters and constraints

The key abstraction is `ParticleGroup`, which materializes a particle selection as
either host indices or device indices, reusable across multiple operations.
"""
module ParticleGroups

using CUDA
using CUDA: CuArray, CuDeviceVector
using ..Backends
using ..Definitions

export ParticleSelection, ParticleGroup,
       All, TypeIDs, Indices,
       resolve, resolve_gpu, materialize,
       count, apply_scalar!, apply_values!, gather, sum_values

# =============================================================================
# Selection Specifications
# =============================================================================

"""
Abstract base for particle selection specifications.

Concrete subtypes include `All`, `TypeIDs`, `Indices`, and can be extended
by users for custom selection logic.
"""
abstract type ParticleSelection end

"""
Select all particles.
"""
struct All <: ParticleSelection end

"""
Select particles by type ID. Accepts a single `Int` or a vector.
"""
struct TypeIDs{I<:Integer} <: ParticleSelection
    ids::Vector{I}
end

TypeIDs(ids::AbstractVector{<:Integer}) = TypeIDs{Int}(Int.(ids))
TypeIDs(id::Integer) = TypeIDs([Int(id)])

"""
Select particles by explicit list of indices.
"""
struct Indices{I<:Integer} <: ParticleSelection
    idx::Vector{I}
end

Indices(idx::AbstractVector{<:Integer}) = Indices{Int}(Int.(idx))
Indices(id::Integer) = Indices([Int(id)])

# =============================================================================
# Materialized Group
# =============================================================================

"""
    ParticleGroup

Materialized particle selection containing both host and device indices.
Reusable across multiple operations without recomputation.

Instances are created via `materialize(selection, state)`.
"""
struct ParticleGroup
    host::Vector{Int}
    device::CuArray{Int32,1}
end

function ParticleGroup(host::Vector{Int})
    dev = CuArray(Int32.(host))
    return ParticleGroup(host, dev)
end

"""
    count(group::ParticleGroup) -> Int

Number of particles in the group.
"""
count(group::ParticleGroup) = length(group.host)

# =============================================================================
# Resolution Helpers
# =============================================================================

"""
    resolve(sel::ParticleSelection, state) -> Vector{Int}

Resolve a selection specification to host indices.
"""
function resolve(::All, state)
    N = length(state.rx)
    return collect(1:N)
end

function resolve(sel::Indices, state)
    idx = Int.(sel.idx)
    _validate_indices(idx, length(state.rx))
    return idx
end

function resolve(sel::TypeIDs, state)
    return Int.(Array(resolve_gpu(sel, state)))
end

"""
    resolve_gpu(sel::ParticleSelection, state) -> CuArray{Int32}

Resolve a selection specification to device indices.
"""
function resolve_gpu(::All, state)
    N = length(state.rx)
    return CuArray(Int32.(collect(1:N)))
end

function resolve_gpu(sel::Indices, state)
    idx = Int.(sel.idx)
    _validate_indices(idx, length(state.rx))
    return CuArray(Int32.(idx))
end

function resolve_gpu(sel::TypeIDs, state)
    N = length(state.rx)
    N == 0 && return CUDA.zeros(Int32, 0)
    ids = unique(Int32.(sel.ids))
    isempty(ids) && return CUDA.zeros(Int32, 0)
    
    mask = CUDA.fill(false, N)
    @inbounds for id in ids
        mask .|= (state.typeid .== id)
    end
    
    idx64 = findall(mask)
    return Int32.(idx64)
end

"""
    materialize(sel::ParticleSelection, state) -> ParticleGroup

Create a reusable materialized group from a selection and state.
"""
function materialize(sel::ParticleSelection, state)
    host = resolve(sel, state)
    dev = resolve_gpu(sel, state)
    return ParticleGroup(host, dev)
end

"""
    count(sel::ParticleSelection, state) -> Int

Number of particles matched by a selection.
"""
count(sel::ParticleSelection, state) = length(resolve(sel, state))
count(sel::TypeIDs, state) = length(resolve_gpu(sel, state))

# =============================================================================
# GPU Kernels for Group Operations
# =============================================================================

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

# =============================================================================
# Operations on Groups
# =============================================================================

"""
    apply_scalar!(dest::CuArray, group::ParticleGroup, value)

Assign `value` to all elements of `dest` indexed by `group`.
"""
function apply_scalar!(dest::CuArray{T,1}, group::ParticleGroup, value::Real) where {T}
    N = length(group.device)
    N == 0 && return dest
    val = T(value)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_scalar_kernel!(dest, group.device, val)
    CUDA.@sync k(dest, group.device, val; threads, blocks)
    return dest
end

function apply_scalar!(dest::AbstractVector{T}, group::ParticleGroup, value::Real) where {T}
    val = T(value)
    @inbounds for j in group.host
        dest[j] = val
    end
    return dest
end

"""
    apply_values!(dest::CuArray, group::ParticleGroup, values)

Assign distinct values per group member.
"""
function apply_values!(dest::CuArray{T,1}, group::ParticleGroup, values::AbstractVector{<:Real}) where {T}
    N = length(group.device)
    @assert length(values) == N "values length $(length(values)) must match group size $(N)"
    N == 0 && return dest
    
    vals_gpu = CuArray(T.(values))
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_vector_kernel!(dest, group.device, vals_gpu)
    CUDA.@sync k(dest, group.device, vals_gpu; threads, blocks)
    return dest
end

function apply_values!(dest::AbstractVector{T}, group::ParticleGroup, values::AbstractVector{<:Real}) where {T}
    @assert length(values) == length(group.host)
    @inbounds for (k, j) in enumerate(group.host)
        dest[j] = T(values[k])
    end
    return dest
end

"""
    gather(src::CuArray, group::ParticleGroup) -> Vector

Collect values from device array indexed by group onto the host.
"""
function gather(src::CuArray{T,1}, group::ParticleGroup) where {T}
    N = length(group.device)
    N == 0 && return T[]
    
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, group.device)
    CUDA.@sync k(tmp, src, group.device; threads, blocks)
    return Array(tmp)
end

function gather(src::AbstractVector{T}, group::ParticleGroup) where {T}
    return T.(src[group.host])
end

"""
    sum_values(src::CuArray, group::ParticleGroup) -> Scalar

Sum values from device array indexed by group.
"""
function sum_values(src::CuArray{T,1}, group::ParticleGroup) where {T<:Real}
    N = length(group.device)
    N == 0 && return zero(T)
    
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, group.device)
    CUDA.@sync k(tmp, src, group.device; threads, blocks)
    return Base.sum(tmp)
end

function sum_values(src::AbstractVector{T}, group::ParticleGroup) where {T<:Real}
    acc = zero(T)
    @inbounds for j in group.host
        acc += src[j]
    end
    return acc
end

# =============================================================================
# Internal Helpers
# =============================================================================

function _validate_indices(idx::Vector{Int}, N::Int)
    for (k, i) in enumerate(idx)
        @assert 1 <= i <= N "Index $(i) at position $(k) out of bounds 1:$(N)"
    end
    nothing
end

end  # module ParticleGroups
