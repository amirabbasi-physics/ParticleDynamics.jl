module Backends

using CUDA

export AbstractBackend, CUDABackend, CPUBackend,
       normalize_backend, ensure_available,
       allocate_vector, allocate_matrix,
       fill_vector, zeros_vector, zeros_matrix,
       from_host, sum_elements, storage_backend

abstract type AbstractBackend end

struct CUDABackend <: AbstractBackend end
struct CPUBackend <: AbstractBackend end

normalize_backend(backend::AbstractBackend) = backend

function normalize_backend(backend::Symbol)
    if backend === :cuda
        return CUDABackend()
    elseif backend === :cpu
        return CPUBackend()
    end
    throw(ArgumentError("Unknown backend=$(backend). Use :cuda or :cpu."))
end

function ensure_available(::CUDABackend)
    CUDA.functional() || throw(ArgumentError("CUDA backend requested but CUDA.functional() == false on this machine."))
    return nothing
end

function ensure_available(::CPUBackend)
    throw(ArgumentError("CPU backend support is not implemented in ParticleDynamics yet. Use backend=:cuda."))
end

allocate_vector(::CUDABackend, ::Type{T}, n::Integer) where {T<:AbstractFloat} =
    CUDA.CuArray{T}(undef, n)

allocate_vector(::CUDABackend, ::Type{Int32}, n::Integer) =
    CUDA.CuArray{Int32}(undef, n)

allocate_matrix(::CUDABackend, ::Type{T}, m::Integer, n::Integer) where {T<:AbstractFloat} =
    CUDA.CuArray{T}(undef, m, n)

allocate_matrix(::CUDABackend, ::Type{Int32}, m::Integer, n::Integer) =
    CUDA.CuArray{Int32}(undef, m, n)

fill_vector(::CUDABackend, value, n::Integer) = CUDA.fill(value, n)
zeros_vector(::CUDABackend, ::Type{T}, n::Integer) where {T} = CUDA.zeros(T, n)
zeros_matrix(::CUDABackend, ::Type{T}, m::Integer, n::Integer) where {T} = CUDA.zeros(T, m, n)
from_host(::CUDABackend, values) = CuArray(values)
sum_elements(::CUDABackend, arr) = CUDA.sum(arr)

storage_backend(x) = error("No storage backend registered for $(typeof(x)).")

end
