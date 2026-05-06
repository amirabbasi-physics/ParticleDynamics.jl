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

function initialize_backend_runtime!()
    _maybe_set_cuda_compat!()
    return nothing
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

# Backend inference is currently limited to build/storage decisions.
# Neighbor, force, and integrator kernels remain CUDA.jl implementations, so
# `backend=:cpu` still errors early until those execution paths are generalized.
storage_backend(x) = error("No storage backend registered for $(typeof(x)).")
storage_backend(::CuArray) = CUDABackend()

# Allow legacy GPUs (e.g. Pascal cc 6.x) to run by pinning to a CUDA runtime
# below the 13.x cutoff on drivers that default to newer runtimes.
function _maybe_set_cuda_compat!()
    mode = get(ENV, "NEQSIMGPU_CUDA_COMPAT", "auto")
    mode == "off" && return

    target = try
        VersionNumber(get(ENV, "NEQSIMGPU_CUDA_LEGACY_VERSION", "12.4.1"))
    catch
        v"12.4.1"
    end

    force = mode == "force"

    try
        dev = CUDA.device()
        cap = CUDA.capability(dev)
        runtime = CUDA.runtime_version()
        needs_downgrade = cap < v"7.5" && runtime >= v"13.0.0"
        if (needs_downgrade || force) && runtime > target
            if !isdefined(CUDA, :set_runtime_version!)
                @warn "CUDA.set_runtime_version! not available; cannot adjust runtime automatically" device=CUDA.name(dev) capability=cap runtime=runtime
                return
            end
            @warn "Legacy GPU detected; attempting to pin CUDA runtime" device=CUDA.name(dev) capability=cap runtime=runtime target mode
            try
                CUDA.set_runtime_version!(target)
                new_runtime = CUDA.runtime_version()
                if new_runtime >= v"13.0.0"
                    @warn "CUDA runtime pin did not take effect; you may still see capability errors (driver may not ship compat libs)" device=CUDA.name(dev) capability=cap runtime=new_runtime target
                else
                    @info "CUDA runtime pinned for legacy GPU" device=CUDA.name(dev) capability=cap runtime=new_runtime target
                end
            catch err
                @warn "Failed to pin CUDA runtime for legacy GPU; driver may be too new or missing compatibility libraries" device=CUDA.name(dev) capability=cap runtime=runtime target error=err
            end
        end
    catch err
        @warn "Legacy CUDA compatibility probe failed; leaving defaults" error=err
    end
end

end
