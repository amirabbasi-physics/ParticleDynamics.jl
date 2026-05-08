"""
    StochasticWorkspace{T}

Integrator-local scratch buffers for stochastic updates (random impulses and
optional OU state). These buffers are carried by integrator specs so the shared
step engine remains integrator-agnostic.
"""
mutable struct StochasticWorkspace{T<:AbstractFloat}
    rf_x::CuArray{T,1}
    rf_y::CuArray{T,1}
    rf_z::Union{Nothing,CuArray{T,1}}
    ou_x::Union{Nothing,CuArray{T,2}}
    ou_y::Union{Nothing,CuArray{T,2}}
    ou_z::Union{Nothing,CuArray{T,2}}
end

"""
    _empty_workspace(T)

Construct a zero-length stochastic workspace used by compatibility constructors
that receive parameter objects without a simulation state.
"""
function _empty_workspace(::Type{T}) where {T<:AbstractFloat}
    backend = Backends.CUDABackend()
    return StochasticWorkspace{T}(Backends.zeros_vector(backend, T, 0),
                                  Backends.zeros_vector(backend, T, 0),
                                  nothing, nothing, nothing, nothing)
end
