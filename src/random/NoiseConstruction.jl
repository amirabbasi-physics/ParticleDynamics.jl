@inline function _mode_vector(::Type{T},
                              value::Union{AbstractVector{<:Real},Real},
                              target::Int,
                              name::AbstractString) where {T<:AbstractFloat}
    if value isa Real
        return fill(T(value), target)
    end
    vals = T.(collect(value))
    length(vals) == target ||
        throw(ArgumentError("$(name) must have length $(target), got $(length(vals))."))
    return vals
end

@inline function _canonical_mode_vectors(::Type{T},
                                         taus::Union{AbstractVector{<:Real},Real},
                                         scales::Union{AbstractVector{<:Real},Real}) where {T<:AbstractFloat}
    tau_vals = taus isa Real ? T[T(taus)] : T.(collect(taus))
    scale_vals = scales isa Real ? T[T(scales)] : T.(collect(scales))
    M = max(length(tau_vals), length(scale_vals))
    tau_vals = _mode_vector(T, tau_vals, M, "OU taus")
    scale_vals = _mode_vector(T, scale_vals, M, "OU scales")
    return tau_vals, scale_vals
end

@inline function _ou_coefficients(backend::Backends.AbstractBackend,
                                  ::Type{T},
                                  dt::T,
                                  tau::AbstractMatrix{T},
                                  scale::AbstractMatrix{T}) where {T<:AbstractFloat}
    coeff_a = Matrix{T}(undef, size(tau))
    coeff_c = Matrix{T}(undef, size(scale))
    @inbounds for j in axes(tau, 2), i in axes(tau, 1)
        τ = tau[i, j]
        s = scale[i, j]
        if τ <= zero(T)
            coeff_a[i, j] = zero(T)
            coeff_c[i, j] = s
        else
            a = exp(-dt / τ)
            coeff_a[i, j] = a
            coeff_c[i, j] = s * sqrt(max(one(T) - a * a, zero(T)))
        end
    end
    return Backends.from_host(backend, coeff_a), Backends.from_host(backend, coeff_c)
end

function _build_single_mode_ou(backend::Backends.AbstractBackend,
                               ::Type{T},
                               noise_scale::CuArray{T,1},
                               corr::CuArray{T,1},
                               dt::Real) where {T<:AbstractFloat}
    corr_host = Array(corr)
    idx_host = findall(!iszero, corr_host)
    isempty(idx_host) && return nothing

    scale_host = Array(noise_scale)
    tau_mat = reshape(T.(corr_host[idx_host]), 1, :)
    scale_mat = reshape(T.(scale_host[idx_host]), 1, :)
    active_idx = Backends.from_host(backend, Int32.(idx_host))
    coeff_a, coeff_c = _ou_coefficients(backend, T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx,
                                     Backends.from_host(backend, tau_mat),
                                     Backends.from_host(backend, scale_mat),
                                     coeff_a, coeff_c)
end

function _build_mode_ou(backend::Backends.AbstractBackend,
                        ::Type{T},
                        active_idx::CuArray{Int32,1},
                        taus::Union{AbstractVector{<:Real},Real},
                        scales::Union{AbstractVector{<:Real},Real},
                        dt::Real) where {T<:AbstractFloat}
    K = length(active_idx)
    K == 0 && return nothing
    tau_vals, scale_vals = _canonical_mode_vectors(T, taus, scales)
    tau_mat = repeat(reshape(tau_vals, :, 1), 1, K)
    scale_mat = repeat(reshape(scale_vals, :, 1), 1, K)
    coeff_a, coeff_c = _ou_coefficients(backend, T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx,
                                     Backends.from_host(backend, tau_mat),
                                     Backends.from_host(backend, scale_mat),
                                     coeff_a, coeff_c)
end

function _refresh_ou_coefficients!(ou::Definitions.OUSpectrum{T}, dt::T) where {T<:AbstractFloat}
    ou.dt == dt && return ou
    coeff_a, coeff_c = _ou_coefficients(Backends.CUDABackend(), T, dt, Array(ou.tau), Array(ou.scale))
    copyto!(ou.coeff_a, coeff_a)
    copyto!(ou.coeff_c, coeff_c)
    ou.dt = dt
    return ou
end
