function _build_brownian_params(st::SimulationState{T};
                                gamma::Union{AbstractVector{<:Real},Real},
                                temperature::Union{AbstractVector{<:Real},Real},
                                noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                                ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                                dt::Real=st.dt) where {T<:AbstractFloat}
    backend = Backends.storage_backend(st)
    N = length(st.rx)
    γ = _device_particle_buffer(backend, T, N, gamma, "gamma")
    Tbuf = _device_particle_buffer(backend, T, N, temperature, "temperature")
    noise = sqrt.(T(2) .* γ .* Tbuf .* T(dt))
    corr = nothing
    ou = nothing
    if ou_scales !== nothing
        noise_corr_time === nothing &&
            throw(ArgumentError("`ou_scales` requires `noise_corr_time` to be provided as OU mode correlation times."))
        ou = _build_mode_ou(backend, T, _all_particle_indices(backend, N), noise_corr_time, ou_scales, dt)
    elseif noise_corr_time !== nothing
        if noise_corr_time isa AbstractVector && length(noise_corr_time) != N
            throw(ArgumentError("Vector `noise_corr_time` without `ou_scales` is interpreted as legacy per-particle single-mode OU and must have length $(N). Pass `ou_scales` as well for a multi-mode spectrum."))
        end
        corr = _device_corr_time_buffer(backend, T, N, noise_corr_time)
        ou = _build_single_mode_ou(backend, T, noise, corr, dt)
    end
    return BrownianIntegrators.BrownianParams{T}(γ, T(dt), noise, corr, ou)
end

function _build_em_params(st::SimulationState{T};
                          gamma::Union{AbstractVector{<:Real},Real},
                          temperature::Union{AbstractVector{<:Real},Real},
                          noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          dt::Real=st.dt) where {T<:AbstractFloat}
    bp = _build_brownian_params(st; gamma=gamma, temperature=temperature,
                                noise_corr_time=noise_corr_time, ou_scales=ou_scales, dt=dt)
    return BrownianIntegrators.EMParams{T}(bp.gamma, bp.dt, bp.noise_scale, bp.corr_time, bp.ou)
end

"""
    eulerheun(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
              dt=st.dt) -> BrownianSpec

Build a midpoint Brownian spec with integrator-owned stochastic buffers.
"""
function eulerheun(st::SimulationState{T};
                   gamma::Union{AbstractVector{<:Real},Real},
                   temperature::Union{AbstractVector{<:Real},Real},
                   noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                   ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                   dt::Real=st.dt) where {T<:AbstractFloat}
    return BrownianSpec(_build_brownian_params(st; gamma=gamma, temperature=temperature,
                                               noise_corr_time=noise_corr_time,
                                               ou_scales=ou_scales, dt=dt))
end

"""
    eulermaruyama(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
                  dt=st.dt) -> EMSpec

Return an Euler-Maruyama spec for overdamped dynamics (`examples/3D_BD.jl`).
"""
function eulermaruyama(st::SimulationState{T};
                       gamma::Union{AbstractVector{<:Real},Real},
                       temperature::Union{AbstractVector{<:Real},Real},
                       noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                       ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                       dt::Real=st.dt) where {T<:AbstractFloat}
    return EMSpec(_build_em_params(st; gamma=gamma, temperature=temperature,
                                   noise_corr_time=noise_corr_time,
                                   ou_scales=ou_scales, dt=dt))
end
