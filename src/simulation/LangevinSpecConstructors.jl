function _build_vv_params(st::SimulationState{T};
                          gamma::Union{AbstractVector{<:Real},Real},
                          temperature::Union{AbstractVector{<:Real},Real},
                          noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          mass::Real=st.mass,
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
    return LangevinIntegrators.VVParams{T}(γ, T(mass), noise; dt=T(dt), corr_time=corr, ou=ou)
end

function _build_baoab_params(st::SimulationState{T};
                             gamma::Union{AbstractVector{<:Real},Real},
                             temperature::Union{AbstractVector{<:Real},Real},
                             noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                             ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                             mass::Real=st.mass,
                             dt::Real=st.dt) where {T<:AbstractFloat}
    vv = _build_vv_params(st; gamma=gamma, temperature=temperature,
                          noise_corr_time=noise_corr_time,
                          ou_scales=ou_scales,
                          mass=mass, dt=dt)
    return LangevinIntegrators.BAOABParams{T}(vv.gamma, vv.mass, vv.noise_scale; dt=vv.dt, corr_time=vv.corr_time, ou=vv.ou)
end

"""
    velocityverlet(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
                   mass=st.mass, dt=st.dt) -> VVSpec

Construct a GJF/Langevin velocity-Verlet spec with integrator-owned stochastic
buffers.
"""
function velocityverlet(st::SimulationState{T};
                        gamma::Union{AbstractVector{<:Real},Real},
                        temperature::Union{AbstractVector{<:Real},Real},
                        noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                        ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                        mass::Real=st.mass,
                        dt::Real=st.dt) where {T<:AbstractFloat}
    return VVSpec(_build_vv_params(st; gamma=gamma, temperature=temperature,
                                   noise_corr_time=noise_corr_time,
                                   ou_scales=ou_scales,
                                   mass=mass, dt=dt))
end

"""
    baoab(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
          mass=st.mass, dt=st.dt) -> BAOABSpec

Construct a BAOAB Langevin spec with integrator-owned stochastic buffers.
"""
function baoab(st::SimulationState{T};
               gamma::Union{AbstractVector{<:Real},Real},
               temperature::Union{AbstractVector{<:Real},Real},
               noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
               ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
               mass::Real=st.mass,
               dt::Real=st.dt) where {T<:AbstractFloat}
    return BAOABSpec(_build_baoab_params(st; gamma=gamma, temperature=temperature,
                                         noise_corr_time=noise_corr_time,
                                         ou_scales=ou_scales,
                                         mass=mass, dt=dt))
end

"""
    baoa(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
         mass=st.mass, dt=st.dt) -> BAOASpec

Legacy BAOA variant (no final B kick).
"""
function baoa(st::SimulationState{T};
              gamma::Union{AbstractVector{<:Real},Real},
              temperature::Union{AbstractVector{<:Real},Real},
              noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
              ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
              mass::Real=st.mass,
              dt::Real=st.dt) where {T<:AbstractFloat}
    return BAOASpec(_build_baoab_params(st; gamma=gamma, temperature=temperature,
                                        noise_corr_time=noise_corr_time,
                                        ou_scales=ou_scales,
                                        mass=mass, dt=dt))
end

"""
    gsm(st; gamma, temperature, noise_corr_time=nothing, ou_scales=nothing,
        mass=st.mass, dt=st.dt) -> GSMSpec

Construct a GSM spec (used by `examples/TwoT_2D_LD_GSM.jl`).
"""
function gsm(st::SimulationState{T};
             gamma::Union{AbstractVector{<:Real},Real},
             temperature::Union{AbstractVector{<:Real},Real},
             noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
             ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
             mass::Real=st.mass,
             dt::Real=st.dt) where {T<:AbstractFloat}
    return GSMSpec(_build_baoab_params(st; gamma=gamma, temperature=temperature,
                                       noise_corr_time=noise_corr_time,
                                       ou_scales=ou_scales,
                                       mass=mass, dt=dt))
end
