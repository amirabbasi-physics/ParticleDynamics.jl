"""
Particle-selection helpers used to assign per-group parameters (temperatures,
frictions, correlation times) as in `examples/TwoT_2D_LD_VV.jl`.
"""
module Filters

using CUDA
using CUDA: CuArray, CuDeviceVector
using ..Backends
import ..ParticleGroups
import ..SimulationCore
using ..Definitions
using ..SimulationCore: SimulationState, IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NVESpec, NHCSpec, CSVRSpec
using ..BrownianIntegrators
using ..LangevinIntegrators

export Filter, All, TypeIDs, Indices, Selection,
       resolve, resolve_gpu, selection, count,
       assign_scalar!, assign_values!, gather, sum,
       set_noise_scale!, set_temperature!, set_friction!, set_corr_time!, set_ou_spectrum!,
       set_thermostat_temperature!, set_thermostat_timescale!, assign_nhc_baths!, assign_csvr_baths!,
       freeze_particles!, unfreeze_particles!

include("filters/FilterSelections.jl")

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

function _set_noise_gamma_kernel!(noise_scale::CuDeviceVector{T},
                                  gamma::CuDeviceVector{T},
                                  idx::CuDeviceVector{Int32},
                                  dt::T, temperature::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    g = gamma[j]
    noise_scale[j] = sqrt(T(2) * g * temperature * dt)
    return
end

function _set_noise_from_gamma!(noise_scale::CuArray{T,1},
                                gamma::CuArray{T,1},
                                idx::CuArray{Int32,1},
                                dt::T, temperature::T) where {T<:AbstractFloat}
    N = length(idx)
    N == 0 && return
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _set_noise_gamma_kernel!(noise_scale, gamma, idx, dt, temperature)
    CUDA.@sync k(noise_scale, gamma, idx, dt, temperature; threads, blocks)
    return
end

function _ensure_corr_time_array(bp::BrownianIntegrators.BrownianParams{T}) where {T<:AbstractFloat}
    if bp.corr_time === nothing
        corr = Backends.fill_vector(Backends.CUDABackend(), zero(T), length(bp.gamma))
        return BrownianIntegrators.BrownianParams{T}(bp.gamma, bp.dt, bp.noise_scale, corr, bp.ou)
    end
    return bp
end

function _ensure_corr_time_array(em::BrownianIntegrators.EMParams{T}) where {T<:AbstractFloat}
    if em.corr_time === nothing
        corr = Backends.fill_vector(Backends.CUDABackend(), zero(T), length(em.gamma))
        return BrownianIntegrators.EMParams{T}(em.gamma, em.dt, em.noise_scale, corr, em.ou)
    end
    return em
end

# -----------------------------------------------------------------------------
# Integrator-spec parameter accessors
# -----------------------------------------------------------------------------

_gamma_view(spec::VVSpec) = spec.params.gamma
_gamma_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.gamma
_gamma_view(spec::BrownianSpec) = spec.params.gamma
_gamma_view(spec::EMSpec) = spec.params.gamma
_gamma_view(spec::NVESpec) =
    throw(ArgumentError("NVE integrator has no stochastic friction buffer. Use a thermostat-bearing or Langevin integrator to control temperature."))
_gamma_view(spec::NHCSpec) =
    throw(ArgumentError("NHC integrator has no per-particle friction buffer. Use set_thermostat_timescale!."))
_gamma_view(spec::CSVRSpec) =
    throw(ArgumentError("CSVR integrator has no per-particle friction buffer. Use set_thermostat_timescale!."))

_noise_scale_view(spec::VVSpec) = spec.params.noise_scale
_noise_scale_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.noise_scale
_noise_scale_view(spec::BrownianSpec) = spec.params.noise_scale
_noise_scale_view(spec::EMSpec) = spec.params.noise_scale
_noise_scale_view(spec::NVESpec) =
    throw(ArgumentError("NVE integrator has no stochastic noise scale."))
_noise_scale_view(spec::NHCSpec) =
    throw(ArgumentError("NHC integrator has no stochastic noise scale."))
_noise_scale_view(spec::CSVRSpec) =
    throw(ArgumentError("CSVR integrator has no stochastic noise scale."))

_corr_time_view(spec::VVSpec) = spec.params.corr_time
_corr_time_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.corr_time
_corr_time_view(spec::BrownianSpec) = spec.params.corr_time
_corr_time_view(spec::EMSpec) = spec.params.corr_time
_corr_time_view(spec::NVESpec) = nothing
_corr_time_view(spec::NHCSpec) = nothing
_corr_time_view(spec::CSVRSpec) = nothing

_ou_view(spec::VVSpec) = spec.params.ou
_ou_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.ou
_ou_view(spec::BrownianSpec) = spec.params.ou
_ou_view(spec::EMSpec) = spec.params.ou
_ou_view(spec::NVESpec) = nothing
_ou_view(spec::NHCSpec) = nothing
_ou_view(spec::CSVRSpec) = nothing

_dt_view(spec::VVSpec) = spec.params.dt
_dt_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.dt
_dt_view(spec::BrownianSpec) = spec.params.dt
_dt_view(spec::EMSpec) = spec.params.dt

function _set_corr_time_view!(spec::VVSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=corr, ou=p.ou)
    return spec
end

function _set_corr_time_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=corr, ou=p.ou)
    return spec
end

function _set_corr_time_view!(spec::BrownianSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, p.dt, p.noise_scale, corr, p.ou)
    return spec
end

function _set_corr_time_view!(spec::EMSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, p.dt, p.noise_scale, corr, p.ou)
    return spec
end

function _set_ou_view!(spec::VVSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=p.corr_time, ou=ou)
    return spec
end

function _set_ou_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=p.corr_time, ou=ou)
    return spec
end

function _set_ou_view!(spec::BrownianSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, p.dt, p.noise_scale, p.corr_time, ou)
    return spec
end

function _set_ou_view!(spec::EMSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, p.dt, p.noise_scale, p.corr_time, ou)
    return spec
end

function _set_dt_view!(spec::VVSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=dt, corr_time=p.corr_time, ou=p.ou)
    return spec
end

function _set_dt_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=dt, corr_time=p.corr_time, ou=p.ou)
    return spec
end

function _set_dt_view!(spec::BrownianSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, dt, p.noise_scale, p.corr_time, p.ou)
    return spec
end

function _set_dt_view!(spec::EMSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, dt, p.noise_scale, p.corr_time, p.ou)
    return spec
end

function _ensure_corr_time_array(spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    corr = _corr_time_view(spec)
    if corr === nothing
        corr = Backends.fill_vector(Backends.CUDABackend(), zero(T), length(_gamma_view(spec)))
        _set_corr_time_view!(spec, corr)
    end
    return corr
end

function _rebuild_single_mode_ou!(spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    corr = _corr_time_view(spec)
    if corr === nothing
        _set_ou_view!(spec, nothing)
        return spec
    end
    ou = SimulationCore._build_single_mode_ou(Backends.CUDABackend(), T, _noise_scale_view(spec), corr, _dt_view(spec))
    _set_ou_view!(spec, ou)
    return spec
end

include("filters/GroupOperationsCompat.jl")
include("filters/StochasticControls.jl")
include("filters/ThermostatControls.jl")
include("filters/FreezeControls.jl")

end # module Filters
