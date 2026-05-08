"""
Unified thermostat operator interface and state management.

This module provides:
- `AbstractThermostat`: base type for all thermostat implementations
- `ThermostatState`: common state interface for all thermostats
- Concrete implementations: `NoseHooverChainThermostat`, `CSVRThermostat`
- Unified parameter assignment and diagnostics

Thermostats are first-class operators that can be:
- Applied to specific particle groups
- Composed with integrators
- Tuned per-bath or per-particle
- Queried for energy exchange and conservation diagnostics
"""
module Thermostats

using CUDA
using CUDA: CuArray
using ..Backends
using ..ParticleGroups

export AbstractThermostat, ThermostatState,
       NoseHooverChainThermostat, CSVRThermostat,
       n_baths, target_temperature, response_time,
       set_target_temperature!, set_response_time!,
       cumulative_energy_exchange

# =============================================================================
# Thermostat Abstractions
# =============================================================================

"""
Abstract base type for all thermostat implementations.
"""
abstract type AbstractThermostat end

"""
Abstract base for thermostat state containers.
"""
abstract type ThermostatState end

# =============================================================================
# Nose-Hoover Chain Thermostat
# =============================================================================

"""
    NHCThermostatState{T}

State container for deterministic Nose-Hoover Chain (NHC) thermostat.

Fields:
- `xi`: chain positions, shape (chain_length, n_baths)
- `eta`: chain velocities, shape (chain_length, n_baths)  
- `chain_force`: accumulated chain forces, shape (chain_length, n_baths)
- `chain_masses`: thermostat inertias Q_j, shape (chain_length, n_baths)
- `target_temperature`: per-bath setpoint, length n_baths
- `particle_bath_id`: bath assignment per particle, length n_particles
- `dof_per_bath`: degrees of freedom per bath
- `kinetic_per_bath`: kinetic energy per bath
- `energy_exchange_per_bath`: cumulative reservoir energy exchange
"""
mutable struct NHCThermostatState{T<:AbstractFloat} <: ThermostatState
    xi::CuArray{T,2}
    eta::CuArray{T,2}
    chain_force::CuArray{T,2}
    chain_masses::CuArray{T,2}
    target_temperature::CuArray{T,1}
    particle_bath_id::CuArray{Int32,1}
    bath_counts::CuArray{Int32,1}
    dof_per_bath::CuArray{T,1}
    kinetic_total_per_bath::CuArray{T,1}
    kinetic_stage_start_per_bath::CuArray{T,1}
    cumulative_energy_exchange_per_bath::CuArray{T,1}
    thermostat_kinetic_per_bath::CuArray{T,1}
    thermostat_potential_per_bath::CuArray{T,1}
    last_velocity_scale_per_bath::CuArray{T,1}
    chain_masses_signature::UInt64
    dof_dirty::Bool
    kinetic_initialized::Bool
end

"""
    NHCThermostatParams{T}

Parameter container for NHC thermostat configuration.

Fields:
- `mass`: particle mass (needed for energy/virial calculations)
- `target_temperature`: per-bath setpoints (mutable)
- `tau`: per-bath response times (mutable)
- `substeps`: number of NHC chain substeps per MD step
- `chain_length`: length of the thermostat chain
- `chain_masses`: Q_j values (pre-computed or recomputed)
- `propagator`: variant of NHC propagation (:legacy, :gromacs, :lammps)
"""
mutable struct NHCThermostatParams{T<:AbstractFloat}
    mass::T
    target_temperature::Vector{T}
    tau::Vector{T}
    substeps::Int
    chain_length::Int
    chain_masses::Matrix{T}
    propagator::Symbol
end

"""
    NoseHooverChainThermostat{T} <: AbstractThermostat

Deterministic Nose-Hoover Chain thermostat with multi-bath support.
"""
mutable struct NoseHooverChainThermostat{T<:AbstractFloat} <: AbstractThermostat
    params::NHCThermostatParams{T}
    state::NHCThermostatState{T}
end

# =============================================================================
# CSVR (Bussi) Thermostat
# =============================================================================

"""
    CSVRThermostatState{T}

State container for canonical-sampling-through-velocity-rescaling (CSVR/Bussi) thermostat.

Fields:
- `target_temperature`: per-bath setpoint
- `tau`: per-bath response times
- `particle_bath_id`: bath assignment per particle
- `bath_counts`: number of particles per bath
- `dof_per_bath`: degrees of freedom per bath
- `kinetic_total_per_bath`: kinetic energy per bath
- `cumulative_energy_exchange_per_bath`: cumulative reservoir energy exchange
- `last_velocity_scale_per_bath`: most recent scaling factor per bath
"""
mutable struct CSVRThermostatState{T<:AbstractFloat} <: ThermostatState
    target_temperature::CuArray{T,1}
    tau::CuArray{T,1}
    particle_bath_id::CuArray{Int32,1}
    bath_counts::CuArray{Int32,1}
    dof_per_bath::CuArray{T,1}
    kinetic_total_per_bath::CuArray{T,1}
    cumulative_energy_exchange_per_bath::CuArray{T,1}
    last_velocity_scale_per_bath::CuArray{T,1}
    dof_dirty::Bool
    kinetic_initialized::Bool
end

"""
    CSVRThermostatParams{T}

Parameter container for CSVR thermostat configuration.

Fields:
- `mass`: particle mass (needed for energy/virial calculations)
- `target_temperature`: per-bath setpoints (mutable)
- `tau`: per-bath response times (mutable)
"""
mutable struct CSVRThermostatParams{T<:AbstractFloat}
    mass::T
    target_temperature::Vector{T}
    tau::Vector{T}
end

"""
    CSVRThermostat{T} <: AbstractThermostat

Stochastic CSVR/Bussi velocity-rescaling thermostat with multi-bath support.
"""
mutable struct CSVRThermostat{T<:AbstractFloat} <: AbstractThermostat
    params::CSVRThermostatParams{T}
    state::CSVRThermostatState{T}
end

# =============================================================================
# Common Interface Methods
# =============================================================================

"""
    n_baths(thermo::AbstractThermostat) -> Int

Number of baths in this thermostat.
"""
n_baths(thermo::NoseHooverChainThermostat) = length(thermo.params.target_temperature)
n_baths(thermo::CSVRThermostat) = length(thermo.params.target_temperature)

"""
    target_temperature(thermo::AbstractThermostat) -> Vector

Per-bath target temperatures.
"""
target_temperature(thermo::NoseHooverChainThermostat) = thermo.params.target_temperature
target_temperature(thermo::CSVRThermostat) = thermo.params.target_temperature

"""
    response_time(thermo::AbstractThermostat) -> Vector

Per-bath response times (tau).
"""
response_time(thermo::NoseHooverChainThermostat) = thermo.params.tau
response_time(thermo::CSVRThermostat) = thermo.params.tau

"""
    cumulative_energy_exchange(thermo::AbstractThermostat) -> CuArray{T}

Cumulative energy exchanged with the thermostat reservoir per bath.
"""
cumulative_energy_exchange(thermo::NoseHooverChainThermostat) = 
    thermo.state.cumulative_energy_exchange_per_bath
cumulative_energy_exchange(thermo::CSVRThermostat) = 
    thermo.state.cumulative_energy_exchange_per_bath

"""
    set_target_temperature!(thermo::AbstractThermostat, T_new)

Update target temperature(s).

Accepts:
- Scalar: apply to all baths
- Vector: length must match n_baths(thermo)
"""
function set_target_temperature!(thermo::NoseHooverChainThermostat{T}, T_new::Real) where {T<:AbstractFloat}
    fill!(thermo.params.target_temperature, T(T_new))
    # Update device buffer
    CUDA.copyto!(thermo.state.target_temperature, thermo.params.target_temperature)
    thermo.state.kinetic_initialized = false
    return thermo
end

function set_target_temperature!(thermo::NoseHooverChainThermostat{T}, T_new::AbstractVector{<:Real}) where {T<:AbstractFloat}
    @assert length(T_new) == length(thermo.params.target_temperature)
    thermo.params.target_temperature .= T.(T_new)
    CUDA.copyto!(thermo.state.target_temperature, thermo.params.target_temperature)
    thermo.state.kinetic_initialized = false
    return thermo
end

function set_target_temperature!(thermo::CSVRThermostat{T}, T_new::Real) where {T<:AbstractFloat}
    fill!(thermo.params.target_temperature, T(T_new))
    CUDA.copyto!(thermo.state.target_temperature, thermo.params.target_temperature)
    thermo.state.kinetic_initialized = false
    return thermo
end

function set_target_temperature!(thermo::CSVRThermostat{T}, T_new::AbstractVector{<:Real}) where {T<:AbstractFloat}
    @assert length(T_new) == length(thermo.params.target_temperature)
    thermo.params.target_temperature .= T.(T_new)
    CUDA.copyto!(thermo.state.target_temperature, thermo.params.target_temperature)
    thermo.state.kinetic_initialized = false
    return thermo
end

"""
    set_response_time!(thermo::AbstractThermostat, tau_new)

Update response time(s).

Accepts:
- Scalar: apply to all baths
- Vector: length must match n_baths(thermo)
"""
function set_response_time!(thermo::NoseHooverChainThermostat{T}, tau_new::Real) where {T<:AbstractFloat}
    fill!(thermo.params.tau, T(tau_new))
    # Note: chain_masses should be recomputed based on new tau
    return thermo
end

function set_response_time!(thermo::NoseHooverChainThermostat{T}, tau_new::AbstractVector{<:Real}) where {T<:AbstractFloat}
    @assert length(tau_new) == length(thermo.params.tau)
    thermo.params.tau .= T.(tau_new)
    return thermo
end

function set_response_time!(thermo::CSVRThermostat{T}, tau_new::Real) where {T<:AbstractFloat}
    fill!(thermo.params.tau, T(tau_new))
    CUDA.copyto!(thermo.state.tau, thermo.params.tau)
    return thermo
end

function set_response_time!(thermo::CSVRThermostat{T}, tau_new::AbstractVector{<:Real}) where {T<:AbstractFloat}
    @assert length(tau_new) == length(thermo.params.tau)
    thermo.params.tau .= T.(tau_new)
    CUDA.copyto!(thermo.state.tau, thermo.params.tau)
    return thermo
end

end  # module Thermostats
