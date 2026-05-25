const INTEGRATOR_ID_UNKNOWN  = UInt8(0)
const INTEGRATOR_ID_LANGEVIN = UInt8(1)
const INTEGRATOR_ID_BROWNIAN = UInt8(2)
const INTEGRATOR_ID_NHC      = UInt8(3)
const INTEGRATOR_ID_CSVR     = UInt8(4)

"""
Concrete spec for the package's Langevin/GJF `velocityverlet` path.
"""
mutable struct VVSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.VVParams{T}
    workspace::StochasticWorkspace{T}
end

"""
BAOAB splitting spec used in `examples/TwoT_2D_LD_BAOAB.jl`.
"""
mutable struct BAOABSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
BAOA splitting (no trailing B) for legacy scripts.
"""
mutable struct BAOASpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Generalized simulation scheme (GSM) spec, which reuses the BAOAB parameter type.
"""
mutable struct GSMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::LangevinIntegrators.BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Midpoint Brownian integrator spec created by [`eulerheun`](@ref).
"""
mutable struct BrownianSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.BrownianParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Euler-Maruyama overdamped spec created by [`eulermaruyama`](@ref).
"""
mutable struct EMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianIntegrators.EMParams{T}
    workspace::StochasticWorkspace{T}
end

"""
    NHCParams{T}

Parameter container for deterministic Nose-Hoover Chain (NHC) thermostatting.
`chain_masses` stores the thermostat inertia values `Q_j`.
"""
mutable struct NHCParams{T<:AbstractFloat}
    mass::T
    target_temperature::Vector{T}
    tau::Vector{T}
    substeps::Int
    chain_length::Int
    chain_masses::Matrix{T}
    propagator::UInt8
end

"""
    NHCWorkspace{T}

Integrator-local Nose-Hoover Chain state (`xi`, `eta`) and reusable scratch
buffers for chain-force evaluation.
"""
mutable struct NHCWorkspace{T<:AbstractFloat}
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
    NHCSpec{T}

Concrete integrator specification for NVT dynamics with a Nose-Hoover Chain
thermostat.
"""
mutable struct NHCSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::NHCParams{T}
    workspace::NHCWorkspace{T}
end

"""
    CSVRParams{T}

Parameter container for the canonical-sampling-through-velocity-rescaling
(CSVR/Bussi) thermostat. The thermostat acts on the kinetic energy of each
assigned bath with target temperature `target_temperature[b]` and response time
`tau[b]`.
"""
mutable struct CSVRParams{T<:AbstractFloat}
    mass::T
    target_temperature::Vector{T}
    tau::Vector{T}
end

"""
    CSVRWorkspace{T}

Integrator-local buffers for fully GPU CSVR thermostatting. The thermostat is
global within each bath, so the workspace stores per-bath reductions and the
cumulative energy exchanged with the coupling reservoir.
"""
mutable struct CSVRWorkspace{T<:AbstractFloat}
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
    CSVRSpec{T}

Concrete integrator specification for deterministic NVT dynamics with a
stochastic global Bussi/CSVR thermostat.
"""
mutable struct CSVRSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::CSVRParams{T}
    workspace::CSVRWorkspace{T}
end

VVSpec(params::LangevinIntegrators.VVParams{T}) where {T<:AbstractFloat} = VVSpec{T}(params, _empty_workspace(T))
BAOABSpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = BAOABSpec{T}(params, _empty_workspace(T))
BAOASpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = BAOASpec{T}(params, _empty_workspace(T))
GSMSpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = GSMSpec{T}(params, _empty_workspace(T))
BrownianSpec(params::BrownianIntegrators.BrownianParams{T}) where {T<:AbstractFloat} = BrownianSpec{T}(params, _empty_workspace(T))
EMSpec(params::BrownianIntegrators.EMParams{T}) where {T<:AbstractFloat} = EMSpec{T}(params, _empty_workspace(T))
NHCSpec(params::NHCParams{T}) where {T<:AbstractFloat} =
    NHCSpec{T}(params,
               _new_nhc_workspace(Backends.CUDABackend(), T,
                                  params.chain_length,
                                  length(params.target_temperature),
                                  0))
CSVRSpec(params::CSVRParams{T}) where {T<:AbstractFloat} =
    CSVRSpec{T}(params,
                _new_csvr_workspace(Backends.CUDABackend(), T,
                                    length(params.target_temperature),
                                    0))

integrator_id(::Union{VVSpec,BAOABSpec,BAOASpec,GSMSpec}) = INTEGRATOR_ID_LANGEVIN
integrator_id(::Union{BrownianSpec,EMSpec}) = INTEGRATOR_ID_BROWNIAN
integrator_id(::NHCSpec) = INTEGRATOR_ID_NHC
integrator_id(::CSVRSpec) = INTEGRATOR_ID_CSVR

integrator_name(::VVSpec) = :velocity_verlet
integrator_name(::BAOABSpec) = :baoab
integrator_name(::BAOASpec) = :baoa
integrator_name(::GSMSpec) = :gsm
integrator_name(::BrownianSpec) = :brownian_midpoint
integrator_name(::EMSpec) = :euler_maruyama
integrator_name(::NHCSpec) = :nose_hoover_chain
integrator_name(::CSVRSpec) = :csvr

stage_sequence(::VVSpec) = (:kick1, :drift, :force, :kick2)
stage_sequence(::BAOABSpec) = (:B1, :A1, :O, :A2, :force, :B2)
stage_sequence(::GSMSpec) = (:B1, :A1, :O, :A2, :force, :B2)
stage_sequence(::BAOASpec) = (:B1, :A1, :O, :A2, :force, :power, :kinetic_refresh)
stage_sequence(::BrownianSpec) = (:midpoint_predict, :midpoint_force, :final_position, :force)
stage_sequence(::EMSpec) = (:em_position, :force)
stage_sequence(::NHCSpec) = (:thermostat_pre, :kick1, :drift, :force, :kick2, :thermostat_post)
stage_sequence(::CSVRSpec) = (:kick1, :drift, :force, :kick2, :thermostat)

@inline function _require_positive_gamma!(gamma::CuArray{T,1}, integrator::AbstractString) where {T<:AbstractFloat}
    gmin = minimum(gamma)
    if !(gmin > zero(T))
        throw(ArgumentError("$(integrator) integrator requires gamma > 0 for all particles."))
    end
    return nothing
end
