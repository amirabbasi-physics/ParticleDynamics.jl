const INTEGRATOR_ID_UNKNOWN  = UInt8(0)
const INTEGRATOR_ID_LANGEVIN = UInt8(1)
const INTEGRATOR_ID_BROWNIAN = UInt8(2)
const INTEGRATOR_ID_NHC      = UInt8(3)
const INTEGRATOR_ID_CSVR     = UInt8(4)
const INTEGRATOR_ID_NVE      = UInt8(5)

# ------------------------------------------------------------------
# Stochastic integrator parameters (the device engine reads these; filters
# update `gamma`/`noise_scale` in place)
# ------------------------------------------------------------------

"""
    VVParams(gamma, mass, noise_scale; dt=1, corr_time=nothing, ou=nothing)

Parameters for the Grønbech-Jensen/Farago (velocity-Verlet) Langevin scheme.
`gamma` and `noise_scale` are CuArrays so `Filters.set_temperature!` can update
them in-place. `corr_time` stores optional
per-particle Ornstein–Uhlenbeck correlation times.
"""
struct VVParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
end
VVParams(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1};
         dt::T=one(T),
         corr_time::Union{Nothing,CuArray{T,1}}=nothing,
         ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} = begin
    @assert corr_time === nothing || length(corr_time) == length(gamma)
    VVParams{T}(gamma, mass, dt, noise_scale, corr_time, ou)
end
VVParams{T}(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1};
            dt::T=one(T),
            corr_time::Union{Nothing,CuArray{T,1}}=nothing,
            ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    VVParams(gamma, mass, noise_scale; dt=dt, corr_time=corr_time, ou=ou)

"""
    BAOABParams(gamma, mass, noise_scale; dt=1, corr_time=nothing, ou=nothing)

Parameter container for the BAOAB/BAOA/GSM splitting schemes; same layout as
[`VVParams`](@ref).
"""
struct BAOABParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
end
BAOABParams(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1};
            dt::T=one(T),
            corr_time::Union{Nothing,CuArray{T,1}}=nothing,
            ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} = begin
    @assert corr_time === nothing || length(corr_time) == length(gamma)
    BAOABParams{T}(gamma, mass, dt, noise_scale, corr_time, ou)
end
BAOABParams{T}(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1};
               dt::T=one(T),
               corr_time::Union{Nothing,CuArray{T,1}}=nothing,
               ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    BAOABParams(gamma, mass, noise_scale; dt=dt, corr_time=corr_time, ou=ou)

"""
    BrownianParams(gamma, noise_scale[, corr_time[, dt[, ou]]])

Parameters for the stochastic midpoint (Euler–Heun) Brownian integrator.
`gamma` and `noise_scale` are CuArrays so filters can assign different
temperatures to different particle groups.
"""
struct BrownianParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
    function BrownianParams{T}(gamma::CuArray{T,1},
                               dt::T,
                               noise_scale::CuArray{T,1},
                               corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                               ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, dt, noise_scale, corr_time, ou)
    end
end

BrownianParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, one(T), noise_scale, nothing, nothing)
BrownianParams(gamma::CuArray{T,1},
               noise_scale::CuArray{T,1},
               corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, one(T), noise_scale, corr_time, nothing)
BrownianParams(gamma::CuArray{T,1},
               noise_scale::CuArray{T,1},
               corr_time::Union{Nothing,CuArray{T,1}},
               dt::T,
               ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, dt, noise_scale, corr_time, ou)

function BrownianParams{T}(gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    γ = T(gamma)
    Δt = T(dt)
    scale = sqrt(T(2) * γ * T(temperature) * Δt)
    return BrownianParams{T}(CUDA.fill(γ, N), Δt, CUDA.fill(scale, N), nothing, nothing)
end
BrownianParams(::Type{T}, gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, temperature, dt, N)
function BrownianParams(gamma::Real, temperature::Real, dt::Real, N::Integer)
    inferred = promote_type(float(typeof(gamma)), float(typeof(temperature)), float(typeof(dt)))
    T = inferred <: AbstractFloat ? inferred : Float32
    return BrownianParams{T}(gamma, temperature, dt, N)
end

"""
    EMParams(gamma, noise_scale[, corr_time[, dt[, ou]]])

Parameters for the Euler–Maruyama Brownian step. Same layout as
[`BrownianParams`](@ref), so `Filters.set_temperature!` and friends operate on
either type.
"""
struct EMParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
    function EMParams{T}(gamma::CuArray{T,1},
                         dt::T,
                         noise_scale::CuArray{T,1},
                         corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                         ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, dt, noise_scale, corr_time, ou)
    end
end

EMParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    EMParams{T}(gamma, one(T), noise_scale, nothing, nothing)
EMParams(gamma::CuArray{T,1},
         noise_scale::CuArray{T,1},
         corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} =
    EMParams{T}(gamma, one(T), noise_scale, corr_time, nothing)
EMParams(gamma::CuArray{T,1},
         noise_scale::CuArray{T,1},
         corr_time::Union{Nothing,CuArray{T,1}},
         dt::T,
         ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    EMParams{T}(gamma, dt, noise_scale, corr_time, ou)

"""
Concrete spec for the package's Langevin/GJF `velocityverlet` path.
"""
mutable struct VVSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::VVParams{T}
    workspace::StochasticWorkspace{T}
end

"""
BAOAB splitting spec.
"""
mutable struct BAOABSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
BAOA splitting (no trailing B half-kick).
"""
mutable struct BAOASpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Generalized simulation scheme (GSM) spec, which reuses the BAOAB parameter type.
"""
mutable struct GSMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BAOABParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Midpoint Brownian integrator spec created by [`eulerheun`](@ref).
"""
mutable struct BrownianSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::BrownianParams{T}
    workspace::StochasticWorkspace{T}
end

"""
Euler-Maruyama overdamped spec created by [`eulermaruyama`](@ref).
"""
mutable struct EMSpec{T<:AbstractFloat} <: IntegratorSpec{T}
    params::EMParams{T}
    workspace::StochasticWorkspace{T}
end

"""
    NVESpec{T}

Concrete integrator specification for deterministic microcanonical (NVE)
dynamics using plain velocity Verlet.
"""
struct NVESpec{T<:AbstractFloat} <: IntegratorSpec{T}
    dt::T
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
    interval::Int    # coupling interval in steps (GROMACS nsttcouple); 1 = every step
end
CSVRParams{T}(mass, target_temperature, tau) where {T<:AbstractFloat} = CSVRParams{T}(mass, target_temperature, tau, 1)

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

VVSpec(params::VVParams{T}) where {T<:AbstractFloat} = VVSpec{T}(params, _empty_workspace(T))
BAOABSpec(params::BAOABParams{T}) where {T<:AbstractFloat} = BAOABSpec{T}(params, _empty_workspace(T))
BAOASpec(params::BAOABParams{T}) where {T<:AbstractFloat} = BAOASpec{T}(params, _empty_workspace(T))
GSMSpec(params::BAOABParams{T}) where {T<:AbstractFloat} = GSMSpec{T}(params, _empty_workspace(T))
BrownianSpec(params::BrownianParams{T}) where {T<:AbstractFloat} = BrownianSpec{T}(params, _empty_workspace(T))
EMSpec(params::EMParams{T}) where {T<:AbstractFloat} = EMSpec{T}(params, _empty_workspace(T))
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
integrator_id(::NVESpec) = INTEGRATOR_ID_NVE

integrator_name(::VVSpec) = :velocity_verlet
integrator_name(::BAOABSpec) = :baoab
integrator_name(::BAOASpec) = :baoa
integrator_name(::GSMSpec) = :gsm
integrator_name(::BrownianSpec) = :brownian_midpoint
integrator_name(::EMSpec) = :euler_maruyama
integrator_name(::NHCSpec) = :nose_hoover_chain
integrator_name(::CSVRSpec) = :csvr
integrator_name(::NVESpec) = :nve


@inline function _require_positive_gamma!(gamma::CuArray{T,1}, integrator::AbstractString) where {T<:AbstractFloat}
    gmin = minimum(gamma)
    if !(gmin > zero(T))
        throw(ArgumentError("$(integrator) integrator requires gamma > 0 for all particles."))
    end
    return nothing
end
