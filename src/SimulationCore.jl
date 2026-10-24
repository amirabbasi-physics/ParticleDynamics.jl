module SimulationCore

using CUDA
import ..IntegratorInterfaces
import ..IntegratorInterfaces: AbstractIntegratorSpec,
                               validate_integrator_inputs!,
                               ensure_integrator_workspace!,
                               integrator_id,
                               integrator_name,
                               collect_integrator_observables
using ..Definitions
using ..Backends
using ..NeighborLists
using ..NonBondedForces
using ..BondedForces
using ..LangevinIntegrators
using ..BrownianIntegrators
using ..Collisions
using ..Electrostatics
using ..ClusterPairs
import ..RigidWaterConstraints
 

const NL_CHECK_STRIDE = 20  # only check NL rebuild every N steps to cut overhead
# Weeks-Chandler-Andersen cutoff factor: r_c = 2^(1/6) * σ ≈ 1.12246 σ
const WCA_RC_FACTOR = 1.122462048309373

# Nonbonded kind tags (host-side routing only)
const NB_KIND_LJ      = UInt8(1)
const NB_KIND_WCA     = UInt8(2)
const NB_KIND_SOFTREP = UInt8(3)
const NB_KIND_NONE    = UInt8(4)   # no short-range vdW term

# Freeze modes
const FREEZE_NONE   = UInt8(0)
const FREEZE_HOLD   = UInt8(1)
const FREEZE_SPRING = UInt8(2)

export SimulationState, build_simulation, step!, step_graph!, sync_unwrapped!, invalidate_forces!, accumulate_virial!, virial_components, virial_tensor
export collect_step_observables
export reset_bath_exchange_accumulators!
export IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NVESpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec
export velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nve, nosehooverchain, csvr
export AbstractExternalPotential, external_forces!, attach_external_potential!, detach_external_potential!
export attach_electrostatics!, detach_electrostatics!, set_charges!, charges, electrostatic_energy, set_box!
export set_nonbonded_backend!, nonbonded_backend, set_vdw_modifier!, vdw_modifier
export attach_rigid_water!, kinetic_dof, constraint_virial_trace, molecular_virial_trace,
       attach_pressure_control!, detach_pressure_control!
export EvaluationRequest, evaluate_configuration!, total_potential_energy, total_virial_trace, total_configurational_virial
export attach_bonded_terms!, detach_bonded_terms!, bonded_energy_components
export minimize!, SteepestDescent, FIRE, MinimizationResult
export set_cell_and_positions!, set_cell_and_molecules!, cell_volume, cell_supported, TrialWorkspace, save_trial_state!, restore_trial_state!
export total_kinetic_energy, mechanical_pressure, MonteCarloVolumeMove, monte_carlo_volume_move!, acceptance_ratio,
       StochasticCellRescalingState, stochastic_cell_rescale!

include("SimulationState.jl")

# -------------------------
# Unified integrator specs
# -------------------------
const IntegratorSpec = AbstractIntegratorSpec

# Stage 8: Extract integrator specs and constructors
include("simulation/StochasticWorkspace.jl")
include("simulation/IntegratorSpecs.jl")
include("simulation/LangevinSpecConstructors.jl")
include("simulation/BrownianSpecConstructors.jl")
include("simulation/ThermostatSpecConstructors.jl")

@inline function _device_particle_buffer(backend::Backends.AbstractBackend,
                                         ::Type{T},
                                         N::Integer,
                                         value::Union{AbstractVector{<:Real},Real},
                                         name::AbstractString) where {T<:AbstractFloat}
    if value isa Real
        return Backends.fill_vector(backend, T(value), N)
    end
    length(value) == N ||
        throw(ArgumentError("$(name) vector must have length $(N), got $(length(value))."))
    return Backends.from_host(backend, T.(value))
end

@inline function _device_corr_time_buffer(backend::Backends.AbstractBackend,
                                          ::Type{T},
                                          N::Integer,
                                          value::Union{Nothing,AbstractVector{<:Real},Real}) where {T<:AbstractFloat}
    value === nothing && return nothing
    return _device_particle_buffer(backend, T, N, value, "noise_corr_time")
end

@inline function _all_particle_indices(backend::Backends.AbstractBackend, N::Integer)
    return Backends.from_host(backend, Int32.(collect(1:N)))
end

include("random/NoiseConstruction.jl")

@inline _is_3d(st::SimulationState) = st.rz !== nothing



"""
    _ensure_workspace_buffers!(workspace, st; require_ou)

Ensure integrator-local stochastic buffers are allocated with the correct shape
for the current simulation state.
"""
function _ensure_workspace_buffers!(workspace::StochasticWorkspace{T},
                                    st::SimulationState{T};
                                    ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
    N = length(st.rx)

    if length(workspace.rf_x) != N
        workspace.rf_x = CUDA.CuArray{T}(undef, N)
        fill!(workspace.rf_x, zero(T))
    end
    if length(workspace.rf_y) != N
        workspace.rf_y = CUDA.CuArray{T}(undef, N)
        fill!(workspace.rf_y, zero(T))
    end

    if _is_3d(st)
        if workspace.rf_z === nothing || length(workspace.rf_z) != N
            workspace.rf_z = CUDA.CuArray{T}(undef, N)
            fill!(workspace.rf_z, zero(T))
        end
    else
        workspace.rf_z = nothing
    end

    if ou !== nothing
        M, K = size(ou.coeff_a)
        if workspace.ou_x === nothing || size(workspace.ou_x) != (M, K)
            workspace.ou_x = CUDA.zeros(T, M, K)
            _initialize_ou_state!(workspace.ou_x, ou.scale)
        end
        if workspace.ou_y === nothing || size(workspace.ou_y) != (M, K)
            workspace.ou_y = CUDA.zeros(T, M, K)
            _initialize_ou_state!(workspace.ou_y, ou.scale)
        end
        if _is_3d(st)
            if workspace.ou_z === nothing || size(workspace.ou_z) != (M, K)
                workspace.ou_z = CUDA.zeros(T, M, K)
                _initialize_ou_state!(workspace.ou_z, ou.scale)
            end
        else
            workspace.ou_z = nothing
        end
    else
        workspace.ou_x = nothing
        workspace.ou_y = nothing
        workspace.ou_z = nothing
    end

    return nothing
end

function _initialize_ou_state!(state::CuArray{T,2},
                               scale::CuArray{T,2}) where {T<:AbstractFloat}
    size(state) == size(scale) || throw(ArgumentError("OU state and scale buffers must have identical shapes."))
    M, K = size(state)
    M == 0 && return state
    state .= scale .* CUDA.randn(T, M, K)
    return state
end

include("simulation/Freeze.jl")

# ==========================================
#  Top-level, non-capturing init kernels
#  (avoid nested functions / closures)
# ==========================================

function _init_vel2_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T},
    inv_mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        scale = sqrt(max(temperature_vec[i] * inv_mass, zero(T)))
        vx[i] = randn(T) * scale
        vy[i] = randn(T) * scale
    end
    return
end

function _init_vel3_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    vz::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T},
    inv_mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        scale = sqrt(max(temperature_vec[i] * inv_mass, zero(T)))
        vx[i] = randn(T) * scale
        vy[i] = randn(T) * scale
        vz[i] = randn(T) * scale
    end
    return
end

function _init_vel2_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T},
    inv_mass_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        scale = sqrt(max(temperature_vec[i] * inv_mass_vec[i], zero(T)))
        vx[i] = randn(T) * scale
        vy[i] = randn(T) * scale
    end
    return
end

function _init_vel3_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    vz::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T},
    inv_mass_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        scale = sqrt(max(temperature_vec[i] * inv_mass_vec[i], zero(T)))
        vx[i] = randn(T) * scale
        vy[i] = randn(T) * scale
        vz[i] = randn(T) * scale
    end
    return
end

function _init_vel2!(vx::CuArray{T,1},
                     vy::CuArray{T,1},
                     temperature_vec::CuArray{T,1},
                     mass::T) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    inv_mass = mass > zero(T) ? inv(mass) : zero(T)
    k = CUDA.@cuda launch=false _init_vel2_kernel!(vx, vy, temperature_vec, inv_mass)
    CUDA.@sync k(vx, vy, temperature_vec, inv_mass; threads, blocks)
    return nothing
end

function _init_vel2!(vx::CuArray{T,1},
                     vy::CuArray{T,1},
                     temperature_vec::CuArray{T,1},
                     inv_mass_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel2_kernel!(vx, vy, temperature_vec, inv_mass_vec)
    CUDA.@sync k(vx, vy, temperature_vec, inv_mass_vec; threads, blocks)
    return nothing
end

function _init_vel3!(vx::CuArray{T,1},
                     vy::CuArray{T,1},
                     vz::CuArray{T,1},
                     temperature_vec::CuArray{T,1},
                     mass::T) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    inv_mass = mass > zero(T) ? inv(mass) : zero(T)
    k = CUDA.@cuda launch=false _init_vel3_kernel!(vx, vy, vz, temperature_vec, inv_mass)
    CUDA.@sync k(vx, vy, vz, temperature_vec, inv_mass; threads, blocks)
    return nothing
end

function _init_vel3!(vx::CuArray{T,1},
                     vy::CuArray{T,1},
                     vz::CuArray{T,1},
                     temperature_vec::CuArray{T,1},
                     inv_mass_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel3_kernel!(vx, vy, vz, temperature_vec, inv_mass_vec)
    CUDA.@sync k(vx, vy, vz, temperature_vec, inv_mass_vec; threads, blocks)
    return nothing
end

include("SimulationBuild.jl")

# TODO: move these includes under observables/ once the Simulation module split is complete.
include("simulation/Virial.jl")
include("simulation/EnergyAccumulation.jl")
include("simulation/SteppingEngine.jl")
include("simulation/ExternalPotential.jl")
include("simulation/ForceEvaluation.jl")
include("simulation/ElectrostaticsCoupling.jl")
include("simulation/ClusterCoupling.jl")
include("simulation/RigidWaterCoupling.jl")
include("simulation/ConfigurationEvaluation.jl")
include("simulation/Minimization.jl")
include("simulation/CellUpdates.jl")
include("simulation/Barostats.jl")



"""
    _refresh_kinetic_buffer!(st)

Refresh the per-particle kinetic-energy buffer `st.Ekin` from the current
velocity field using the mass representation owned by `st`.
"""
function _refresh_kinetic_buffer!(st::SimulationState{T}) where {T<:AbstractFloat}
    if st.mass_particle === nothing
        return _refresh_kinetic_buffer!(st, st.mass)
    end
    mp = st.mass_particle::CuArray{T,1}
    if _is_3d(st)
        @. st.Ekin = T(0.5) * mp * (st.vx * st.vx + st.vy * st.vy + st.vz * st.vz)
    else
        @. st.Ekin = T(0.5) * mp * (st.vx * st.vx + st.vy * st.vy)
    end
    return nothing
end

"""
    _refresh_kinetic_buffer!(st, mass)

Refresh the per-particle kinetic-energy buffer `st.Ekin` from the current
velocity field using an explicit uniform mass override.
"""
function _refresh_kinetic_buffer!(st::SimulationState{T}, mass::T) where {T<:AbstractFloat}
    if _is_3d(st)
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy + st.vz * st.vz)
    else
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy)
    end
    return nothing
end



function _require_positive_inertial_mass!(st::SimulationState{T},
                                          integrator::AbstractString) where {T<:AbstractFloat}
    if st.mass_particle === nothing
        st.mass > zero(T) ||
            throw(ArgumentError("$(integrator) integrator requires mass > 0 for all particles."))
    else
        minimum(st.mass_particle::CuArray{T,1}) > zero(T) ||
            throw(ArgumentError("$(integrator) integrator requires mass > 0 for all particles."))
    end
    return nothing
end



include("simulation/NHCStepper.jl")
include("simulation/CSVRStepper.jl")

using ..Definitions: _require_stochastic_dt!

function validate_integrator_inputs!(spec::VVSpec{T}, st::SimulationState{T}, dt) where {T<:AbstractFloat}
    _require_stochastic_dt!(spec.params, dt)
    _require_positive_inertial_mass!(st, string(integrator_name(spec)))
    return nothing
end

function validate_integrator_inputs!(spec::NVESpec{T}, st::SimulationState{T}, dt) where {T<:AbstractFloat}
    _require_positive_inertial_mass!(st, string(integrator_name(spec)))
    return nothing
end

function validate_integrator_inputs!(spec::Union{BAOABSpec,BAOASpec,GSMSpec}, st, dt)
    _require_stochastic_dt!(spec.params, dt)
    _require_positive_gamma!(spec.params.gamma, string(integrator_name(spec)))
    _require_positive_inertial_mass!(st, string(integrator_name(spec)))
    return nothing
end

function validate_integrator_inputs!(spec::BrownianSpec, st, dt)
    _require_stochastic_dt!(spec.params, dt)
    _require_positive_gamma!(spec.params.gamma, "Brownian midpoint")
    return nothing
end

function validate_integrator_inputs!(spec::EMSpec, st, dt)
    _require_stochastic_dt!(spec.params, dt)
    _require_positive_gamma!(spec.params.gamma, "Euler-Maruyama")
    return nothing
end

function validate_integrator_inputs!(spec::NHCSpec{T}, st, dt) where {T<:AbstractFloat}
    p = spec.params
    _require_positive_inertial_mass!(st, "NHC")
    p.substeps >= 1 || throw(ArgumentError("NHC requires substeps >= 1."))
    p.chain_length >= 1 || throw(ArgumentError("NHC requires chain_length >= 1."))
    (p.propagator == NHC_PROPAGATOR_LEGACY ||
     p.propagator == NHC_PROPAGATOR_GROMACS ||
     p.propagator == NHC_PROPAGATOR_LAMMPS) ||
        throw(ArgumentError("NHC requires a supported propagator id, got $(p.propagator)."))
    nbaths = length(p.target_temperature)
    nbaths >= 1 || throw(ArgumentError("NHC requires at least one bath."))
    length(p.tau) == nbaths ||
        throw(ArgumentError("NHC requires length(tau) == length(target_temperature)."))
    size(p.chain_masses, 1) == p.chain_length ||
        throw(ArgumentError("NHC chain_masses first dimension must equal chain_length."))
    size(p.chain_masses, 2) == nbaths ||
        throw(ArgumentError("NHC chain_masses second dimension must equal number of baths."))
    @inbounds for b in 1:nbaths
        p.target_temperature[b] > zero(T) ||
            throw(ArgumentError("NHC requires target_temperature[$(b)] > 0."))
        p.tau[b] > zero(T) || throw(ArgumentError("NHC requires tau[$(b)] > 0."))
    end
    @inbounds for j in 1:p.chain_length, b in 1:nbaths
        p.chain_masses[j, b] > zero(T) ||
            throw(ArgumentError("NHC chain mass Q[$(j), bath=$(b)] must be > 0."))
    end
    return nothing
end

function validate_integrator_inputs!(spec::CSVRSpec{T}, st, dt) where {T<:AbstractFloat}
    p = spec.params
    _require_positive_inertial_mass!(st, "CSVR")
    nbaths = length(p.target_temperature)
    nbaths >= 1 || throw(ArgumentError("CSVR requires at least one bath."))
    length(p.tau) == nbaths ||
        throw(ArgumentError("CSVR requires length(tau) == length(target_temperature)."))
    @inbounds for b in 1:nbaths
        p.target_temperature[b] > zero(T) ||
            throw(ArgumentError("CSVR requires target_temperature[$(b)] > 0."))
        p.tau[b] > zero(T) || throw(ArgumentError("CSVR requires tau[$(b)] > 0."))
    end
    return nothing
end

function ensure_integrator_workspace!(spec::VVSpec{T}, st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::Union{BrownianSpec{T},EMSpec{T}},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    _ensure_workspace_buffers!(spec.workspace, st; ou=spec.params.ou)
    return nothing
end

function ensure_integrator_workspace!(spec::NHCSpec{T},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    M = p.chain_length
    B = length(p.target_temperature)
    N = length(st.rx)

    if size(ws.xi) != (M, B)
        ws.xi = CUDA.zeros(T, M, B)
        ws.eta = CUDA.zeros(T, M, B)
        ws.chain_force = CUDA.zeros(T, M, B)
        ws.chain_masses = CUDA.zeros(T, M, B)
        ws.chain_masses_signature = UInt64(0)
        ws.kinetic_initialized = false
    end

    if length(ws.target_temperature) != B
        ws.target_temperature = CUDA.zeros(T, B)
    end
    copyto!(ws.target_temperature, p.target_temperature)

    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    if length(ws.bath_counts) != B
        ws.bath_counts = CUDA.zeros(Int32, B)
        ws.dof_dirty = true
    end
    if length(ws.dof_per_bath) != B
        ws.dof_per_bath = CUDA.zeros(T, B)
        ws.dof_dirty = true
    end
    if length(ws.kinetic_total_per_bath) != B
        ws.kinetic_total_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.kinetic_stage_start_per_bath) != B
        ws.kinetic_stage_start_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.cumulative_energy_exchange_per_bath) != B
        ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.thermostat_kinetic_per_bath) != B
        ws.thermostat_kinetic_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.thermostat_potential_per_bath) != B
        ws.thermostat_potential_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.last_velocity_scale_per_bath) != B
        ws.last_velocity_scale_per_bath = CUDA.fill(one(T), B)
    end

    sig = _nhc_chain_masses_signature(p.chain_masses)
    if ws.chain_masses_signature != sig
        copyto!(ws.chain_masses, p.chain_masses)
        ws.chain_masses_signature = sig
        ws.kinetic_initialized = false
    end

    if st.step == 0
        ws.kinetic_initialized = false
    end
    return nothing
end

function ensure_integrator_workspace!(spec::CSVRSpec{T},
                                      st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    B = length(p.target_temperature)
    N = length(st.rx)

    if length(ws.target_temperature) != B
        ws.target_temperature = CUDA.zeros(T, B)
    end
    copyto!(ws.target_temperature, p.target_temperature)

    if length(ws.tau) != B
        ws.tau = CUDA.zeros(T, B)
    end
    copyto!(ws.tau, p.tau)

    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    if length(ws.bath_counts) != B
        ws.bath_counts = CUDA.zeros(Int32, B)
        ws.dof_dirty = true
    end
    if length(ws.dof_per_bath) != B
        ws.dof_per_bath = CUDA.zeros(T, B)
        ws.dof_dirty = true
    end
    if length(ws.kinetic_total_per_bath) != B
        ws.kinetic_total_per_bath = CUDA.zeros(T, B)
        ws.kinetic_initialized = false
    end
    if length(ws.cumulative_energy_exchange_per_bath) != B
        ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, B)
    end
    if length(ws.last_velocity_scale_per_bath) != B
        ws.last_velocity_scale_per_bath = CUDA.fill(one(T), B)
    end

    if st.step == 0
        ws.kinetic_initialized = false
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Minimal deterministic (NVE-family) half-kick kernels.
#
# These deliberately do *not* maintain `Ekin`/`dU`: deterministic MD defines
# no per-particle stochastic heat/work channel, and the kinetic-energy buffer
# is refreshed lazily at sampling time (`_refresh_kinetic_buffer!`).
# Arithmetic stays in the state's native precision; consumer GPUs execute
# Float64 at a small fraction of Float32 throughput.
# ---------------------------------------------------------------------------

function _nve_kick2_kernel!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                            fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                            c::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = muladd(c, fx[i], vx[i])
        vy[i] = muladd(c, fy[i], vy[i])
    end
    return
end

function _nve_kick3_kernel!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                            fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                            c::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = muladd(c, fx[i], vx[i])
        vy[i] = muladd(c, fy[i], vy[i])
        vz[i] = muladd(c, fz[i], vz[i])
    end
    return
end









@inline _device_scalar(x::CuArray{T,1}) where {T<:AbstractFloat} = T(Array(x)[1])
include("simulation/Observables.jl")

end # module
