module Simulation

using CUDA
import ..IntegratorInterfaces
import ..IntegratorInterfaces: AbstractIntegratorSpec,
                               validate_integrator_inputs!,
                               ensure_integrator_workspace!,
                               integrator_id,
                               integrator_name,
                               stage_sequence,
                               execute_integrator_stage!,
                               collect_integrator_observables
using ..Definitions
using ..NeighborLists
using ..NonBondedForces
using ..BondedForces
using ..LangevinIntegrators
using ..BrownianIntegrators
using ..Collisions
 

const NL_CHECK_STRIDE = 20  # only check NL rebuild every N steps to cut overhead
# Weeks-Chandler-Andersen cutoff factor: r_c = 2^(1/6) * σ ≈ 1.12246 σ
const WCA_RC_FACTOR = 1.122462048309373

# Nonbonded kind tags (host-side routing only)
const NB_KIND_LJ      = UInt8(1)
const NB_KIND_WCA     = UInt8(2)
const NB_KIND_SOFTREP = UInt8(3)

# Freeze modes
const FREEZE_NONE   = UInt8(0)
const FREEZE_HOLD   = UInt8(1)
const FREEZE_SPRING = UInt8(2)

export SimulationState, build_simulation, step!, step_graph!, zero_forces!, sync_unwrapped!, accumulate_virial!, virial_components, virial_tensor
export run_integrator_step!, collect_step_observables, thermostatted_dof, thermostatted_particle_mask
export IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NHCParams, NHCSpec, CSVRParams, CSVRSpec
export velocityverlet, baoab, baoa, gsm, eulerheun, eulermaruyama, nosehooverchain, csvr

# =========================
#   Simulation state (SoA)
# =========================
"""
    SimulationState{T}

Structure-of-arrays storage for all GPU-resident buffers required to evolve a
non-equilibrium simulation. Users normally do not construct this type manually;
[`build_simulation`](@ref) assembles a fully initialized instance with concrete
CuArray fields and consistent dimensionality inferred from the `box` argument.

Key fields that user code may read or mutate:
- `rx, ry[, rz]`, `vx, vy[, vz]`, `fx, fy[, fz]`: positions, velocities, and
  force accumulators in GPU memory. Arrays are sized `N` and remain allocated
  for the entire simulation so that stepping is allocation-free.
- `rx_unwrap, ry_unwrap[, rz_unwrap]`: optional unwrapped positions that track
  continuous motion across periodic boundaries (enabled via `unwrapped_positions`).
- `nbh`: neighbor matrix (either dense cell list or sentinel all-pairs) built
  with [`NeighborLists.build_neighbors_*`](@ref) using the requested cutoff,
  skin, and capacity.
- `mass`, `dt`: universal particle mass and nominal build-time timestep used by
  deterministic integrators and by explicit stochastic spec constructors.
- `Epot`, `Ekin`, `virial`, `dq`, `dU` plus the corresponding `*_accum` buffers:
  energy, virial, and heat observables that can be sampled directly on the GPU.
- `virial_nonbonded`, `virial_bonded`, `virial_tensor`, `virial_tensor_accum`:
  GPU-resident configurational virial tensors. These store the raw pairwise
  virial `W = Σ r_ij ⊗ F_ij` using the same minimum-image displacement used in
  the force kernels. The scalar `virial` field stores the trace of `W` for
  backward compatibility with existing examples. These buffers are refreshed on
  force evaluations with `compute_energy=true`.
- `typeid`: per-particle type ids used by `Filters` and mixed-size LJ kernels.
- `freeze_*`: optional buffers used by the freeze/tether helpers in `Filters`.
- `coll_*`: optional buffers that appear when collision counting is enabled via
  `enable_collision_counting!`.

All other fields are internal implementation details (neighbor rebuild state,
bond lists, cached LJ parameters, etc.) and should be treated as read-only.
"""
mutable struct SimulationState{T<:AbstractFloat}
    # SoA arrays
    rx::CuArray{T,1}; ry::CuArray{T,1}
    rz::Union{Nothing,CuArray{T,1}}
    rx_unwrap::Union{Nothing,CuArray{T,1}}
    ry_unwrap::Union{Nothing,CuArray{T,1}}
    rz_unwrap::Union{Nothing,CuArray{T,1}}
    vx::CuArray{T,1}; vy::CuArray{T,1}
    vz::Union{Nothing,CuArray{T,1}}
    fx::CuArray{T,1}; fy::CuArray{T,1}
    fz::Union{Nothing,CuArray{T,1}}

    # previous forces (to avoid per-step allocations)
    f0x::CuArray{T,1}; f0y::CuArray{T,1}
    f0z::Union{Nothing,CuArray{T,1}}

    # per-particle type id
    typeid::CuArray{Int32,1}

    # box (stored directly; no splatting)
    box2::Union{Definitions.Box2{T},Nothing}
    box3::Union{Definitions.Box3{T},Nothing}

    # neighbor list
    nbh::NeighborLists.AbstractNeighborMatrix
    neigh_interval::Int

    # pair params (global LJ)
    pair_lj::Definitions.LJParams{T}
    # optional per-particle size (mixed interactions)
    sigma_particle::Union{Nothing,CuArray{T,1}}
    rcut_factor::T
    # optional per-type pair parameters
    sigma_pair::Union{Nothing,CuArray{T,2}}
    epsilon_pair::Union{Nothing,CuArray{T,2}}
    rcut_pair::Union{Nothing,CuArray{T,2}}
    
    # bonded interactions (optional)
    bonds::Union{Nothing,BondedForces.BondList}
    bonding::Union{Nothing,Definitions.BondPotential{T}}

    # universal integration metadata
    mass::T
    dt::T

    # observables buffers
    Epot::CuArray{T,1}
    dq::CuArray{T,1}
    dU::CuArray{T,1}
    Ekin::CuArray{T,1}
    virial::CuArray{T,1}
    virial_nonbonded::CuArray{T,2}
    virial_bonded::CuArray{T,2}
    virial_tensor::CuArray{T,2}
    # interval accumulators (GPU) to avoid host reductions each step
    Epot_accum::CuArray{T,1}
    Ekin_accum::CuArray{T,1}
    virial_accum::CuArray{T,1}
    virial_tensor_accum::CuArray{T,2}

    # misc
    step::Int
    # last integrator used: 1=Langevin, 2=Brownian, 3=NHC, 4=CSVR, 0=unknown
    last_integrator::UInt8
    nb_kind::UInt8
    softrep::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    # freeze controls (optional)
    freeze_mode::UInt8
    freeze_until::Int
    freeze_include_energy::Bool
    freeze_mask::Union{Nothing,CuArray{UInt8,1}}
    freeze_k::T
    freeze_rx::Union{Nothing,CuArray{T,1}}
    freeze_ry::Union{Nothing,CuArray{T,1}}
    freeze_rz::Union{Nothing,CuArray{T,1}}
    # Collision counting (optional)
    coll_enabled::Bool
    coll_prev::Union{Nothing,CuArray{UInt8,1}}
    coll_counts::Union{Nothing,CuArray{Int64,1}}
    coll_bins::Union{Nothing,CuArray{Int32,2}}
end

# -------------------------
# Unified integrator specs
# -------------------------
const IntegratorSpec = AbstractIntegratorSpec

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

@inline function _device_particle_buffer(::Type{T},
                                         N::Integer,
                                         value::Union{AbstractVector{<:Real},Real},
                                         name::AbstractString) where {T<:AbstractFloat}
    if value isa Real
        return CUDA.fill(T(value), N)
    end
    length(value) == N ||
        throw(ArgumentError("$(name) vector must have length $(N), got $(length(value))."))
    return CuArray(T.(value))
end

@inline function _device_corr_time_buffer(::Type{T},
                                          N::Integer,
                                          value::Union{Nothing,AbstractVector{<:Real},Real}) where {T<:AbstractFloat}
    value === nothing && return nothing
    return _device_particle_buffer(T, N, value, "noise_corr_time")
end

@inline function _all_particle_indices(N::Integer)
    return CuArray(Int32.(collect(1:N)))
end

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

@inline function _ou_coefficients(::Type{T},
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
    return CuArray(coeff_a), CuArray(coeff_c)
end

function _build_single_mode_ou(::Type{T},
                               noise_scale::CuArray{T,1},
                               corr::CuArray{T,1},
                               dt::Real) where {T<:AbstractFloat}
    corr_host = Array(corr)
    idx_host = findall(!iszero, corr_host)
    isempty(idx_host) && return nothing

    scale_host = Array(noise_scale)
    tau_mat = reshape(T.(corr_host[idx_host]), 1, :)
    scale_mat = reshape(T.(scale_host[idx_host]), 1, :)
    active_idx = CuArray(Int32.(idx_host))
    coeff_a, coeff_c = _ou_coefficients(T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx, CuArray(tau_mat), CuArray(scale_mat), coeff_a, coeff_c)
end

function _build_mode_ou(::Type{T},
                        active_idx::CuArray{Int32,1},
                        taus::Union{AbstractVector{<:Real},Real},
                        scales::Union{AbstractVector{<:Real},Real},
                        dt::Real) where {T<:AbstractFloat}
    K = length(active_idx)
    K == 0 && return nothing
    tau_vals, scale_vals = _canonical_mode_vectors(T, taus, scales)
    tau_mat = repeat(reshape(tau_vals, :, 1), 1, K)
    scale_mat = repeat(reshape(scale_vals, :, 1), 1, K)
    coeff_a, coeff_c = _ou_coefficients(T, T(dt), tau_mat, scale_mat)
    return Definitions.OUSpectrum{T}(T(dt), active_idx, CuArray(tau_mat), CuArray(scale_mat), coeff_a, coeff_c)
end

@inline function _compat_corr_time(::Type{T},
                                   N::Integer,
                                   taus::AbstractVector{T},
                                   scales::AbstractVector{T}) where {T<:AbstractFloat}
    length(taus) == 1 && length(scales) == 1 || return nothing
    return CUDA.fill(taus[1], N)
end

function _refresh_ou_coefficients!(ou::Definitions.OUSpectrum{T}, dt::T) where {T<:AbstractFloat}
    ou.dt == dt && return ou
    coeff_a, coeff_c = _ou_coefficients(T, dt, Array(ou.tau), Array(ou.scale))
    copyto!(ou.coeff_a, coeff_a)
    copyto!(ou.coeff_c, coeff_c)
    ou.dt = dt
    return ou
end

function _build_vv_params(st::SimulationState{T};
                          gamma::Union{AbstractVector{<:Real},Real},
                          temperature::Union{AbstractVector{<:Real},Real},
                          noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                          mass::Real=st.mass,
                          dt::Real=st.dt) where {T<:AbstractFloat}
    N = length(st.rx)
    γ = _device_particle_buffer(T, N, gamma, "gamma")
    Tbuf = _device_particle_buffer(T, N, temperature, "temperature")
    noise = CuArray(sqrt.(T(2) .* γ .* Tbuf .* T(dt)))
    corr = nothing
    ou = nothing
    if ou_scales !== nothing
        noise_corr_time === nothing &&
            throw(ArgumentError("`ou_scales` requires `noise_corr_time` to be provided as OU mode correlation times."))
        ou = _build_mode_ou(T, _all_particle_indices(N), noise_corr_time, ou_scales, dt)
        tau_vals, scale_vals = _canonical_mode_vectors(T, noise_corr_time, ou_scales)
        corr = _compat_corr_time(T, N, tau_vals, scale_vals)
    elseif noise_corr_time !== nothing
        if noise_corr_time isa AbstractVector && length(noise_corr_time) != N
            throw(ArgumentError("Vector `noise_corr_time` without `ou_scales` is interpreted as legacy per-particle single-mode OU and must have length $(N). Pass `ou_scales` as well for a multi-mode spectrum."))
        end
        corr = _device_corr_time_buffer(T, N, noise_corr_time)
        ou = _build_single_mode_ou(T, noise, corr, dt)
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

function _build_brownian_params(st::SimulationState{T};
                                gamma::Union{AbstractVector{<:Real},Real},
                                temperature::Union{AbstractVector{<:Real},Real},
                                noise_corr_time::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                                ou_scales::Union{Nothing,AbstractVector{<:Real},Real}=nothing,
                                dt::Real=st.dt) where {T<:AbstractFloat}
    N = length(st.rx)
    γ = _device_particle_buffer(T, N, gamma, "gamma")
    Tbuf = _device_particle_buffer(T, N, temperature, "temperature")
    noise = CuArray(sqrt.(T(2) .* γ .* Tbuf .* T(dt)))
    corr = nothing
    ou = nothing
    if ou_scales !== nothing
        noise_corr_time === nothing &&
            throw(ArgumentError("`ou_scales` requires `noise_corr_time` to be provided as OU mode correlation times."))
        ou = _build_mode_ou(T, _all_particle_indices(N), noise_corr_time, ou_scales, dt)
        tau_vals, scale_vals = _canonical_mode_vectors(T, noise_corr_time, ou_scales)
        corr = _compat_corr_time(T, N, tau_vals, scale_vals)
    elseif noise_corr_time !== nothing
        if noise_corr_time isa AbstractVector && length(noise_corr_time) != N
            throw(ArgumentError("Vector `noise_corr_time` without `ou_scales` is interpreted as legacy per-particle single-mode OU and must have length $(N). Pass `ou_scales` as well for a multi-mode spectrum."))
        end
        corr = _device_corr_time_buffer(T, N, noise_corr_time)
        ou = _build_single_mode_ou(T, noise, corr, dt)
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
Euler–Maruyama overdamped spec created by [`eulermaruyama`](@ref).
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

const NHC_PROPAGATOR_LEGACY  = UInt8(1)
const NHC_PROPAGATOR_GROMACS = UInt8(2)
const NHC_PROPAGATOR_LAMMPS  = UInt8(3)

@inline function _nhc_propagator_id(propagator::Symbol)
    if propagator === :legacy
        return NHC_PROPAGATOR_LEGACY
    elseif propagator === :gromacs
        return NHC_PROPAGATOR_GROMACS
    elseif propagator === :lammps
        return NHC_PROPAGATOR_LAMMPS
    end
    throw(ArgumentError("Unsupported NHC propagator $(propagator). Expected :legacy, :gromacs, or :lammps."))
end

@inline function _nhc_propagator_name(propagator::UInt8)
    if propagator == NHC_PROPAGATOR_LEGACY
        return :legacy
    elseif propagator == NHC_PROPAGATOR_GROMACS
        return :gromacs
    elseif propagator == NHC_PROPAGATOR_LAMMPS
        return :lammps
    end
    return :unknown
end

NHCParams{T}(mass::T,
             target_temperature::Vector{T},
             tau::Vector{T},
             substeps::Int,
             chain_length::Int,
             chain_masses::Matrix{T}) where {T<:AbstractFloat} =
    NHCParams{T}(mass,
                 target_temperature,
                 tau,
                 substeps,
                 chain_length,
                 chain_masses,
                 NHC_PROPAGATOR_GROMACS)

NHCParams(mass::T,
          target_temperature::Vector{T},
          tau::Vector{T},
          substeps::Int,
          chain_length::Int,
          chain_masses::Matrix{T}) where {T<:AbstractFloat} =
    NHCParams{T}(mass,
                 target_temperature,
                 tau,
                 substeps,
                 chain_length,
                 chain_masses)

@inline function _default_nhc_chain_masses(::Type{T},
                                           dof::Int,
                                           target_temperature::T,
                                           tau::T,
                                           chain_length::Int) where {T<:AbstractFloat}
    @assert chain_length >= 1
    base = target_temperature * tau * tau
    masses = Vector{T}(undef, chain_length)
    masses[1] = max(one(T), T(dof)) * base
    @inbounds for j in 2:chain_length
        masses[j] = base
    end
    return masses
end

@inline function _nhc_chain_masses_signature(masses::AbstractArray{T}) where {T<:AbstractFloat}
    sig = hash(size(masses))
    @inbounds for q in masses
        sig = hash(q, sig)
    end
    return sig
end

@inline function _new_nhc_workspace(::Type{T},
                                    chain_length::Int,
                                    nbaths::Int,
                                    nparticles::Int) where {T<:AbstractFloat}
    return NHCWorkspace{T}(CUDA.zeros(T, chain_length, nbaths),
                           CUDA.zeros(T, chain_length, nbaths),
                           CUDA.zeros(T, chain_length, nbaths),
                           CUDA.zeros(T, chain_length, nbaths),
                           CUDA.zeros(T, nbaths),
                           CUDA.fill(Int32(1), nparticles),
                           CUDA.zeros(Int32, nbaths),
                           CUDA.zeros(T, nbaths),
                           CUDA.zeros(T, nbaths),
                           CUDA.zeros(T, nbaths),
                           CUDA.zeros(T, nbaths),
                           CUDA.fill(one(T), nbaths),
                           UInt64(0),
                           true,
                           false)
end

@inline function _new_csvr_workspace(::Type{T},
                                     nbaths::Int,
                                     nparticles::Int) where {T<:AbstractFloat}
    return CSVRWorkspace{T}(CUDA.zeros(T, nbaths),
                            CUDA.zeros(T, nbaths),
                            CUDA.fill(Int32(1), nparticles),
                            CUDA.zeros(Int32, nbaths),
                            CUDA.zeros(T, nbaths),
                            CUDA.zeros(T, nbaths),
                            CUDA.zeros(T, nbaths),
                            CUDA.fill(one(T), nbaths),
                            true,
                            false)
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

Return an Euler–Maruyama spec for overdamped dynamics (`examples/3D_BD.jl`).
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

"""
    nosehooverchain(st; temperature=1, tau=1, chain_length=5, substeps=5,
                    mass=st.mass, chain_masses=nothing, propagator=:gromacs) -> NHCSpec

Create a deterministic NVT integrator specification using a Nose-Hoover Chain.
When `chain_masses` is omitted, masses are initialized from `(dof, T, tau)`
using the standard `Q₁ = g T τ²`, `Qⱼ = T τ² (j>1)` rule. `propagator`
selects the chain update scheme: `:gromacs` is the package default and uses
a GPU port of the reversible Suzuki-Yoshida propagator used by GROMACS,
`:lammps` uses a GPU port of the reversible `tloop` chain update used by
LAMMPS, and `:legacy` preserves the original package implementation. For
`propagator=:gromacs`, `substeps=5` matches the default fifth-order outer
repetition used there; for `propagator=:lammps`, `substeps` corresponds to
LAMMPS `tloop` and `substeps=1` matches the LAMMPS default.
"""
function nosehooverchain(st::SimulationState{T};
                         temperature::Union{Nothing,Real}=nothing,
                         tau::Union{Nothing,Real}=nothing,
                         temperatures::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         taus::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         chain_length::Integer=5,
                         substeps::Integer=5,
                         mass::Real=st.mass,
                         chain_masses::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}}=nothing,
                         propagator::Symbol=:gromacs) where {T<:AbstractFloat}
    chain_length >= 1 || throw(ArgumentError("chain_length must be >= 1, got $(chain_length)."))
    substeps >= 1 || throw(ArgumentError("substeps must be >= 1, got $(substeps)."))
    massT = T(mass)
    massT > zero(T) || throw(ArgumentError("NHC mass must be > 0."))
    propagator_id = _nhc_propagator_id(propagator)

    if temperatures !== nothing && temperature !== nothing
        throw(ArgumentError("Provide either `temperature` or `temperatures`, not both."))
    end
    if taus !== nothing && tau !== nothing
        throw(ArgumentError("Provide either `tau` or `taus`, not both."))
    end

    temp_vec = if temperatures === nothing
        [T(something(temperature, one(T)))]
    else
        T.(temperatures)
    end
    tau_vec = if taus === nothing
        [T(something(tau, one(T)))]
    else
        T.(taus)
    end

    length(temp_vec) == length(tau_vec) ||
        throw(ArgumentError("temperatures and taus must have identical lengths."))
    nbaths = length(temp_vec)
    nbaths >= 1 || throw(ArgumentError("NHC requires at least one bath."))

    @inbounds for (b, Tb) in pairs(temp_vec)
        Tb > zero(T) || throw(ArgumentError("NHC target temperature for bath $(b) must be > 0."))
    end
    @inbounds for (b, τb) in pairs(tau_vec)
        τb > zero(T) || throw(ArgumentError("NHC tau for bath $(b) must be > 0."))
    end

    dof_total = (_is_3d(st) ? 3 : 2) * length(st.rx)
    dof_guess = max(1, cld(dof_total, nbaths))

    masses = if chain_masses === nothing
        out = Matrix{T}(undef, Int(chain_length), nbaths)
        @inbounds for b in 1:nbaths
            col = _default_nhc_chain_masses(T, dof_guess, temp_vec[b], tau_vec[b], Int(chain_length))
            out[:, b] = col
        end
        out
    else
        if chain_masses isa AbstractVector
            length(chain_masses) == chain_length ||
                throw(ArgumentError("Vector chain_masses length must equal chain_length ($(chain_length))."))
            v = T.(chain_masses)
            repeat(reshape(v, Int(chain_length), 1), 1, nbaths)
        else
            cm = T.(chain_masses)
            size(cm, 1) == chain_length ||
                throw(ArgumentError("Matrix chain_masses first dimension must equal chain_length ($(chain_length))."))
            size(cm, 2) == nbaths ||
                throw(ArgumentError("Matrix chain_masses second dimension must equal number of baths ($(nbaths))."))
            cm
        end
    end
    @inbounds for j in axes(masses, 1), b in axes(masses, 2)
        masses[j, b] > zero(T) ||
            throw(ArgumentError("NHC chain mass Q[$(j), bath=$(b)] must be > 0."))
    end

    params = NHCParams{T}(massT,
                          temp_vec,
                          tau_vec,
                          Int(substeps),
                          Int(chain_length),
                          masses,
                          propagator_id)
    return NHCSpec{T}(params, _new_nhc_workspace(T, Int(chain_length), nbaths, length(st.rx)))
end

"""
    csvr(st; temperature=1, tau=1, mass=st.mass) -> CSVRSpec
    csvr(st; temperatures, taus, mass=st.mass) -> CSVRSpec

Create a deterministic MD integrator using the Bussi canonical-sampling through
velocity rescaling (CSVR) thermostat. The thermostat acts on one or more baths
defined by filter assignments, with one global velocity-rescaling factor drawn
per bath and per timestep.
"""
function csvr(st::SimulationState{T};
              temperature::Union{Nothing,Real}=nothing,
              tau::Union{Nothing,Real}=nothing,
              temperatures::Union{Nothing,AbstractVector{<:Real}}=nothing,
              taus::Union{Nothing,AbstractVector{<:Real}}=nothing,
              mass::Real=st.mass) where {T<:AbstractFloat}
    if temperatures !== nothing && temperature !== nothing
        throw(ArgumentError("Provide either `temperature` or `temperatures`, not both."))
    end
    if taus !== nothing && tau !== nothing
        throw(ArgumentError("Provide either `tau` or `taus`, not both."))
    end

    massT = T(mass)
    massT > zero(T) || throw(ArgumentError("CSVR mass must be > 0."))

    temp_vec = if temperatures === nothing
        [T(something(temperature, one(T)))]
    else
        T.(temperatures)
    end
    tau_vec = if taus === nothing
        [T(something(tau, one(T)))]
    else
        T.(taus)
    end

    length(temp_vec) == length(tau_vec) ||
        throw(ArgumentError("temperatures and taus must have identical lengths."))
    nbaths = length(temp_vec)
    nbaths >= 1 || throw(ArgumentError("CSVR requires at least one bath."))

    @inbounds for (b, Tb) in pairs(temp_vec)
        Tb > zero(T) || throw(ArgumentError("CSVR target temperature for bath $(b) must be > 0."))
    end
    @inbounds for (b, τb) in pairs(tau_vec)
        τb > zero(T) || throw(ArgumentError("CSVR tau for bath $(b) must be > 0."))
    end

    params = CSVRParams{T}(massT, temp_vec, tau_vec)
    return CSVRSpec{T}(params, _new_csvr_workspace(T, nbaths, length(st.rx)))
end
"""
    zero_forces!(st)

Fill `st.fx`, `st.fy`, (`st.fz` if present) with zeros. Called by the tests
before running isolated force kernels and safe to use between manual force
evaluations.
"""
function zero_forces!(st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(st.fx, zero(T)); fill!(st.fy, zero(T))
    st.fz === nothing || fill!(st.fz, zero(T))
    return nothing
end

"""
    sync_unwrapped!(st)

Copy the wrapped positions into the unwrapped buffers. Call this after manually
overwriting `st.rx`/`st.ry`/`st.rz` when `unwrapped_positions=true`.
"""
function sync_unwrapped!(st::SimulationState{T}) where {T<:AbstractFloat}
    st.rx_unwrap === nothing && return st
    copyto!(st.rx_unwrap, st.rx)
    copyto!(st.ry_unwrap, st.ry)
    if st.rz !== nothing && st.rz_unwrap !== nothing
        copyto!(st.rz_unwrap, st.rz)
    end
    return st
end

@inline function _require_positive_gamma!(gamma::CuArray{T,1}, integrator::AbstractString) where {T<:AbstractFloat}
    gmin = minimum(gamma)
    if !(gmin > zero(T))
        throw(ArgumentError("$(integrator) integrator requires gamma > 0 for all particles."))
    end
    return nothing
end

# -------------------------
# Bond helpers (2D / 3D)
# -------------------------
function _apply_bonds2!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1},
                        E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool,
                        V::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        V === nothing || fill!(V, zero(T))
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
            else
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, fx, fy, E, V, st.bonds, st.box2::Definitions.Box2{T}, p)
            end
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.fene_forces_soa!(st.rx, st.ry, fx, fy, E, st.bonds, st.box2::Definitions.Box2{T}, p)
            else
                BondedForces.fene_forces_soa!(st.rx, st.ry, fx, fy, E, V, st.bonds, st.box2::Definitions.Box2{T}, p)
            end
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, fx, fy, st.bonds, st.box2::Definitions.Box2{T}, p)
        end
    end
    return
end

function _apply_bonds3!(st::SimulationState{T}, fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool,
                        V::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    if (st.bonds === nothing) || (st.bonding === nothing)
        V === nothing || fill!(V, zero(T))
        return
    end
    if st.bonding isa Definitions.HarmonicBond{T}
        p = (st.bonding::Definitions.HarmonicBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
            else
                BondedForces.harmonic_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, V, st.bonds, st.box3::Definitions.Box3{T}, p)
            end
        else
            BondedForces.harmonic_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    elseif st.bonding isa Definitions.FENEBond{T}
        p = (st.bonding::Definitions.FENEBond{T}).params
        if compute_energy && E !== nothing
            if V === nothing
                BondedForces.fene_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, st.bonds, st.box3::Definitions.Box3{T}, p)
            else
                BondedForces.fene_forces_soa!(st.rx, st.ry, st.rz, fx, fy, fz, E, V, st.bonds, st.box3::Definitions.Box3{T}, p)
            end
        else
            BondedForces.fene_forces_soa_noE!(st.rx, st.ry, st.rz, fx, fy, fz, st.bonds, st.box3::Definitions.Box3{T}, p)
        end
    end
    return
end

function _compute_final_nonbonded2!(st::SimulationState{T}, compute_energy::Bool) where {T<:AbstractFloat}
    if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
        if compute_energy
            NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box2::Definitions.Box2,
                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        else
            NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        end
    elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
        if compute_energy
            NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box2::Definitions.Box2,
                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        else
            NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                     st.nbh, st.box2::Definitions.Box2,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        end
    elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
        if compute_energy
            NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                  st.nbh, st.box2::Definitions.Box2,
                                                  st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        else
            NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                      st.nbh, st.box2::Definitions.Box2,
                                                      st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        end
    elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
        if compute_energy
            NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                  st.nbh, st.box2::Definitions.Box2,
                                                  st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        else
            NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                      st.nbh, st.box2::Definitions.Box2,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        end
    elseif st.nb_kind == NB_KIND_LJ
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                               st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                    st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
            end
        else
            if st.bonds === nothing
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                        st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
            end
        end
    elseif st.nb_kind == NB_KIND_WCA
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                     st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
            end
        else
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                    st.nbh, st.box2::Definitions.Box2, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                         st.nbh, st.bonds, st.box2::Definitions.Box2, st.pair_lj)
            end
        end
    else
        @assert st.softrep !== nothing "softrep params missing"
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                         st.nbh, st.box2::Definitions.Box2, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                              st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
            end
        else
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                             st.nbh, st.box2::Definitions.Box2, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy,
                                                                  st.nbh, st.bonds, st.box2::Definitions.Box2, st.softrep)
            end
        end
    end
    return nothing
end

function _compute_final_nonbonded3!(st::SimulationState{T}, compute_energy::Bool) where {T<:AbstractFloat}
    if st.nb_kind == NB_KIND_LJ && st.sigma_pair !== nothing
        if compute_energy
            NonBondedForces.lj_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box3::Definitions.Box3,
                                                 st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        else
            NonBondedForces.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        end
    elseif st.nb_kind == NB_KIND_LJ && st.sigma_particle !== nothing
        if compute_energy
            NonBondedForces.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.box3::Definitions.Box3,
                                                 st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        else
            NonBondedForces.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                     st.nbh, st.box3::Definitions.Box3,
                                                     st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        end
    elseif st.nb_kind == NB_KIND_WCA && st.sigma_pair !== nothing
        if compute_energy
            NonBondedForces.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                  st.nbh, st.box3::Definitions.Box3,
                                                  st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        else
            NonBondedForces.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                      st.nbh, st.box3::Definitions.Box3,
                                                      st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
        end
    elseif st.nb_kind == NB_KIND_WCA && st.sigma_particle !== nothing
        if compute_energy
            NonBondedForces.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                  st.nbh, st.box3::Definitions.Box3,
                                                  st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        else
            NonBondedForces.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                      st.nbh, st.box3::Definitions.Box3,
                                                      st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
        end
    elseif st.nb_kind == NB_KIND_LJ
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                               st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                    st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
            end
        else
            if st.bonds === nothing
                NonBondedForces.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                   st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                        st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
            end
        end
    elseif st.nb_kind == NB_KIND_WCA
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                     st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
            end
        else
            if st.bonds === nothing
                NonBondedForces.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                    st.nbh, st.box3::Definitions.Box3, st.pair_lj)
            else
                NonBondedForces.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                         st.nbh, st.bonds, st.box3::Definitions.Box3, st.pair_lj)
            end
        end
    else
        @assert st.softrep !== nothing "softrep params missing"
        if compute_energy
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                         st.nbh, st.box3::Definitions.Box3, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                              st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
            end
        else
            if st.bonds === nothing
                NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                             st.nbh, st.box3::Definitions.Box3, st.softrep)
            else
                NonBondedForces.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                                  st.nbh, st.bonds, st.box3::Definitions.Box3, st.softrep)
            end
        end
    end
    return nothing
end

function _finalize_force_eval2!(st::SimulationState{T}, compute_energy::Bool, freeze_spring::Bool) where {T<:AbstractFloat}
    _apply_bonds2!(st, st.fx, st.fy, compute_energy ? st.Epot : nothing, compute_energy,
                   compute_energy ? st.virial_bonded : nothing)
    if freeze_spring
        _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy,
                              compute_energy ? st.Epot : nothing, compute_energy)
    end
    if compute_energy
        _combine_virial!(st)
    end
    return nothing
end

function _finalize_force_eval3!(st::SimulationState{T}, compute_energy::Bool, freeze_spring::Bool) where {T<:AbstractFloat}
    _apply_bonds3!(st, st.fx, st.fy, st.fz, compute_energy ? st.Epot : nothing, compute_energy,
                   compute_energy ? st.virial_bonded : nothing)
    if freeze_spring
        _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                              compute_energy ? st.Epot : nothing, compute_energy)
    end
    if compute_energy
        _combine_virial!(st)
    end
    return nothing
end

# -------------------------
# Freeze helpers
# -------------------------

@inline function _freeze_active!(st::SimulationState)
    if st.freeze_mode == FREEZE_NONE
        return false
    end
    if st.freeze_until >= 0 && st.step >= st.freeze_until
        st.freeze_mode = FREEZE_NONE
        st.freeze_until = -1
        return false
    end
    return true
end

function _freeze_hold2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
        end
    end
    return
end

function _freeze_hold3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
            rz[i] = az[i]
        end
    end
    return
end

function _freeze_hold2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold2_kernel!(rx, ry, mask, ax, ay)
    k(rx, ry, mask, ax, ay; threads, blocks)
    return nothing
end

function _freeze_hold3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold3_kernel!(rx, ry, rz, mask, ax, ay, az)
    k(rx, ry, rz, mask, ax, ay, az; threads, blocks)
    return nothing
end

function _freeze_spring2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
        end
    end
    return
end

function _freeze_spring2_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy)
        end
    end
    return
end

function _freeze_spring3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
        end
    end
    return
end

function _freeze_spring3_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy + dz * dz)
        end
    end
    return
end

function _freeze_spring2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_kernel!(rx, ry, fx, fy, mask, ax, ay, k)
    ker(rx, ry, fx, fy, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring2_energy!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_energy_kernel!(rx, ry, fx, fy, Epot, mask, ax, ay, k)
    ker(rx, ry, fx, fy, Epot, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_kernel!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _freeze_spring3_energy!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_energy_kernel!(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    return _freeze_hold2!(rx, ry, mask, ax, ay)
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    return _freeze_hold3!(rx, ry, rz, mask, ax, ay, az)
end

function _apply_freeze_hold_unwrap!(st::SimulationState{T}) where {T<:AbstractFloat}
    rxu = st.rx_unwrap
    ryu = st.ry_unwrap
    if rxu === nothing || ryu === nothing
        return nothing
    end
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    if st.rz_unwrap === nothing
        return _freeze_hold2!(rxu, ryu, mask, ax, ay)
    end
    az = st.freeze_rz
    az === nothing && return nothing
    return _freeze_hold3!(rxu, ryu, st.rz_unwrap, mask, ax, ay, az)
end

function _apply_freeze_hold_positions!(st::SimulationState{T}) where {T<:AbstractFloat}
    if st.rz === nothing
        _apply_freeze_hold!(st, st.rx, st.ry)
    else
        _apply_freeze_hold!(st, st.rx, st.ry, st.rz)
    end
    _apply_freeze_hold_unwrap!(st)
    return nothing
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring2_energy!(rx, ry, fx, fy, E, mask, ax, ay, k)
    end
    return _freeze_spring2!(rx, ry, fx, fy, mask, ax, ay, k)
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring3_energy!(rx, ry, rz, fx, fy, fz, E, mask, ax, ay, az, k)
    end
    return _freeze_spring3!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
end

# ==========================================
#  Top-level, non-capturing init kernels
#  (avoid nested functions / closures)
# ==========================================

function _init_vel2_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(T) * sqrt(temperature_vec[i])
        vy[i] = randn(T) * sqrt(temperature_vec[i])
    end
    return
end

function _init_vel3_kernel!(
    vx::CuDeviceVector{T},
    vy::CuDeviceVector{T},
    vz::CuDeviceVector{T},
    temperature_vec::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vx[i] = randn(T) * sqrt(temperature_vec[i])
        vy[i] = randn(T) * sqrt(temperature_vec[i])
        vz[i] = randn(T) * sqrt(temperature_vec[i])
    end
    return
end

# Host launchers
function _init_vel2!(vx::CuArray{T,1}, vy::CuArray{T,1}, temperature_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel2_kernel!(vx, vy, temperature_vec)
    CUDA.@sync k(vx, vy, temperature_vec; threads, blocks)
    return nothing
end

function _init_vel3!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}, temperature_vec::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _init_vel3_kernel!(vx, vy, vz, temperature_vec)
    CUDA.@sync k(vx, vy, vz, temperature_vec; threads, blocks)
    return nothing
end

# =========================
#   Build simulation
# =========================
"""
    build_simulation(; N, box, cutoff=1, skin=0.4, cap=Int32(96),
                      neigh_interval=20, use_neighborlist=true,
                      epsilon=1, sigma=1, gamma=1, temperature=1,
                      noise_corr_time=nothing, dt=0.001,
                      mass=1, bonds=nothing, bonding=nothing,
                      nonbonded=:lj, softrep_params=nothing,
                      precision=:f32, unwrapped_positions::Bool=false)

Construct a [`SimulationState`](@ref) with GPU-resident SoA arrays and a
neighbor list configured for the requested potential. All inputs are keyword
arguments so that scripts can copy known-good parameter sets directly from the
`examples/` directory without ambiguity.

Key behaviors:
- The simulation dimensionality (2D vs 3D) is inferred from the length of
  `box`. Positions/velocities/forces allocate the corresponding CuArrays.
- For `nonbonded = :wca` the neighbor cutoff is forced to the physical WCA
  value `r_c = 2^(1/6) σ` even if a larger `cutoff` was passed, guaranteeing
  that kernels reuse the validated parameter sets from the packaged examples.
- `temperature` accepts either a scalar or length-`N` vector and is used to
  initialize velocities. Stochastic integrator parameters are constructed
  later via explicit specs such as [`velocityverlet`](@ref) and
  [`eulermaruyama`](@ref).
- Initial velocities are drawn from a Maxwell–Boltzmann distribution at
  `temperature` and then centered to remove center-of-mass drift.
- When `unwrapped_positions=true`, additional `rx_unwrap`/`ry_unwrap`/`rz_unwrap`
  buffers track continuous positions across periodic boundaries.

Example (mirrors `examples/2D_example.jl`, scaled down to N=4096 for testing):

```julia
st = build_simulation(N=4096, box=(250f0, 250f0),
                      cutoff=Float32(2^(1/6)), skin=Float32(2^(1/6))/2,
                      cap=Int32(250), neigh_interval=50,
                      epsilon=1f4, sigma=1f0,
                      gamma=615f0, temperature=10f0,
                      dt=1f-5, nonbonded=:wca, precision=:f32)
vv = velocityverlet(st; gamma=615f0, temperature=10f0, dt=1f-5)
step!(st, vv, 1f-5; compute_energy=false)
```

Returns a fully initialized `SimulationState`. Stochastic controls such as
`noise_corr_time` now belong to the explicit integrator spec rather than the
core simulation state.
"""
function build_simulation(;N::Int,
                           box,
                           cutoff::Real=1.0,
                           skin::Real=0.4,
                           cap::Int32=Int32(96),
                           neigh_interval::Int=20,
                           use_neighborlist::Bool=true,
                           epsilon::Real=1,
                           sigma::Real=1,
                           gamma::Union{AbstractVector{<:Real},Real,Nothing}=1,
                           temperature::Union{AbstractVector{<:Real},Real}=1,
                           noise_corr_time::Union{AbstractVector{<:Real},Real,Nothing}=nothing,
                           dt::Real=0.001,
                           mass::Real=1,
                           bonds::Union{Nothing,Vector{Tuple{Int32,Int32}}}=nothing,
                           bonding::Union{Nothing,Definitions.BondPotential}=nothing,
                           nonbonded::Symbol = :lj,
                           softrep_params::Union{Nothing,Definitions.SoftRepulsiveParams{<:Real}}=nothing,
                           precision::Symbol = :f32,
                           unwrapped_positions::Bool = false)

    # Dimension from box
    D = length(box)

    if precision == :f32
        T = Float32
    elseif precision == :f64
        T = Float64
    else
        error("Unknown precision=$(precision). Use :f32 or :f64")
    end

    epsilonT = T(epsilon)
    sigmaT   = T(sigma)
    requested_cutoff = T(cutoff)
    nb_cutoff = (nonbonded === :wca) ? (sigmaT * T(WCA_RC_FACTOR)) : requested_cutoff
    rcut_factor = sigmaT == zero(T) ? T(1) : nb_cutoff / sigmaT

    # Allocate SoA buffers
    rx = CUDA.CuArray{T}(undef, N); ry = CUDA.CuArray{T}(undef, N)
    vx = CUDA.CuArray{T}(undef, N); vy = CUDA.CuArray{T}(undef, N)
    fx = CUDA.CuArray{T}(undef, N); fy = CUDA.CuArray{T}(undef, N)
    rz = nothing; vz = nothing; fz = nothing
    rx_unwrap = unwrapped_positions ? CUDA.CuArray{T}(undef, N) : nothing
    ry_unwrap = unwrapped_positions ? CUDA.CuArray{T}(undef, N) : nothing
    rz_unwrap = nothing

    # previous forces
    f0x = CUDA.CuArray{T}(undef, N)
    f0y = CUDA.CuArray{T}(undef, N)
    f0z = nothing

    if D == 3
        rz  = CUDA.CuArray{T}(undef, N)
        vz  = CUDA.CuArray{T}(undef, N)
        fz  = CUDA.CuArray{T}(undef, N)
        f0z = CUDA.CuArray{T}(undef, N)
        if unwrapped_positions
            rz_unwrap = CUDA.CuArray{T}(undef, N)
        end
    end

    fill!(rx, zero(T)); fill!(ry, zero(T))
    rz === nothing || fill!(rz, zero(T))
    rx_unwrap === nothing || fill!(rx_unwrap, zero(T))
    ry_unwrap === nothing || fill!(ry_unwrap, zero(T))
    rz_unwrap === nothing || fill!(rz_unwrap, zero(T))

    fill!(fx, zero(T)); fill!(fy, zero(T)); fz === nothing || fill!(fz, zero(T))
    fill!(f0x, zero(T)); fill!(f0y, zero(T)); f0z === nothing || fill!(f0z, zero(T))

    # Maxwell-Boltzmann initial velocities on GPU
    temperature_vec = _device_particle_buffer(T, N, temperature, "temperature")

    if D == 2
        _init_vel2!(vx, vy, temperature_vec)
    else
        _init_vel3!(vx, vy, vz, temperature_vec)
    end

    # Remove center-of-mass drift from the initial velocities
    if N > 0
        Vx = CUDA.sum(vx) / T(N)
        Vy = CUDA.sum(vy) / T(N)
        @. vx = vx - Vx
        @. vy = vy - Vy
        if D == 3
            Vz = CUDA.sum(vz) / T(N)
            @. vz = vz - Vz
        end
    end

    typeid = CUDA.fill(Int32(1), N)

    # Neighbors (dense cell-list or sentinel all-pairs)
    if use_neighborlist
        if D == 2
            nbh = NeighborLists.build_neighbors_dense!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_dense!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    else
        if D == 2
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry; box=(T(box[1]), T(box[2])), cutoff=nb_cutoff, cap, skin=T(skin))
        else
            nbh = NeighborLists.build_neighbors_allpairs!(rx, ry, rz; box=(T(box[1]), T(box[2]), T(box[3])), cutoff=nb_cutoff, cap, skin=T(skin))
        end
    end

    lj = Definitions.LJParams{T}(epsilonT, sigmaT, nb_cutoff)
    noise_corr_time === nothing ||
        throw(ArgumentError("build_simulation no longer accepts `noise_corr_time`; pass it to an explicit integrator constructor such as velocityverlet(st; gamma=..., temperature=..., noise_corr_time=..., dt=...)."))

    Epot = CUDA.CuArray{T}(undef, N); fill!(Epot, zero(T))
    dq   = CUDA.CuArray{T}(undef, N); fill!(dq, zero(T))
    dU   = CUDA.CuArray{T}(undef, N); fill!(dU, zero(T))
    Ekin = CUDA.CuArray{T}(undef, N); fill!(Ekin, zero(T))
    virial = CUDA.CuArray{T}(undef, N); fill!(virial, zero(T))
    nvirial = D == 2 ? 3 : 6
    virial_nonbonded = CUDA.CuArray{T}(undef, N, nvirial); fill!(virial_nonbonded, zero(T))
    virial_bonded = CUDA.CuArray{T}(undef, N, nvirial); fill!(virial_bonded, zero(T))
    virial_tensor = CUDA.CuArray{T}(undef, N, nvirial); fill!(virial_tensor, zero(T))
    # interval accumulators (GPU)
    Epot_accum = CUDA.CuArray{T}(undef, N); fill!(Epot_accum, zero(T))
    Ekin_accum = CUDA.CuArray{T}(undef, N); fill!(Ekin_accum, zero(T))
    virial_accum = CUDA.CuArray{T}(undef, N); fill!(virial_accum, zero(T))
    virial_tensor_accum = CUDA.CuArray{T}(undef, N, nvirial); fill!(virial_tensor_accum, zero(T))

    # Build bonds (if provided)
    local bondlist
    if bonds === nothing
        bondlist = nothing
    else
        bondlist = BondedForces.build_bondlist(N, bonds)
    end

    # Determine nonbonded kind and soft-rep params
    local nb_tag::UInt8
    local srp::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    if nonbonded === :lj
        nb_tag = NB_KIND_LJ
        srp = nothing
    elseif nonbonded === :wca
        nb_tag = NB_KIND_WCA
        srp = nothing
    elseif nonbonded === :soft_repulsive || nonbonded === :softrep || nonbonded === :soft
        nb_tag = NB_KIND_SOFTREP
        srp = softrep_params === nothing ? Definitions.SoftRepulsiveParams{T}(epsilonT, sigmaT) : softrep_params
    else
        error("Unknown nonbonded=:$(nonbonded). Use :lj, :wca, or :soft_repulsive")
    end

    # Resolve bonded potential from provided options (new unified or legacy)
    local bond_spec
    if bonding !== nothing
        # if provided, attempt to cast inner params to T
        if bonding isa Definitions.HarmonicBond
            p = (bonding::Definitions.HarmonicBond).params
            bond_spec = Definitions.HarmonicBond{T}(Definitions.HarmonicBondParams{T}(T(p.k), T(p.r0)))
        elseif bonding isa Definitions.FENEBond
            p = (bonding::Definitions.FENEBond).params
            bond_spec = Definitions.FENEBond{T}(Definitions.FENEParams{T}(T(p.k), T(p.R0)))
        else
            bond_spec = nothing
        end
    else
        bond_spec = nothing
    end

    # Construct with boxes set to nothing; assign after
    st = SimulationState(rx, ry, rz, rx_unwrap, ry_unwrap, rz_unwrap, vx, vy, vz, fx, fy, fz,
                         f0x, f0y, f0z,
                         typeid,
                         nothing,   # box2
                         nothing,   # box3
                         nbh, neigh_interval, lj,
                         nothing, rcut_factor,
                         nothing, nothing, nothing,
                         bondlist, bond_spec,
                         T(mass), T(dt),
                         Epot, dq, dU, Ekin, virial, virial_nonbonded, virial_bonded, virial_tensor,
                         Epot_accum, Ekin_accum, virial_accum, virial_tensor_accum,
                         0, UInt8(0), nb_tag, srp,
                         FREEZE_NONE, -1, true, nothing, zero(T), nothing, nothing, nothing,
                         false, nothing, nothing, nothing) 

    # Assign the appropriate box directly (no extra tuple layer)
    if D == 2
        st.box2 = (T(box[1]), T(box[2]))
        st.box3 = nothing
    else
        st.box2 = nothing
        st.box3 = (T(box[1]), T(box[2]), T(box[3]))
    end

    return st
end

# =========================
#   Virial (GPU)
# =========================
@inline _virial_ncomponents(st::SimulationState) = st.rz === nothing ? 3 : 6

virial_components(st::SimulationState) =
    st.rz === nothing ? (:xx, :yy, :xy) : (:xx, :yy, :zz, :xy, :xz, :yz)

@inline function _virial_buffer(st::SimulationState, part::Symbol, accumulated::Bool)
    if accumulated
        part === :total || throw(ArgumentError("only the total virial tensor has an accumulated buffer"))
        return st.virial_tensor_accum
    end
    if part === :total
        return st.virial_tensor
    elseif part === :nonbonded
        return st.virial_nonbonded
    elseif part === :bonded
        return st.virial_bonded
    end
    throw(ArgumentError("unknown virial part $(part); use :total, :nonbonded, or :bonded"))
end

"""
    virial_tensor(st; part=:total, accumulated=false)

Return the particle-summed configurational virial tensor components as a
`NamedTuple`. The stored sign convention is the raw configurational virial
`W = Σ r_ij ⊗ F_ij`, so pressure/stress formulas may need an additional minus
sign depending on convention.

Component ordering follows:
- 2D: `(:xx, :yy, :xy)`
- 3D: `(:xx, :yy, :zz, :xy, :xz, :yz)`

The primary buffers remain GPU-resident in `SimulationState`; this helper
reduces the selected tensor to a small host-side summary for diagnostics.
Virial buffers are refreshed by force evaluations with `compute_energy=true`.
"""
function virial_tensor(st::SimulationState{T}; part::Symbol=:total, accumulated::Bool=false) where {T<:AbstractFloat}
    comps = vec(Array(CUDA.sum(_virial_buffer(st, part, accumulated); dims=1)))
    if st.rz === nothing
        return (xx=comps[1], yy=comps[2], xy=comps[3])
    end
    return (xx=comps[1], yy=comps[2], zz=comps[3], xy=comps[4], xz=comps[5], yz=comps[6])
end

function _combine_virial2_kernel!(virial::CuDeviceVector{T},
                                  total::CuDeviceMatrix{T},
                                  nonbonded::CuDeviceMatrix{T},
                                  bonded::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        vxx = nonbonded[i, 1] + bonded[i, 1]
        vyy = nonbonded[i, 2] + bonded[i, 2]
        vxy = nonbonded[i, 3] + bonded[i, 3]
        total[i, 1] = vxx
        total[i, 2] = vyy
        total[i, 3] = vxy
        virial[i] = vxx + vyy
    end
    return
end

function _combine_virial3_kernel!(virial::CuDeviceVector{T},
                                  total::CuDeviceMatrix{T},
                                  nonbonded::CuDeviceMatrix{T},
                                  bonded::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        vxx = nonbonded[i, 1] + bonded[i, 1]
        vyy = nonbonded[i, 2] + bonded[i, 2]
        vzz = nonbonded[i, 3] + bonded[i, 3]
        vxy = nonbonded[i, 4] + bonded[i, 4]
        vxz = nonbonded[i, 5] + bonded[i, 5]
        vyz = nonbonded[i, 6] + bonded[i, 6]
        total[i, 1] = vxx
        total[i, 2] = vyy
        total[i, 3] = vzz
        total[i, 4] = vxy
        total[i, 5] = vxz
        total[i, 6] = vyz
        virial[i] = vxx + vyy + vzz
    end
    return
end

function _combine_virial!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.virial)
    threads = min(256, N)
    blocks = cld(N, threads)
    if st.rz === nothing
        k = CUDA.@cuda launch=false _combine_virial2_kernel!(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded)
        k(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded; threads, blocks)
    else
        k = CUDA.@cuda launch=false _combine_virial3_kernel!(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded)
        k(st.virial, st.virial_tensor, st.virial_nonbonded, st.virial_bonded; threads, blocks)
    end
    return nothing
end

# =========================
#   Energy accumulation (GPU)
# =========================
function _accumulate_energies!(Ekin_accum, Epot_accum, Ekin, Epot)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(Ekin); if i > N; return; end
    @inbounds begin
        Ekin_accum[i] += Ekin[i]
        Epot_accum[i] += Epot[i]
    end
    return
end

"""
    accumulate_energies!(st)

Add the instantaneous `Ekin`/`Epot` buffers into their per-interval accumulators.
Called once per logging interval in `examples/TwoT_2D_LD_VV.jl` before computing
entropy production.
"""
function accumulate_energies!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.Ekin)
    threads = min(256, N)
    blocks  = cld(N, threads)
    k = CUDA.@cuda launch=false _accumulate_energies!(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot)
    k(st.Ekin_accum, st.Epot_accum, st.Ekin, st.Epot; threads, blocks)
    return nothing
end

function _accumulate_virial2!(virial_accum::CuDeviceVector{T},
                              virial::CuDeviceVector{T},
                              virial_tensor_accum::CuDeviceMatrix{T},
                              virial_tensor::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        virial_accum[i] += virial[i]
        virial_tensor_accum[i, 1] += virial_tensor[i, 1]
        virial_tensor_accum[i, 2] += virial_tensor[i, 2]
        virial_tensor_accum[i, 3] += virial_tensor[i, 3]
    end
    return
end

function _accumulate_virial3!(virial_accum::CuDeviceVector{T},
                              virial::CuDeviceVector{T},
                              virial_tensor_accum::CuDeviceMatrix{T},
                              virial_tensor::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(virial); if i > N; return; end
    @inbounds begin
        virial_accum[i] += virial[i]
        virial_tensor_accum[i, 1] += virial_tensor[i, 1]
        virial_tensor_accum[i, 2] += virial_tensor[i, 2]
        virial_tensor_accum[i, 3] += virial_tensor[i, 3]
        virial_tensor_accum[i, 4] += virial_tensor[i, 4]
        virial_tensor_accum[i, 5] += virial_tensor[i, 5]
        virial_tensor_accum[i, 6] += virial_tensor[i, 6]
    end
    return
end

"""
    accumulate_virial!(st)

Add the instantaneous virial trace and total virial tensor into their
per-interval accumulators. `st.virial` remains the scalar trace of the
configurational virial tensor for backward compatibility.
"""
function accumulate_virial!(st::SimulationState{T}) where {T<:AbstractFloat}
    N = length(st.virial)
    threads = min(256, N)
    blocks  = cld(N, threads)
    if st.rz === nothing
        k = CUDA.@cuda launch=false _accumulate_virial2!(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor)
        k(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor; threads, blocks)
    else
        k = CUDA.@cuda launch=false _accumulate_virial3!(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor)
        k(st.virial_accum, st.virial, st.virial_tensor_accum, st.virial_tensor; threads, blocks)
    end
    return nothing
end

# =========================
#   Stage-driven stepping
# =========================

const INTEGRATOR_ID_UNKNOWN  = UInt8(0)
const INTEGRATOR_ID_LANGEVIN = UInt8(1)
const INTEGRATOR_ID_BROWNIAN = UInt8(2)
const INTEGRATOR_ID_NHC      = UInt8(3)
const INTEGRATOR_ID_CSVR     = UInt8(4)

"""
    _empty_workspace(T)

Construct a zero-length stochastic workspace used by compatibility constructors
that receive parameter objects without a simulation state.
"""
function _empty_workspace(::Type{T}) where {T<:AbstractFloat}
    return StochasticWorkspace{T}(CUDA.zeros(T, 0), CUDA.zeros(T, 0), nothing, nothing, nothing, nothing)
end

VVSpec(params::LangevinIntegrators.VVParams{T}) where {T<:AbstractFloat} = VVSpec{T}(params, _empty_workspace(T))
BAOABSpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = BAOABSpec{T}(params, _empty_workspace(T))
BAOASpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = BAOASpec{T}(params, _empty_workspace(T))
GSMSpec(params::LangevinIntegrators.BAOABParams{T}) where {T<:AbstractFloat} = GSMSpec{T}(params, _empty_workspace(T))
BrownianSpec(params::BrownianIntegrators.BrownianParams{T}) where {T<:AbstractFloat} = BrownianSpec{T}(params, _empty_workspace(T))
EMSpec(params::BrownianIntegrators.EMParams{T}) where {T<:AbstractFloat} = EMSpec{T}(params, _empty_workspace(T))
NHCSpec(params::NHCParams{T}) where {T<:AbstractFloat} =
    NHCSpec{T}(params,
               _new_nhc_workspace(T,
                                  params.chain_length,
                                  length(params.target_temperature),
                                  0))
CSVRSpec(params::CSVRParams{T}) where {T<:AbstractFloat} =
    CSVRSpec{T}(params,
                _new_csvr_workspace(T,
                                    length(params.target_temperature),
                                    0))

@inline _is_3d(st::SimulationState) = st.rz !== nothing

@inline function _stage_tracing_enabled()
    flag = lowercase(get(ENV, "NEQSIMGPU_STAGE_TRACE", "0"))
    return flag == "1" || flag == "true" || flag == "yes" || flag == "on"
end

"""
    _trace_integrator_stage!(st, spec, stage_tag; force_evaluated=false,
                             rebuild_applied=false)

Emit an optional per-stage trace line for debugging and diagnostics. Tracing is
controlled by `ENV["NEQSIMGPU_STAGE_TRACE"]` and is disabled by default.
"""
function _trace_integrator_stage!(st::SimulationState,
                                  spec::IntegratorSpec,
                                  stage_tag::Symbol;
                                  force_evaluated::Bool=false,
                                  rebuild_applied::Bool=false)
    _stage_tracing_enabled() || return nothing
    @info "integrator stage" integrator=integrator_name(spec) stage=stage_tag step=st.step force_evaluated rebuild_applied
    return nothing
end

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
        end
        if workspace.ou_y === nothing || size(workspace.ou_y) != (M, K)
            workspace.ou_y = CUDA.zeros(T, M, K)
        end
        if _is_3d(st)
            if workspace.ou_z === nothing || size(workspace.ou_z) != (M, K)
                workspace.ou_z = CUDA.zeros(T, M, K)
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

"""
    _swap_force_slots!(st)

Swap the active force slot (`f`) and reference force slot (`f0`).
"""
function _swap_force_slots!(st::SimulationState)
    st.f0x, st.fx = st.fx, st.f0x
    st.f0y, st.fy = st.fy, st.f0y
    if _is_3d(st)
        st.f0z, st.fz = st.fz, st.f0z
    end
    return nothing
end

"""
    _swap_midpoint_position_slots!(st)

Swap physical coordinates (`r`) with midpoint scratch coordinates (`v`) used by
Brownian midpoint-style integrators.
"""
function _swap_midpoint_position_slots!(st::SimulationState)
    st.rx, st.vx = st.vx, st.rx
    st.ry, st.vy = st.vy, st.ry
    if _is_3d(st)
        st.rz, st.vz = st.vz, st.rz
    end
    return nothing
end

"""
    evaluate_forces_into_f!(st, compute_energy; freeze_spring=false)

Evaluate nonbonded + bonded + optional freeze-spring contributions into the
active force slot (`f`).
"""
function evaluate_forces_into_f!(st::SimulationState{T},
                                 compute_energy::Bool;
                                 freeze_spring::Bool=false) where {T<:AbstractFloat}
    if _is_3d(st)
        _compute_final_nonbonded3!(st, compute_energy)
        _finalize_force_eval3!(st, compute_energy, freeze_spring)
    else
        _compute_final_nonbonded2!(st, compute_energy)
        _finalize_force_eval2!(st, compute_energy, freeze_spring)
    end
    return nothing
end

"""
    evaluate_forces_into_f0!(st, compute_energy; freeze_spring=false)

Evaluate forces into the reference force slot (`f0`) without changing the
external slot ownership.
"""
function evaluate_forces_into_f0!(st::SimulationState{T},
                                  compute_energy::Bool;
                                  freeze_spring::Bool=false) where {T<:AbstractFloat}
    _swap_force_slots!(st)
    try
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    finally
        _swap_force_slots!(st)
    end
    return nothing
end

"""
    evaluate_midpoint_forces_into_f0!(st; freeze_spring=false)

Evaluate forces at midpoint coordinates (stored in `vx,vy[,vz]`) into the
reference force slot (`f0`). This helper is used by Brownian midpoint / EM.
"""
function evaluate_midpoint_forces_into_f0!(st::SimulationState{T};
                                           freeze_spring::Bool=false) where {T<:AbstractFloat}
    _swap_midpoint_position_slots!(st)
    _swap_force_slots!(st)
    try
        evaluate_forces_into_f!(st, false; freeze_spring=freeze_spring)
    finally
        _swap_force_slots!(st)
        _swap_midpoint_position_slots!(st)
    end
    return nothing
end

"""
    plan_neighbor_rebuild!(st, dt) -> Bool

Determine whether neighbor lists should be rebuilt on this step.
"""
function plan_neighbor_rebuild!(st::SimulationState{T}, dt::T) where {T<:AbstractFloat}
    do_check = (st.step % st.neigh_interval == 0)
    do_check || return false
    if _is_3d(st)
        return NeighborLists.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                                            skin=st.nbh.skin,
                                            Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                                            step=st.step)
    end
    return NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                        skin=st.nbh.skin,
                                        Lx=st.box2[1], Ly=st.box2[2],
                                        step=st.step)
end

"""
    apply_neighbor_rebuild_if_needed!(st, rebuild_needed)

Apply neighbor-list rebuild and integrator-independent post-rebuild hooks.
"""
function apply_neighbor_rebuild_if_needed!(st::SimulationState,
                                           rebuild_needed::Bool)
    rebuild_needed || return nothing
    if _is_3d(st)
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    else
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
    end
    _collisions_reinit_on_rebuild!(st)
    return nothing
end

"""
    prepare_previous_force_buffers!(st, spec)

Prepare any force-buffer reuse needed before stage execution.
"""
function prepare_previous_force_buffers!(st::SimulationState,
                                         spec::IntegratorSpec)
    return nothing
end

function prepare_previous_force_buffers!(st::SimulationState,
                                         spec::Union{VVSpec,BAOABSpec,BAOASpec,GSMSpec,NHCSpec,CSVRSpec})
    if st.step > 1
        _swap_force_slots!(st)
    end
    return nothing
end

"""
    ensure_reference_forces_ready!(st, spec, compute_energy, freeze_spring)

Ensure the required reference force buffers for the selected integrator are
initialized before stage execution.
"""
function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::IntegratorSpec,
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    return nothing
end

function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::Union{VVSpec,BAOABSpec,BAOASpec,GSMSpec,NHCSpec,CSVRSpec},
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    if st.step <= 1
        evaluate_forces_into_f0!(st, compute_energy; freeze_spring=freeze_spring)
    end
    return nothing
end

function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::Union{BrownianSpec,EMSpec},
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    if st.step == 0
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    end
    return nothing
end

"""
    apply_post_position_hooks!(st, stage_tag; freeze_hold=false)

Apply integrator-independent post-position hooks after a position-changing
stage.
"""
function apply_post_position_hooks!(st::SimulationState,
                                    stage_tag::Symbol;
                                    freeze_hold::Bool=false)
    if freeze_hold
        _apply_freeze_hold_positions!(st)
    end
    _collisions_update_after_positions!(st)
    return nothing
end

"""
    finalize_step_accounting!(st, spec, compute_energy)

Perform integrator-independent end-of-step bookkeeping.
"""
function finalize_step_accounting!(st::SimulationState,
                                   spec::IntegratorSpec,
                                   compute_energy::Bool)
    return nothing
end

function finalize_step_accounting!(st::SimulationState{T},
                                   spec::Union{NHCSpec{T},CSVRSpec{T}},
                                   compute_energy::Bool) where {T<:AbstractFloat}
    # Deterministic thermostatted MD paths do not define per-particle
    # stochastic heat/work channels.
    fill!(st.dq, zero(T))
    fill!(st.dU, zero(T))
    return nothing
end

"""
    finalize_step_counter!(st, integrator_id)

Finalize step counters shared across integrators.
"""
function finalize_step_counter!(st::SimulationState,
                                id::UInt8)
    st.last_integrator = id
    st.step += 1
    return nothing
end

"""
    _prepare_langevin_noise!(spec, st, dt)

Draw Langevin stochastic impulses into the spec workspace.
"""
function _prepare_langevin_noise!(params::Union{LangevinIntegrators.VVParams{T},LangevinIntegrators.BAOABParams{T}},
                                  workspace::StochasticWorkspace{T},
                                  st::SimulationState{T},
                                  dt::T) where {T<:AbstractFloat}
    params.ou !== nothing && _refresh_ou_coefficients!(params.ou, dt)
    if _is_3d(st)
        LangevinIntegrators.vv_prepare_noise!(workspace.rf_x, workspace.rf_y, params.noise_scale;
                                              beta_z=workspace.rf_z,
                                              ou=params.ou,
                                              state_x=workspace.ou_x,
                                              state_y=workspace.ou_y,
                                              state_z=workspace.ou_z)
    else
        LangevinIntegrators.vv_prepare_noise!(workspace.rf_x, workspace.rf_y, params.noise_scale;
                                              beta_z=nothing,
                                              ou=params.ou,
                                              state_x=workspace.ou_x,
                                              state_y=workspace.ou_y,
                                              state_z=nothing)
    end
    return nothing
end

"""
    _prepare_brownian_noise!(params, workspace, st, dt)

Draw Brownian noise into the spec workspace.
"""
function _prepare_brownian_noise!(params::Union{BrownianIntegrators.BrownianParams{T},BrownianIntegrators.EMParams{T}},
                                  workspace::StochasticWorkspace{T},
                                  st::SimulationState{T},
                                  dt::T) where {T<:AbstractFloat}
    params.ou !== nothing && _refresh_ou_coefficients!(params.ou, dt)
    if _is_3d(st)
        BrownianIntegrators.bd_prepare_noise_3d!(workspace.rf_x, workspace.rf_y, workspace.rf_z;
                                                 noise_scale=params.noise_scale,
                                                 ou=params.ou,
                                                 state_x=workspace.ou_x,
                                                 state_y=workspace.ou_y,
                                                 state_z=workspace.ou_z)
    else
        BrownianIntegrators.bd_prepare_noise_2d!(workspace.rf_x, workspace.rf_y;
                                                 noise_scale=params.noise_scale,
                                                 ou=params.ou,
                                                 state_x=workspace.ou_x,
                                                 state_y=workspace.ou_y)
    end
    return nothing
end

"""
    _refresh_kinetic_buffer!(st, mass)

Refresh the per-particle kinetic-energy buffer `st.Ekin` from the current
velocity field.
"""
function _refresh_kinetic_buffer!(st::SimulationState{T}, mass::T) where {T<:AbstractFloat}
    if _is_3d(st)
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy + st.vz * st.vz)
    else
        @. st.Ekin = T(0.5) * mass * (st.vx * st.vx + st.vy * st.vy)
    end
    return nothing
end

"""
    _refresh_kinetic_energy!(st, mass) -> T

Compatibility helper that refreshes `st.Ekin` and returns its total on host.
"""
function _refresh_kinetic_energy!(st::SimulationState{T}, mass::T) where {T<:AbstractFloat}
    _refresh_kinetic_buffer!(st, mass)
    return T(CUDA.sum(st.Ekin))
end

function _sum_into_scalar_kernel!(out::CuDeviceVector{T},
                                  src::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(src)
        @inbounds CUDA.@atomic out[1] += src[i]
    end
    return nothing
end

"""
    _reduce_sum_to_device!(out, src)

Reduce `src` into the one-element device buffer `out` without host transfers.
"""
function _reduce_sum_to_device!(out::CuArray{T,1},
                                src::CuArray{T,1}) where {T<:AbstractFloat}
    fill!(out, zero(T))
    N = length(src)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _sum_into_scalar_kernel!(out, src)
    k(out, src; threads, blocks)
    return nothing
end

function _nhc_active_bath_counts_kernel!(bath_counts::CuDeviceVector{Int32},
                                         particle_bath_id::CuDeviceVector{Int32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(bath_counts)
                CUDA.@atomic bath_counts[b] += Int32(1)
            end
        end
    end
    return nothing
end

function _nhc_active_bath_counts_with_freeze_kernel!(bath_counts::CuDeviceVector{Int32},
                                                     particle_bath_id::CuDeviceVector{Int32},
                                                     freeze_mask::CuDeviceVector{UInt8})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            if freeze_mask[i] == UInt8(0)
                b = Int(particle_bath_id[i])
                if 1 <= b <= length(bath_counts)
                    CUDA.@atomic bath_counts[b] += Int32(1)
                end
            end
        end
    end
    return nothing
end

function _nhc_finalize_dof_kernel!(dof_per_bath::CuDeviceVector{T},
                                   bath_counts::CuDeviceVector{Int32},
                                   dim::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if b <= length(dof_per_bath)
        @inbounds dof_per_bath[b] = dim * T(bath_counts[b])
    end
    return nothing
end

function _update_bath_dof!(bath_counts::CuArray{Int32,1},
                           dof_per_bath::CuArray{T,1},
                           particle_bath_id::CuArray{Int32,1},
                           st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(bath_counts, Int32(0))
    fill!(dof_per_bath, zero(T))

    N = length(particle_bath_id)
    if N > 0
        threads = min(256, N)
        blocks = cld(N, threads)
        if st.freeze_mode != FREEZE_NONE && st.freeze_mask !== nothing
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_with_freeze_kernel!(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1})
            k(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1}; threads, blocks)
        else
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_kernel!(bath_counts, particle_bath_id)
            k(bath_counts, particle_bath_id; threads, blocks)
        end
    end

    B = length(dof_per_bath)
    if B > 0
        threads = min(256, B)
        blocks = cld(B, threads)
        dim = T(_is_3d(st) ? 3 : 2)
        k = CUDA.@cuda launch=false _nhc_finalize_dof_kernel!(dof_per_bath, bath_counts, dim)
        k(dof_per_bath, bath_counts, dim; threads, blocks)
    end
    return nothing
end

function _nhc_update_dof_per_bath!(spec::NHCSpec{T},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _csvr_update_dof_per_bath!(spec::CSVRSpec{T},
                                    st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                             Ekin::CuDeviceVector{T},
                                             particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Ekin)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += Ekin[i]
            end
        end
    end
    return nothing
end

function _nhc_reduce_kinetic_by_bath!(kinetic_total_per_bath::CuArray{T,1},
                                      Ekin::CuArray{T,1},
                                      particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(Ekin)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath, Ekin, particle_bath_id)
    k(kinetic_total_per_bath, Ekin, particle_bath_id; threads, blocks)
    return nothing
end

"""
    _ensure_nhc_kinetic_initialized!(spec, st)

Initialize per-bath kinetic totals and `st.Ekin` once before the first NHC
thermostat stage if no prior kick/thermostat stage has populated them.
"""
function _ensure_nhc_kinetic_initialized!(spec::NHCSpec{T},
                                          st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st, spec.params.mass)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

function _ensure_csvr_kinetic_initialized!(spec::CSVRSpec{T},
                                           st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st, spec.params.mass)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

function _nhc_apply_stage_scale2_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

function _nhc_apply_stage_scale3_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 vz::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                vz[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_stage_scale!(st, stage_scale_per_bath, particle_bath_id)

Apply the net thermostat scale per bath for one NHC stage to velocities and to
`st.Ekin` in a single GPU pass.
"""
function _nhc_apply_stage_scale!(st::SimulationState{T},
                                 stage_scale_per_bath::CuArray{T,1},
                                 particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    N = length(st.vx)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale2_by_bath_kernel!(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    end
    return nothing
end

@inline function _csvr_gamma_sample(shape::T) where {T<:AbstractFloat}
    shape > zero(T) || return zero(T)
    if shape < one(T)
        u = max(rand(T), eps(T))
        return _csvr_gamma_sample(shape + one(T)) * u^(inv(shape))
    end

    d = shape - T(1) / T(3)
    c = inv(sqrt(T(9) * d))
    while true
        x = randn(T)
        v = one(T) + c * x
        v <= zero(T) && continue
        v3 = v * v * v
        u = rand(T)
        x2 = x * x
        if u < one(T) - T(0.0331) * x2 * x2
            return d * v3
        end
        if log(max(u, eps(T))) < T(0.5) * x2 + d * (one(T) - v3 + log(v3))
            return d * v3
        end
    end
end

@inline function _csvr_chisq(::Type{T}, dof::Int) where {T<:AbstractFloat}
    dof <= 0 && return zero(T)
    return T(2) * _csvr_gamma_sample(T(dof) / T(2))
end

function _csvr_thermostat_stage_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                        cumulative_energy_exchange_per_bath::CuDeviceVector{T},
                                        last_velocity_scale_per_bath::CuDeviceVector{T},
                                        dof_per_bath::CuDeviceVector{T},
                                        target_temperature::CuDeviceVector{T},
                                        tau::CuDeviceVector{T},
                                        stage_dt::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            Kold = kinetic_total_per_bath[b]
            Ttarget = target_temperature[b]
            τ = tau[b]
            if !(Kold > eps(T)) || !(Ttarget > zero(T)) || !(τ > zero(T))
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            ndof = max(1, Int(floor(dof + T(0.5))))
            Kbar = T(0.5) * dof * Ttarget
            c1 = exp(-stage_dt / τ)
            c2 = ((one(T) - c1) * Kbar) / (Kold * T(ndof))
            r1 = randn(T)
            r2 = _csvr_chisq(T, ndof - 1)
            cross = sqrt(max(c1 * c2, zero(T)))
            scale2 = c1 + c2 * (r1 * r1 + r2) + T(2) * r1 * cross
            scale2 = max(scale2, zero(T))
            scale = sqrt(scale2)
            Knew = Kold * scale2

            kinetic_total_per_bath[b] = Knew
            cumulative_energy_exchange_per_bath[b] += Kold - Knew
            last_velocity_scale_per_bath[b] = scale
        end
    end
    return nothing
end

function _run_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                     stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    k = CUDA.@cuda launch=false _csvr_thermostat_stage_kernel!(ws.kinetic_total_per_bath,
                                                               ws.cumulative_energy_exchange_per_bath,
                                                               ws.last_velocity_scale_per_bath,
                                                               ws.dof_per_bath,
                                                               ws.target_temperature,
                                                               ws.tau,
                                                               stage_dt)
    k(ws.kinetic_total_per_bath,
      ws.cumulative_energy_exchange_per_bath,
      ws.last_velocity_scale_per_bath,
      ws.dof_per_bath,
      ws.target_temperature,
      ws.tau,
      stage_dt;
      threads,
      blocks)
    return nothing
end

function _apply_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                       st::SimulationState{T},
                                       stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _csvr_update_dof_per_bath!(spec, st)
    _ensure_csvr_kinetic_initialized!(spec, st)
    _run_csvr_thermostat_stage!(spec, stage_dt)
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end

function _nhc_half_kick2_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new)
            vx[i] = vx_new
            vy[i] = vy_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

function _nhc_half_kick3_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         vz::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         fz::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            vz_new = vz[i] + coef * fz[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new)
            vx[i] = vx_new
            vy[i] = vy_new
            vz[i] = vz_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_half_kick!(st, fx, fy, fz, dt, mass, kinetic_total_per_bath, particle_bath_id)

Apply a deterministic half-force kick `v <- v + (dt / (2m)) f`, refresh
`st.Ekin`, and reduce per-bath kinetic energies into `kinetic_total_per_bath`.
"""
function _nhc_apply_half_kick!(st::SimulationState{T},
                               fx::CuArray{T,1},
                               fy::CuArray{T,1},
                               fz::Union{Nothing,CuArray{T,1}},
                               dt::T,
                               mass::T,
                               kinetic_total_per_bath::CuArray{T,1},
                               particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(st.vx)
    N == 0 && return nothing
    coef = dt / (T(2) * mass)
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_half_kick3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1},
                                                                     fx, fy, fz::CuArray{T,1},
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy, st.vz::CuArray{T,1},
          fx, fy, fz::CuArray{T,1},
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_half_kick2_by_bath_kernel!(st.vx, st.vy,
                                                                     fx, fy,
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy,
          fx, fy,
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    end
    return nothing
end

"""
    _nhc_drift_positions!(st, dt)

Advance positions by one full deterministic drift under periodic boundaries.
Implemented through the existing BAOAB A-kernel using `2dt` so the effective
drift is exactly `dt`.
"""
function _nhc_drift_positions!(st::SimulationState{T}, dt::T) where {T<:AbstractFloat}
    drift_dt = T(2) * dt
    if _is_3d(st)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                        st.vx, st.vy, st.vz,
                                        drift_dt, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap,
                                        unwrapped_z=st.rz_unwrap)
    else
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                        st.vx, st.vy,
                                        drift_dt, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap)
    end
    return nothing
end

@inline function _nhc_gromacs_sy_weight(::Type{T}, idx::Int32) where {T<:AbstractFloat}
    if idx == Int32(3)
        return T(-0.186929716880426)
    elseif idx >= Int32(1) && idx <= Int32(5)
        return T(0.2967324292201065)
    end
    return zero(T)
end

function _nhc_chain_stage_legacy_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                scale = exp(-h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                for j in 1:M
                    eta[j, b] += h * xi[j, b]
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_gromacs_kernel!(xi::CuDeviceMatrix{T},
                                          eta::CuDeviceMatrix{T},
                                          chain_masses::CuDeviceMatrix{T},
                                          kinetic_total_per_bath::CuDeviceVector{T},
                                          thermostat_kinetic_per_bath::CuDeviceVector{T},
                                          thermostat_potential_per_bath::CuDeviceVector{T},
                                          last_velocity_scale_per_bath::CuDeviceVector{T},
                                          dof_per_bath::CuDeviceVector{T},
                                          target_temperature::CuDeviceVector{T},
                                          stage_dt::T,
                                          substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                for sy_idx in Int32(1):Int32(5)
                    time_step = stage_dt * _nhc_gromacs_sy_weight(T, sy_idx) / T(ns)

                    for j in M:-1:1
                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end

                    system_scale = exp(-time_step * xi[1, b])
                    total_scale *= system_scale
                    K *= system_scale * system_scale

                    for j in 1:M
                        eta[j, b] += time_step * xi[j, b]

                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_lammps_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            quarter_h = h / T(4)
            eighth_h = h / T(8)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for j in 2:M
                chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
            end

            for _ in 1:ns
                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in M:-1:2
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end

                expfac1 = M > 1 ? exp(-eighth_h * xi[2, b]) : one(T)
                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                scale = exp(-half_h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in 1:M
                    eta[j, b] += half_h * xi[j, b]
                end

                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                for j in 2:M
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _run_nhc_chain_stage!(spec::NHCSpec{T},
                               stage_dt::T) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    if p.propagator == NHC_PROPAGATOR_LEGACY
        k = CUDA.@cuda launch=false _nhc_chain_stage_legacy_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_GROMACS
        k = CUDA.@cuda launch=false _nhc_chain_stage_gromacs_kernel!(ws.xi,
                                                                     ws.eta,
                                                                     ws.chain_masses,
                                                                     ws.kinetic_total_per_bath,
                                                                     ws.thermostat_kinetic_per_bath,
                                                                     ws.thermostat_potential_per_bath,
                                                                     ws.last_velocity_scale_per_bath,
                                                                     ws.dof_per_bath,
                                                                     ws.target_temperature,
                                                                     stage_dt,
                                                                     Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_LAMMPS
        k = CUDA.@cuda launch=false _nhc_chain_stage_lammps_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    else
        throw(ArgumentError("Unsupported NHC propagator id $(p.propagator)."))
    end
    return nothing
end

"""
    _apply_nhc_thermostat_stage!(spec, st, stage_dt)

Propagate the NHC chain and apply deterministic velocity scaling over
`stage_dt`, internally split into `spec.params.substeps`.
"""
function _apply_nhc_thermostat_stage!(spec::NHCSpec{T},
                                      st::SimulationState{T},
                                      stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _nhc_update_dof_per_bath!(spec, st)
    _ensure_nhc_kinetic_initialized!(spec, st)
    _run_nhc_chain_stage!(spec, stage_dt)
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end

# -----------------------------------------------------------------------------
# Integrator protocol implementations
# -----------------------------------------------------------------------------

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
stage_sequence(::EMSpec) = (:midpoint_predict, :midpoint_force, :final_position, :force)
stage_sequence(::NHCSpec) = (:thermostat_pre, :kick1, :drift, :force, :kick2, :thermostat_post)
stage_sequence(::CSVRSpec) = (:kick1, :drift, :force, :kick2, :thermostat)

function validate_integrator_inputs!(spec::Union{BAOABSpec,BAOASpec,GSMSpec}, st, dt)
    _require_positive_gamma!(spec.params.gamma, string(integrator_name(spec)))
    return nothing
end

function validate_integrator_inputs!(spec::BrownianSpec, st, dt)
    _require_positive_gamma!(spec.params.gamma, "Brownian midpoint")
    return nothing
end

function validate_integrator_inputs!(spec::EMSpec, st, dt)
    _require_positive_gamma!(spec.params.gamma, "Euler-Maruyama")
    return nothing
end

function validate_integrator_inputs!(spec::NHCSpec{T}, st, dt) where {T<:AbstractFloat}
    p = spec.params
    p.mass > zero(T) || throw(ArgumentError("NHC requires mass > 0."))
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
    p.mass > zero(T) || throw(ArgumentError("CSVR requires mass > 0."))
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

function execute_integrator_stage!(spec::VVSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    params = spec.params
    ws = spec.workspace

    if stage_tag === :kick1
        _prepare_langevin_noise!(params, ws, st, dt)
    elseif stage_tag === :drift
        if _is_3d(st)
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry, st.rz,
                                                  st.vx, st.vy, st.vz,
                                                  st.f0x, st.f0y, st.f0z,
                                                  ws.rf_x, ws.rf_y, ws.rf_z,
                                                  params, dt, st.box3::Definitions.Box3;
                                                  unwrapped_x=st.rx_unwrap,
                                                  unwrapped_y=st.ry_unwrap,
                                                  unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.vv_positions_soa!(st.rx, st.ry,
                                                  st.vx, st.vy,
                                                  st.f0x, st.f0y,
                                                  ws.rf_x, ws.rf_y,
                                                  params, dt, st.box2::Definitions.Box2;
                                                  unwrapped_x=st.rx_unwrap,
                                                  unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        if _is_3d(st)
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy, st.vz,
                                                   st.f0x, st.f0y, st.f0z,
                                                   st.fx, st.fy, st.fz,
                                                   ws.rf_x, ws.rf_y, ws.rf_z,
                                                   st.dq, st.dU, st.Ekin,
                                                   params, dt)
        else
            LangevinIntegrators.vv_velocities_soa!(st.vx, st.vy,
                                                   st.f0x, st.f0y,
                                                   st.fx, st.fy,
                                                   ws.rf_x, ws.rf_y,
                                                   st.dq, st.dU, st.Ekin,
                                                   params, dt)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function _execute_baoab_family_stage!(params::LangevinIntegrators.BAOABParams{T},
                                      ws::StochasticWorkspace{T},
                                      st::SimulationState{T},
                                      stage_tag::Symbol,
                                      dt::T;
                                      compute_energy::Bool,
                                      freeze_hold::Bool,
                                      freeze_spring::Bool) where {T<:AbstractFloat}
    if stage_tag === :B1
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.f0x, st.f0y, st.f0z,
                                            params, dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.f0x, st.f0y,
                                            params, dt, st.Ekin, st.dU)
        end
    elseif stage_tag === :A1 || stage_tag === :A2
        if _is_3d(st)
            LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                            st.vx, st.vy, st.vz,
                                            dt, st.box3::Definitions.Box3;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap,
                                            unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                            st.vx, st.vy,
                                            dt, st.box2::Definitions.Box2;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, stage_tag === :A1 ? :after_drift : :after_final_position;
                                   freeze_hold=freeze_hold)
    elseif stage_tag === :O
        _prepare_langevin_noise!(params, ws, st, dt)
        if _is_3d(st)
            LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz,
                                             ws.rf_x, ws.rf_y, ws.rf_z,
                                             params, dt, st.dq)
        else
            LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy,
                                             ws.rf_x, ws.rf_y,
                                             params, dt, st.dq)
        end
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :B2
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.fx, st.fy, st.fz,
                                            params, dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.fx, st.fy,
                                            params, dt, st.Ekin, st.dU)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for BAOAB-family integrator."))
    end
    return nothing
end

function execute_integrator_stage!(spec::Union{BAOABSpec{T},GSMSpec{T}},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_baoab_family_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                        compute_energy=compute_energy,
                                        freeze_hold=freeze_hold,
                                        freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::BAOASpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    params = spec.params
    ws = spec.workspace

    if stage_tag === :B1
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.f0x, st.f0y, st.f0z,
                                            params, T(2) * dt, st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.f0x, st.f0y,
                                            params, T(2) * dt, st.Ekin, st.dU)
        end
    elseif stage_tag === :A1 || stage_tag === :A2
        if _is_3d(st)
            LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                            st.vx, st.vy, st.vz,
                                            dt, st.box3::Definitions.Box3;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap,
                                            unwrapped_z=st.rz_unwrap)
        else
            LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                            st.vx, st.vy,
                                            dt, st.box2::Definitions.Box2;
                                            unwrapped_x=st.rx_unwrap,
                                            unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, stage_tag === :A1 ? :after_drift : :after_final_position;
                                   freeze_hold=freeze_hold)
    elseif stage_tag === :O
        _prepare_langevin_noise!(params, ws, st, dt)
        if _is_3d(st)
            LangevinIntegrators.baoab_OU_3d!(st.vx, st.vy, st.vz,
                                             ws.rf_x, ws.rf_y, ws.rf_z,
                                             params, dt, st.dq)
        else
            LangevinIntegrators.baoab_OU_2d!(st.vx, st.vy,
                                             ws.rf_x, ws.rf_y,
                                             params, dt, st.dq)
        end
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :power
        if _is_3d(st)
            LangevinIntegrators.cons_power_3d!(st.vx, st.vy, st.vz,
                                               st.fx, st.fy, st.fz,
                                               st.dU)
        else
            LangevinIntegrators.cons_power_2d!(st.vx, st.vy,
                                               st.fx, st.fy,
                                               st.dU)
        end
    elseif stage_tag === :kinetic_refresh
        if _is_3d(st)
            LangevinIntegrators.baoab_B_3d!(st.vx, st.vy, st.vz,
                                            st.fx, st.fy, st.fz,
                                            params, T(0), st.Ekin, st.dU)
        else
            LangevinIntegrators.baoab_B_2d!(st.vx, st.vy,
                                            st.fx, st.fy,
                                            params, T(0), st.Ekin, st.dU)
        end
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function _execute_brownian_midpoint_stage!(params::Union{BrownianIntegrators.BrownianParams{T},BrownianIntegrators.EMParams{T}},
                                           ws::StochasticWorkspace{T},
                                           st::SimulationState{T},
                                           stage_tag::Symbol,
                                           dt::T;
                                           compute_energy::Bool,
                                           freeze_hold::Bool,
                                           freeze_spring::Bool) where {T<:AbstractFloat}
    if stage_tag === :midpoint_predict
        _prepare_brownian_noise!(params, ws, st, dt)
        if _is_3d(st)
            BrownianIntegrators.bd_midpoint_positions_3d!(st.rx, st.ry, st.rz,
                                                          st.fx, st.fy, st.fz,
                                                          ws.rf_x, ws.rf_y, ws.rf_z,
                                                          st.vx, st.vy, st.vz,
                                                          params.gamma, params.noise_scale,
                                                          dt, st.box3::Definitions.Box3)
            if freeze_hold
                _apply_freeze_hold!(st, st.vx, st.vy, st.vz)
            end
        else
            BrownianIntegrators.bd_midpoint_positions_2d!(st.rx, st.ry,
                                                          st.fx, st.fy,
                                                          ws.rf_x, ws.rf_y,
                                                          st.vx, st.vy,
                                                          params.gamma, params.noise_scale,
                                                          dt, st.box2::Definitions.Box2)
            if freeze_hold
                _apply_freeze_hold!(st, st.vx, st.vy)
            end
        end
    elseif stage_tag === :midpoint_force
        evaluate_midpoint_forces_into_f0!(st; freeze_spring=freeze_spring)
    elseif stage_tag === :final_position
        if _is_3d(st)
            BrownianIntegrators.bd_finish_step_3d!(st.rx, st.ry, st.rz,
                                                   st.f0x, st.f0y, st.f0z,
                                                   ws.rf_x, ws.rf_y, ws.rf_z,
                                                   params.gamma, params.noise_scale,
                                                   dt, st.dq, st.dU,
                                                   st.box3::Definitions.Box3;
                                                   unwrapped_x=st.rx_unwrap,
                                                   unwrapped_y=st.ry_unwrap,
                                                   unwrapped_z=st.rz_unwrap)
        else
            BrownianIntegrators.bd_finish_step_2d!(st.rx, st.ry,
                                                   st.f0x, st.f0y,
                                                   ws.rf_x, ws.rf_y,
                                                   params.gamma, params.noise_scale,
                                                   dt, st.dq, st.dU,
                                                   st.box2::Definitions.Box2;
                                                   unwrapped_x=st.rx_unwrap,
                                                   unwrapped_y=st.ry_unwrap)
        end
        apply_post_position_hooks!(st, :after_final_position; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for Brownian-family integrator."))
    end

    return nothing
end

function execute_integrator_stage!(spec::BrownianSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_brownian_midpoint_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                             compute_energy=compute_energy,
                                             freeze_hold=freeze_hold,
                                             freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::EMSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    return _execute_brownian_midpoint_stage!(spec.params, spec.workspace, st, stage_tag, dt;
                                             compute_energy=compute_energy,
                                             freeze_hold=freeze_hold,
                                             freeze_spring=freeze_spring)
end

function execute_integrator_stage!(spec::NHCSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace

    if stage_tag === :thermostat_pre
        _apply_nhc_thermostat_stage!(spec, st, dt / T(2))
    elseif stage_tag === :kick1
        _nhc_apply_half_kick!(st, st.f0x, st.f0y, st.f0z, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :drift
        _nhc_drift_positions!(st, dt)
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        _nhc_apply_half_kick!(st, st.fx, st.fy, st.fz, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :thermostat_post
        _apply_nhc_thermostat_stage!(spec, st, dt / T(2))
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

function execute_integrator_stage!(spec::CSVRSpec{T},
                                   st::SimulationState{T},
                                   dt::T,
                                   stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace

    if stage_tag === :kick1
        _nhc_apply_half_kick!(st, st.f0x, st.f0y, st.f0z, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :drift
        _nhc_drift_positions!(st, dt)
        apply_post_position_hooks!(st, :after_drift; freeze_hold=freeze_hold)
    elseif stage_tag === :force
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    elseif stage_tag === :kick2
        _nhc_apply_half_kick!(st, st.fx, st.fy, st.fz, dt, p.mass, ws.kinetic_total_per_bath, ws.particle_bath_id)
    elseif stage_tag === :thermostat
        _apply_csvr_thermostat_stage!(spec, st, dt)
    else
        throw(ArgumentError("Unsupported stage $(stage_tag) for $(integrator_name(spec))."))
    end

    return nothing
end

"""
    run_integrator_step!(st, spec, dt; compute_energy=true)

Shared stage-driven step engine. Integrator-independent orchestration lives
here, while stage-specific work is delegated through the integrator protocol.
"""
function run_integrator_step!(st::SimulationState{T},
                              spec::IntegratorSpec{T},
                              dt::T;
                              compute_energy::Bool=true) where {T<:AbstractFloat}
    validate_integrator_inputs!(spec, st, dt)

    freeze_active = _freeze_active!(st)
    freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
    freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING

    rebuild_needed = plan_neighbor_rebuild!(st, dt)
    apply_neighbor_rebuild_if_needed!(st, rebuild_needed)

    prepare_previous_force_buffers!(st, spec)
    ensure_reference_forces_ready!(st, spec, compute_energy, freeze_spring)
    ensure_integrator_workspace!(spec, st)

    for stage_tag in stage_sequence(spec)
        _trace_integrator_stage!(st, spec, stage_tag;
                                 force_evaluated=(stage_tag == :force),
                                 rebuild_applied=rebuild_needed)
        execute_integrator_stage!(spec, st, dt, stage_tag;
                                  compute_energy=compute_energy,
                                  freeze_hold=freeze_hold,
                                  freeze_spring=freeze_spring)
    end

    finalize_step_accounting!(st, spec, compute_energy)
    finalize_step_counter!(st, integrator_id(spec))
    return nothing
end

# -----------------------------------------------------------------------------
# Public stepping API
# -----------------------------------------------------------------------------

"""
    step!(st, spec, dt; compute_energy=true)

Canonical explicit stepping API.
"""
function step!(st::SimulationState{T},
               spec::IntegratorSpec{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return run_integrator_step!(st, spec, T(dt); compute_energy=compute_energy)
end

"""
    step_graph!(st, spec, dt; compute_energy=true)

Graph-entry stepping API with explicit integrator selection. It currently
shares the same stage-driven engine as [`step!`](@ref).
"""
function step_graph!(st::SimulationState{T},
                     spec::IntegratorSpec{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return run_integrator_step!(st, spec, T(dt); compute_energy=compute_energy)
end

"""
    step!(st, vv, dt; compute_energy=true)

Compatibility wrapper for explicit `VVParams` stepping.
"""
function step!(st::SimulationState{T},
               vv::LangevinIntegrators.VVParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, VVSpec(vv), dt; compute_energy=compute_energy)
end

"""
    step!(st, bao, dt; compute_energy=true)

Compatibility wrapper for explicit `BAOABParams` stepping.
"""
function step!(st::SimulationState{T},
               bao::LangevinIntegrators.BAOABParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, BAOABSpec(bao), dt; compute_energy=compute_energy)
end

"""
    step!(st, bp, dt; compute_energy=true)

Compatibility wrapper for explicit Brownian midpoint stepping.
"""
function step!(st::SimulationState{T},
               bp::BrownianIntegrators.BrownianParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, BrownianSpec(bp), dt; compute_energy=compute_energy)
end

"""
    step!(st, em, dt; compute_energy=true)

Compatibility wrapper for explicit Euler–Maruyama stepping.
"""
function step!(st::SimulationState{T},
               em::BrownianIntegrators.EMParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, EMSpec(em), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, vv, dt; compute_energy=true)

Compatibility wrapper for explicit `VVParams` graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     vv::LangevinIntegrators.VVParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, VVSpec(vv), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, bao, dt; compute_energy=true)

Compatibility wrapper for explicit `BAOABParams` graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     bao::LangevinIntegrators.BAOABParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, BAOABSpec(bao), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, bp, dt; compute_energy=true)

Compatibility wrapper for explicit Brownian midpoint graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     bp::BrownianIntegrators.BrownianParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, BrownianSpec(bp), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, em, dt; compute_energy=true)

Compatibility wrapper for explicit Euler-Maruyama graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     em::BrownianIntegrators.EMParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, EMSpec(em), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, nhc, dt; compute_energy=true)

Compatibility wrapper for explicit Nose-Hoover chain graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     nhc::NHCParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, NHCSpec(nhc), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, csvr_params, dt; compute_energy=true)

Compatibility wrapper for explicit CSVR graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     csvr_params::CSVRParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, CSVRSpec(csvr_params), dt; compute_energy=compute_energy)
end

"""
    step!(st, nhc, dt; compute_energy=true)

Compatibility wrapper for explicit Nose-Hoover Chain parameter stepping.
"""
function step!(st::SimulationState{T},
               nhc::NHCParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st,
                 NHCSpec{T}(nhc, _new_nhc_workspace(T,
                                                    nhc.chain_length,
                                                    length(nhc.target_temperature),
                                                    length(st.rx))),
                 dt;
                 compute_energy=compute_energy)
end

"""
    step!(st, csvr_params, dt; compute_energy=true)

Compatibility wrapper for explicit CSVR parameter stepping.
"""
function step!(st::SimulationState{T},
               csvr_params::CSVRParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st,
                 CSVRSpec{T}(csvr_params,
                             _new_csvr_workspace(T,
                                                 length(csvr_params.target_temperature),
                                                 length(st.rx))),
                 dt;
                 compute_energy=compute_energy)
end

"""
    step_bd!(st, dt, bp; compute_energy=true)

Deprecated thin wrapper. Use `step!(st, bp, dt; ...)`.
"""
function step_bd!(st::SimulationState{T},
                  dt::Real,
                  bp::BrownianIntegrators.BrownianParams{T};
                  compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, bp, dt; compute_energy=compute_energy)
end

# -----------------------------------------------------------------------------
# Observables and thermostat metadata
# -----------------------------------------------------------------------------

"""
    thermostatted_particle_mask(st, spec)

Return a mask of particles thermostatted by `spec`. `nothing` means all
particles are currently thermostatted.
"""
function thermostatted_particle_mask(st::SimulationState,
                                     spec::IntegratorSpec)
    if st.freeze_mask === nothing || st.freeze_mode == FREEZE_NONE
        return nothing
    end
    return ifelse.(st.freeze_mask .== UInt8(0), UInt8(1), UInt8(0))
end

function thermostatted_particle_mask(st::SimulationState,
                                     spec::NHCSpec)
    if st.freeze_mask === nothing || st.freeze_mode == FREEZE_NONE
        return nothing
    end
    return ifelse.(st.freeze_mask .== UInt8(0), UInt8(1), UInt8(0))
end

"""
    thermostatted_dof(st, spec) -> Int

Return the physical degrees of freedom acted on by the thermostat.
"""
function thermostatted_dof(st::SimulationState,
                           spec::IntegratorSpec)
    D = _is_3d(st) ? 3 : 2
    mask = thermostatted_particle_mask(st, spec)
    if mask === nothing
        return D * length(st.rx)
    end
    ntherm = Int(CUDA.sum(Int32.(mask)))
    return D * ntherm
end

function thermostatted_dof(st::SimulationState,
                           spec::NHCSpec)
    D = _is_3d(st) ? 3 : 2
    mask = thermostatted_particle_mask(st, spec)
    if mask === nothing
        return D * length(st.rx)
    end
    ntherm = Int(CUDA.sum(Int32.(mask)))
    return D * ntherm
end

@inline _device_scalar(x::CuArray{T,1}) where {T<:AbstractFloat} = T(Array(x)[1])

function collect_integrator_observables(spec::NHCSpec{T},
                                        st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    _nhc_update_dof_per_bath!(spec, st)
    Ekin_total = T(CUDA.sum(st.Ekin))
    Epot_total = T(CUDA.sum(st.Epot))
    dof_b = Array(ws.dof_per_bath)
    K_b = Array(ws.kinetic_total_per_bath)
    target_b = p.target_temperature
    tau_b = p.tau
    scale_b = Array(ws.last_velocity_scale_per_bath)
    total_dof = sum(dof_b)

    weighted_target = zero(T)
    weighted_error = zero(T)
    if total_dof > zero(T)
        weighted_target = sum(target_b .* dof_b) / total_dof
        @inbounds for b in eachindex(target_b)
            dof = dof_b[b]
            if dof > zero(T)
                Tinst = T(2) * K_b[b] / dof
                weighted_error += (Tinst - target_b[b]) * dof
            end
        end
        weighted_error /= total_dof
    end

    thermostat_kin = T(sum(Array(ws.thermostat_kinetic_per_bath)))
    thermostat_pot = T(sum(Array(ws.thermostat_potential_per_bath)))
    if total_dof > zero(T)
        acc = zero(T)
        @inbounds for b in eachindex(scale_b)
            if dof_b[b] > zero(T)
                acc += log(max(scale_b[b], eps(T))) * dof_b[b]
            end
        end
        stage_scale = exp(acc / total_dof)
    else
        stage_scale = one(T)
    end

    ext_h = Epot_total + Ekin_total + thermostat_kin + thermostat_pot
    return (thermostat_kind=:nhc,
            target_temperature=weighted_target,
            thermostat_timescale=(isempty(tau_b) ? zero(T) : T(sum(tau_b) / length(tau_b))),
            chain_length=p.chain_length,
            chain_substeps=p.substeps,
            nhc_propagator=_nhc_propagator_name(p.propagator),
            nhc_num_baths=length(target_b),
            nhc_dof_total=total_dof,
            physical_kinetic=Ekin_total,
            thermostat_kinetic=thermostat_kin,
            thermostat_potential=thermostat_pot,
            extended_hamiltonian=ext_h,
            thermostat_temperature_error=weighted_error,
            nhc_velocity_scale=stage_scale)
end

function collect_integrator_observables(spec::CSVRSpec{T},
                                        st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    _csvr_update_dof_per_bath!(spec, st)
    _ensure_csvr_kinetic_initialized!(spec, st)
    Ekin_total = T(CUDA.sum(st.Ekin))
    Epot_total = T(CUDA.sum(st.Epot))
    dof_b = Array(ws.dof_per_bath)
    K_b = Array(ws.kinetic_total_per_bath)
    target_b = p.target_temperature
    tau_b = p.tau
    scale_b = Array(ws.last_velocity_scale_per_bath)
    total_dof = sum(dof_b)

    weighted_target = zero(T)
    weighted_error = zero(T)
    if total_dof > zero(T)
        weighted_target = sum(target_b .* dof_b) / total_dof
        @inbounds for b in eachindex(target_b)
            dof = dof_b[b]
            if dof > zero(T)
                Tinst = T(2) * K_b[b] / dof
                weighted_error += (Tinst - target_b[b]) * dof
            end
        end
        weighted_error /= total_dof
    end

    thermostat_energy = T(sum(Array(ws.cumulative_energy_exchange_per_bath)))
    if total_dof > zero(T)
        acc = zero(T)
        @inbounds for b in eachindex(scale_b)
            if dof_b[b] > zero(T)
                acc += log(max(scale_b[b], eps(T))) * dof_b[b]
            end
        end
        stage_scale = exp(acc / total_dof)
    else
        stage_scale = one(T)
    end

    ext_h = Epot_total + Ekin_total + thermostat_energy
    return (thermostat_kind=:csvr,
            target_temperature=weighted_target,
            thermostat_timescale=(isempty(tau_b) ? zero(T) : T(sum(tau_b) / length(tau_b))),
            csvr_num_baths=length(target_b),
            csvr_dof_total=total_dof,
            physical_kinetic=Ekin_total,
            thermostat_energy=thermostat_energy,
            extended_hamiltonian=ext_h,
            thermostat_temperature_error=weighted_error,
            csvr_velocity_scale=stage_scale)
end

"""
    collect_step_observables(st, spec) -> NamedTuple

Collect universal and integrator-specific diagnostics for writers and
post-processing.
"""
function collect_step_observables(st::SimulationState{T},
                                  spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    Epot_total = T(CUDA.sum(st.Epot))
    Ekin_total = T(CUDA.sum(st.Ekin))
    virial_total = T(CUDA.sum(st.virial))
    dq_total = T(CUDA.sum(st.dq))
    dU_total = T(CUDA.sum(st.dU))

    base = (step=st.step,
            integrator=integrator_name(spec),
            Epot_total=Epot_total,
            Ekin_total=Ekin_total,
            Etot=Epot_total + Ekin_total,
            virial_total=virial_total,
            dq_total=dq_total,
            dU_total=dU_total,
            Qtot=dq_total,
            thermostatted_dof=thermostatted_dof(st, spec))

    return merge(base, collect_integrator_observables(spec, st))
end

end # module
