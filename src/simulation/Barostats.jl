# =========================
#   Isotropic pressure control (3D): Monte Carlo volume moves and C-rescale
# =========================
#
# Both act at complete-step boundaries on all particle coordinates
# (`scale=:atoms`); velocities are never scaled. Host work per event is a few
# scalars: the RNG draws, Float64 energy/virial totals and the decision.

using Random: Xoshiro, randn, rand
import Random

# Counter-based streams: event k draws from Xoshiro seeded with (seed, k), so a
# restart needs only the seed and the counters, and particle order or the
# number of other random draws never affects the barostat.
_event_seed(seed::UInt64, k::Integer) = seed ⊻ (UInt64(k) * 0x9e3779b97f4a7c15)
_reseed!(rng::Xoshiro, seed::UInt64, k::Integer) = Random.seed!(rng, _event_seed(seed, k))

"""
    total_kinetic_energy(st) -> Float64

Σ m v²/2 over all particles (device reduction in Float64).
"""
function total_kinetic_energy(st::SimulationState)
    d3 = _is_3d(st)
    v2 = if d3
        CUDA.mapreduce((x, y, z) -> Float64(x)^2 + Float64(y)^2 + Float64(z)^2, +, st.vx, st.vy, st.vz)
    else
        CUDA.mapreduce((x, y) -> Float64(x)^2 + Float64(y)^2, +, st.vx, st.vy)
    end
    has_uniform_particle_mass(st) && return 0.5 * Float64(st.mass) * v2
    m = st.mass_particle
    w = if d3
        CUDA.mapreduce((mi, x, y, z) -> Float64(mi) * (Float64(x)^2 + Float64(y)^2 + Float64(z)^2), +, m, st.vx, st.vy, st.vz)
    else
        CUDA.mapreduce((mi, x, y) -> Float64(mi) * (Float64(x)^2 + Float64(y)^2), +, m, st.vx, st.vy)
    end
    return 0.5 * w
end

"""
    mechanical_pressure(st) -> Float64

Instantaneous `(2K + tr W)/(D V)` with the configurational virial `W = Σ r⊗F`
(including global terms). Refreshes stale virials.
"""
mechanical_pressure(st::SimulationState) =
    (2 * total_kinetic_energy(st) + total_virial_trace(st)) / ((_is_3d(st) ? 3 : 2) * cell_volume(st))

abstract type PressureControl end

function _check_barostat_state(st::SimulationState, name; molecular::Bool=false)
    # Rigid water: Monte Carlo moves and C-rescale scale molecules.
    molecular || st.rigid_water === nothing ||
        throw(ArgumentError("$(name) does not support rigid water (molecular pressure); use MonteCarloVolumeMove"))
    _is_3d(st) || throw(ArgumentError("$(name) is implemented for isotropic 3D cells only."))
    st.external_potential === nothing || throw(ArgumentError("$(name) does not support external potentials."))
    st.freeze_mode == FREEZE_NONE || throw(ArgumentError("$(name) requires all particles mobile (no freeze controls)."))
    # electrostatics: validated with cell changes on the device engine
    st.tag === nothing || st.last_reorder_step < 0 ||
        throw(ArgumentError("$(name) requires spatial reordering to be off."))
    return nothing
end

"""
    MonteCarloVolumeMove(st; pressure, kT, log_volume_step=0.01, seed)

Isotropic log-volume Monte Carlo move: `η ~ U(-δ, δ)`, `V' = V e^η`, all
coordinates scaled by `e^(η/3)`, accepted with
`min(1, exp(-(U'-U + P(V'-V))/kT + (N+1)η))`. Energies in Float64. With rigid
water the move is molecular ([`set_cell_and_molecules!`](@ref)) and `N` counts
each water once.
"""
mutable struct MonteCarloVolumeMove{T} <: PressureControl
    pressure::Float64
    kT::Float64
    log_volume_step::Float64
    seed::UInt64
    rng::Xoshiro
    trial::TrialWorkspace{T}
    attempts::Int
    accepts::Int
    rejected_domain::Int
    # tuning (optional, bounded batches; frozen afterwards)
    tune_batches::Int
    batch_attempts::Int
    batch_accepts::Int
    # committed diagnostics
    volume_change::Float64
    energy_change::Float64
    pv_work::Float64
end

function MonteCarloVolumeMove(st::SimulationState{T}; pressure::Real, kT::Real, log_volume_step::Real=0.01,
                              seed::Integer, tune_batches::Integer=0) where {T}
    _check_barostat_state(st, "MonteCarloBarostat"; molecular=true)
    isfinite(pressure) || throw(ArgumentError("pressure must be finite"))
    kT > 0 && isfinite(kT) || throw(ArgumentError("kT must be positive"))
    0 < log_volume_step < 1 || throw(ArgumentError("log_volume_step must lie in (0, 1)"))
    return MonteCarloVolumeMove{T}(Float64(pressure), Float64(kT), Float64(log_volume_step), UInt64(seed),
                                   Xoshiro(UInt64(seed)), TrialWorkspace(st), 0, 0, 0,
                                   Int(tune_batches), 0, 0, 0.0, 0.0, 0.0)
end

"""
    monte_carlo_volume_move!(st, mc) -> Bool

Attempt one volume move. On rejection the state (cell, coordinates, forces,
energies, virials, validity) is restored exactly; the RNG and attempt count
still advance. Invalid trial cells or geometries are rejections.
"""
function monte_carlo_volume_move!(st::SimulationState{T}, mc::MonteCarloVolumeMove{T}) where {T}
    U = total_potential_energy(st)          # ensures valid energies at the committed state
    V = cell_volume(st)
    N = scaled_units(st)
    _reseed!(mc.rng, mc.seed, mc.attempts)
    η = mc.log_volume_step * (2 * rand(mc.rng) - 1)
    logu = log(rand(mc.rng))
    mc.attempts += 1
    Vt = V * exp(η)
    s = exp(η / 3)
    box = _current_box(st)
    newbox = ntuple(d -> box[d] * T(s), 3)
    accepted = false
    if cell_supported(st, newbox) === nothing
        save_trial_state!(mc.trial, st)
        _set_cell_scaled!(st, newbox)
        Ut = try
            # Only the cell and coordinates changed (`moved`): the device engine that
            # owns the state evaluates the trial in place.
            evaluate_current_configuration!(st, true; freeze_spring=st.force_freeze_spring, moved=true)
            CUDA.mapreduce(Float64, +, st.Epot)
        catch err
            err isa DomainError || (restore_trial_state!(mc.trial, st); rethrow())
            NaN
        end
        Vt = cell_volume(st)            # exact volume of the stored box
        log_acc = -(Ut - U + mc.pressure * (Vt - V)) / mc.kT + (N + 1) * log(Vt / V)
        if isfinite(log_acc) && logu < min(0.0, log_acc)
            accepted = true
            mc.accepts += 1
            mc.volume_change += Vt - V
            mc.energy_change += Ut - U
            mc.pv_work += mc.pressure * (Vt - V)
        else
            isfinite(Ut) || (mc.rejected_domain += 1)
            restore_trial_state!(mc.trial, st)
        end
    else
        mc.rejected_domain += 1
    end
    _mc_tune!(mc, accepted)
    return accepted
end

function _mc_tune!(mc::MonteCarloVolumeMove, accepted::Bool)
    mc.tune_batches > 0 || return nothing
    mc.batch_attempts += 1
    mc.batch_accepts += accepted
    if mc.batch_attempts == 100
        rate = mc.batch_accepts / 100
        # toward a 40% acceptance, bounded change per batch
        mc.log_volume_step = clamp(mc.log_volume_step * clamp(rate / 0.4, 0.5, 2.0), 1e-6, 0.5)
        mc.batch_attempts = 0
        mc.batch_accepts = 0
        mc.tune_batches -= 1
    end
    return nothing
end

acceptance_ratio(mc::MonteCarloVolumeMove) = mc.attempts == 0 ? NaN : mc.accepts / mc.attempts

"""
    StochasticCellRescaling(st; pressure, kT, compressibility, tau, seed)

Isotropic stochastic cell rescaling (Bernetti & Bussi 2020), unscaled-momentum
formulation in `λ = √V`:

    Pbaro = N kT/V + tr(W)/(3V)
    dλ = κλ/(2τ) [Pbaro - P + kT/(2V)] dt + √(kT κ/(2τ)) dW

integrated with one Euler–Maruyama step of the elapsed MD time per event.
Coordinates follow the cell; velocities are not scaled. With rigid water, N counts
molecules, `W` is the molecular virial ([`molecular_virial_trace`](@ref)) and each
molecule is translated with its centre of mass ([`set_cell_and_molecules!`](@ref)).
"""
mutable struct StochasticCellRescalingState <: PressureControl
    pressure::Float64
    kT::Float64
    compressibility::Float64
    tau::Float64
    seed::UInt64
    rng::Xoshiro
    events::Int
    volume_change::Float64
    last_pressure::Float64
end

function StochasticCellRescalingState(st::SimulationState; pressure::Real, kT::Real, compressibility::Real,
                                      tau::Real, seed::Integer)
    _check_barostat_state(st, "StochasticCellRescaling"; molecular=true)
    isfinite(pressure) || throw(ArgumentError("pressure must be finite"))
    kT > 0 && isfinite(kT) || throw(ArgumentError("kT must be positive"))
    compressibility > 0 && isfinite(compressibility) || throw(ArgumentError("compressibility must be positive"))
    tau > 0 && isfinite(tau) || throw(ArgumentError("tau must be positive"))
    return StochasticCellRescalingState(Float64(pressure), Float64(kT), Float64(compressibility), Float64(tau),
                                        UInt64(seed), Xoshiro(UInt64(seed)), 0, 0.0, NaN)
end

"""
    stochastic_cell_rescale!(st, cr, h; compute_energy=true)

One C-rescale event over elapsed MD time `h`. Uses a fresh virial at the
current coordinates, scales the cell and coordinates, and re-evaluates
forces (and energies when requested) at the new cell. A nonpositive or
unsupported trial cell is an integration failure: the state is left
unchanged and a `DomainError` is thrown.
"""
function stochastic_cell_rescale!(st::SimulationState{T}, cr::StochasticCellRescalingState, h::Real;
                                  compute_energy::Bool=true) where {T}
    h > 0 && isfinite(h) || throw(ArgumentError("elapsed time must be positive"))
    V = cell_volume(st)
    # Rigid water: molecules are the scaled units, with the molecular
    # virial; otherwise particles and the configurational virial.
    N = scaled_units(st)
    W = molecular_virial_trace(st)
    Θ = cr.kT; κ = cr.compressibility; τ = cr.tau
    Pbaro = N * Θ / V + W / (3V)
    cr.last_pressure = Pbaro
    λ = sqrt(V)
    drift = κ * λ / (2τ) * (Pbaro - cr.pressure + Θ / (2V))
    amp = sqrt(Θ * κ / (2τ))
    _reseed!(cr.rng, cr.seed, cr.events)
    R = randn(cr.rng)
    cr.events += 1
    λt = λ + h * drift + sqrt(h) * amp * R
    (isfinite(λt) && λt > 0) || throw(DomainError(λt, "C-rescale produced a nonpositive volume; reduce tau/compressibility ratio or the event interval"))
    s = cbrt(λt^2 / V)
    box = _current_box(st)
    newbox = ntuple(d -> box[d] * T(s), 3)
    reason = cell_supported(st, newbox)
    reason === nothing || throw(DomainError(newbox, "C-rescale trial cell unsupported: $(reason)"))
    _set_cell_scaled!(st, newbox)
    cr.volume_change += cell_volume(st) - V
    evaluate_current_configuration!(st, compute_energy; freeze_spring=st.force_freeze_spring, moved=true)
    return st
end

# ------------------------- device-resident pressure control -------------------------

"""Pressure control applied inside device step graphs: `control` every `interval` steps."""
struct DevicePressureControl
    control::StochasticCellRescalingState
    interval::Int
end

"""
    attach_pressure_control!(st, control::StochasticCellRescalingState; interval) -> st

Device-resident pressure control: from the next `step!` on, the device engine applies the
C-rescale event of `control` every `interval` steps inside its captured step graphs, with
no host synchronization (the virial of the step before each event, the Gaussian draw,
the scale, molecular scaling and the mesh refresh all run on the device). The host box
(`st.box3`) and the counters of `control` are refreshed at host boundaries (observables,
evaluations, writers, `synchronize_state!`). Supported on the cluster-pair backend (3D)
with NVE, CSVR or Nosé–Hoover integration; `interval ≥ 2`.
[`detach_pressure_control!`](@ref) removes it. The host event
[`stochastic_cell_rescale!`](@ref) stays available for other states.
"""
function attach_pressure_control!(st::SimulationState, control::StochasticCellRescalingState; interval::Integer)
    interval >= 2 || throw(ArgumentError("interval must be at least 2 (the step before an event provides the virial)"))
    st.pressure_control = DevicePressureControl(control, Int(interval))
    invalidate_forces!(st)   # the next step captures graphs with the event variant
    return st
end

"""
    detach_pressure_control!(st) -> st

Remove the device-resident pressure control (the host box is synchronized first).
"""
function detach_pressure_control!(st::SimulationState)
    st.pressure_control === nothing && return st
    _engine_boundary!(st.engine, st)
    st.pressure_control = nothing
    invalidate_forces!(st)
    return st
end

# ---------------------------- restart records ----------------------------

const _MC_RECORD = 1.0
const _CR_RECORD = 2.0

"""
    pressure_control_record(c) -> Vector{Float64}

Counters, tuned parameters and committed diagnostics needed to continue the
pressure-control stream after a restart (the seed itself is stored with the
counters so a mismatching descriptor is detected).
"""
pressure_control_record(c::MonteCarloVolumeMove) =
    Float64[_MC_RECORD, reinterpret(Float64, c.seed), c.attempts, c.accepts, c.rejected_domain, c.tune_batches,
            c.batch_attempts, c.batch_accepts, c.log_volume_step, c.volume_change, c.energy_change, c.pv_work]
pressure_control_record(c::StochasticCellRescalingState) =
    Float64[_CR_RECORD, reinterpret(Float64, c.seed), c.events, c.volume_change]

function restore_pressure_control!(c::MonteCarloVolumeMove, r::AbstractVector{<:Real})
    (length(r) == 12 && r[1] == _MC_RECORD) || throw(ArgumentError("not a Monte Carlo barostat record"))
    reinterpret(UInt64, Float64(r[2])) == c.seed || throw(ArgumentError("barostat record seed does not match the descriptor"))
    c.attempts, c.accepts, c.rejected_domain = Int(r[3]), Int(r[4]), Int(r[5])
    c.tune_batches, c.batch_attempts, c.batch_accepts = Int(r[6]), Int(r[7]), Int(r[8])
    c.log_volume_step = Float64(r[9])
    c.volume_change, c.energy_change, c.pv_work = Float64(r[10]), Float64(r[11]), Float64(r[12])
    return c
end

function restore_pressure_control!(c::StochasticCellRescalingState, r::AbstractVector{<:Real})
    (length(r) == 4 && r[1] == _CR_RECORD) || throw(ArgumentError("not a C-rescale barostat record"))
    reinterpret(UInt64, Float64(r[2])) == c.seed || throw(ArgumentError("barostat record seed does not match the descriptor"))
    c.events = Int(r[3])
    c.volume_change = Float64(r[4])
    return c
end
