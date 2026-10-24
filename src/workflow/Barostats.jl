# ------------------------------------------------------------------
# Workflow pressure control
# ------------------------------------------------------------------

"""
    Barostat

Abstract global pressure-control descriptor, passed as
`Integrator(...; barostat=...)`.
"""
abstract type Barostat end

"""
    StochasticCellRescaling(; pressure, kT, compressibility, tau, interval=10, seed, mode=:isotropic)

Isotropic stochastic cell rescaling (Bernetti & Bussi 2020) applied every
`interval` steps over the elapsed MD time. Units: `pressure` E/L³, `kT` E,
`compressibility` L³/E, `tau` time. Positions follow the cell; velocities are
not scaled. Requires `scheme=BAOAB()` with one `Langevin` method on all
particles at the same `kT`.
"""
@kwdef struct StochasticCellRescaling <: Barostat
    pressure
    kT
    compressibility
    tau
    interval::Int = 10
    seed
    mode::Symbol = :isotropic
end

"""
    MonteCarloBarostat(; pressure, kT, interval=25, log_volume_step=0.01, seed, scale=:atoms, tune=false)

Isotropic Monte Carlo volume moves in log V with the `(N+1)` Jacobian, one
attempt every `interval` steps. `tune=true` adapts `log_volume_step` toward 40%
acceptance during the first 10 batches of 100 attempts and then freezes it.
Same thermal requirements as [`StochasticCellRescaling`](@ref).
"""
@kwdef struct MonteCarloBarostat <: Barostat
    pressure
    kT
    interval::Int = 25
    log_volume_step = 0.01
    seed
    scale::Symbol = :atoms
    tune::Bool = false
end

mutable struct BarostatContext{C}
    control::C
    interval::Int
    last_time::Float64
end

function _barostat_group_count(sim, ref)
    group = ref isa Symbol ? only(g for g in sim.groups if g.name === ref) : ref
    return length(_selection_indices(sim.system, group.selection))
end

function _validate_barostat_dynamics(sim, b::Barostat)
    integ = sim.integrator
    name = nameof(typeof(b))
    b.interval >= 1 || throw(ArgumentError("$(name) interval must be ≥ 1."))
    length(sim.system.box) == 3 || throw(ArgumentError("$(name) supports isotropic 3D systems only."))
    integ.scheme isa BAOAB ||
        throw(ArgumentError("$(name) currently requires `scheme=BAOAB()` (one equilibrium Langevin bath)."))
    methods = integ.methods
    (length(methods) == 1 && only(methods) isa Langevin) ||
        throw(ArgumentError("$(name) requires exactly one Langevin method covering all particles."))
    m = only(methods)
    _barostat_group_count(sim, m.group) == length(sim.system) ||
        throw(ArgumentError("$(name) requires the Langevin method to cover all particles (partial-system pressure control is not supported)."))
    (m.kT isa Real && m.gamma isa Real && m.gamma > 0) ||
        throw(ArgumentError("$(name) requires a uniform positive Langevin friction and a single bath temperature."))
    isapprox(Float64(m.kT), Float64(b.kT); rtol=1e-12) ||
        throw(ArgumentError("$(name) kT=$(b.kT) must equal the Langevin bath kT=$(m.kT)."))
    # long-range electrostatics: validated with cell changes on the device engine
    return nothing
end

function _build_barostat_control(sim, b::StochasticCellRescaling)
    b.mode === :isotropic || throw(ArgumentError("StochasticCellRescaling mode=$(b.mode) is not implemented; use :isotropic."))
    return SimulationCore.StochasticCellRescalingState(sim.state; pressure=b.pressure, kT=b.kT,
        compressibility=b.compressibility, tau=b.tau, seed=b.seed)
end

function _build_barostat_control(sim, b::MonteCarloBarostat)
    b.scale === :atoms || throw(ArgumentError("MonteCarloBarostat scale=$(b.scale) is not implemented; use :atoms."))
    return SimulationCore.MonteCarloVolumeMove(sim.state; pressure=b.pressure, kT=b.kT,
        log_volume_step=b.log_volume_step, seed=b.seed, tune_batches=(b.tune ? 10 : 0))
end

function _prepare_barostat!(sim)
    b = sim.integrator === nothing ? nothing : sim.integrator.barostat
    if b === nothing
        pop!(sim.metadata, :barostat_context, nothing)
        return nothing
    end
    b isa Barostat || throw(ArgumentError("Integrator barostat must be a Barostat; got $(typeof(b))."))
    sim.state === nothing && throw(ArgumentError("Barostats require a built simulation state."))
    old = get(sim.metadata, :barostat_context, nothing)
    if old !== nothing && get(sim.metadata, :barostat_descriptor, nothing) === b
        return nothing       # keep RNG stream and counters across repeated prepare!
    end
    _validate_barostat_dynamics(sim, b)
    control = _build_barostat_control(sim, b)
    sim.metadata[:barostat_descriptor] = b
    sim.metadata[:barostat_context] = BarostatContext(control, b.interval, _workflow_time(sim))
    return nothing
end

_barostat_event!(sim, st) = _barostat_event!(sim, st, get(sim.metadata, :barostat_context, nothing))
_barostat_event!(sim, st, ::Nothing) = nothing

function _barostat_event!(sim, st, ctx::BarostatContext{<:SimulationCore.MonteCarloVolumeMove})
    st.step % ctx.interval == 0 || return nothing
    SimulationCore.monte_carlo_volume_move!(st, ctx.control)
    ctx.last_time = _workflow_time(sim)
    return nothing
end

function _barostat_event!(sim, st, ctx::BarostatContext{<:SimulationCore.StochasticCellRescalingState})
    st.step % ctx.interval == 0 || return nothing
    t = _workflow_time(sim)
    SimulationCore.stochastic_cell_rescale!(st, ctx.control, t - ctx.last_time; compute_energy=true)
    ctx.last_time = t
    return nothing
end

"""
    barostat_diagnostics(sim) -> NamedTuple

Current volume and pressure-control counters. Monte Carlo: attempts,
acceptance ratio, current `log_volume_step`, rejected invalid trials, and the
committed volume change, potential-energy change and `P·ΔV`. C-rescale:
events, committed volume change and the last barostat pressure.
"""
function barostat_diagnostics(sim)
    ctx = get(sim.metadata, :barostat_context, nothing)
    ctx === nothing && throw(ArgumentError("No barostat is active."))
    st = sim.state
    c = ctx.control
    V = SimulationCore.cell_volume(st)
    if c isa SimulationCore.MonteCarloVolumeMove
        return (volume=V, attempts=c.attempts, acceptance=SimulationCore.acceptance_ratio(c),
                log_volume_step=c.log_volume_step, rejected_invalid=c.rejected_domain,
                volume_change=c.volume_change, energy_change=c.energy_change, pv_work=c.pv_work)
    end
    return (volume=V, events=c.events, volume_change=c.volume_change, last_pressure=c.last_pressure)
end

"""
    barostat_state_record(sim) -> Vector{Float64}

Restart record of the active barostat (written to GSD frames as
`log/barostat/state` by `GSDWriter`): the pressure-control counters plus the
time of the last event.
"""
function barostat_state_record(sim)
    ctx = get(sim.metadata, :barostat_context, nothing)
    ctx === nothing && throw(ArgumentError("No barostat is active."))
    return vcat(SimulationCore.pressure_control_record(ctx.control), ctx.last_time)
end

"""
    restore_barostat!(sim, record) -> sim

Continue a barostat stream from a [`barostat_state_record`](@ref), e.g.
`read_gsd_log(path, "log/barostat/state")` of the restart frame. Call after
`prepare!` with the same barostat descriptor (the seed is checked).
"""
function restore_barostat!(sim, record::AbstractVector{<:Real})
    sim.prepared || prepare!(sim)
    ctx = get(sim.metadata, :barostat_context, nothing)
    ctx === nothing && throw(ArgumentError("No barostat is active."))
    SimulationCore.restore_pressure_control!(ctx.control, @view record[1:end-1])
    ctx.last_time = Float64(record[end])
    return sim
end
