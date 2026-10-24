using CUDA
using ..SimulationCore
using ..Collisions

"""
    Observable

Abstract workflow observable descriptor.
"""
abstract type Observable end

"""
    ThermodynamicObservable(group; name=:thermo)

Compute thermodynamic quantities for a workflow group.
"""
@kwdef struct ThermodynamicObservable <: Observable
    group
    name::Symbol = :thermo
end
ThermodynamicObservable(group; name::Symbol=:thermo) = ThermodynamicObservable(group=group, name=name)

"""
    BathExchangeObservable(; name=:bath)

Compute bath heat and entropy-exchange diagnostics.
"""
@kwdef struct BathExchangeObservable <: Observable
    name::Symbol = :bath
end

"""
    VirialObservable(group; name=:virial)

Compute scalar virial diagnostics for a workflow group.
"""
@kwdef struct VirialObservable <: Observable
    group
    name::Symbol = :virial
end
VirialObservable(group; name::Symbol=:virial) = VirialObservable(group=group, name=name)

"""
    CollisionObservable(; name=:collisions)

Compute collision-count diagnostics from the low-level collision counters.
"""
@kwdef struct CollisionObservable <: Observable
    name::Symbol = :collisions
end

"""
    MSDObservable(group; name=:msd, reference=:start)

Mean-squared-displacement observable for a workflow group.
"""
@kwdef struct MSDObservable <: Observable
    group
    name::Symbol = :msd
    reference = :start
end
MSDObservable(group; name::Symbol=:msd, reference=:start) = MSDObservable(group=group, name=name, reference=reference)

"""
    VACFObservable(group; name=:vacf, reference=:start)

Velocity-autocorrelation observable for a workflow group.
"""
@kwdef struct VACFObservable <: Observable
    group
    name::Symbol = :vacf
    reference = :start
end
VACFObservable(group; name::Symbol=:vacf, reference=:start) = VACFObservable(group=group, name=name, reference=reference)

observable_name(obs::Observable) = getfield(obs, :name)

observable_default_fields(::ThermodynamicObservable) = (
    :temperature,
    :kinetic_energy,
    :potential_energy,
    :total_energy,
    :virial,
    :kinetic_energy_accumulated,
    :potential_energy_accumulated,
    :virial_accumulated,
)
observable_default_fields(::BathExchangeObservable) = (
    :heat,
    :entropy,
    :entropy_production_rate,
    :temperature_error,
    :extended_hamiltonian,
)
observable_default_fields(::VirialObservable) = (:virial, :virial_accumulated)
observable_default_fields(::CollisionObservable) = (:counts, :rate, :pair_counts, :pair_rates)
observable_default_fields(::MSDObservable) = (:msd, :elapsed_time, :elapsed_steps)
observable_default_fields(::VACFObservable) = (:vacf, :elapsed_time, :elapsed_steps)

function observable_requires_energy(obs::Observable, fields::AbstractVector{Symbol})
    return any(field -> _observable_field_requires_energy(obs, field), fields)
end

_observable_field_requires_energy(::ThermodynamicObservable, ::Symbol) = true
_observable_field_requires_energy(::VirialObservable, ::Symbol) = true
_observable_field_requires_energy(::BathExchangeObservable, field::Symbol) =
    field in (:extended_hamiltonian, :temperature_error, :thermostat_kinetic, :thermostat_potential)
_observable_field_requires_energy(::CollisionObservable, ::Symbol) = false
_observable_field_requires_energy(::MSDObservable, ::Symbol) = false
_observable_field_requires_energy(::VACFObservable, ::Symbol) = false

function observable_has_interval_fields(obs::Observable, fields::AbstractVector{Symbol})
    return any(field -> _observable_field_is_interval(obs, field), fields)
end

_observable_field_is_interval(::ThermodynamicObservable, field::Symbol) =
    field in (:kinetic_energy_accumulated, :potential_energy_accumulated, :virial_accumulated)
_observable_field_is_interval(::BathExchangeObservable, field::Symbol) =
    field in (:heat, :entropy, :entropy_production_rate, :temperature_error, :extended_hamiltonian,
              :thermostat_kinetic, :thermostat_potential)
_observable_field_is_interval(::VirialObservable, field::Symbol) =
    field in (:virial_accumulated,)
_observable_field_is_interval(::CollisionObservable, ::Symbol) = true
_observable_field_is_interval(::MSDObservable, ::Symbol) = false
_observable_field_is_interval(::VACFObservable, ::Symbol) = false

_observable_key(obs::Observable) = (
    Symbol(nameof(typeof(obs))),
    observable_name(obs),
    hasproperty(obs, :group) ? _workflow_group_key(getfield(obs, :group)) : nothing,
)

_workflow_group_key(group::Group) = (group.domain, group.name)
_workflow_group_key(group::Symbol) = (:particles, group)

function _ensure_observable_registry!(sim)
    return get!(sim.metadata, :workflow_observables) do
        Dict{Any,Dict{Symbol,Any}}()
    end
end

_workflow_time(sim) = get(sim.metadata, :workflow_time, sim.state === nothing ? 0.0 : Float64(sim.state.step) * _current_dt(sim))

function _set_interval_reference!(sim)
    sim.metadata[:workflow_interval_reference_step] = sim.state === nothing ? 0 : sim.state.step
    sim.metadata[:workflow_interval_reference_time] = _workflow_time(sim)
    return sim
end

function _interval_elapsed(sim)
    dt = _current_dt(sim)
    current_time = _workflow_time(sim)
    reference_time = get(sim.metadata, :workflow_interval_reference_time, current_time - dt)
    elapsed = current_time - reference_time
    return elapsed > 0 ? elapsed : dt
end

function _current_dt(sim)
    if sim.integrator !== nothing && hasproperty(sim.integrator, :dt)
        return Float64(sim.integrator.dt)
    elseif sim.state !== nothing
        return Float64(sim.state.dt)
    else
        return 0.0
    end
end

_is_3d_state(st::SimulationState) = st.rz !== nothing
_state_dimension(st::SimulationState) = _is_3d_state(st) ? 3 : 2

function _position_views(st::SimulationState)
    rx = st.rx_unwrap === nothing ? st.rx : st.rx_unwrap
    ry = st.ry_unwrap === nothing ? st.ry : st.ry_unwrap
    rz = if st.rz === nothing
        nothing
    elseif st.rz_unwrap === nothing
        st.rz
    else
        st.rz_unwrap
    end
    return rx, ry, rz
end

function _group_filter(sim, ref)
    materialized = get(sim.metadata, :materialized_groups, nothing)
    materialized === nothing && throw(ArgumentError("Workflow groups are not prepared. Call prepare!(sim) first."))
    if ref isa Group
        return get(materialized, ref.name) do
            materialize_group(sim.system, ref)
        end
    elseif ref isa Symbol
        return get(materialized, ref) do
            throw(ArgumentError("Unknown workflow group $(ref)."))
        end
    else
        throw(ArgumentError("Workflow observables require a Group or Symbol group reference; got $(typeof(ref))."))
    end
end

function _group_count(sim, filter)
    st = sim.state::SimulationState
    return Filters.count(st, filter)
end

function _mean_group_value(src, sim, filter)
    count = _group_count(sim, filter)
    count == 0 && return zero(eltype(src))
    return Filters.sum(src, sim.state, filter) / count
end

function _ensure_observable_context!(sim, obs::Observable)
    registry = _ensure_observable_registry!(sim)
    key = _observable_key(obs)
    ctx = get!(registry, key) do
        Dict{Symbol,Any}(:observable => obs)
    end
    if sim.state !== nothing
        if obs isa Union{MSDObservable,VACFObservable}
            if !haskey(ctx, :reference_step)
                _refresh_reference_observable!(sim, obs, ctx)
            end
        elseif obs isa CollisionObservable
            st = sim.state
            if !st.coll_enabled
                Collisions.enable_collision_counting!(st; ntypes=length(sim.system.types))
            end
            compiled = get(sim.metadata, :compiled_forces, nothing)
            if compiled isa CompiledForces && haskey(compiled.metadata, :cutoff_pair)
                Collisions.set_collision_pair_cutoffs!(st, compiled.metadata[:cutoff_pair])
            end
        end
    end
    return ctx
end

function prepare_observables!(sim, observables=sim.observables)
    for obs in observables
        _ensure_observable_context!(sim, obs)
    end
    _set_interval_reference!(sim)
    return sim
end

# The engine reorders particle storage (slot k holds particle `st.tag[k]`), so
# references kept across steps are stored in particle-id order and gathered back
# into the current slot order when evaluated.
_by_particle_id(st, x) = x === nothing ? nothing : (st.tag === nothing ? copy(x) : (y = similar(x); y[st.tag] = x; y))
_in_slot_order(st, ref) = ref === nothing ? nothing : (st.tag === nothing ? ref : ref[st.tag])

function _refresh_reference_observable!(sim, obs::Observable, ctx::Dict{Symbol,Any})
    st = sim.state
    st === nothing && return ctx
    rx, ry, rz = _position_views(st)
    if obs isa MSDObservable
        ctx[:reference_step] = st.step
        ctx[:reference_time] = _workflow_time(sim)
        ctx[:reference_rx] = _by_particle_id(st, rx)
        ctx[:reference_ry] = _by_particle_id(st, ry)
        ctx[:reference_rz] = _by_particle_id(st, rz)
    elseif obs isa VACFObservable
        ctx[:reference_step] = st.step
        ctx[:reference_time] = _workflow_time(sim)
        ctx[:reference_vx] = _by_particle_id(st, st.vx)
        ctx[:reference_vy] = _by_particle_id(st, st.vy)
        ctx[:reference_vz] = _by_particle_id(st, st.vz)
    end
    return ctx
end

function _reset_interval_buffers!(sim)
    st = sim.state
    st === nothing && return sim
    T = eltype(st.rx)
    fill!(st.Epot_accum, zero(T))
    fill!(st.Ekin_accum, zero(T))
    fill!(st.virial_accum, zero(T))
    fill!(st.virial_tensor_accum, zero(T))
    fill!(st.virial_global_accum, zero(T))
    st.rigid_water === nothing || fill!(st.rigid_water.virial_accum, zero(T))
    if sim.lowlevel_integrator !== nothing
        SimulationCore.reset_bath_exchange_accumulators!(st, sim.lowlevel_integrator)
    else
        fill!(st.dq, zero(T))
        fill!(st.dU, zero(T))
    end
    if st.coll_enabled
        Collisions.collisions_reset_counts!(st)
    end
    _set_interval_reference!(sim)
    return sim
end

function reset_observables!(sim)
    _reset_interval_buffers!(sim)
    registry = get(sim.metadata, :workflow_observables, Dict{Any,Dict{Symbol,Any}}())
    for ctx in values(registry)
        obs = ctx[:observable]
        if obs isa Union{MSDObservable,VACFObservable}
            _refresh_reference_observable!(sim, obs, ctx)
        end
    end
    return sim
end

function _thermodynamic_data(sim, obs::ThermodynamicObservable)
    st = sim.state::SimulationState
    if sim.lowlevel_integrator !== nothing
        SimulationCore._ensure_sample_buffers!(st, sim.lowlevel_integrator)
    end
    filter = _group_filter(sim, obs.group)
    dim = _state_dimension(st)
    n = _group_count(sim, filter)
    kinetic = Filters.sum(st.Ekin, st, filter)
    potential = Filters.sum(st.Epot, st, filter)
    virial = _group_virial(st, filter, n, false)
    dof = dim * n
    if st.rigid_water !== nothing
        n == length(st.rx) || throw(ArgumentError("Rigid-water temperature requires a whole-system group"))
        dof = SimulationCore.kinetic_dof(st)
    end
    temperature = dof == 0 ? zero(eltype(st.rx)) : (2 * kinetic / dof)
    kinetic_acc = Filters.sum(st.Ekin_accum, st, filter)
    potential_acc = Filters.sum(st.Epot_accum, st, filter)
    virial_acc = _group_virial(st, filter, n, true)
    return (
        temperature = temperature,
        kinetic_energy = kinetic,
        potential_energy = potential,
        total_energy = kinetic + potential,
        virial = virial,
        kinetic_energy_accumulated = kinetic_acc,
        potential_energy_accumulated = potential_acc,
        virial_accumulated = virial_acc,
    )
end

function _bath_exchange_data(sim, ::BathExchangeObservable)
    st = sim.state
    st === nothing && throw(ArgumentError("BathExchangeObservable requires a prepared SimulationState."))
    spec = sim.lowlevel_integrator
    spec === nothing && throw(ArgumentError("BathExchangeObservable requires a prepared low-level integrator."))
    obs = SimulationCore.collect_step_observables(st, spec)
    elapsed = _interval_elapsed(sim)
    heat = hasproperty(obs, :bath_heat_total) ? obs.bath_heat_total : zero(eltype(st.rx))
    entropy = hasproperty(obs, :bath_entropy_total) ? obs.bath_entropy_total : zero(eltype(st.rx))
    return (
        heat = heat,
        entropy = entropy,
        entropy_production_rate = elapsed > 0 ? entropy / elapsed : zero(eltype(st.rx)),
        temperature_error = hasproperty(obs, :thermostat_temperature_error) ? obs.thermostat_temperature_error : zero(eltype(st.rx)),
        extended_hamiltonian = hasproperty(obs, :extended_hamiltonian) ? obs.extended_hamiltonian : obs.Etot,
        thermostat_kinetic = hasproperty(obs, :thermostat_kinetic) ? obs.thermostat_kinetic : zero(eltype(st.rx)),
        thermostat_potential = hasproperty(obs, :thermostat_potential) ? obs.thermostat_potential : zero(eltype(st.rx)),
    )
end

function _virial_data(sim, obs::VirialObservable)
    st = sim.state::SimulationState
    filter = _group_filter(sim, obs.group)
    n = _group_count(sim, filter)
    return (
        virial = _group_virial(st, filter, n, false),
        virial_accumulated = _group_virial(st, filter, n, true),
    )
end

# Per-particle virial sum of a group. The global (reciprocal electrostatic)
# virial has no per-particle decomposition: it is added for groups covering
# every particle, and the value is NaN for strict subgroups (requests for it
# are rejected in `sample_observable`).
function _group_virial(st::SimulationState, filter, n::Integer, accumulated::Bool)
    local_sum = Filters.sum(accumulated ? st.virial_accum : st.virial, st, filter)
    if st.rigid_water !== nothing
        n == length(st.rx) || return oftype(local_sum, NaN)
        local_sum += SimulationCore.constraint_virial_trace(st; accumulated)
    end
    SimulationCore.has_global_virial(st) || return local_sum
    n == length(st.rx) || return oftype(local_sum, NaN)
    return local_sum + SimulationCore.global_virial_trace(st; accumulated=accumulated)
end

# Energies/virials sampled from stale buffers would silently describe earlier
# coordinates; refresh them for the current configuration first.
function _refresh_energies_for!(sim, obs::Observable, fields)
    st = sim.state::SimulationState
    st.energy_valid && return nothing
    observable_requires_energy(obs, collect(Symbol, fields)) || return nothing
    obs isa BathExchangeObservable && return nothing     # integrator-owned accounting
    SimulationCore.ensure_energies!(st)
    return nothing
end

_observable_field_uses_virial(::ThermodynamicObservable, field::Symbol) = field in (:virial, :virial_accumulated)
_observable_field_uses_virial(::VirialObservable, ::Symbol) = true
_observable_field_uses_virial(::Observable, ::Symbol) = false

function _reject_subgroup_global_virial(sim, obs::Observable, fields)
    st = sim.state::SimulationState
    (SimulationCore.has_global_virial(st) || st.rigid_water !== nothing) || return nothing
    hasproperty(obs, :group) || return nothing
    any(field -> _observable_field_uses_virial(obs, field), fields) || return nothing
    filter = _group_filter(sim, getfield(obs, :group))
    _group_count(sim, filter) == length(st.rx) && return nothing
    throw(ArgumentError("Observable $(observable_name(obs)) requests a subgroup virial, but electrostatic or rigid-water virials require a whole-system group. Use all particles or omit the virial fields."))
end

function _collision_pair_labels(sim)
    labels = Symbol[]
    for i in eachindex(sim.system.types)
        for j in i:length(sim.system.types)
            push!(labels, Symbol(string(sim.system.types[i], "_", sim.system.types[j])))
        end
    end
    return labels
end

function _collision_data(sim, ::CollisionObservable)
    st = sim.state
    st === nothing && throw(ArgumentError("CollisionObservable requires a prepared SimulationState."))
    counts = Collisions.collisions_read_counts!(st)
    labels = _collision_pair_labels(sim)
    named = Dict{Symbol,Int64}()
    rate_named = Dict{Symbol,Float64}()
    elapsed = _interval_elapsed(sim)
    for (idx, label) in pairs(labels)
        count = idx <= length(counts) ? counts[idx] : Int64(0)
        named[label] = count
        rate_named[label] = count / elapsed
    end
    return (
        counts = Base.sum(counts),
        rate = Base.sum(counts) / elapsed,
        pair_counts = named,
        pair_rates = rate_named,
    )
end

function _msd_data(sim, obs::MSDObservable)
    st = sim.state
    st === nothing && throw(ArgumentError("MSDObservable requires a prepared SimulationState."))
    ctx = _ensure_observable_context!(sim, obs)
    filter = _group_filter(sim, obs.group)
    count = _group_count(sim, filter)
    count == 0 && throw(ArgumentError("MSDObservable group $(observable_name(obs)) selects no particles."))
    rx, ry, rz = _position_views(st)
    x0, y0, z0 = (_in_slot_order(st, ctx[k]) for k in (:reference_rx, :reference_ry, :reference_rz))
    dr2 = if rz === nothing
        (rx .- x0).^2 .+ (ry .- y0).^2
    else
        (rx .- x0).^2 .+ (ry .- y0).^2 .+ (rz .- z0).^2
    end
    elapsed_steps = st.step - get(ctx, :reference_step, st.step)
    elapsed_time = _workflow_time(sim) - get(ctx, :reference_time, _workflow_time(sim))
    return (
        msd = Filters.sum(dr2, st, filter) / count,
        elapsed_time = elapsed_time,
        elapsed_steps = elapsed_steps,
    )
end

function _vacf_data(sim, obs::VACFObservable)
    st = sim.state
    st === nothing && throw(ArgumentError("VACFObservable requires a prepared SimulationState."))
    ctx = _ensure_observable_context!(sim, obs)
    filter = _group_filter(sim, obs.group)
    count = _group_count(sim, filter)
    count == 0 && throw(ArgumentError("VACFObservable group $(observable_name(obs)) selects no particles."))
    vx0, vy0, vz0 = (_in_slot_order(st, ctx[k]) for k in (:reference_vx, :reference_vy, :reference_vz))
    corr = if st.vz === nothing
        st.vx .* vx0 .+ st.vy .* vy0
    else
        st.vx .* vx0 .+ st.vy .* vy0 .+ st.vz .* vz0
    end
    elapsed_steps = st.step - get(ctx, :reference_step, st.step)
    elapsed_time = _workflow_time(sim) - get(ctx, :reference_time, _workflow_time(sim))
    return (
        vacf = Filters.sum(corr, st, filter) / count,
        elapsed_time = elapsed_time,
        elapsed_steps = elapsed_steps,
    )
end

_observable_data(sim, obs::ThermodynamicObservable) = _thermodynamic_data(sim, obs)
_observable_data(sim, obs::BathExchangeObservable) = _bath_exchange_data(sim, obs)
_observable_data(sim, obs::VirialObservable) = _virial_data(sim, obs)
_observable_data(sim, obs::CollisionObservable) = _collision_data(sim, obs)
_observable_data(sim, obs::MSDObservable) = _msd_data(sim, obs)
_observable_data(sim, obs::VACFObservable) = _vacf_data(sim, obs)

function _normalize_observable_fields(obs::Observable, fields)
    if fields === nothing
        return collect(observable_default_fields(obs))
    end
    host = Symbol.(collect(fields))
    isempty(host) && throw(ArgumentError("Observable field list must not be empty."))
    return host
end

function sample_observable(sim, obs::Observable; fields=nothing)
    sim.state === nothing && throw(ArgumentError("Workflow observables require a prepared SimulationState."))
    selected = _normalize_observable_fields(obs, fields)
    _reject_subgroup_global_virial(sim, obs, selected)
    _refresh_energies_for!(sim, obs, selected)
    data = _observable_data(sim, obs)
    values = map(field -> begin
        hasproperty(data, field) || throw(ArgumentError("Observable $(observable_name(obs)) does not provide field $(field)."))
        getproperty(data, field)
    end, selected)
    return NamedTuple{Tuple(selected)}(values)
end
