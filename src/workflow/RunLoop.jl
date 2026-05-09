using Random
using CUDA
using Printf: @sprintf
using ..Backends
using ..NeighborLists
using ..SimulationCore

"""
    state(sim)

Return the prepared low-level [`SimulationState`](@ref) backing a workflow
[`Simulation`](@ref).
"""
function state(sim)
    return sim.state
end

function _workflow_backend(device)
    if device === nothing
        return Backends.CUDABackend()
    elseif device isa Union{Symbol,Backends.AbstractBackend}
        return Backends.normalize_backend(device)
    else
        throw(ArgumentError("Unsupported workflow device $(typeof(device)). Use `nothing`, `:cuda`, `:cpu`, or a Backends.AbstractBackend."))
    end
end

function _workflow_precision_symbol(precision)
    if precision === :f32 || precision === Float32
        return :f32
    elseif precision === :f64 || precision === Float64
        return :f64
    else
        throw(ArgumentError("Unsupported workflow precision $(precision). Use :f32, :f64, Float32, or Float64."))
    end
end

function _workflow_state_dtype(sim)
    return _precision_type(sim.precision)
end

function _workflow_build_dt(sim, ::Type{T}) where {T<:AbstractFloat}
    if sim.integrator !== nothing && hasproperty(sim.integrator, :dt)
        return T(sim.integrator.dt)
    elseif sim.state !== nothing
        return T(sim.state.dt)
    else
        return T(get(sim.metadata, :dt, 1.0e-3))
    end
end

function _workflow_mass(system::ParticleSystem, ::Type{T}) where {T<:AbstractFloat}
    masses = system.masses
    masses === nothing && return one(T)
    if masses isa Real
        massT = T(masses)
        massT >= zero(T) || throw(ArgumentError("ParticleSystem masses must be >= 0."))
        return massT
    end
    if masses isa AbstractDict
        host = T[masses[system.types[Int(id)]] for id in system.typeids]
    elseif masses isa AbstractVector
        host = T.(collect(masses))
        length(host) == length(system) ||
            throw(ArgumentError("ParticleSystem mass vectors must have length $(length(system))."))
    else
        throw(ArgumentError("Unsupported ParticleSystem mass container $(typeof(masses))."))
    end
    isempty(host) && return one(T)
    all(mi -> mi >= zero(T), host) || throw(ArgumentError("ParticleSystem masses must be >= 0."))
    return all(mi -> isapprox(mi, host[1]; atol=zero(T), rtol=zero(T)), host) ? host[1] : host
end

function _workflow_has_explicit_nonzero_masses(system::ParticleSystem)
    masses = system.masses
    masses === nothing && return false
    if masses isa Real
        return !iszero(masses)
    elseif masses isa AbstractDict
        return any(v -> !iszero(v), values(masses))
    elseif masses isa AbstractVector
        return any(v -> !iszero(v), masses)
    end
    return true
end

function _workflow_has_explicit_nonpositive_masses(system::ParticleSystem)
    masses = system.masses
    masses === nothing && return false
    if masses isa Real
        return !(masses > 0)
    elseif masses isa AbstractDict
        return any(v -> !(v > 0), values(masses))
    elseif masses isa AbstractVector
        return any(v -> !(v > 0), masses)
    end
    return true
end

function _workflow_validate_mass_model!(sim)
    sim.integrator === nothing && return nothing
    methods = hasproperty(sim.integrator, :methods) ? sim.integrator.methods : Method[]
    isempty(methods) && return nothing
    scheme = _normalize_scheme(sim.integrator.scheme, methods)
    if all(method -> method isa ConstantVolume, methods) || _scheme_family(scheme) == :langevin
        _workflow_has_explicit_nonpositive_masses(sim.system) ||
            return nothing
        throw(ArgumentError("Inertial MD and Langevin dynamics require strictly positive particle masses. Zero masses are only supported for Brownian dynamics, where masses are ignored."))
    end
    get(sim.metadata, :workflow_warned_brownian_masses_ignored, false) && return nothing
    all(method -> method isa Union{Brownian,ActiveOrnsteinUhlenbeck}, methods) || return nothing
    _workflow_has_explicit_nonzero_masses(sim.system) || return nothing
    @warn "Brownian dynamics ignores particle masses; nonzero `ParticleSystem.masses` will not affect the simulation. Set masses to zero if you want that metadata to reflect the overdamped model."
    sim.metadata[:workflow_warned_brownian_masses_ignored] = true
    return nothing
end

function _workflow_seed_temperature(sim, ::Type{T}) where {T<:AbstractFloat}
    sim.system.velocities === nothing || return zero(T)
    sim.integrator === nothing && return zero(T)
    methods = hasproperty(sim.integrator, :methods) ? sim.integrator.methods : Method[]
    temps = T[]
    for method in methods
        if method isa Union{Langevin,Brownian,ActiveOrnsteinUhlenbeck}
            push!(temps, T(method.kT))
        elseif method isa ConstantVolume && method.thermostat !== nothing
            push!(temps, T(method.thermostat.kT))
        end
    end
    return isempty(temps) ? zero(T) : Base.sum(temps) / T(length(temps))
end

function _workflow_needs_unwrapped_positions(sim)
    for obs in sim.observables
        obs isa MSDObservable && return true
    end
    for writer in sim.writers
        writer isa GSDWriter && writer.write_unwrapped && return true
    end
    return false
end

function _default_build_force_kwargs(::Type{T}) where {T<:AbstractFloat}
    return Dict{Symbol,Any}(
        :nonbonded => :wca,
        :epsilon => zero(T),
        :sigma => one(T),
        :cutoff => T(SimulationCore.WCA_RC_FACTOR),
        :use_neighborlist => false,
    )
end

function _copy_system_into_state!(st::SimulationState, system::ParticleSystem)
    T = eltype(st.rx)
    D = length(system.box)
    copyto!(st.rx, T[p[1] for p in system.positions])
    copyto!(st.ry, T[p[2] for p in system.positions])
    if D == 3
        st.rz !== nothing || throw(ArgumentError("3D ParticleSystem requires a 3D SimulationState."))
        copyto!(st.rz, T[p[3] for p in system.positions])
    end

    copyto!(st.typeid, Int32.(system.typeids))

    if system.velocities !== nothing
        copyto!(st.vx, T[v[1] for v in system.velocities])
        copyto!(st.vy, T[v[2] for v in system.velocities])
        if D == 3
            st.vz !== nothing || throw(ArgumentError("3D ParticleSystem velocities require a 3D SimulationState."))
            copyto!(st.vz, T[v[3] for v in system.velocities])
        end
    end

    if haskey(system.metadata, :step)
        st.step = Int(system.metadata[:step])
    end
    return st
end

function _refresh_neighbor_state!(st::SimulationState)
    if st.rz === nothing
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
    else
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    end
    return st
end

function _warn_if_neighbor_capacity_saturated!(sim)
    get(sim.metadata, :workflow_neighbor_capacity_warned, false) && return nothing
    st = sim.state
    st === nothing && return nothing
    nbh = st.nbh
    nbh isa NeighborLists.AllPairsNeighborMatrix && return nothing
    max_count = Int(maximum(nbh.counts))
    cap = Int(nbh.cap)
    max_count < cap && return nothing
    style = nbh isa NeighborLists.StencilNeighborMatrix ? "typed stencil" : "cell-list"
    @warn "Neighbor list reached the configured per-particle capacity during the initial build. This can truncate neighbors and yield incorrect forces; increase `CellList.capacity`." style cap max_count
    sim.metadata[:workflow_neighbor_capacity_warned] = true
    return nothing
end

function _validate_writer_state_requirements(sim)
    st = sim.state
    st === nothing && return sim
    for writer in sim.writers
        if writer isa GSDWriter && writer.write_unwrapped && st.rx_unwrap === nothing
            throw(ArgumentError("GSDWriter(write_unwrapped=true) requires unwrapped position buffers. Build the workflow Simulation without a prebuilt state, or provide a low-level state built with `unwrapped_positions=true`."))
        end
    end
    return sim
end

function _apply_workflow_seed!(seed)
    seed === nothing && return nothing
    Random.seed!(seed)
    CUDA.seed!(UInt64(seed))
    return nothing
end

function _build_workflow_state!(sim, compiled_forces::Union{Nothing,CompiledForces})
    T = _workflow_state_dtype(sim)
    backend = _workflow_backend(sim.device)
    dtT = _workflow_build_dt(sim, T)
    kwargs = Dict{Symbol,Any}(
        :N => length(sim.system),
        :box => Tuple(sim.system.box),
        :gamma => one(T),
        :temperature => _workflow_seed_temperature(sim, T),
        :dt => dtT,
        :mass => _workflow_mass(sim.system, T),
        :backend => backend,
        :precision => _workflow_precision_symbol(sim.precision),
        :unwrapped_positions => _workflow_needs_unwrapped_positions(sim),
    )
    merge!(kwargs, compiled_forces === nothing ? _default_build_force_kwargs(T) : compiled_forces.build_kwargs)

    st = SimulationCore.build_simulation(; kwargs...)
    _copy_system_into_state!(st, sim.system)
    compiled_forces === nothing || post_build!(compiled_forces, st)
    if st.rx_unwrap !== nothing
        bonds = sim.system.topology.bonds
        if isempty(bonds)
            SimulationCore.sync_unwrapped!(st)
        else
            SimulationCore.initialize_unwrapped_from_bonds!(st, bonds)
        end
    end
    _refresh_neighbor_state!(st)
    sim.state = st
    sim.metadata[:workflow_backend] = backend
    sim.metadata[:workflow_precision] = _workflow_precision_symbol(sim.precision)
    sim.metadata[:workflow_time] = Float64(st.step) * Float64(st.dt)
    _warn_if_neighbor_capacity_saturated!(sim)
    return st
end

"""
    prepare!(sim)

Prepare a workflow [`Simulation`](@ref) by building or validating its
low-level state, materializing groups, compiling workflow forces/integrators,
and preparing observable and writer contexts.
"""
function prepare!(sim)
    _apply_workflow_seed!(sim.seed)
    _workflow_validate_mass_model!(sim)
    built_state = false

    compiled_forces = nothing
    if sim.integrator !== nothing && hasproperty(sim.integrator, :forces) && !isempty(sim.integrator.forces)
        compiled_forces = compile_forces(sim.system, sim.integrator.forces; precision=sim.precision)
        sim.metadata[:compiled_forces] = compiled_forces
    else
        pop!(sim.metadata, :compiled_forces, nothing)
    end
    if sim.integrator !== nothing && hasproperty(sim.integrator, :methods) && !isempty(sim.integrator.methods)
        sim.metadata[:compiled_integrator] = compile_integrator(sim.system, sim.integrator; precision=sim.precision)
    else
        pop!(sim.metadata, :compiled_integrator, nothing)
    end

    if sim.state === nothing
        _build_workflow_state!(sim, compiled_forces)
        built_state = true
    elseif !haskey(sim.metadata, :workflow_time)
        sim.metadata[:workflow_time] = Float64(sim.state.step) * _current_dt(sim)
    end

    materialized = Dict{Symbol,Any}()
    for group in sim.groups
        materialized[group.name] = materialize_group(sim.system, group)
    end
    sim.metadata[:materialized_groups] = materialized

    if sim.state !== nothing
        if haskey(sim.metadata, :compiled_forces) && !built_state
            post_build!(sim.metadata[:compiled_forces], sim.state)
            _warn_if_neighbor_capacity_saturated!(sim)
        end
        if haskey(sim.metadata, :compiled_integrator)
            sim.lowlevel_integrator = build_lowlevel_integrator(sim.metadata[:compiled_integrator],
                                                                sim.state;
                                                                system=sim.system,
                                                                materialized_groups=materialized)
        end
    end
    _validate_writer_state_requirements(sim)
    prepare_observables!(sim)
    prepare_writers!(sim)
    sim.prepared = true
    return sim
end

"""
    reset_step!(sim, step=0)

Reset the visible workflow step counter.
"""
function reset_step!(sim, step::Integer=0)
    if sim.state !== nothing
        sim.state.step = Int(step)
        sim.metadata[:workflow_time] = Float64(step) * _current_dt(sim)
        sim.metadata[:workflow_interval_reference_step] = sim.state.step
        sim.metadata[:workflow_interval_reference_time] = sim.metadata[:workflow_time]
    end
    return sim
end

function _effective_dt(sim, stage::Stage)
    return stage.dt === nothing ? _current_dt(sim) : Float64(stage.dt)
end

function _stage_compute_energy(sim, stage::Stage, next_step::Int)
    if stage.compute_energy === :auto
        return active_writer_requires_energy(sim, next_step)
    elseif stage.compute_energy isa Bool
        return stage.compute_energy
    else
        throw(ArgumentError("Stage.compute_energy must be :auto or Bool; got $(stage.compute_energy)."))
    end
end

function _build_stage_integrator(sim, stage::Stage)
    if stage.dt === nothing
        return sim.lowlevel_integrator, _effective_dt(sim, stage), false
    end
    sim.integrator !== nothing || throw(ArgumentError("Stage dt override requires a workflow Integrator."))
    haskey(sim.metadata, :compiled_integrator) || throw(ArgumentError("Stage dt override requires prepared workflow integrator metadata."))
    stage_integrator = Integrator(
        dt=stage.dt,
        scheme=sim.integrator.scheme,
        forces=sim.integrator.forces,
        methods=sim.integrator.methods,
        metadata=copy(sim.integrator.metadata),
    )
    compiled = compile_integrator(sim.system, stage_integrator; precision=sim.precision)
    spec = build_lowlevel_integrator(compiled,
                                     sim.state;
                                     system=sim.system,
                                     materialized_groups=sim.metadata[:materialized_groups])
    return spec, Float64(stage.dt), true
end

function _maybe_print_stage_banner(stage::Stage, dt::Float64)
    if stage.progress && stage.steps >= 1000
        println("Running stage $(stage.name) for $(stage.steps) steps (dt=$(dt)).")
    end
end

function _maybe_print_stage_end(stage::Stage, executed::Int, elapsed_s::Float64)
    if stage.progress && stage.steps >= 1000
        steps_per_second = elapsed_s > 0 ? executed / elapsed_s : Inf
        println("Finished stage $(stage.name) after $(executed) step(s) in $(@sprintf("%.3f", elapsed_s)) s ($(@sprintf("%.2f", steps_per_second)) steps/s).")
    end
end

function _maybe_print_stage_interval(stage::Stage, current_step::Int, total_steps::Int, interval_steps::Int, interval_elapsed_s::Float64)
    stage.progress || return nothing
    steps_per_second = interval_elapsed_s > 0 ? interval_steps / interval_elapsed_s : Inf
    println("Stage $(stage.name): step $(current_step)/$(total_steps) ($(@sprintf("%.2f", 100 * current_step / max(total_steps, 1)))%) at $(@sprintf("%.2f", steps_per_second)) steps/s.")
    return nothing
end

"""
    run!(sim, nsteps)
    run!(sim, stage::Stage)
    run!(sim, stages::AbstractVector{<:Stage})

Run a workflow [`Simulation`](@ref). `run!` owns the timestep loop, prepares
the simulation on first use, and drives scheduled writers automatically.
"""
function run!(sim, nsteps::Integer)
    return run!(sim, Stage(:run, steps=Int(nsteps); progress=false))
end

function run!(sim, stages::AbstractVector{<:Stage})
    for stage in stages
        run!(sim, stage)
    end
    return sim
end

function run!(sim, stage::Stage)
    sim.prepared || prepare!(sim)
    sim.state === nothing && throw(ArgumentError("Simulation state is not initialized. Stage 10 has not built a low-level state yet; pass `state=` explicitly for now."))
    sim.lowlevel_integrator === nothing && throw(ArgumentError("Low-level integrator is not initialized. Provide workflow methods or a prepared low-level integrator."))

    st = sim.state
    original_neigh_interval = st.neigh_interval
    original_integrator = sim.lowlevel_integrator
    executed = 0

    stage.reset_observables && reset_observables!(sim)
    stage.reset_step === nothing || reset_step!(sim, stage.reset_step)

    local_spec, local_dt, overridden_spec = _build_stage_integrator(sim, stage)
    stage.reset_step === nothing || (sim.metadata[:workflow_time] = Float64(stage.reset_step) * local_dt)
    stage.neighbor_rebuild_interval === nothing || (st.neigh_interval = Int(stage.neighbor_rebuild_interval))
    sim.lowlevel_integrator = local_spec

    _maybe_print_stage_banner(stage, local_dt)
    write_initial_frames!(sim)

    t0_ns = time_ns()
    last_report_ns = t0_ns
    last_report_step = st.step
    try
        for _ in 1:stage.steps
            next_step = st.step + 1
            compute_energy = _stage_compute_energy(sim, stage, next_step)
            step!(st, local_spec, local_dt; compute_energy=compute_energy)
            sim.metadata[:workflow_time] = _workflow_time(sim) + local_dt
            executed += 1

            output_status = write_scheduled_outputs!(sim, st.step)
            output_status.interval_consumed && _reset_interval_buffers!(sim)
            if output_status.wrote_any
                interval_elapsed_s = (time_ns() - last_report_ns) / 1.0e9
                interval_steps = st.step - last_report_step
                _maybe_print_stage_interval(stage, executed, stage.steps, interval_steps, interval_elapsed_s)
                last_report_ns = time_ns()
                last_report_step = st.step
            end

            if (time_ns() - t0_ns) / 1.0e9 >= stage.max_seconds
                break
            end
        end
    finally
        st.neigh_interval = original_neigh_interval
        sim.lowlevel_integrator = overridden_spec ? original_integrator : local_spec
        close_writers!(sim)
    end

    elapsed_s = (time_ns() - t0_ns) / 1.0e9
    _maybe_print_stage_end(stage, executed, elapsed_s)
    return sim
end
