function state(sim)
    return sim.state
end

function prepare!(sim)
    if sim.state !== nothing && !haskey(sim.metadata, :workflow_time)
        sim.metadata[:workflow_time] = Float64(sim.state.step) * _current_dt(sim)
    end
    materialized = Dict{Symbol,Any}()
    for group in sim.groups
        materialized[group.name] = materialize_group(sim.system, group)
    end
    sim.metadata[:materialized_groups] = materialized
    if sim.integrator !== nothing && hasproperty(sim.integrator, :forces) && !isempty(sim.integrator.forces)
        sim.metadata[:compiled_forces] = compile_forces(sim.system, sim.integrator.forces; precision=sim.precision)
    end
    if sim.integrator !== nothing && hasproperty(sim.integrator, :methods) && !isempty(sim.integrator.methods)
        sim.metadata[:compiled_integrator] = compile_integrator(sim.system, sim.integrator; precision=sim.precision)
    end
    if sim.state !== nothing
        if haskey(sim.metadata, :compiled_forces)
            post_build!(sim.metadata[:compiled_forces], sim.state)
        end
        if haskey(sim.metadata, :compiled_integrator)
            sim.lowlevel_integrator = build_lowlevel_integrator(sim.metadata[:compiled_integrator],
                                                                sim.state;
                                                                system=sim.system,
                                                                materialized_groups=materialized)
        end
    end
    prepare_observables!(sim)
    prepare_writers!(sim)
    sim.prepared = true
    return sim
end

function reset_step!(sim, step::Integer=0)
    if sim.state !== nothing
        sim.state.step = Int(step)
        sim.metadata[:workflow_time] = Float64(step) * _current_dt(sim)
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

function _maybe_print_stage_end(stage::Stage, executed::Int)
    if stage.progress && stage.steps >= 1000
        println("Finished stage $(stage.name) after $(executed) step(s).")
    end
end

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

    t0 = time()
    try
        for _ in 1:stage.steps
            next_step = st.step + 1
            compute_energy = _stage_compute_energy(sim, stage, next_step)
            step!(st, local_spec, local_dt; compute_energy=compute_energy)
            sim.metadata[:workflow_time] = _workflow_time(sim) + local_dt
            executed += 1

            interval_consumed = write_scheduled_outputs!(sim, st.step)
            interval_consumed && _reset_interval_buffers!(sim)

            if time() - t0 >= stage.max_seconds
                break
            end
        end
    finally
        st.neigh_interval = original_neigh_interval
        sim.lowlevel_integrator = overridden_spec ? original_integrator : local_spec
        close_writers!(sim)
    end

    _maybe_print_stage_end(stage, executed)
    return sim
end
