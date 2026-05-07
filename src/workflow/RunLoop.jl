function state(sim::Simulation)
    return sim.state
end

function prepare!(sim::Simulation)
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
    sim.prepared = true
    return sim
end

function reset_step!(sim::Simulation, step::Integer=0)
    if sim.state !== nothing
        sim.state.step = Int(step)
    end
    return sim
end

function run!(sim::Simulation, nsteps::Integer; 
              every::Integer=1,
              verbose::Bool=false)
    !sim.prepared && throw(ArgumentError("Simulation must be prepared before running. Call prepare!(sim) first."))
    sim.state === nothing && throw(ArgumentError("Simulation state is not initialized."))
    sim.lowlevel_integrator === nothing && throw(ArgumentError("Lowlevel integrator is not initialized."))
    
    # Run the integration loop
    for step in 1:nsteps
        step!(sim.state, sim.lowlevel_integrator, sim.integrator.dt; compute_energy=true)
        
        # Collect observables at specified intervals
        if mod(step, every) == 0 && verbose
            println("Step $step / $nsteps")
        end
    end
    
    return sim
end
