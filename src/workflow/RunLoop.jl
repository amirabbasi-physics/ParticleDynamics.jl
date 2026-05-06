function state(sim::Simulation)
    return sim.state
end

function prepare!(sim::Simulation)
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
    sim.prepared = true
    return sim
end

function reset_step!(sim::Simulation, step::Integer=0)
    if sim.state !== nothing
        sim.state.step = Int(step)
    end
    return sim
end

function reset_observables!(sim::Simulation)
    return sim
end

function run!(sim::Simulation, args...)
    throw(ArgumentError("The high-level workflow run loop is not implemented yet."))
end
