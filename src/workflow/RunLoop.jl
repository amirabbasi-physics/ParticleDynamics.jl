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
