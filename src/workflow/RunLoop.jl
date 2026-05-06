function state(sim::Simulation)
    return sim.state
end

function prepare!(sim::Simulation)
    sim.prepared = sim.state !== nothing
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
