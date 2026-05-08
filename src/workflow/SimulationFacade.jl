"""
    Simulation(system; groups=Groups(), integrator=nothing, observables=Observable[], writers=Writer[], device=nothing, precision=Float64, seed=nothing, state=nothing, lowlevel_integrator=nothing, prepared=false, metadata=Dict())

High-level workflow simulation object. It owns the particle system, groups,
integrator, observables, writers, run metadata, and optionally a prepared
low-level [`SimulationState`](@ref).
"""
@kwdef mutable struct Simulation
    system::ParticleSystem
    groups::Groups = Groups()
    integrator = nothing
    observables::Vector{Observable} = Observable[]
    writers::Vector{Writer} = Writer[]
    device = nothing
    precision = Float64
    seed = nothing
    state::Union{Nothing,SimulationState} = nothing
    lowlevel_integrator = nothing
    prepared::Bool = false
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

function Simulation(system::ParticleSystem;
                    groups::Groups=Groups(),
                    integrator=nothing,
                    observables=Observable[],
                    writers=Writer[],
                    device=nothing,
                    precision=Float64,
                    seed=nothing,
                    state::Union{Nothing,SimulationState}=nothing,
                    lowlevel_integrator=nothing,
                    prepared::Bool=false,
                    metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())
    observables_norm = observables isa Observable ? Observable[observables] : Observable[collect(observables)...]
    writers_norm = writers isa Writer ? Writer[writers] : Writer[collect(writers)...]
    return Simulation(system,
                      groups,
                      integrator,
                      observables_norm,
                      writers_norm,
                      device,
                      precision,
                      seed,
                      state,
                      lowlevel_integrator,
                      prepared,
                      metadata)
end
