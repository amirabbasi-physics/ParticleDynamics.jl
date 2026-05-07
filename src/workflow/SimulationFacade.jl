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
                    observables::AbstractVector{<:Observable}=Observable[],
                    writers::AbstractVector{<:Writer}=Writer[],
                    device=nothing,
                    precision=Float64,
                    seed=nothing,
                    state::Union{Nothing,SimulationState}=nothing,
                    lowlevel_integrator=nothing,
                    prepared::Bool=false,
                    metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())
    return Simulation(system=system,
                      groups=groups,
                      integrator=integrator,
                      observables=Observable[observables...],
                      writers=Writer[writers...],
                      device=device,
                      precision=precision,
                      seed=seed,
                      state=state,
                      lowlevel_integrator=lowlevel_integrator,
                      prepared=prepared,
                      metadata=metadata)
end
