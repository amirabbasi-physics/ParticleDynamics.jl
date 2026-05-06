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
