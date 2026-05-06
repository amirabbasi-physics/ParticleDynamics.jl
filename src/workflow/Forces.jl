abstract type Force end

@kwdef mutable struct ForceField
    forces::Vector{Force} = Force[]
end

Base.length(ff::ForceField) = length(ff.forces)
Base.iterate(ff::ForceField, state::Int=1) =
    state > length(ff.forces) ? nothing : (ff.forces[state], state + 1)

function add!(ff::ForceField, force::Force)
    push!(ff.forces, force)
    return ff
end

@kwdef struct LennardJones <: Force
    epsilon
    sigma
    cutoff
    pairs = :all
    mode = :standard
    neighborlist = nothing
end

@kwdef struct WCA <: Force
    epsilon
    sigma
    cutoff = nothing
    pairs = :all
    neighborlist = nothing
end

@kwdef struct SoftRepulsive <: Force
    epsilon
    sigma
    cutoff
    pairs = :all
    params = nothing
    neighborlist = nothing
end

@kwdef struct HarmonicBondForce <: Force
    k
    r0
    type = :default
end

@kwdef struct FENEBondForce <: Force
    k
    R0
    type = :default
end
