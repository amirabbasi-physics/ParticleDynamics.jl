"""
    Stage(name; steps, dt=nothing, neighbor_rebuild_interval=nothing, compute_energy=:auto, reset_observables=false, reset_step=nothing, progress=true, max_seconds=Inf)

Describe a named block of simulation steps for [`run!`](@ref).
"""
@kwdef struct Stage
    name::Symbol
    steps::Int
    dt = nothing
    neighbor_rebuild_interval = nothing
    compute_energy = :auto
    reset_observables::Bool = false
    reset_step::Union{Nothing,Int} = nothing
    progress::Bool = true
    max_seconds::Real = Inf
end

Stage(name::Symbol; kwargs...) = Stage(; name=name, kwargs...)
