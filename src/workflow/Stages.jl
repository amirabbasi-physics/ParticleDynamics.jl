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
