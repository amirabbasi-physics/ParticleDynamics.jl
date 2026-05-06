abstract type Writer end

@kwdef struct TableWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    observables::Vector{Any} = Any[]
    mode::Symbol = :replace
    delimiter::String = ","
    format::Symbol = :scientific
    append::Bool = false
end

@kwdef struct GSDWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    group = nothing
    write_start::Bool = true
    mode::Symbol = :replace
    append::Bool = false
    types = :automatic
    diameter = :automatic
    write_unwrapped::Bool = false
    sync_on_write::Bool = true
    write_forces::Bool = false
    observables::Vector{Any} = Any[]
end
