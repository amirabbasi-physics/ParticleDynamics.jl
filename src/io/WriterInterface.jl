# =======================================================================
# Writer interface and lightweight in-memory logger
# =======================================================================

abstract type Writer end

"""
Lightweight logger that records selected observables in host memory every
`every` steps. Useful for quick ad-hoc diagnostics during testing.
"""
mutable struct InMemoryLogger <: Writer
    every::Int
    data::Dict{String, Vector}
    steps::Vector{Int}
end

function InMemoryLogger(data_keys; every::Int=1)
    data = Dict{String, Vector}()
    for key in data_keys
        data[string(key)] = Vector{Any}()
    end
    return InMemoryLogger(every, data, Int[])
end

function write!(w::InMemoryLogger, _simulation, step::Int, _dt::Real)
    if step % w.every != 0
        return
    end
    push!(w.steps, step)
    return nothing
end
