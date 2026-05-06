abstract type AbstractSchedule end

struct Every <: AbstractSchedule
    interval::Int
    function Every(interval::Integer)
        interval >= 1 || throw(ArgumentError("Every(interval) requires interval >= 1."))
        new(Int(interval))
    end
end

struct AtSteps <: AbstractSchedule
    steps::Vector{Int}
    function AtSteps(steps::AbstractVector{<:Integer})
        host = sort(unique(Int.(steps)))
        all(>=(0), host) || throw(ArgumentError("AtSteps requires nonnegative step indices."))
        new(host)
    end
end

struct Between <: AbstractSchedule
    first::Int
    last::Int
    every::Int
    function Between(first::Integer, last::Integer; every::Integer)
        first_i = Int(first)
        last_i = Int(last)
        every_i = Int(every)
        first_i <= last_i || throw(ArgumentError("Between(first, last) requires first <= last."))
        every_i >= 1 || throw(ArgumentError("Between(...; every) requires every >= 1."))
        new(first_i, last_i, every_i)
    end
end
