abstract type AbstractSchedule end

"""
    Every(interval)

Trigger a workflow writer or observable action every `interval` steps.
"""
struct Every <: AbstractSchedule
    interval::Int
    function Every(interval::Integer)
        interval >= 1 || throw(ArgumentError("Every(interval) requires interval >= 1."))
        new(Int(interval))
    end
end

"""
    AtSteps(steps)

Trigger a workflow action at an explicit set of step indices.
"""
struct AtSteps <: AbstractSchedule
    steps::Vector{Int}
    function AtSteps(steps::AbstractVector{<:Integer})
        host = sort(unique(Int.(steps)))
        all(>=(0), host) || throw(ArgumentError("AtSteps requires nonnegative step indices."))
        new(host)
    end
end

"""
    Between(first, last; every)

Trigger a workflow action on a regular interval between two step bounds.
"""
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

schedule_matches(schedule::Every, step::Integer) = (Int(step) % schedule.interval) == 0
schedule_matches(schedule::AtSteps, step::Integer) = Int(step) in schedule.steps
function schedule_matches(schedule::Between, step::Integer)
    host = Int(step)
    schedule.first <= host <= schedule.last || return false
    return ((host - schedule.first) % schedule.every) == 0
end

function normalize_schedule(; every=nothing, schedule=nothing, default=Every(1))
    if every !== nothing && schedule !== nothing
        throw(ArgumentError("Specify either `every` or `schedule`, not both."))
    elseif schedule !== nothing
        schedule isa AbstractSchedule || throw(ArgumentError("schedule must be an AbstractSchedule; got $(typeof(schedule))."))
        return schedule
    elseif every !== nothing
        if every isa AbstractSchedule
            return every
        else
            return Every(Int(every))
        end
    else
        return default
    end
end
