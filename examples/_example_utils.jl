function maybe_override_int(default::Real, env_name::AbstractString; lower::Int=1)
    default_int = max(lower, round(Int, default))
    value = strip(get(ENV, env_name, ""))
    isempty(value) && return default_int
    parsed = tryparse(Int, value)
    return parsed === nothing ? default_int : max(lower, parsed)
end

function maybe_override_interval(default::Real, nsteps::Integer; env_name::AbstractString="SIM_LOG_INTERVAL")
    return min(maybe_override_int(default, env_name; lower=1), max(1, Int(nsteps)))
end

function maybe_override_runtime()
    value = strip(get(ENV, "SIM_MAX_SECONDS", ""))
    isempty(value) && return Inf
    parsed = tryparse(Float64, value)
    return parsed === nothing ? Inf : parsed
end
