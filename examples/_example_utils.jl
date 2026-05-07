using Random

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

function maybe_override_float(default::Real, env_name::AbstractString; lower::Real=-Inf)
    value = strip(get(ENV, env_name, ""))
    isempty(value) && return max(lower, float(default))
    parsed = tryparse(Float64, value)
    return parsed === nothing ? max(lower, float(default)) : max(lower, parsed)
end

function maybe_override_bool(default::Bool, env_name::AbstractString)
    value = lowercase(strip(get(ENV, env_name, "")))
    isempty(value) && return default
    if value in ("1", "true", "yes", "y", "on")
        return true
    elseif value in ("0", "false", "no", "n", "off")
        return false
    end
    return default
end

function square_lattice_positions(N::Integer, box::NTuple{2,<:Real})
    n_side = ceil(Int, sqrt(Float64(max(N, 1))))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    return [
        (
            (mod(i - 1, n_side) + 0.5) * spacing_x - box[1] / 2,
            (div(i - 1, n_side) + 0.5) * spacing_y - box[2] / 2,
        )
        for i in 1:N
    ]
end

function simple_cubic_positions(N::Integer, box::NTuple{3,<:Real})
    n_side = ceil(Int, cbrt(Float64(max(N, 1))))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    spacing_z = box[3] / n_side
    n_side_sq = n_side^2
    return [
        begin
            linear = i - 1
            ix = mod(linear, n_side)
            iy = mod(div(linear, n_side), n_side)
            iz = div(linear, n_side_sq)
            (
                (ix + 0.5) * spacing_x - box[1] / 2,
                (iy + 0.5) * spacing_y - box[2] / 2,
                (iz + 0.5) * spacing_z - box[3] / 2,
            )
        end
        for i in 1:N
    ]
end

function linear_chain_positions(N::Integer;
                                spacing::Real=0.97,
                                origin::Tuple{<:Real,<:Real}=(0.0, 0.0))
    x0, y0 = origin
    return [(x0 + (i - 1) * spacing, y0) for i in 1:N]
end

single_particle_groups() = begin
    all_particles = Group(:all, AllSelection())
    all_particles, Groups(all_particles)
end

function two_type_particle_groups(; cold::Symbol=:cold,
                                  hot::Symbol=:hot,
                                  all_name::Symbol=:all,
                                  cold_type::Symbol=:C,
                                  hot_type::Symbol=:H)
    cold_group = Group(cold, TypeSelection(cold_type))
    hot_group = Group(hot, TypeSelection(hot_type))
    all_group = Group(all_name, AllSelection())
    return cold_group, hot_group, all_group, Groups(cold_group, hot_group, all_group)
end

function random_typeids(N::Integer; fractions::Dict{Symbol,<:Real}, seed::Integer=0x5A17)
    N > 0 || return Int32[]
    syms = collect(keys(fractions))
    isempty(syms) && throw(ArgumentError("fractions must not be empty."))
    vals = Float64[fractions[sym] for sym in syms]
    total = sum(vals)
    total > 0 || throw(ArgumentError("fractions must sum to a positive value."))
    vals ./= total

    counts = floor.(Int, vals .* N)
    remainder = N - sum(counts)
    if remainder > 0
        order = sortperm(vals .* N .- counts; rev=true)
        for idx in order[1:remainder]
            counts[idx] += 1
        end
    end

    host = Vector{Int32}(undef, N)
    offset = 1
    for (tid, count) in enumerate(counts)
        for _ in 1:count
            host[offset] = Int32(tid)
            offset += 1
        end
    end
    Random.seed!(seed)
    return host[randperm(N)]
end

function run_equilibration!(sim;
                            warmup_steps::Integer=0,
                            warmup_dt=nothing,
                            warmup_neighbor_rebuild_interval=nothing,
                            init_steps::Integer=0,
                            relax_steps::Integer=0,
                            progress::Bool=false)
    warmup_steps > 0 && run!(sim, Stage(:warmup, steps=Int(warmup_steps);
                                        dt=warmup_dt,
                                        neighbor_rebuild_interval=warmup_neighbor_rebuild_interval,
                                        progress=progress))
    init_steps > 0 && run!(sim, Stage(:init, steps=Int(init_steps); progress=progress))
    relax_steps > 0 && run!(sim, Stage(:relax, steps=Int(relax_steps); progress=progress))
    return sim
end

function prepare_production!(sim;
                             reset_observables_before_production::Bool=true,
                             reset_step_before_production::Bool=true)
    prepare!(sim)
    reset_observables_before_production && reset_observables!(sim)
    reset_step_before_production && reset_step!(sim, 0)
    return sim
end

function run_stage_sequence!(sim;
                             warmup_steps::Integer=0,
                             warmup_dt=nothing,
                             warmup_neighbor_rebuild_interval=nothing,
                             init_steps::Integer=0,
                             relax_steps::Integer=0,
                             production_steps::Integer,
                             reset_observables_before_production::Bool=true,
                             reset_step_before_production::Bool=true,
                             progress::Bool=false,
                             max_seconds=maybe_override_runtime())
    warmup_steps > 0 && run!(sim, Stage(:warmup, steps=Int(warmup_steps);
                                        dt=warmup_dt,
                                        neighbor_rebuild_interval=warmup_neighbor_rebuild_interval,
                                        progress=progress))
    init_steps > 0 && run!(sim, Stage(:init, steps=Int(init_steps); progress=progress))
    relax_steps > 0 && run!(sim, Stage(:relax, steps=Int(relax_steps); progress=progress))

    if reset_observables_before_production
        reset_observables!(sim)
    end
    if reset_step_before_production
        reset_step!(sim, 0)
    end

    return run!(sim, Stage(:production, steps=Int(production_steps);
                            progress=progress,
                            max_seconds=max_seconds))
end
