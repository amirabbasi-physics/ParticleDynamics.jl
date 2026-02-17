module ParamsFromExamples

export collect_example_params, recommended_test_params

const DEFAULTS = (
    brownian = (dt=2.5e-4, gamma=10.0, temperature=1.0, mass=1.0, tau=0.5, boxL=50.0, integrator=:eulerheun),
    langevin = (dt=1.0e-5, gamma=615.985, temperature=50.0, mass=1.0, tau=0.5, boxL=125.0, integrator=:velocityverlet),
    ou = (dt=1.0e-3, gamma=10.0, temperature=5.0, mass=1.0, tau=0.5, boxL=125.0, integrator=:velocityverlet),
)

const PARAM_KEYS = (:dt, :gamma, :temperature, :temp, :T, :mass, :tau, :corr_time, :noise_corr_time)
const PARAM_RE = Regex(
    "^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+|f[+-]?\\d+)?)\\s*(?:#.*)?\$",
    "m",
)
const BOX_RE = Regex(
    "^\\s*box\\s*=\\s*\\(\\s*([^,\\)]+)\\s*,\\s*([^,\\)]+)(?:\\s*,\\s*([^,\\)]+))?\\s*\\)\\s*(?:#.*)?\$",
    "m",
)

const INTEGRATOR_PATTERNS = [
    (:eulerheun, r"\beulerheun\s*\("),
    (:eulermaruyama, r"\beulermaruyama\s*\("),
    (:velocityverlet, r"\bvelocityverlet\s*\("),
    (:baoab, r"\bbaoab\s*\("),
    (:baoa, r"\bbaoa\s*\("),
    (:gsm, r"\bgsm\s*\("),
]

@inline function _try_parse_number(raw::AbstractString)
    s = replace(strip(raw), "_" => "")
    # Julia Float32 exponent syntax: 2.0f-4, 1f0, ...
    s = replace(s, r"([0-9.])f([+-]?[0-9]+)" => s"\\1e\\2")
    return try
        parse(Float64, s)
    catch
        nothing
    end
end

function _median_or_default(values::Vector{Float64}, default::Float64)
    isempty(values) && return default
    s = sort(values)
    n = length(s)
    if isodd(n)
        return s[(n + 1) ÷ 2]
    end
    i = n ÷ 2
    return (s[i] + s[i + 1]) / 2
end

function _mode_or_default(values::Vector{Symbol}, default::Symbol)
    isempty(values) && return default
    counts = Dict{Symbol,Int}()
    for v in values
        counts[v] = get(counts, v, 0) + 1
    end
    best = default
    bestc = -1
    for (k, c) in counts
        if c > bestc || (c == bestc && string(k) < string(best))
            best = k
            bestc = c
        end
    end
    return best
end

function _match_integrators(text::String)
    hits = Symbol[]
    for (name, pat) in INTEGRATOR_PATTERNS
        occursin(pat, text) && push!(hits, name)
    end
    return hits
end

function _collect_numeric_fields(text::String)
    vals = Dict{Symbol,Vector{Float64}}()
    for k in PARAM_KEYS
        vals[k] = Float64[]
    end
    for m in eachmatch(PARAM_RE, text)
        key = Symbol(m.captures[1])
        if haskey(vals, key)
            num = _try_parse_number(m.captures[2])
            num === nothing || push!(vals[key], num)
        end
    end
    return vals
end

function _collect_box_lengths(text::String)
    out = Float64[]
    for m in eachmatch(BOX_RE, text)
        nums = Float64[]
        for i in 1:3
            c = m.captures[i]
            c === nothing && continue
            v = _try_parse_number(c)
            v === nothing || push!(nums, abs(v))
        end
        isempty(nums) || append!(out, nums)
    end
    return out
end

function _group_for_file(path::String, integrators::Vector{Symbol}, has_ou_terms::Bool)
    b = !isempty(intersect(integrators, [:eulerheun, :eulermaruyama])) ||
        occursin(r"(?:^|/)(?:.*BD.*|.*brownian.*)\.jl$"i, path)
    l = !isempty(intersect(integrators, [:velocityverlet, :baoab, :baoa, :gsm])) ||
        occursin(r"(?:^|/)(?:.*LD.*|.*baoab.*|.*baoa.*|.*gsm.*)\.jl$"i, path)
    o = has_ou_terms || occursin(r"(?:^|/).*OU.*\.jl$"i, path)
    return (brownian=b, langevin=l, ou=o)
end

function collect_example_params(; examples_dir::String=joinpath(dirname(@__DIR__), "examples"))
    files = sort(filter(f -> endswith(f, ".jl"), mapreduce(x -> [joinpath(x[1], f) for f in x[3]], vcat, walkdir(examples_dir))))
    read_fail = String[]
    parse_fail = String[]

    data = Dict(
        :brownian => Dict(:dt=>Float64[], :gamma=>Float64[], :temperature=>Float64[], :mass=>Float64[], :tau=>Float64[], :boxL=>Float64[], :integrator=>Symbol[]),
        :langevin => Dict(:dt=>Float64[], :gamma=>Float64[], :temperature=>Float64[], :mass=>Float64[], :tau=>Float64[], :boxL=>Float64[], :integrator=>Symbol[]),
        :ou => Dict(:dt=>Float64[], :gamma=>Float64[], :temperature=>Float64[], :mass=>Float64[], :tau=>Float64[], :boxL=>Float64[], :integrator=>Symbol[]),
    )

    for f in files
        txt = try
            read(f, String)
        catch
            push!(read_fail, f)
            continue
        end

        vals = _collect_numeric_fields(txt)
        box_vals = _collect_box_lengths(txt)
        integrators = _match_integrators(txt)

        tau_vals = vcat(vals[:tau], vals[:corr_time], vals[:noise_corr_time])
        temp_vals = vcat(vals[:temperature], vals[:temp], vals[:T])
        has_ou_terms = !isempty(tau_vals)
        group = _group_for_file(f, integrators, has_ou_terms)

        any_numeric = false
        for key in (:dt, :gamma, :mass)
            any_numeric |= !isempty(vals[key])
        end
        any_numeric |= !isempty(temp_vals) || !isempty(tau_vals) || !isempty(box_vals)
        any_numeric |= !isempty(integrators)
        any_numeric || push!(parse_fail, f)

        for (gname, active) in pairs(group)
            active || continue
            d = data[gname]
            append!(d[:dt], vals[:dt])
            append!(d[:gamma], vals[:gamma])
            append!(d[:temperature], temp_vals)
            append!(d[:mass], vals[:mass])
            append!(d[:tau], tau_vals)
            append!(d[:boxL], box_vals)
            append!(d[:integrator], integrators)
        end
    end

    return (
        files=files,
        read_failures=read_fail,
        parse_failures=parse_fail,
        data=data,
    )
end

function _recommend_group(group::Dict{Symbol,Vector}, defaults)
    return (
        dt=_median_or_default(group[:dt], defaults.dt),
        gamma=_median_or_default(group[:gamma], defaults.gamma),
        temperature=_median_or_default(group[:temperature], defaults.temperature),
        mass=_median_or_default(group[:mass], defaults.mass),
        tau=_median_or_default(group[:tau], defaults.tau),
        boxL=_median_or_default(group[:boxL], defaults.boxL),
        integrator=_mode_or_default(group[:integrator], defaults.integrator),
    )
end

function recommended_test_params(; examples_dir::String=joinpath(dirname(@__DIR__), "examples"))
    parsed = collect_example_params(; examples_dir)
    rec = (
        brownian=_recommend_group(parsed.data[:brownian], DEFAULTS.brownian),
        langevin=_recommend_group(parsed.data[:langevin], DEFAULTS.langevin),
        ou=_recommend_group(parsed.data[:ou], DEFAULTS.ou),
    )
    return merge(rec, (parse_report=(
        files_scanned=length(parsed.files),
        read_failures=parsed.read_failures,
        parse_failures=parsed.parse_failures,
        source_files=parsed.files,
    ),))
end

end # module
