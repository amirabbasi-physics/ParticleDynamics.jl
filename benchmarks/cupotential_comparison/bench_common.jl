# ==============================================================================
# Common Accuracy, NVE Energy-Drift, Virial & GPU Timing Utilities
# Zero external dependencies beyond ParticleDynamics's own Project.toml.
# ==============================================================================
module BenchCommon

using CUDA
using LinearAlgebra
using Printf
using Random
using StaticArrays
using ParticleDynamics
using ParticleDynamics: SimulationCore

export relative_rms_error, max_component_error, virial_matrix,
       measure_step_throughput, measure_nve_drift, write_json, write_csv_summary

"""
    relative_rms_error(A, Aref) -> Float64

Relative root-mean-square error `‖A − Aref‖₂ / max(‖Aref‖₂, 1e-30)`.
"""
function relative_rms_error(A, Aref)
    num = sum(abs2, Float64.(A) .- Float64.(Aref))
    den = max(sum(abs2, Float64.(Aref)), 1e-60)
    return sqrt(num / den)
end

max_component_error(A, Aref) = maximum(abs.(Float64.(A) .- Float64.(Aref)))

"""
    virial_matrix(st) -> SMatrix{3,3,Float64,9}

Full 3×3 configurational virial tensor `W_ab = Σ r_a F_b` (including reciprocal-space
global virial terms) at the state's current coordinates after an energy-enabled evaluation.
"""
function virial_matrix(st)
    W = virial_tensor(st)
    return @SMatrix [
        Float64(W.xx) Float64(W.xy) Float64(W.xz);
        Float64(W.xy) Float64(W.yy) Float64(W.yz);
        Float64(W.xz) Float64(W.yz) Float64(W.zz)
    ]
end

"""
    evaluate_snapshot(st) -> (energy::Float64, forces::Matrix{Float64}, virial::SMatrix{3,3,Float64,9})

Evaluate conservative energy, `3×N` forces (in build-time particle id order if `st.tag !== nothing`),
and `3×3` total configurational virial on `st`.
"""
function evaluate_snapshot(st::SimulationState)
    SimulationCore.invalidate_forces!(st)
    SimulationCore.evaluate_forces_into_f!(st, true)
    CUDA.synchronize()
    fx = Float64.(Array(st.fx))
    fy = Float64.(Array(st.fy))
    fz = Float64.(Array(st.fz))
    F_slot = vcat(fx', fy', fz')
    if st.tag !== nothing
        tag = Array(st.tag)
        F = similar(F_slot)
        F[:, tag] = F_slot
    else
        F = F_slot
    end
    E = Float64(total_potential_energy(st))
    W = virial_matrix(st)
    return E, F, W
end

"""
    measure_step_throughput(st, spec, dt; warmup=200, steps=2000, trials=5) -> NamedTuple

Measure steady-state device-engine stepping throughput (`steps_per_sec`, `ms_per_step`,
and `ns_per_day` when `dt_ps` is supplied in picoseconds).
"""
function measure_step_throughput(st::SimulationState{T}, spec, dt::Real;
                                 warmup::Int=200, steps::Int=2_000, trials::Int=5,
                                 dt_ps::Union{Nothing,Real}=nothing) where {T}
    dtT = T(dt)
    for _ in 1:warmup
        step!(st, spec, dtT; compute_energy=false)
    end
    CUDA.synchronize()

    times_s = Vector{Float64}(undef, trials)
    for r in 1:trials
        t0 = CUDA.@elapsed begin
            for _ in 1:steps
                step!(st, spec, dtT; compute_energy=false)
            end
        end
        times_s[r] = t0
    end
    sort!(times_s)
    med_s = times_s[cld(trials, 2)]
    sps = steps / med_s
    ms_step = 1e3 * med_s / steps
    ns_day = dt_ps === nothing ? NaN : (sps * Float64(dt_ps) * 86.4)   # 1 ps = 1e-3 ns; 86400 s/day -> ×86.4
    return (steps_per_sec=sps, ms_per_step=ms_step, ns_per_day=ns_day,
            min_ms_per_step=1e3 * first(times_s) / steps)
end

"""
    measure_nve_drift(st, dt; equil_steps=2000, prod_steps=50000, sample_every=100,
                      dof=nothing, kBT=1.0, time_unit_ns=1e-3) -> NamedTuple

Run NVE from `st` and fit ordinary least-squares slope of total energy per degree of freedom
in units of `k_B T / dof / ns` (or per simulation time unit if `time_unit_ns = 1.0`),
plus block-bootstrap 95% confidence bounds and short-time fluctuation RMS.
"""
function measure_nve_drift(st::SimulationState{T}, dt::Real;
                           equil_steps::Int=2_000, prod_steps::Int=50_000,
                           sample_every::Int=100, dof::Union{Nothing,Int}=nothing,
                           kBT::Real=1.0, time_unit_ns::Real=1e-3) where {T}
    dtT = T(dt)
    spec = nve(st; dt=dtT)
    ndof = dof === nothing ? kinetic_dof(st; remove_com=true) : dof

    for _ in 1:equil_steps
        step!(st, spec, dtT; compute_energy=false)
    end
    CUDA.synchronize()

    nsamp = prod_steps ÷ sample_every
    ts = Vector{Float64}(undef, nsamp + 1)
    es = Vector{Float64}(undef, nsamp + 1)

    step_zero_forces_eval!(st) = begin
        SimulationCore.invalidate_forces!(st)
        SimulationCore.evaluate_forces_into_f!(st, true)
        Float64(total_potential_energy(st)) + Float64(total_kinetic_energy(st))
    end

    ts[1] = 0.0
    es[1] = step_zero_forces_eval!(st)
    for k in 1:nsamp
        for _ in 1:(sample_every - 1)
            step!(st, spec, dtT; compute_energy=false)
        end
        step!(st, spec, dtT; compute_energy=true)
        CUDA.synchronize()
        ts[k + 1] = k * sample_every * Float64(dt) * Float64(time_unit_ns)
        es[k + 1] = Float64(total_potential_energy(st)) + Float64(total_kinetic_energy(st))
    end

    # Normalize reduced energy per DOF in kBT units: e_norm(t) = (E(t) - E(0)) / (ndof * kBT)
    y = (es .- es[1]) ./ (ndof * Float64(kBT))
    tmean = sum(ts) / length(ts)
    ymean = sum(y) / length(y)
    sxx = sum((t - tmean)^2 for t in ts)
    sxy = sum((ts[i] - tmean) * (y[i] - ymean) for i in eachindex(ts))
    slope = sxy / max(sxx, 1e-30)   # (kBT / dof / ns)
    resid = [y[i] - (ymean + slope * (ts[i] - tmean)) for i in eachindex(ts)]
    fluct_rms = sqrt(sum(abs2, resid) / max(length(resid), 1))

    # Moving-block bootstrap (10 blocks) for 95% upper bound on |drift|
    nblocks = min(10, length(ts) ÷ 2)
    bsize = max(1, length(ts) ÷ nblocks)
    rng = MersenneTwister(0x5EED2026)
    boots = Vector{Float64}(undef, 400)
    for b in 1:400
        idxs = Int[]
        while length(idxs) < length(ts)
            start = rand(rng, 1:(length(ts) - bsize + 1))
            append!(idxs, start:(start + bsize - 1))
        end
        resize!(idxs, length(ts))
        tb = ts[idxs]; yb = y[idxs]
        tm = sum(tb) / length(tb); ym = sum(yb) / length(yb)
        denom = sum((t - tm)^2 for t in tb)
        boots[b] = denom > 0 ? sum((tb[i] - tm) * (yb[i] - ym) for i in eachindex(tb)) / denom : slope
    end
    sort!(abs.(boots))
    abs_ci95 = abs.(boots)[clamp(round(Int, 0.95 * length(boots)), 1, length(boots))]

    return (drift_kBT_dof_ns=slope, abs_drift_kBT_dof_ns=abs(slope),
            abs_drift_ci95_kBT_dof_ns=abs_ci95, fluct_rms_kBT_dof=fluct_rms,
            E0=es[1], Eend=es[end], ndof=ndof)
end

# Minimal JSON serializer (handles Dict, NamedTuple, Vector, Tuple, Number, String, Symbol, Bool, Nothing)
function _to_json(io::IO, x, indent::Int=0)
    pad = "  "^indent
    pad1 = "  "^(indent + 1)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        isfinite(x) ? @printf(io, "%.12g", x) : print(io, "null")
    elseif x isa Union{AbstractString,Symbol}
        s = replace(String(x), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")
        print(io, '"', s, '"')
    elseif x isa Union{AbstractVector,Tuple}
        if isempty(x)
            print(io, "[]")
        else
            println(io, "[")
            for (i, v) in enumerate(x)
                print(io, pad1)
                _to_json(io, v, indent + 1)
                println(io, i < length(x) ? "," : "")
            end
            print(io, pad, "]")
        end
    elseif x isa Union{AbstractDict,NamedTuple}
        ks = collect(keys(x))
        if isempty(ks)
            print(io, "{}")
        else
            println(io, "{")
            for (i, k) in enumerate(ks)
                print(io, pad1, '"', String(Symbol(k)), "\": ")
                _to_json(io, x[k], indent + 1)
                println(io, i < length(ks) ? "," : "")
            end
            print(io, pad, "}")
        end
    else
        _to_json(io, string(x), indent)
    end
end

function write_json(path::AbstractString, data)
    mkpath(dirname(path))
    open(path, "w") do io
        _to_json(io, data, 0)
        println(io)
    end
    return path
end

function write_csv_summary(path::AbstractString, rows::AbstractVector)
    isempty(rows) && return path
    mkpath(dirname(path))
    cols = collect(keys(first(rows)))
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            vals = map(cols) do c
                v = r[c]
                v isa AbstractFloat ? @sprintf("%.8g", v) : string(v)
            end
            println(io, join(vals, ","))
        end
    end
    return path
end

end # module BenchCommon
