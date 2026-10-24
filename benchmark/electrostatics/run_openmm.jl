#!/usr/bin/env julia
#
# OpenMM 8 (CUDA platform, mixed precision, PME) on the equilibrated SPC/E
# water series. ewaldErrorTolerance is calibrated on the 24k frame A to the
# measured Δ (same Float64 reference and metric as run_water.jl), checked on
# frame B, and timed as GPU-resident Verlet steps (flexible water,
# W1 = LJ switched off, W2 = LJ on). Requires the caches of run_water.jl.

using Printf, Serialization
include(joinpath(@__DIR__, "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness

const WORK = ENV["PD_BENCH_WORK"]
const SERIES = joinpath(ENV["PD_ESP_DATA"], "gromacs_bench", "single_core_bench")
const TARGETS = (1e-3, 1e-4)
const SIZES = length(ARGS) >= 1 ? split(ARGS[1], ",") :
              ["10_10_10", "20_20_20", "30_30_30", "40_40_40", "50_50_50", "60_60_60", "70_70_70"]

water = deserialize(joinpath(WORK, "water_results.jls"))
const STORE = joinpath(WORK, "openmm_results.jls")
results = isfile(STORE) ? deserialize(STORE) : Dict{Any,Any}()
function cached(f, key)
    haskey(results, key) && return results[key]
    v = f()
    results[key] = v
    serialize(STORE, results)
    return v
end
top_of(sz) = joinpath(SERIES, sz, "topol.top")
gA, gB = water[("equilibrated", "20_20_20")]
FrefA = water[("reference", "w24000 frame a")]; FrefB = water[("reference", "w24000 frame b")]
wdir = mkpath(joinpath(WORK, "openmm"))

for Δ in TARGETS
    c = cached(("calib", Δ)) do
        for tol in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6)
            r = openmm_water(gA, top_of("20_20_20"), "forces", tol; workdir=wdir)
            e = relerr(r.F, FrefA)
            @printf("  OpenMM tol %.0e: Δ = %.3e (α %.3f, grid %s)\n", tol, e, r.alpha, string(r.grid))
            e <= Δ && return (tol=tol, err=e)
        end
        nothing
    end
    c === nothing && continue
    cached(("heldout", Δ)) do
        relerr(openmm_water(gB, top_of("20_20_20"), "forces", c.tol; workdir=wdir).F, FrefB)
    end
    @printf("Δ=%.0e: OpenMM ewaldErrorTolerance %.0e, Δ %.3e / held-out %.3e\n", Δ, c.tol, c.err, results[("heldout", Δ)])
    for sz in SIZES
        gro, _ = water[("equilibrated", sz)]
        for mode in ("w1", "w2")
            r = cached((mode, sz, Δ)) do
                openmm_water(gro, top_of(sz), mode, c.tol; workdir=wdir, nsteps=sz in ("60_60_60", "70_70_70") ? 500 : 2000)
            end
            @printf("  %-9s %s %.3f ms/step (grid %s)\n", sz, mode, r.ms_per_step, string(r.grid))
        end
    end
end
