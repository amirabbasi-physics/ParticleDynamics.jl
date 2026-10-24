# ESP settings at a target Coulomb force error (match_accuracy.jl protocol): order P, split
# tolerance (prolate split shape c) and mesh oversampling s (mesh = ⌈L·c·s/(π r_c)⌉, FFT-friendly);
# window shape from tolerance/2, at least the minimum for P. Prints Δ on frame A for every setting.
# Usage: julia --project=<PD> esp_scan.jl FRAME_A.gro TOP TARGET OUT.toml
using CUDA, LinearAlgebra, Printf, TOML, ParticleDynamics
include(joinpath(@__DIR__, "..", "electrostatics", "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics
const GMX = get(ENV, "PD_GMX_GPU", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")
groA, top, target, out = ARGS[1], ARGS[2], parse(Float64, ARGS[3]), ARGS[4]
fA = gromacs_fixture(groA; top, gmx=GMX)
Fref = pd_forces(pd_state(fA, esp_spec(fA; rc=1.0, tolerance=1e-10); T=Float64))
function spec(tol, P, s)
    c = ES.prolate_shape_for_tolerance(tol)
    cw = max(ES.prolate_shape_for_tolerance(tol / 2), pi * (P + 0.5) / 2 + 1e-3)
    m = ntuple(d -> max(ES.fft_friendly(ceil(Int, fA.box[d] * c * s / pi)), P), 3)
    return ES.ElectrostaticsSpec(coulomb_prefactor=fA.prefactor, real_cutoff=1.0, split=ES.ProlateSplit(1.0, c),
        window=ES.ProlateWindow(P, cw), mesh=m, influence=:esp), m
end
rows = []
for P in (4, 5), tol in (0.8e-3, 1.0e-3, 1.33e-3, 1.8e-3, 2.5e-3), s in (1.0, 1.15, 1.3, 1.5, 1.75)
    sp, m = spec(tol, P, s)
    Δ = try relerr(pd_forces(pd_state(fA, sp; T=Float32)), Fref) catch err; err isa ArgumentError || rethrow(); Inf end
    @printf("P %d  tol %.2e  s %.2f  mesh %-14s Δ %.3e %s\n", P, tol, s, string(m), Δ, Δ <= target ? "ok" : "")
    push!(rows, Dict("P" => P, "tol" => tol, "s" => s, "mesh" => collect(m), "delta" => Δ, "ok" => Δ <= target))
    GC.gc(); CUDA.reclaim()
end
open(io -> TOML.print(io, Dict("target" => target, "rows" => rows)), out, "w")
