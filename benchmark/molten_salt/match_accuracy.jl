# Molten NaCl: matched Coulomb force accuracy, PME vs ESP.
# Protocol of benchmark/water/match_accuracy.jl:
#   Δ = (Σ‖F − F_ref‖² / Σ‖F_ref‖²)^½ on the Coulomb forces only (frozen frames), against the
#   ParticleDynamics Float64 ESP reference (ε = 1e-10).
#   Target: GROMACS production PME (ewald-rtol 1e-5, fourierspacing 0.12 nm, pme-order 4) on frame A.
#   PD PME at GROMACS' α and grid (same algorithm: reported, not tuned).
#   PD ESP: the loosest tolerance ε meeting the target on frame A (shape, width, order and mesh
#   from the ESP rules), re-measured on the held-out frame B.
# Usage: julia --project=<PD> match_accuracy.jl FRAME_A.gro FRAME_B.gro TOP WORKDIR OUT.toml
using CUDA, LinearAlgebra, Printf, TOML, ParticleDynamics
include(joinpath(@__DIR__, "..", "electrostatics", "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics
const GMX = get(ENV, "PD_GMX_GPU", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")

groA, groB, top, work, out = ARGS
mkpath(work)
fA = gromacs_fixture(groA; top, gmx=GMX); fB = gromacs_fixture(groB; top, gmx=GMX)
reference(f) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=1e-10); T=Float64))
FrefA = reference(fA); FrefB = reference(fB)
rec = Dict{String,Any}("frame_a" => groA, "frame_b" => groB, "metric" => "relative RMS Coulomb force error")

setG = Dict(:rc => 1.0, :spacing => 0.12, :order => 4, :rtol => 1e-5)
FgA, _, _ = gromacs_coulomb(joinpath(work, "gmx_A"), groA, top, setG; gmx=GMX, gpu=true)
FgB, _, _ = gromacs_coulomb(joinpath(work, "gmx_B"), groB, top, setG; gmx=GMX, gpu=true)
target = relerr(FgA, FrefA)
rec["target"] = target
rec["GROMACS PME"] = Dict("knob" => "ewald-rtol 1e-5, fourierspacing 0.12, order 4", "delta_a" => target,
                          "delta_b" => relerr(FgB, FrefB))
@printf("target Δ (GROMACS production PME) = %.3e  (held-out %.3e)\n", target, relerr(FgB, FrefB))

function ewald_alpha(rtol, rc)   # erfc(α r_c) = rtol (GROMACS' definition)
    lo, hi = 0.0, 20.0 / rc
    for _ in 1:200
        mid = (lo + hi) / 2
        ES._host_erfc(mid * rc) > rtol ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
α = ewald_alpha(1e-5, 1.0)
gmesh(f) = ntuple(d -> ES.fft_friendly(ceil(Int, f.box[d] / 0.12 - 1e-9)), 3)
pdpme(f) = pd_forces(pd_state(f, pme_spec(f; rc=1.0, alpha=α, mesh=gmesh(f), order=4); T=Float32))
rec["PD PME"] = Dict("knob" => "alpha $(α), mesh $(gmesh(fA)), order 4", "alpha" => α, "mesh" => collect(gmesh(fA)),
                     "delta_a" => relerr(pdpme(fA), FrefA), "delta_b" => relerr(pdpme(fB), FrefB))
@printf("PD PME: Δ_A = %.3e  Δ_B = %.3e  (α = %.5f, mesh %s)\n", rec["PD PME"]["delta_a"], rec["PD PME"]["delta_b"], α, gmesh(fA))

function refine(knobs, err; iters::Int=4)
    prev = nothing
    for k in knobs
        e = err(k)
        @printf("    knob %-12.6g Δ = %.3e\n", k, e)
        if e <= target
            prev === nothing && return k, e
            good, egood, bad = k, e, prev
            for _ in 1:iters
                mid = sqrt(good * bad)
                em = err(mid)
                @printf("    knob %-12.6g Δ = %.3e\n", mid, em)
                em <= target ? ((good, egood) = (mid, em)) : (bad = mid)
            end
            return good, egood
        end
        prev = k
    end
    return knobs[end], err(knobs[end])
end
println("PD ESP (tolerance):")
pdesp(f, ε) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=ε); T=Float32))
# tolerances whose window cannot hold the rule's order are invalid settings: counted as failing
safe(f) = k -> (try f(k) catch err; err isa ArgumentError || rethrow(); Inf end)
kE, eE = refine(target .* (4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.65, 0.5), safe(ε -> relerr(pdesp(fA, ε), FrefA)))
sE = esp_spec(fA; rc=1.0, tolerance=kE)
rec["PD ESP"] = Dict("knob" => "tolerance $(kE)", "tolerance" => kE, "delta_a" => eE, "delta_b" => relerr(pdesp(fB, kE), FrefB),
                     "mesh" => collect(sE.mesh), "order" => sE.window.order)
open(io -> TOML.print(io, rec), out, "w")
println("\nsummary (target ", @sprintf("%.3e", target), "):")
for k in ("GROMACS PME", "PD PME", "PD ESP")
    @printf("  %-12s %-60s Δ_A %.3e  Δ_B %.3e\n", k, rec[k]["knob"], rec[k]["delta_a"], rec[k]["delta_b"])
end
@printf("  PD ESP mesh %s, order %d\n", Tuple(sE.mesh), sE.window.order)
