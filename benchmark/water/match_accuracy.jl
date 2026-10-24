# Matched Coulomb force accuracy across engines.
#
# Metric: Δ = (Σ‖F − F_ref‖² / Σ‖F_ref‖²)^½ on the Coulomb forces only (frozen frame),
# against the ParticleDynamics Float64 ESP reference (ε = 1e-10), the harness of
# benchmark/electrostatics (GROMACS Coulomb forces by the εᵣ = 1/2 single points).
#
# Target: Δ of GROMACS' production PME (ewald-rtol 1e-5, fourierspacing 0.12 nm,
# pme-order 4) on frame A (the shared water state). Each engine is then set to the cheapest
# knob value whose Δ on frame A meets the target ("first meeting"), and every choice is
# re-measured on the held-out frame B:
#   PD PME     GROMACS' α and grid (same algorithm): Δ is reported, not tuned
#   OpenMM     ewaldErrorTolerance (OpenMM chooses α and the grid)
#   LAMMPS     PPPM accuracy (LAMMPS chooses g_ewald and the grid; order 5)
#   PD ESP     tolerance ε (shape, width and mesh from the paper's rules)
# ESP of GROMACS-ESP and LAMMPS-ESP (CPU) is tuned by match_accuracy_esp.jl.
#
# Usage: julia --project=<PD> match_accuracy.jl FRAME_A.gro FRAME_B.gro TOP WORKDIR OUT.toml
using CUDA, LinearAlgebra, Printf, TOML, ParticleDynamics
include(joinpath(@__DIR__, "..", "electrostatics", "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics
const GMX = get(ENV, "PD_GMX_GPU", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")
const LMP_TO_KJ_NM = 4.184 * 10   # LAMMPS real units: kcal/mol/Å → kJ/mol/nm

groA, groB, top, work, out = ARGS
mkpath(work)
fA = gromacs_fixture(groA; top, gmx=GMX); fB = gromacs_fixture(groB; top, gmx=GMX)
reference(f) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=1e-10); T=Float64))
FrefA = reference(fA); FrefB = reference(fB)
rec = Dict{String,Any}("frame_a" => groA, "frame_b" => groB, "metric" => "relative RMS Coulomb force error")

# --- target: GROMACS production PME ---------------------------------------------------
setG = Dict(:rc => 1.0, :spacing => 0.12, :order => 4, :rtol => 1e-5)
FgA, _, _ = gromacs_coulomb(joinpath(work, "gmx_A"), groA, top, setG; gmx=GMX, gpu=true)
FgB, _, _ = gromacs_coulomb(joinpath(work, "gmx_B"), groB, top, setG; gmx=GMX, gpu=true)
target = relerr(FgA, FrefA)
rec["target"] = target
rec["GROMACS PME"] = Dict("knob" => "ewald-rtol 1e-5, fourierspacing 0.12, order 4", "delta_a" => target,
                          "delta_b" => relerr(FgB, FrefB))
@printf("target Δ (GROMACS production PME) = %.3e  (held-out %.3e)\n", target, relerr(FgB, FrefB))

# --- PD PME at GROMACS' α and grid ---------------------------------------------------------
function ewald_alpha(rtol, rc)   # erfc(α r_c) = rtol (GROMACS' definition)
    lo, hi = 0.0, 20.0 / rc
    for _ in 1:200
        mid = (lo + hi) / 2
        ES._host_erfc(mid * rc) > rtol ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
α = ewald_alpha(1e-5, 1.0)
gmesh(f) = ntuple(d -> ceil(Int, f.box[d] / 0.12), 3)   # GROMACS' grid at 0.12 nm (factors incl. 7 are allowed)
pdpme(f) = pd_forces(pd_state(f, pme_spec(f; rc=1.0, alpha=α, mesh=gmesh(f), order=4); T=Float32))
rec["PD PME"] = Dict("knob" => "alpha $(α), mesh $(gmesh(fA)), order 4", "alpha" => α, "spacing" => 0.12,
                     "delta_a" => relerr(pdpme(fA), FrefA), "delta_b" => relerr(pdpme(fB), FrefB))
@printf("PD PME: Δ_A = %.3e  Δ_B = %.3e  (α = %.5f, mesh %s)\n", rec["PD PME"]["delta_a"], rec["PD PME"]["delta_b"], α, gmesh(fA))

# --- tuned engines: first knob value meeting the target on frame A -------------------------
# Loosest knob meeting the target: scan `knobs` (loose → tight) for the first value that
# meets it, then bisect geometrically between it and the last failing value, so that every
# engine ends close to the target rather than one coarse step below it.
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
                @printf("    knob %-12.6g Δ = %.3e (bisect)\n", mid, em)
                em <= target ? ((good, egood) = (mid, em)) : (bad = mid)
            end
            return good, egood
        end
        prev = k
    end
    return knobs[end], err(knobs[end])
end
topdir = dirname(top)
println("OpenMM PME (ewaldErrorTolerance):")
omm(f, gro, tol, tag) = openmm_water(gro, top, "forces", tol; workdir=joinpath(work, "omm_$(tag)")).F
kO, eO = refine((1e-3, 7e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1.5e-4, 1e-4, 7e-5, 5e-5), t -> relerr(omm(fA, groA, t, "A"), FrefA))
rO = openmm_water(groA, top, "forces", kO; workdir=joinpath(work, "omm_A"))
rec["OpenMM PME"] = Dict("knob" => "ewaldErrorTolerance $(kO)", "tolerance" => kO, "delta_a" => eO,
                         "delta_b" => relerr(omm(fB, groB, kO, "B"), FrefB), "alpha" => rO.alpha, "grid" => collect(rO.grid))
println("LAMMPS PPPM (accuracy):")
dataA = lammps_data(joinpath(work, "lmpA.data"), fA; length_scale=10.0)
dataB = lammps_data(joinpath(work, "lmpB.data"), fB; length_scale=10.0)
lmp(data, acc, tag) = lammps_run(joinpath(work, "lmp"), "lmp_$(tag)_$(acc)", data; rc=10.0, accuracy=acc, nsteps=1)
kL, eL = refine((1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1.5e-5, 1e-5, 7e-6, 5e-6),
                       a -> relerr(lmp(dataA, a, "A").F .* LMP_TO_KJ_NM, FrefA))
rL = lmp(dataA, kL, "A")
rec["LAMMPS PPPM"] = Dict("knob" => "pppm accuracy $(kL), order 5", "accuracy" => kL, "delta_a" => eL,
                          "delta_b" => relerr(lmp(dataB, kL, "B").F .* LMP_TO_KJ_NM, FrefB), "gewald" => rL.gewald, "grid" => collect(rL.grid))
println("PD ESP (tolerance):")
pdesp(f, ε) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=ε); T=Float32))
kE, eE = refine(target .* (4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.65, 0.5), ε -> relerr(pdesp(fA, ε), FrefA))
rec["PD ESP"] = Dict("knob" => "tolerance $(kE)", "tolerance" => kE, "delta_a" => eE, "delta_b" => relerr(pdesp(fB, kE), FrefB))

open(io -> TOML.print(io, rec), out, "w")
println("\nsummary (target ", @sprintf("%.3e", target), "):")
for k in ("GROMACS PME", "PD PME", "OpenMM PME", "LAMMPS PPPM", "PD ESP")
    @printf("  %-12s %-50s Δ_A %.3e  Δ_B %.3e\n", k, rec[k]["knob"], rec[k]["delta_a"], rec[k]["delta_b"])
end
