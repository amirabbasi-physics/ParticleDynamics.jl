# Matched Coulomb force accuracy for the authors' CPU ESP forks, at the same
# target Δ as the PME family (match_accuracy.jl, read from its TOML).
#   GROMACS-ESP  ewald-rtol = the target relative force error (the fork's definition),
#                pme-order 5 (the fork's recommendation for ~4 digits); grid chosen internally
#   LAMMPS-ESP   kspace_style esp ACCURACY (pair coul/esp); grid and order chosen internally
# Δ is measured exactly as for the other engines (Coulomb-only forces, frozen frames A and
# B, against the ParticleDynamics Float64 ESP reference).
#
# Usage: julia --project=<PD> match_accuracy_esp.jl FRAME_A.gro FRAME_B.gro TOP WORKDIR MATCH.toml OUT.toml
using CUDA, LinearAlgebra, Printf, TOML, ParticleDynamics
include(joinpath(@__DIR__, "..", "electrostatics", "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const GMX = get(ENV, "PD_GMX_GPU", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")
const GMX_ESP = get(ENV, "PD_GMX_ESP", "/net/storage/abbaa90/gromacs-esp-cpu/bin/gmx")
const LMP_ESP = get(ENV, "PD_LMP_ESP", "/net/storage/abbaa90/lammps-esp-cpu/bin/lmp")
const LMP_TO_KJ_NM = 4.184 * 10   # LAMMPS real units: kcal/mol/Å → kJ/mol/nm

groA, groB, top, work, matchfile, out = ARGS
mkpath(work)
target = TOML.parsefile(matchfile)["target"]
fA = gromacs_fixture(groA; top, gmx=GMX); fB = gromacs_fixture(groB; top, gmx=GMX)
reference(f) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=1e-10); T=Float64))
FrefA = reference(fA); FrefB = reference(fB)
rec = Dict{String,Any}("target" => target, "metric" => "relative RMS Coulomb force error")
@printf("target Δ = %.3e\n", target)

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
knobs = target .* (2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1)

println("GROMACS-ESP (ewald-rtol = force tolerance, pme-order 5):")
gesp(gro, tol, tag) = gromacs_coulomb(joinpath(work, "gesp_$(tag)_$(tol)"), gro, top,
    Dict(:rc => 1.0, :spacing => 0.12, :order => 5, :rtol => tol); gmx=GMX_ESP, gpu=false)[1]
kG, eG = refine(knobs, t -> relerr(gesp(groA, t, "A"), FrefA))
rec["GROMACS-ESP"] = Dict("knob" => "ewald-rtol $(kG), pme-order 5", "tolerance" => kG, "delta_a" => eG,
                          "delta_b" => relerr(gesp(groB, kG, "B"), FrefB))

println("LAMMPS-ESP (kspace_style esp ACCURACY):")
dataA = lammps_data(joinpath(work, "lespA.data"), fA; length_scale=10.0)
dataB = lammps_data(joinpath(work, "lespB.data"), fB; length_scale=10.0)
lesp(data, acc, tag) = lammps_run(joinpath(work, "lesp"), "lesp_$(tag)_$(acc)", data; rc=10.0, accuracy=acc, nsteps=1,
    kokkos=false, lmp=LMP_ESP, pair="coul/esp 10.0", kspace="esp $(acc)", order=nothing)
kL, eL = refine(knobs, a -> relerr(lesp(dataA, a, "A").F .* LMP_TO_KJ_NM, FrefA))
rec["LAMMPS-ESP"] = Dict("knob" => "esp accuracy $(kL)", "accuracy" => kL, "delta_a" => eL,
                         "delta_b" => relerr(lesp(dataB, kL, "B").F .* LMP_TO_KJ_NM, FrefB))

open(io -> TOML.print(io, rec), out, "w")
for k in ("GROMACS-ESP", "LAMMPS-ESP")
    @printf("  %-12s %-36s Δ_A %.3e  Δ_B %.3e\n", k, rec[k]["knob"], rec[k]["delta_a"], rec[k]["delta_b"])
end
