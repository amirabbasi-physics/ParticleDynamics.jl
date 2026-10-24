# ParticleDynamics: the fastest PME and ESP settings that meet the common target Δ
# (match_accuracy.jl), chosen like OpenMM's and LAMMPS' (their cheapest setting at the
# target) instead of copying GROMACS' α and grid. `tune_electrostatics(...; objective=:time)`
# searches order, α and mesh spacing (PME) or ε and window width (ESP) for the fastest complete
# force evaluation on the timed system (LJ + Coulomb, cluster backend, Float32). Every choice is
# then re-measured with the common harness on frames A and B (exceptions included), as for the
# other engines.
# Usage: julia --project=<PD> tune_pd_electrostatics.jl TUNE_STATE.gro TUNE_TOP FRAME_A.gro FRAME_B.gro TOP MATCH.toml OUT.toml
using CUDA, LinearAlgebra, Printf, TOML, ParticleDynamics
include(joinpath(@__DIR__, "..", "electrostatics", "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics
const WF = ParticleDynamics.Workflow
const GMX = get(ENV, "PD_GMX_GPU", "/net/storage/abbaa90/gromacs-gpu/bin/gmx")
const SIGMA_O = 0.316557; const EPS_O = 0.650629

groT, topT, groA, groB, top, matchfile, out = ARGS
target = TOML.parsefile(matchfile)["target"]
fT = spatially_sorted(gromacs_fixture(groT; top=topT, gmx=GMX); cell=1.0)
fA = gromacs_fixture(groA; top, gmx=GMX); fB = gromacs_fixture(groB; top, gmx=GMX)
reference(f) = pd_forces(pd_state(f, esp_spec(f; rc=1.0, tolerance=1e-10); T=Float64))
FrefA = reference(fA); FrefB = reference(fB)
types(f) = [q < 0 ? 1 : 2 for q in f.q]
spacing(force, mesh, box) = WF._rebuild_force(force; mesh=nothing, mesh_spacing=maximum(box ./ mesh))
rec = Dict{String,Any}("target" => target, "tuning_state" => groT)
for (name, force) in (("PD PME", PME(coulomb_prefactor=fT.prefactor, real_cutoff=1.0, alpha=1.0)),
                      ("PD ESP", ESP(coulomb_prefactor=fT.prefactor, real_cutoff=1.0)))
    spec0 = pme_spec(fT; rc=1.0, alpha=3.0, mesh=ntuple(d -> ES.fft_friendly(ceil(Int, fT.box[d] / 0.14)), 3), order=4)
    st = pd_state_lj(fT, spec0; types=types(fT), sigma=[SIGMA_O 0.0; 0.0 0.0], epsilon=[EPS_O 0.0; 0.0 0.0],
                     rc_lj=1.0, T=Float32, backend=:cluster)
    t = tune_electrostatics(st, force; target, objective=:time)
    st = nothing; GC.gc(); CUDA.reclaim()
    chosen = spacing(t.force, t.mesh, fT.box)
    spec(f) = WF.electrostatics_spec(WF._rebuild_force(chosen; net_charge=:background))
    stA = pd_state(fA, spec(fA); T=Float32, backend=:cluster)
    dA = relerr(pd_forces(stA), FrefA)
    meshA = stA.electrostatics.ws.mesh      # the spacing resolved for the 10k box
    stA = nothing; GC.gc(); CUDA.reclaim()
    dB = relerr(pd_forces(pd_state(fB, spec(fB); T=Float32, backend=:cluster)), FrefB)
    s = spec(fA)
    rec[name] = Dict("descriptor" => string(chosen), "tune_error" => t.error, "tune_time_ms" => t.time_ms,
        "tune_mesh" => collect(t.mesh), "spacing_nm" => maximum(fT.box ./ t.mesh), "delta_a" => dA, "delta_b" => dB,
        "order" => s.window.order,
        "split" => string(typeof(s.split)), "window" => string(typeof(s.window)), "mesh_at_10k" => collect(meshA),
        # scalars that rebuild the spec in pd_npt.jl (Gaussian α, or the prolate shapes)
        "alpha" => s.split isa ES.GaussianSplit ? s.split.alpha : NaN,
        "split_shape" => s.split isa ES.ProlateSplit ? s.split.shape : NaN,
        "window_shape" => s.window isa ES.ProlateWindow ? s.window.shape : NaN)
    @printf("%s: %s\n  tuning error %.3e, force time %.3f ms (mesh %s); Δ_A %.3e  Δ_B %.3e\n", name, string(chosen),
            t.error, t.time_ms, t.mesh, dA, dB)
    flush(stdout)
end
open(io -> TOML.print(io, rec), out, "w")
