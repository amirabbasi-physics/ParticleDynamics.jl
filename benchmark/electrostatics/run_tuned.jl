#!/usr/bin/env julia
#
# Tuned head-to-head on the equilibrated SPC/E water series: each engine with
# its own run-time optimisation at the measured accuracy target.
#
#   PD       tune_electrostatics(...; objective=:time) on the timed system
#            itself (cluster-pair backend, LJ + Coulomb): order, Ewald α and
#            mesh for PME/PPPM, ε and window width for ESP; the fastest
#            setting that meets Δ on that configuration wins.
#   GROMACS  the calibrated setting (spacing meeting Δ, order 4, rtol 1e-5)
#            in its production configuration: automatic Verlet buffer
#            (verlet-buffer-tolerance 0.005, which sets rlist). Its PME load
#            balancing (`-tunepme`) does not complete on this single-GPU,
#            GPU-update setup: with rlist fixed at r_c it never ends, and with
#            the automatic buffer a full run ends with "final = initial", so
#            the production timing is taken with `-notunepme`.
#
# Every chosen setting is then validated on the held-out 24k frame B against
# the Float64 reference of run_water.jl (same Δ metric as all other tables).
# Requires the caches of run_water.jl. Same environment variables.

using CUDA, LinearAlgebra, Printf, Serialization, ParticleDynamics, Dates, TOML, Statistics
include(joinpath(@__DIR__, "../gromacs/Protocol.jl"))
include(joinpath(@__DIR__, "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics
const WF = ParticleDynamics.Workflow

const DATA = ENV["PD_ESP_DATA"]
const WORK = ENV["PD_BENCH_WORK"]
const GMX = ENV["PD_GMX_GPU"]
const SERIES = joinpath(DATA, "gromacs_bench", "single_core_bench")
const SIGMA_O = 0.316557
const EPS_O = 0.650194
const TARGETS = (1e-3, 1e-4)
const SIZES = length(ARGS) >= 1 ? split(ARGS[1], ",") : ["20_20_20", "40_40_40", "70_70_70"]
const RUN = Protocol.fresh_directory(get(ENV, "PD_BENCH_RUN",
    joinpath(WORK, "runs", Dates.format(now(), "yyyymmdd-HHMMSS-sss"))))
const REPEATS = parse(Int, get(ENV, "PD_BENCH_REPEATS", "3"))
REPEATS > 0 || error("PD_BENCH_REPEATS must be positive")
CUDA.functional() || error("CUDA device unavailable; no water performance result can be recorded")
open(joinpath(RUN, "provenance.toml"), "w") do io
    TOML.print(io, Dict("workload"=>"frozen W2 nonbonded forces versus full GROMACS MD steps; NOT NPT",
        "source_sha256"=>Protocol.source_fingerprint(normpath(joinpath(@__DIR__, "../.."))),
        "julia"=>string(VERSION), "gpu"=>CUDA.name(CUDA.device()),
        "gromacs_version"=>read(`$GMX --version`, String), "repeats"=>REPEATS,
        "candidate_cache"=>joinpath(WORK, "tuned_results.jls"), "created"=>string(now())); sorted=true)
end
fresh = Dict{Any,Any}()
gmx_errors = Dict{Float64,Float64}()

water = deserialize(joinpath(WORK, "water_results.jls"))
const STORE = joinpath(WORK, "tuned_results.jls")
results = isfile(STORE) ? deserialize(STORE) : Dict{Any,Any}()
function cached(f, key)
    haskey(results, key) && return results[key]
    v = f()
    results[key] = v
    serialize(STORE, results)
    return v
end

top_of(sz) = joinpath(SERIES, sz, "topol.top")
frames(sz) = water[("equilibrated", sz)]
f24b = gromacs_fixture(frames("20_20_20")[2]; top=top_of("20_20_20"))
Frefb = water[("reference", "w24000 frame b")]
water_types(f) = [q < 0 ? 1 : 2 for q in f.q]

# descriptor with the chosen mesh expressed as a spacing (transferable)
function as_spacing(force, mesh, box)
    h = maximum(box ./ mesh)
    return WF._rebuild_force(force; mesh=nothing, mesh_spacing=h)
end

for sz in SIZES
    gro, _ = frames(sz)
    fs = spatially_sorted(gromacs_fixture(gro; top=top_of(sz)); cell=1.0)
    N = length(fs)
    for Δ in TARGETS
        # ---------- PD ----------
        for (name, force) in (("PD PME", PME(coulomb_prefactor=fs.prefactor, real_cutoff=1.0, alpha=1.0)),
                              ("PD PPPM-ad", PPPM(coulomb_prefactor=fs.prefactor, real_cutoff=1.0, alpha=1.0, differentiation=:ad)),
                              ("PD PPPM-ik", PPPM(coulomb_prefactor=fs.prefactor, real_cutoff=1.0, alpha=1.0, differentiation=:ik)),
                              ("PD ESP", ESP(coulomb_prefactor=fs.prefactor, real_cutoff=1.0)))
            r = cached((name, N, Δ)) do
                spec0 = pme_spec(fs; rc=1.0, alpha=3.0, mesh=ntuple(d -> ES.fft_friendly(ceil(Int, fs.box[d] / 0.14)), 3), order=4)
                st = pd_state_lj(fs, spec0; types=water_types(fs), sigma=[SIGMA_O 0.0; 0.0 0.0], epsilon=[EPS_O 0.0; 0.0 0.0],
                                 rc_lj=1.0, T=Float32, backend=:cluster)
                t = tune_electrostatics(st, force; target=Δ, objective=:time)
                st = nothing; GC.gc(); CUDA.reclaim()
                (force=as_spacing(t.force, t.mesh, fs.box), error=t.error, time_ms=t.time_ms, mesh=t.mesh)
            end
            heldout = let
                spec = WF.electrostatics_spec(WF._rebuild_force(r.force; net_charge=:background))
                relerr(pd_forces(pd_state(f24b, spec; T=Float32, backend=:cluster)), Frefb)
            end
            isfinite(heldout) && heldout <= Δ || error("$name fails held-out target: $heldout > $Δ")
            GC.gc(); CUDA.reclaim()
            # Cache the selected parameters, never the reported performance.
            # Preserve the actual tuned mesh on the timed size.
            spec = WF.electrostatics_spec(WF._rebuild_force(r.force; mesh=r.mesh, mesh_spacing=nothing))
            st = pd_state_lj(fs, spec; types=water_types(fs), sigma=[SIGMA_O 0.0; 0.0 0.0],
                epsilon=[EPS_O 0.0; 0.0 0.0], rc_lj=1.0, T=Float32, backend=:cluster)
            timings = [pd_nonbonded_time(st) for _ in 1:REPEATS]
            measured = median(t.median for t in timings)
            fresh[(name,N,Δ)] = (time_ms=measured, timings=timings, mesh=r.mesh)
            fresh[(name,N,Δ,"heldout")] = heldout
            serialize(joinpath(RUN,"measurements.jls"),fresh)
            st = nothing; GC.gc(); CUDA.reclaim()
            @printf("N=%-8d Δ=%.0e %-11s fresh force %.3f ms (held-out %.2e, mesh %s)\n", N, Δ, name,
                    measured, heldout, string(r.mesh))
        end
        # ---------- GROMACS production setting, fresh repeats ----------
        c = water[("calib", "gromacs", Δ)]
        c === nothing && continue
        gerror = get!(gmx_errors, Δ) do
            settings = Dict{Symbol,Any}(:rc=>1.0, :spacing=>c.knob, :order=>4, :rtol=>1e-5)
            F, _, _ = gromacs_coulomb(joinpath(RUN,"heldout_$(Δ)"), frames("20_20_20")[2],
                top_of("20_20_20"), settings; gmx=GMX, gpu=true)
            err = relerr(F,Frefb)
            isfinite(err) && err <= Δ || error("GROMACS fails held-out target: $err > $Δ")
            err
        end
        g = let
            wdir = mkpath(joinpath(RUN, "w$(N)"))
            samples = map(1:REPEATS) do rep
                ms, log = gromacs_step_time(wdir, "pme_prod_$(Δ)_r$(rep)", gro, top_of(sz),
                                        Dict{Symbol,Any}(:rc => 1.0, :spacing => c.knob, :order => 4, :rtol => 1e-5,
                                                         :auto_buffer => true);
                                        gmx=GMX, nsteps=N <= 100_000 ? 8000 : 4000)
                Protocol.require_gpu_offload(log; pme=true)
                cyc = gromacs_cycle_accounting(log)
                (step=ms, gpu_estimate=cyc === nothing ? NaN : cyc.gpu_estimate)
            end
            (step=median(s.step for s in samples), gpu_estimate=median(s.gpu_estimate for s in samples), samples=samples,
             tuned=(grid=Tuple(ES.fft_friendly.(ceil.(Int, fs.box ./ c.knob))), rcoulomb=1.0))
        end
        fresh[("gromacs tuned", N, Δ)] = g
        fresh[("gromacs tuned", N, Δ, "heldout")] = gerror
        serialize(joinpath(RUN,"measurements.jls"),fresh)
        @printf("N=%-8d Δ=%.0e GROMACS tuned: step %.3f ms, GPU est. %.3f ms, %s, held-out %s\n", N, Δ, g.step, g.gpu_estimate,
                string(g.tuned), string(get(fresh, ("gromacs tuned", N, Δ, "heldout"), "–")))
    end
end

open(joinpath(RUN, "tuned_report.md"), "w") do io
    println(io, "Fresh frozen-coordinate W2 diagnostics, not a full MD/NPT comparison. PD times exclude integration, constraints, thermostat, barostat and amortized neighbor builds. GROMACS includes an MD step. Held-out errors are remeasured on the 24k frame B for every engine.\n")
    results = fresh
    println(io, "| N | Δ | GROMACS step ms | GROMACS GPU est. ms | PD PME force ms | PD PPPM-ad force ms | PD PPPM-ik force ms | PD ESP force ms | held-out Δ (GROMACS / PME / PPPM-ad / PPPM-ik / ESP) |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    Ns = sort!(unique([k[2] for k in keys(results) if k isa Tuple && length(k) == 3 && k[1] == "PD PME"]))
    for N in Ns, Δ in TARGETS
        g = get(results, ("gromacs tuned", N, Δ), nothing)
        t(name) = haskey(results, (name, N, Δ)) ? @sprintf("%.3f", results[(name, N, Δ)].time_ms) : "–"
        h(key) = haskey(results, key) ? @sprintf("%.1e", results[key]) : "–"
        @printf(io, "| %d | %.0e | %s | %s | %s | %s | %s | %s | %s / %s / %s / %s / %s |\n", N, Δ,
                g === nothing ? "–" : @sprintf("%.3f", g.step), g === nothing ? "–" : @sprintf("%.3f", g.gpu_estimate),
                t("PD PME"), t("PD PPPM-ad"), t("PD PPPM-ik"), t("PD ESP"),
                h(("gromacs tuned", N, Δ, "heldout")), h(("PD PME", N, Δ, "heldout")), h(("PD PPPM-ad", N, Δ, "heldout")),
                h(("PD PPPM-ik", N, Δ, "heldout")), h(("PD ESP", N, Δ, "heldout")))
    end
end
println(read(joinpath(RUN, "tuned_report.md"), String))
