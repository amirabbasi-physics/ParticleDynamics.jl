#!/usr/bin/env julia
#
# Head-to-head electrostatics benchmark on the paper's heterogeneous systems
# (membrane 81,743 atoms; lysozyme 1,036,152; LiTFSI 1,011,392) on one GPU,
# at the paper's target errors (4e-4, 2e-4, 1e-5) and each system's cutoff.
# Same environment variables as run_water.jl.
#
# Engines and what is measured (W1 = electrostatics only):
#   PD          PME (order 4), PPPM-ik (order 5), ESP — Float32, frozen coordinates
#   LAMMPS      KOKKOS/CUDA mixed precision, coul/long + pppm, frozen coordinates
#               (systems whose exceptions are full exclusions only)
#   GROMACS     GPU, mixed precision: measured Coulomb error (εᵣ trick) and the
#               W1 estimate (full GPU-resident MD step − same step with zero
#               charges) for systems with a topology; the membrane ships only
#               run inputs, so its GROMACS entry is the archived benchMEM-PME
#               input (error via identical-settings SPME, timing = full MD step).

using CUDA, LinearAlgebra, Printf, Serialization, ParticleDynamics
include(joinpath(@__DIR__, "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics

const DATA = ENV["PD_ESP_DATA"]
const WORK = mkpath(ENV["PD_BENCH_WORK"])
const GMX = ENV["PD_GMX_GPU"]
const KJ_PER_KCAL = 4.184

const SYSTEMS = [
    (name="membrane", gro=joinpath(DATA, "gromacs_bench/membrane_scaling_bench/gro-pdb/output.gro"), gro_b=nothing,
     top=nothing, tpr=joinpath(DATA, "gromacs_bench/membrane_scaling_bench/benchMEM-PME.tpr"), rc=1.0, target=4e-4),
    (name="lysozyme", gro=joinpath(DATA, "LysoProtein/final.gro"), gro_b=joinpath(DATA, "LysoProtein/pswf.gro"),
     top=joinpath(DATA, "LysoProtein/topol27.top"), tpr=nothing, rc=0.9, target=2e-4),
    (name="LiTFSI", gro=joinpath(DATA, "Li-ion-Electrolyte/last_frame.gro"), gro_b=joinpath(DATA, "Li-ion-Electrolyte/pswf64.gro"),
     top=joinpath(DATA, "Li-ion-Electrolyte/pswf64.top"), tpr=nothing, rc=0.9, target=1e-5),
]

const STORE = joinpath(WORK, "heterogeneous_results.jls")
results = isfile(STORE) ? deserialize(STORE) : Dict{Any,Any}()
save!() = serialize(STORE, results)
function cached(f, key)
    haskey(results, key) && return results[key]
    v = f()
    results[key] = v
    save!()
    return v
end

mesh_for(f, h) = ntuple(d -> ES.fft_friendly(ceil(Int, f.box[d] / h)), 3)
first_meeting(target, knobs, err_of) = begin
    for k in knobs
        e = err_of(k)
        e <= target && return (knob=k, err=e)
    end
    nothing
end
const SPACINGS = (0.25, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.075, 0.07,
                  0.065, 0.06, 0.055, 0.05, 0.045, 0.04)

only_full_exclusions(f) = all(e -> e[3] == 0.0, f.exceptions)

for sys in SYSTEMS
    wdir = mkpath(joinpath(WORK, sys.name))
    load(gro) = sys.top === nothing ? gromacs_fixture(gro; tpr=sys.tpr) : gromacs_fixture(gro; top=sys.top)
    fa = load(sys.gro)
    fb = sys.gro_b === nothing ? nothing : load(sys.gro_b)
    println(fa)
    rc = sys.rc
    Δ = sys.target
    α = ES.alpha_from_tolerance(rc, 1e-5)
    refa = cached((sys.name, "reference a")) do
        pd_forces(pd_state(fa, esp_spec(fa; rc=rc, tolerance=1e-10); T=Float64))
    end
    refb = fb === nothing ? nothing : cached((sys.name, "reference b")) do
        pd_forces(pd_state(fb, esp_spec(fb; rc=rc, tolerance=1e-10); T=Float64))
    end
    GC.gc(); CUDA.reclaim()

    pdspec(name, f, k) = name == "PD PME" ? pme_spec(f; rc=rc, alpha=α, mesh=mesh_for(f, k), order=4) :
                         name == "PD PPPM-ik" ? pppm_spec(f; rc=rc, alpha=α, mesh=mesh_for(f, k), order=5) :
                         esp_spec(f; rc=rc, tolerance=k)
    knobs(name) = name == "PD ESP" ? Δ .* (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05) : SPACINGS

    # ---- ParticleDynamics ----
    for name in ("PD PME", "PD PPPM-ik", "PD ESP")
        c = cached((sys.name, "calib", name)) do
            first_meeting(Δ, knobs(name), k -> begin
                st = pd_state(fa, pdspec(name, fa, k); T=Float32)
                e = relerr(pd_forces(st), refa)
                st = nothing; GC.gc(); CUDA.reclaim()
                e
            end)
        end
        c === nothing && continue
        fb === nothing || cached((sys.name, "heldout", name)) do
            relerr(pd_forces(pd_state(fb, pdspec(name, fb, c.knob); T=Float32)), refb)
        end
        cached((sys.name, "time", name)) do
            fs = spatially_sorted(fa; cell=rc)
            st = pd_state(fs, pdspec(name, fs, c.knob); T=Float32)
            t = pd_times(st)
            st = nothing; GC.gc(); CUDA.reclaim()
            t
        end
    end

    # ---- LAMMPS (full exclusions only) ----
    if only_full_exclusions(fa)
        data = lammps_data(joinpath(wdir, "a.data"), fa; length_scale=10.0)
        c = cached((sys.name, "calib", "lammps")) do
            first_meeting(Δ, (1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7),
                          acc -> relerr(lammps_run(wdir, "lmp_$(acc)", data; rc=10rc, accuracy=acc, nsteps=1).F .*
                                        (KJ_PER_KCAL * 10), refa))
        end
        if c !== nothing
            fb === nothing || cached((sys.name, "heldout", "lammps")) do
                datab = lammps_data(joinpath(wdir, "b.data"), fb; length_scale=10.0)
                relerr(lammps_run(wdir, "lmpb_$(c.knob)", datab; rc=10rc, accuracy=c.knob, nsteps=1).F .*
                       (KJ_PER_KCAL * 10), refb)
            end
            cached((sys.name, "time", "lammps")) do
                fs = spatially_sorted(fa; cell=rc)
                datas = lammps_data(joinpath(wdir, "a_sorted.data"), fs; length_scale=10.0)
                r = lammps_run(wdir, "lmp_time", datas; rc=10rc, accuracy=c.knob, nsteps=100, dump_forces=false)
                (pair=r.pair_ms, kspace=r.kspace_ms, total=r.pair_ms + r.kspace_ms, grid=r.grid)
            end
        end
    end

    # ---- GROMACS ----
    if sys.top !== nothing
        c = cached((sys.name, "calib", "gromacs")) do
            first_meeting(Δ, SPACINGS, sp -> begin
                set = Dict{Symbol,Any}(:rc => rc, :spacing => sp, :order => 4, :rtol => 1e-5)
                relerr(gromacs_coulomb(joinpath(wdir, "gmx_$(sp)"), sys.gro, sys.top, set; gmx=GMX, gpu=true)[1], refa)
            end)
        end
        if c !== nothing
            fb === nothing || cached((sys.name, "heldout", "gromacs")) do
                set = Dict{Symbol,Any}(:rc => rc, :spacing => c.knob, :order => 4, :rtol => 1e-5)
                relerr(gromacs_coulomb(joinpath(wdir, "gmxb_$(c.knob)"), sys.gro_b, sys.top, set; gmx=GMX, gpu=true)[1], refb)
            end
            cached((sys.name, "time", "gromacs")) do
                zq = zero_charge_topology(sys.gro, sys.top, joinpath(wdir, "zeroq"); gmx=GMX)
                base = gromacs_step_time(wdir, "lj_only", sys.gro, zq, Dict{Symbol,Any}(:rc => rc, :charges_off => true);
                                         gmx=GMX, nsteps=2000)[1]
                full = gromacs_step_time(wdir, "pme", sys.gro, sys.top,
                    Dict{Symbol,Any}(:rc => rc, :spacing => c.knob, :order => 4, :rtol => 1e-5); gmx=GMX, nsteps=2000)[1]
                (full_step=full, lj_only_step=base, coulomb_estimate=full - base)
            end
        end
    else
        # archived run input only: identical-settings SPME gives its Coulomb error
        cached((sys.name, "calib", "gromacs (benchMEM-PME settings)")) do
            ts = Engines.tpr_settings(sys.tpr)
            spec = pme_spec(fa; rc=ts.rc, alpha=ES.alpha_from_tolerance(ts.rc, ts.rtol), mesh=ts.grid, order=ts.order)
            (knob="order $(ts.order), grid $(ts.grid)", err=relerr(pd_forces(pd_state(fa, spec; T=Float64)), refa))
        end
    end
    @printf("%s done\n", sys.name)
end

open(joinpath(WORK, "heterogeneous_report.md"), "w") do io
    println(io, "| System | target Δ | engine | setting | Δ (tuning frame) | Δ (held-out frame) | W1 time (ms) | breakdown |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for sys in SYSTEMS
        for eng in ("PD PME", "PD PPPM-ik", "PD ESP", "lammps", "gromacs", "gromacs (benchMEM-PME settings)")
            c = get(results, (sys.name, "calib", eng), nothing)
            c === nothing && continue
            h = get(results, (sys.name, "heldout", eng), nothing)
            t = get(results, (sys.name, "time", eng), nothing)
            tstr, bstr = if t === nothing
                "–", ""
            elseif startswith(eng, "PD")
                @sprintf("%.3f", t.force), @sprintf("real %.3f, spread %.3f, FFT %.3f, gather/other %.3f; mesh %s", t.real, t.spread, t.fft, t.gather_other, string(t.mesh))
            elseif eng == "lammps"
                @sprintf("%.3f", t.total), @sprintf("Pair %.3f, Kspace %.3f; grid %s", t.pair, t.kspace, string(t.grid))
            else
                @sprintf("%.3f (est.)", t.coulomb_estimate), @sprintf("full MD step %.3f, LJ-only step %.3f", t.full_step, t.lj_only_step)
            end
            @printf(io, "| %s | %.0e | %s | %s | %.3e | %s | %s | %s |\n", sys.name, sys.target, eng, string(c.knob), c.err,
                    h === nothing ? "–" : @sprintf("%.3e", h), tstr, bstr)
        end
    end
end
println(read(joinpath(WORK, "heterogeneous_report.md"), String))
