#!/usr/bin/env julia
#
# Head-to-head electrostatics benchmark on the paper's SPC/E water series
# (Liang et al. 2026, Zenodo 19559644, gromacs_bench/single_core_bench) on one GPU.
#
#   PD_ESP_DATA    path of the unpacked ESP_Data directory
#   PD_BENCH_WORK  work/results directory (outside the repository)
#   PD_GMX_GPU     GPU GROMACS binary; PD_GMX_DOUBLE: double-precision GROMACS
#   PD_LMP_KOKKOS  KOKKOS/CUDA LAMMPS (mixed precision for timings)
#
# Accuracy: every engine's own measured relative RMS Coulomb force error Δ
# against a converged reference (ParticleDynamics Float64 ESP, ε = 1e-10,
# validated against GROMACS double precision) on the 24,000-atom box; the
# coarsest setting meeting each target is then timed on every size.
#
# Workloads (frozen coordinates unless stated):
#   W1  electrostatics only: PD real space + mesh; LAMMPS coul/long + pppm
#       (synchronized Pair + Kspace); GROMACS estimated as (full MD step) −
#       (same step with zero charges).
#   W2  all nonbonded (SPC/E O–O LJ + Coulomb, r_c = 1 nm): one PD force
#       evaluation; LAMMPS lj/cut/coul/long + pppm; GROMACS GPU-resident MD
#       step (includes integration and SETTLE).
#
# Usage: julia --project=. benchmark/electrostatics/run_water.jl [max_atoms]

using CUDA, LinearAlgebra, Printf, Serialization, ParticleDynamics
include(joinpath(@__DIR__, "Benchmark.jl"))
using .Fixtures, .Engines, .PDHarness
const ES = ParticleDynamics.Electrostatics

const DATA = ENV["PD_ESP_DATA"]
const WORK = mkpath(ENV["PD_BENCH_WORK"])
const GMX = ENV["PD_GMX_GPU"]
const SERIES = joinpath(DATA, "gromacs_bench", "single_core_bench")
const SIZES = [("10_10_10", "New1000Box.gro"), ("20_20_20", "New8000Box.gro"), ("30_30_30", "New27000Box.gro"),
               ("40_40_40", "New64000Box.gro"), ("50_50_50", "New125000Box.gro"), ("60_60_60", "New216000Box.gro"),
               ("70_70_70", "New343000Box.gro")]
const TARGETS = (1e-3, 1e-4)
const KJ_PER_KCAL = 4.184
# SPC/E (oplsaa.ff/spce.itp): O–O Lennard-Jones only
const SIGMA_O = 0.316557          # nm
const EPS_O = 0.650194            # kJ/mol

const STORE = joinpath(WORK, "water_results.jls")
results = isfile(STORE) ? deserialize(STORE) : Dict{Any,Any}()
save!() = serialize(STORE, results)
function cached(f, key)
    haskey(results, key) && return results[key]
    v = f()
    results[key] = v
    save!()
    return v
end

# The archived boxes are perfect lattices; use equilibrated liquid frames
# (derived fixtures) produced by short GPU NVT runs.
function paths(sz; frame=:a)
    gro0, top = joinpath(SERIES, sz[1], sz[2]), joinpath(SERIES, sz[1], "topol.top")
    a, b = cached(("equilibrated", sz[1])) do
        equilibrate(joinpath(WORK, "equil", sz[1]), gro0, top; gmx=GMX)
    end
    return (frame === :a ? a : b), top
end
water_types(f) = [q < 0 ? 1 : 2 for q in f.q]                     # 1 = O, 2 = H

# ---------------------------------------------------------------------------
# accuracy calibration on the 24,000-atom box
# ---------------------------------------------------------------------------
gro24, top24 = paths(SIZES[2])
f24 = gromacs_fixture(gro24; top=top24)
Fref = cached(("reference", "w24000 frame a")) do
    pd_forces(pd_state(f24, esp_spec(f24; rc=1.0, tolerance=1e-10); T=Float64))
end
gro24b, _ = paths(SIZES[2]; frame=:b)
f24b = gromacs_fixture(gro24b; top=top24)
Frefb = cached(("reference", "w24000 frame b")) do
    pd_forces(pd_state(f24b, esp_spec(f24b; rc=1.0, tolerance=1e-10); T=Float64))
end

function first_meeting(target, knobs, err_of)
    for k in knobs
        e = err_of(k)
        e <= target && return (knob=k, err=e)
    end
    return nothing
end

const GMX_SPACINGS = (0.30, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10,
                      0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06)
calib = Dict{Any,Any}()
α1 = ES.alpha_from_tolerance(1.0, 1e-5)
mesh_for(f, h) = ntuple(d -> ES.fft_friendly(ceil(Int, f.box[d] / h)), 3)
pd_spec(name, f, k) = name == "PD PME" ? pme_spec(f; rc=1.0, alpha=α1, mesh=mesh_for(f, k), order=4) :
                      name == "PD PPPM-ik" ? pppm_spec(f; rc=1.0, alpha=α1, mesh=mesh_for(f, k), order=5) :
                      name == "PD ESP rc=1.0" ? esp_spec(f; rc=1.0, tolerance=k) : esp_spec(f; rc=0.5, tolerance=k)

for Δ in TARGETS
    calib[("gromacs", Δ)] = cached(("calib", "gromacs", Δ)) do
        first_meeting(Δ, GMX_SPACINGS,
                      sp -> begin
            set = Dict{Symbol,Any}(:rc => 1.0, :spacing => sp, :order => 4, :rtol => 1e-5)
            F, _, _ = gromacs_coulomb(joinpath(WORK, "calib", "gmx_$(sp)"), gro24, top24, set; gmx=GMX, gpu=true)
            relerr(F, Fref)
        end)
    end
    calib[("lammps", Δ)] = cached(("calib", "lammps", Δ)) do
        data = lammps_data(joinpath(mkpath(joinpath(WORK, "calib")), "w24000.data"), f24; length_scale=10.0)
        first_meeting(Δ, (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6),
                      acc -> begin
            r = lammps_run(joinpath(WORK, "calib"), "lmp_$(acc)", data; rc=10.0, accuracy=acc, nsteps=1)
            relerr(r.F .* (KJ_PER_KCAL * 10), Fref)
        end)
    end
    for (name, mk, knobs) in (
            ("PD PME", h -> pme_spec(f24; rc=1.0, alpha=α1, mesh=ntuple(d -> ES.fft_friendly(ceil(Int, f24.box[d] / h)), 3), order=4),
             GMX_SPACINGS),
            ("PD PPPM-ik", h -> pppm_spec(f24; rc=1.0, alpha=α1, mesh=ntuple(d -> ES.fft_friendly(ceil(Int, f24.box[d] / h)), 3), order=5),
             GMX_SPACINGS),
            ("PD ESP rc=1.0", ε -> esp_spec(f24; rc=1.0, tolerance=ε), Δ .* (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1)),
            ("PD ESP rc=0.5", ε -> esp_spec(f24; rc=0.5, tolerance=ε), Δ .* (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1)))
        calib[(name, Δ)] = cached(("calib", name, Δ)) do
            first_meeting(Δ, knobs, k -> relerr(pd_forces(pd_state(f24, mk(k); T=Float32)), Fref))
        end
    end
end
# held-out frame B: error of every chosen setting on an independent configuration
heldout = Dict{Any,Any}()
for Δ in TARGETS
    c = calib[("gromacs", Δ)]
    c === nothing || (heldout[("gromacs", Δ)] = cached(("heldout", "gromacs", Δ)) do
        set = Dict{Symbol,Any}(:rc => 1.0, :spacing => c.knob, :order => 4, :rtol => 1e-5)
        relerr(gromacs_coulomb(joinpath(WORK, "heldout", "gmx_$(c.knob)"), gro24b, top24, set; gmx=GMX, gpu=true)[1], Frefb)
    end)
    c = calib[("lammps", Δ)]
    c === nothing || (heldout[("lammps", Δ)] = cached(("heldout", "lammps", Δ)) do
        data = lammps_data(joinpath(mkpath(joinpath(WORK, "heldout")), "w24000b.data"), f24b; length_scale=10.0)
        r = lammps_run(joinpath(WORK, "heldout"), "lmp_$(c.knob)", data; rc=10.0, accuracy=c.knob, nsteps=1)
        relerr(r.F .* (KJ_PER_KCAL * 10), Frefb)
    end)
    for name in ("PD PME", "PD PPPM-ik", "PD ESP rc=1.0", "PD ESP rc=0.5")
        c = calib[(name, Δ)]
        c === nothing && continue
        heldout[(name, Δ)] = cached(("heldout", name, Δ)) do
            relerr(pd_forces(pd_state(f24b, pd_spec(name, f24b, c.knob); T=Float32)), Frefb)
        end
    end
end

# the cluster-pair backend at the calibrated settings (same physics; its Δ is
# re-measured so the timings below are at verified accuracy)
for Δ in TARGETS, name in ("PD PME", "PD PPPM-ik", "PD ESP rc=1.0", "PD ESP rc=0.5")
    c = calib[(name, Δ)]
    c === nothing && continue
    heldout[(name * " cluster", Δ)] = cached(("heldout", name * " cluster", Δ)) do
        relerr(pd_forces(pd_state(f24b, pd_spec(name, f24b, c.knob); T=Float32, backend=:cluster)), Frefb)
    end
end

println("Calibration (24,000-atom SPC/E water, equilibrated; measured Δ on tuning frame A / held-out frame B):")
for ((eng, Δ), c) in sort!(collect(calib); by=x -> (x[1][2], x[1][1]), rev=true)
    @printf("  Δ=%.0e %-15s knob=%-8s measured %.3e / %.3e\n", Δ, eng, c === nothing ? "-" : string(c.knob),
            c === nothing ? NaN : c.err, get(heldout, (eng, Δ), NaN))
end

# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------
maxatoms = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)

for sz in SIZES
    gro, top = paths(sz)
    f = gromacs_fixture(gro; top=top)
    N = length(f)
    N > maxatoms && continue
    fs = spatially_sorted(f; cell=1.0)
    types = water_types(fs)
    wdir = mkpath(joinpath(WORK, "w$(N)"))
    nsteps = N <= 100_000 ? 5000 : 2000
    zq = cached(("zero_charge_top", N)) do
        zero_charge_topology(gro, top, joinpath(wdir, "zeroq"); gmx=GMX)
    end
    base = cached(("gromacs_lj_only", N)) do
        gromacs_step_time(wdir, "lj_only", gro, zq, Dict{Symbol,Any}(:rc => 1.0, :charges_off => true); gmx=GMX, nsteps=nsteps)[1]
    end
    for Δ in TARGETS
        c = calib[("gromacs", Δ)]
        if c !== nothing
            cached(("gromacs", N, Δ)) do
                full = gromacs_step_time(wdir, "pme_$(Δ)", gro, top,
                    Dict{Symbol,Any}(:rc => 1.0, :spacing => c.knob, :order => 4, :rtol => 1e-5); gmx=GMX, nsteps=nsteps)[1]
                (full_step=full, lj_only_step=base, coulomb_estimate=full - base)
            end
        end
        c = calib[("lammps", Δ)]
        if c !== nothing
            data = lammps_data(joinpath(wdir, "water.data"), fs; types=types, length_scale=10.0)
            cached(("lammps W1", N, Δ)) do
                r = lammps_run(wdir, "lmp_w1_$(Δ)", data; rc=10.0, accuracy=c.knob, nsteps=100, dump_forces=false)
                (pair=r.pair_ms, kspace=r.kspace_ms, total=r.pair_ms + r.kspace_ms, grid=r.grid)
            end
            cached(("lammps W2", N, Δ)) do
                r = lammps_run(wdir, "lmp_w2_$(Δ)", data; rc=10.0, accuracy=c.knob, nsteps=100, dump_forces=false,
                               pair="lj/cut/coul/long 10.0",
                               pair_coeffs=[@sprintf("1 1 %.6f %.6f", EPS_O / KJ_PER_KCAL, 10 * SIGMA_O), "* 2 0.0 0.0"])
                (pair=r.pair_ms, kspace=r.kspace_ms, total=r.pair_ms + r.kspace_ms, grid=r.grid)
            end
        end
        for name in ("PD PME", "PD PPPM-ik", "PD ESP rc=1.0", "PD ESP rc=0.5")
            c = calib[(name, Δ)]
            c === nothing && continue
            cached((name * " W1", N, Δ)) do
                st = pd_state(fs, pd_spec(name, fs, c.knob); T=Float32)
                t = pd_times(st)
                st = nothing; GC.gc(); CUDA.reclaim()
                t
            end
            # W2 needs a cell list for the LJ pair table: boxes under three cells
            # per dimension at r_c + skin (the 3,000-atom box) are reported as n/a
            all(L -> floor(L / 1.1) >= 3, fs.box) || continue
            cached((name * " W2", N, Δ)) do
                st = pd_state_lj(fs, pd_spec(name, fs, c.knob); types=types,
                                 sigma=[SIGMA_O 0.0; 0.0 0.0], epsilon=[EPS_O 0.0; 0.0 0.0], rc_lj=1.0, T=Float32)
                t = pd_nonbonded_time(st)
                st = nothing; GC.gc(); CUDA.reclaim()
                t
            end
        end
        # cluster-pair backend (fused LJ + real-space Coulomb, mesh overlapped)
        for name in ("PD PME", "PD PPPM-ik", "PD ESP rc=1.0", "PD ESP rc=0.5")
            c = calib[(name, Δ)]
            c === nothing && continue
            cached((name * " W1 cluster", N, Δ)) do
                st = pd_state(fs, pd_spec(name, fs, c.knob); T=Float32, backend=:cluster)
                t = pd_nonbonded_time(st)
                st = nothing; GC.gc(); CUDA.reclaim()
                t
            end
            cached((name * " W2 cluster", N, Δ)) do
                st = pd_state_lj(fs, pd_spec(name, fs, c.knob); types=types, sigma=[SIGMA_O 0.0; 0.0 0.0],
                                 epsilon=[EPS_O 0.0; 0.0 0.0], rc_lj=1.0, T=Float32, backend=:cluster)
                t = pd_nonbonded_time(st)
                st = nothing; GC.gc(); CUDA.reclaim()
                t
            end
        end
        for (key, tag) in ((("gromacs cycles", N, Δ), "pme_$(Δ)"), (("gromacs cycles lj_only", N), "lj_only"))
            haskey(results, ("gromacs", N, Δ)) || continue
            cached(key) do
                gromacs_cycle_accounting(read(joinpath(wdir, tag * ".log"), String))
            end
        end
        @printf("N=%-8d Δ=%.0e done\n", N, Δ)
    end
end

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
open(joinpath(WORK, "water_report.md"), "w") do io
    println(io, "| N | Δ | GROMACS step (W2) | GROMACS Coulomb est. (W1) | LAMMPS W2 | LAMMPS W1 | PD PME W1 | PD PPPM-ik W1 | PD ESP r_c=1.0 W1 | PD ESP r_c=0.5 W1 | PD PME W2 | PD ESP r_c=1.0 W2 |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sz in SIZES, Δ in TARGETS
        Ns = [k[2] for k in keys(results) if k isa Tuple && length(k) == 3 && k[1] == "PD PME W1" && k[3] == Δ]
        isempty(Ns) && continue
    end
    allN = sort!(unique([k[2] for k in keys(results) if k isa Tuple && length(k) == 3 && k[1] == "PD PME W1"]))
    g(key, field) = haskey(results, key) && results[key] !== nothing ? @sprintf("%.3f", getproperty(results[key], field)) : "–"
    for N in allN, Δ in TARGETS
        println(io, "| $(N) | $(Δ) | ", g(("gromacs", N, Δ), :full_step), " | ", g(("gromacs", N, Δ), :coulomb_estimate),
                " | ", g(("lammps W2", N, Δ), :total), " | ", g(("lammps W1", N, Δ), :total),
                " | ", g(("PD PME W1", N, Δ), :force), " | ", g(("PD PPPM-ik W1", N, Δ), :force),
                " | ", g(("PD ESP rc=1.0 W1", N, Δ), :force), " | ", g(("PD ESP rc=0.5 W1", N, Δ), :force),
                " | ", g(("PD PME W2", N, Δ), :median), " | ", g(("PD ESP rc=1.0 W2", N, Δ), :median), " |")
    end
end
open(joinpath(WORK, "water_report.md"), "a") do io
    g(key, field) = haskey(results, key) && results[key] !== nothing ? @sprintf("%.3f", getproperty(results[key], field)) : "–"
    allN = sort!(unique([k[2] for k in keys(results) if k isa Tuple && length(k) == 3 && k[1] == "PD PME W2 cluster"]))
    println(io, "\nCluster-pair backend. W2 = LJ + Coulomb, W1 = Coulomb only; ms per force evaluation (PD) or per MD step (GROMACS).\n")
    println(io, "| N | Δ | GROMACS step | GROMACS GPU est. | LAMMPS W2 | PD PME W2 | PD PPPM-ik W2 | PD ESP 1.0 W2 | PD ESP 0.5 W2 | PD PME W1 | PD PPPM-ik W1 | PD ESP 1.0 W1 | PD ESP 0.5 W1 |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N in allN, Δ in TARGETS
        println(io, "| $(N) | $(Δ) | ", g(("gromacs", N, Δ), :full_step), " | ", g(("gromacs cycles", N, Δ), :gpu_estimate),
                " | ", g(("lammps W2", N, Δ), :total),
                " | ", g(("PD PME W2 cluster", N, Δ), :median), " | ", g(("PD PPPM-ik W2 cluster", N, Δ), :median),
                " | ", g(("PD ESP rc=1.0 W2 cluster", N, Δ), :median), " | ", g(("PD ESP rc=0.5 W2 cluster", N, Δ), :median),
                " | ", g(("PD PME W1 cluster", N, Δ), :median), " | ", g(("PD PPPM-ik W1 cluster", N, Δ), :median),
                " | ", g(("PD ESP rc=1.0 W1 cluster", N, Δ), :median), " | ", g(("PD ESP rc=0.5 W1 cluster", N, Δ), :median), " |")
    end
    println(io, "\nHeld-out Δ of the cluster backend (24k frame B): ",
            join([@sprintf("%s %.0e: %.2e", k[1], k[2], v) for (k, v) in heldout if endswith(k[1], "cluster")], "; "))
end
println(read(joinpath(WORK, "water_report.md"), String))
