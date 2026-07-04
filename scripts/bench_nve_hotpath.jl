#!/usr/bin/env julia
#
# NVE hot-path micro-benchmark.
#
# Reproduces the exact build used by the external package benchmark
# (`benchmarking_packages_nve/scripts/run_particledynamics_benchmark.jl`)
# and times (a) whole steps and (b) each hot-path component in isolation,
# so that optimization work can be attributed kernel by kernel.
#
# Usage:
#   julia --project=. scripts/bench_nve_hotpath.jl <N> <density_label> [nsteps] [--shuffle]
#
# `--shuffle` randomly permutes the particle order of the initial condition.
# A fresh IC stores particles in lattice order (array index correlates with
# position), which flatters cache locality; after ~1e5 production steps of
# liquid diffusion that correlation is gone. Shuffling reproduces the
# steady-state ordering that the recorded production benchmarks actually ran
# in, so it is the honest baseline for a code without spatial reordering.
#
# Example:
#   julia --project=. scripts/bench_nve_hotpath.jl 1000000 medium 1000 --shuffle

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using CUDA
using ParticleDynamics
using Printf
using Random
using Statistics
using TOML

const IC_ROOT = "/net/storage/abbaa90/benchmarking_packages_nve/initial_conditions"

function read_float32_triplets(path::AbstractString, N::Int)
    values = open(path, "r") do io
        buffer = Vector{Float32}(undef, 3N)
        read!(io, buffer)
        buffer
    end
    return reshape(values, 3, N)
end

function build_state(N::Int, density_label::String; shuffle::Bool=false)
    ic_dir = joinpath(IC_ROOT, "N$(N)_density_$(density_label)")
    metadata = TOML.parsefile(joinpath(ic_dir, "metadata.toml"))
    box_length = Float64(metadata["box_length"])
    dt = 0.0025f0

    positions = read_float32_triplets(joinpath(ic_dir, "positions_float32.bin"), N)
    momenta = read_float32_triplets(joinpath(ic_dir, "momenta_float32.bin"), N)
    positions .-= Float32(box_length / 2)
    if shuffle
        perm = randperm(MersenneTwister(1234), N)
        positions = positions[:, perm]
        momenta = momenta[:, perm]
    end

    st = ParticleDynamics.build_simulation(
        N=N,
        box=(Float32(box_length), Float32(box_length), Float32(box_length)),
        cutoff=2.5f0,
        skin=0.55f0,
        cap=Int32(128),
        neigh_interval=10,
        use_neighborlist=true,
        epsilon=1.0f0,
        sigma=1.0f0,
        gamma=1.0f0,
        temperature=0.85f0,
        dt=dt,
        mass=Float32(metadata["argon_mass"]),
        nonbonded=:lj,
        backend=:cuda,
        precision=:f32,
        unwrapped_positions=true,
    )
    copyto!(st.rx, vec(positions[1, :]))
    copyto!(st.ry, vec(positions[2, :]))
    copyto!(st.rz, vec(positions[3, :]))
    copyto!(st.vx, vec(momenta[1, :]))
    copyto!(st.vy, vec(momenta[2, :]))
    copyto!(st.vz, vec(momenta[3, :]))
    ParticleDynamics.sync_unwrapped!(st)
    ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    spec = ParticleDynamics.SimulationCore.nve(st; dt=dt)
    ParticleDynamics.SimulationCore.evaluate_forces_into_f!(st, true)
    return st, spec, dt
end

# Time `f()` repeated `reps` times with full device sync; report per-call ms.
function time_component(f, reps::Int)
    f()                       # warm compile
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:reps
            f()
        end
    end
    return 1000 * t / reps
end

function main()
    N = parse(Int, ARGS[1])
    density_label = ARGS[2]
    positional = filter(a -> !startswith(a, "--"), ARGS)
    nsteps = length(positional) >= 3 ? parse(Int, positional[3]) : 1000
    shuffle = "--shuffle" in ARGS

    st, spec, dt = build_state(N, density_label; shuffle)
    sc = ParticleDynamics.SimulationCore
    nl = ParticleDynamics.NeighborLists

    @printf("device            : %s\n", CUDA.name(CUDA.device()))
    @printf("N                 : %d\n", N)
    @printf("density           : %s\n", density_label)
    @printf("eltype            : %s\n", eltype(st.rx))
    @printf("shuffled order    : %s\n", shuffle)

    # Warm up: run real steps so the neighbor list settles into steady state.
    warm = min(200, nsteps)
    for _ in 1:warm
        ParticleDynamics.step!(st, spec, dt; compute_energy=false)
    end
    CUDA.synchronize()

    counts = Array(st.nbh.counts)
    @printf("neighbors mean/max: %.1f / %d (cap %d)\n", mean(counts), maximum(counts), Int(st.nbh.cap))

    # --- whole-step throughput --------------------------------------------
    t = CUDA.@elapsed begin
        for _ in 1:nsteps
            ParticleDynamics.step!(st, spec, dt; compute_energy=false)
        end
    end
    @printf("\nfull step         : %8.3f ms/step  (%.1f steps/s)\n", 1000t / nsteps, nsteps / t)

    # --- components --------------------------------------------------------
    ms_force = time_component(100) do
        sc.evaluate_forces_into_f!(st, false)
    end
    ms_force_E = time_component(50) do
        sc.evaluate_forces_into_f!(st, true)
    end
    ms_kick = time_component(100) do
        sc._deterministic_half_kick!(st, st.f0x, st.f0y, st.f0z, dt)
    end
    ms_drift = time_component(100) do
        sc._deterministic_drift_positions!(st, dt)
    end
    ms_fills = time_component(100) do
        fill!(st.dq, 0.0f0)
        fill!(st.dU, 0.0f0)
    end
    ms_check = time_component(50) do
        nl.update_needed!(st.nbh, st.rx, st.ry, st.rz;
                          skin=st.nbh.skin,
                          Lx=st.box3[1], Ly=st.box3[2], Lz=st.box3[3],
                          step=st.step)
    end
    ms_rebuild = time_component(20) do
        nl.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
    end

    println("\ncomponents (isolated, device-synced):")
    @printf("  force (noE)     : %8.3f ms\n", ms_force)
    @printf("  force (E+virial): %8.3f ms\n", ms_force_E)
    @printf("  half kick       : %8.3f ms   (x2 per step)\n", ms_kick)
    @printf("  drift           : %8.3f ms\n", ms_drift)
    @printf("  dq/dU fills     : %8.3f ms\n", ms_fills)
    @printf("  neigh check     : %8.3f ms   (every %d steps)\n", ms_check, st.neigh_interval)
    @printf("  neigh rebuild   : %8.3f ms\n", ms_rebuild)

    est = ms_force + 2ms_kick + ms_drift + ms_fills + ms_check / st.neigh_interval
    @printf("\nsum w/o rebuild   : %8.3f ms/step\n", est)
end

main()
