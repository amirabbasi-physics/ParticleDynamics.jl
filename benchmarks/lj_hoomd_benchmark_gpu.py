#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HOOMD-blue Langevin LJ Benchmark - Extended Version
Benchmarks Lennard-Jones fluid simulation with Langevin dynamics
Writes outputs next to this script:
  - hoomd_trajectory_gpu.gsd
  - hoomd_benchmark_results_gpu.txt
"""

import hoomd
import numpy as np
import time
import matplotlib.pyplot as plt  # (unused; kept if you add plots later)
from pathlib import Path


def create_fcc_lattice(n_unit_cells: int, a: float) -> np.ndarray:
    """Create FCC lattice with n_unit_cells^3 unit cells, lattice parameter a."""
    positions = []
    basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
    for i in range(n_unit_cells):
        for j in range(n_unit_cells):
            for k in range(n_unit_cells):
                for atom in basis:
                    x = a * (i + atom[0])
                    y = a * (j + atom[1])
                    z = a * (k + atom[2])
                    positions.append([x, y, z])
    return np.array(positions, dtype=float)


def benchmark_lj_langevin():
    print("=== HOOMD-blue Langevin LJ Benchmark ===")
    print(f"HOOMD-blue version: {hoomd.version.version}")

    # I/O paths anchored to this script's directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    traj_path = SCRIPT_DIR / "hoomd_trajectory_gpu.gsd"
    txt_path = SCRIPT_DIR / "hoomd_benchmark_results_gpu.txt"

    print(f"Current working directory (CWD): {Path.cwd()}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Trajectory will be saved to: {traj_path}")
    print(f"Results will be saved to:    {txt_path}")

    # Simulation parameters (match your NonEqSim.jl setup)
    N = 10000
    density = 0.85
    temperature = 1.0
    dt = 0.005
    n_equilibrate = 0
    n_production = 100000
    trajectory_interval = 10000  # save frames every 250 steps

    # LJ parameters
    epsilon = 1.0
    sigma = 1.0
    r_cut = 2.5 * sigma

    # Langevin damping parameter
    gamma = 1.0

    # Box size from density and number of particles
    volume = N / density
    box_length = volume ** (1.0 / 3.0)

    print(f"N = {N}")
    print(f"Density = {density}")
    print(f"Box length = {box_length}")
    print(f"Temperature = {temperature}")
    print(f"dt = {dt}")
    print(f"Langevin gamma = {gamma}")
    print(f"LJ parameters: ε={epsilon}, σ={sigma}, r_cut={r_cut}")
    print(f"Equilibration steps = {n_equilibrate}")
    print(f"Production steps = {n_production}")
    print(f"Trajectory interval = {trajectory_interval}")

    # Create FCC lattice
    n_unit_cells = int(round((N / 4) ** (1.0 / 3.0)))
    lattice_param = box_length / n_unit_cells
    positions = create_fcc_lattice(n_unit_cells, lattice_param)

    # Center the lattice in the box
    positions = np.asarray(positions, dtype=float)
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    positions -= center

    # Trim to exact N particles
    positions = positions[:N]

    print(f"Created lattice with {len(positions)} particles")

    # Random seed for reproducible velocities
    np.random.seed(42)
    velocity_scale = np.sqrt(temperature)
    velocities = np.random.randn(N, 3) * velocity_scale
    velocities -= velocities.mean(axis=0)  # remove COM motion

    # Initialize HOOMD
    gpu = hoomd.device.GPU()
    simulation = hoomd.Simulation(device=gpu, seed=42)

    # Create snapshot
    snapshot = hoomd.Snapshot()
    snapshot.particles.N = N
    snapshot.particles.types = ["A"]
    snapshot.configuration.box = [box_length, box_length, box_length, 0, 0, 0]
    snapshot.particles.typeid[:] = 0
    snapshot.particles.position[:] = positions
    snapshot.particles.velocity[:] = velocities
    simulation.create_state_from_snapshot(snapshot)

    print(f"Using device: {simulation.device}")

    # Neighbor list & LJ potential
    nl = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nl)
    lj.params[("A", "A")] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[("A", "A")] = r_cut

    # Langevin integrator
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=temperature)
    integrator = hoomd.md.Integrator(dt=dt, methods=[langevin], forces=[lj])
    simulation.operations.integrator = integrator

    # Thermo
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermo)

    # GSD trajectory writer (absolute path)
    gsd_writer = hoomd.write.GSD(
        filename=str(traj_path),
        trigger=hoomd.trigger.Periodic(trajectory_interval),
        mode="wb",
    )
    simulation.operations.writers.append(gsd_writer)

    print("\n=== Starting Equilibration ===")

    eq_start_time = time.time()
    for step in range(n_equilibrate):
        simulation.run(1)
        if (step + 1) % 100 == 0:
            print(f"Equilibration step {step + 1}/{n_equilibrate}")
    eq_time = time.time() - eq_start_time
    eq_sps = n_equilibrate / eq_time
    print(f"Equilibration completed: {eq_time:.3f} seconds")
    print(f"Equilibration performance: {eq_sps:.3f} steps/second")

    # Ensure energies are updated
    simulation.run(0)
    initial_ke = thermo.kinetic_energy
    initial_pe = thermo.potential_energy

    print("\n=== Starting Production Run ===")
    print(f"Initial KE: {initial_ke}")
    print(f"Initial PE: {initial_pe}")

    prod_start_time = time.time()
    for step in range(n_production):
        simulation.run(1)
        if (step + 1) % trajectory_interval == 0:
            ke = thermo.kinetic_energy
            pe = thermo.potential_energy
            print(f"Step {step + 1}: KE = {ke}, PE = {pe}")
        if (step + 1) % 10000 == 0:
            print(f"Production step {step + 1}/{n_production}")
    prod_time = time.time() - prod_start_time
    prod_sps = n_production / prod_time

    final_ke = thermo.kinetic_energy
    final_pe = thermo.potential_energy

    print("\n=== Production Run Results ===")
    print(f"Production time: {prod_time:.3f} seconds")
    print(f"Production performance: {prod_sps:.3f} steps/second")
    print(f"Final KE: {final_ke}")
    print(f"Final PE: {final_pe}")

    print("\n=== Overall Performance ===")
    total_time = eq_time + prod_time
    total_steps = n_equilibrate + n_production
    overall_sps = total_steps / total_time
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Overall performance: {overall_sps:.3f} steps/second")

    # Write human-readable TXT (next to the script)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# HOOMD-blue Langevin LJ Benchmark - Results\n")
        f.write(f"hoomd_version           = {hoomd.version.version}\n")
        f.write(f"script_dir              = {SCRIPT_DIR}\n")
        f.write(f"trajectory_file         = {traj_path.name}\n")
        f.write("\n")
        f.write("[parameters]\n")
        f.write(f"N                       = {N}\n")
        f.write(f"density                 = {density}\n")
        f.write(f"temperature             = {temperature}\n")
        f.write(f"dt                      = {dt}\n")
        f.write(f"gamma                   = {gamma}\n")
        f.write(f"epsilon                 = {epsilon}\n")
        f.write(f"sigma                   = {sigma}\n")
        f.write(f"r_cut                   = {r_cut}\n")
        f.write(f"box_length              = {box_length}\n")
        f.write(f"n_equilibrate           = {n_equilibrate}\n")
        f.write(f"n_production            = {n_production}\n")
        f.write(f"trajectory_interval     = {trajectory_interval}\n")
        f.write("\n")
        f.write("[timing]\n")
        f.write(f"equilibration_time_s    = {eq_time}\n")
        f.write(f"equilibration_sps       = {eq_sps}\n")
        f.write(f"production_time_s       = {prod_time}\n")
        f.write(f"production_sps          = {prod_sps}\n")
        f.write(f"total_time_s            = {total_time}\n")
        f.write(f"overall_sps             = {overall_sps}\n")
        f.write("\n")
        f.write("[energies]\n")
        f.write(f"initial_ke              = {initial_ke}\n")
        f.write(f"initial_pe              = {initial_pe}\n")
        f.write(f"final_ke                = {final_ke}\n")
        f.write(f"final_pe                = {final_pe}\n")

    print(f"\nResults saved to: {txt_path}")
    print(f"Trajectory saved to:    {traj_path}")

    return overall_sps, final_ke, final_pe


if __name__ == "__main__":
    benchmark_lj_langevin()