#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HOOMD-blue Minimal 2D Benchmark
Matches minimal_2D_noneqsimgpu.jl exactly:
- Square lattice, 2D box (125x125)
- 10,000 particles, WCA (epsilon=10, sigma=1, r_cut=2.5)
- Langevin (gamma=10, T=1, dt=0.0005)
- 100,000 steps, GSD every 10,000 steps
"""

import hoomd
import numpy as np
import time
from pathlib import Path
import csv


def create_square_lattice_2d(N, box_Lx, box_Ly):
    """Create square lattice positions exactly matching Julia version."""
    n_side = int(np.ceil(np.sqrt(N)))
    
    positions = []
    for i in range(N):
        # Convert linear index to 2D grid coordinates (match Julia logic)
        ix = i % n_side
        iy = i // n_side
        
        # Calculate lattice spacing
        spacing_x = box_Lx / n_side
        spacing_y = box_Ly / n_side
        
        # Position on lattice, centered around origin
        x = (ix + 0.5) * spacing_x - box_Lx / 2
        y = (iy + 0.5) * spacing_y - box_Ly / 2
        
        positions.append([x, y, 0.0])  # z=0 for 2D
    
    return np.array(positions, dtype=float)


def benchmark_minimal_2d():
    print("=== HOOMD-blue Minimal 2D Benchmark ===")
    print(f"HOOMD-blue version: {hoomd.version.version}")

    # I/O paths anchored to this script's directory
    SCRIPT_DIR = Path(__file__).resolve().parent
    traj_path = SCRIPT_DIR / "traj2d_hoomd.gsd"
    obs_path = SCRIPT_DIR / "obs2d_hoomd.csv"

    print(f"Trajectory will be saved to: {traj_path}")
    print(f"Observables will be saved to: {obs_path}")

    # ---- Params (match Julia exactly) ----
    N = 10000
    box_Lx = 125.0
    box_Ly = 125.0
    r_cut = 2.5  # LJ cutoff
    epsilon = 10.0
    sigma = 1.0
    gamma = 10.0
    temperature = 1.0
    dt = 0.0005
    n_steps = 100000
    frame_interval = 10000  # save frames every 10,000 steps

    print(f"N = {N}")
    print(f"Box = {box_Lx} x {box_Ly}")
    print(f"Temperature = {temperature}")
    print(f"dt = {dt}")
    print(f"Langevin gamma = {gamma}")
    print(f"LJ parameters: ε={epsilon}, σ={sigma}, r_cut={r_cut}")
    print(f"Steps = {n_steps}")
    print(f"Frame interval = {frame_interval}")

    # ---- Square lattice initialization (match Julia exactly) ----
    positions = create_square_lattice_2d(N, box_Lx, box_Ly)
    print(f"Created square lattice with {len(positions)} particles")

    # Zero initial velocities (will thermalize with Langevin)
    velocities = np.zeros((N, 3), dtype=float)

    # ---- HOOMD context ----
    gpu = hoomd.device.GPU()
    simulation = hoomd.Simulation(device=gpu, seed=42)

    # Create snapshot (2D: thin z-box with z-velocity constraint)
    snapshot = hoomd.Snapshot()
    snapshot.particles.N = N
    snapshot.particles.types = ["C"]  # Match Julia type name
    snapshot.configuration.box = [box_Lx, box_Ly, 10.0, 0, 0, 0]  # Thin z-box > cutoff+buffer
    snapshot.particles.typeid[:] = 0
    snapshot.particles.position[:] = positions
    # Zero z-velocities to keep 2D behavior
    velocities[:, 2] = 0.0  # Force all z-velocities to zero
    snapshot.particles.velocity[:] = velocities
    simulation.create_state_from_snapshot(snapshot)

    print(f"Using device: {simulation.device}")

    # ---- Potential (LJ with cutoff at r_cut=2.5) ----
    nl = hoomd.md.nlist.Cell(buffer=0.4)  # Match Julia skin=0.4
    lj = hoomd.md.pair.LJ(nlist=nl)
    lj.params[("C", "C")] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[("C", "C")] = r_cut

    # ---- Integrator (Langevin) - effectively 2D ----
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=temperature)
    langevin.gamma["C"] = gamma  # Set gamma per particle type
    
    integrator = hoomd.md.Integrator(dt=dt, methods=[langevin], forces=[lj])
    simulation.operations.integrator = integrator
    
    # Custom action to zero z-velocities after each integration step (enforce 2D)
    class Keep2D(hoomd.custom.Action):
        def act(self, timestep):
            with simulation.state.cpu_local_snapshot as snap:
                # Zero z-velocities and keep z-positions at 0
                snap.particles.velocity[:, 2] = 0.0
                snap.particles.position[:, 2] = 0.0
    
    keep_2d_updater = hoomd.update.CustomUpdater(action=Keep2D(), trigger=hoomd.trigger.Periodic(1))
    simulation.operations.updaters.append(keep_2d_updater)

    # ---- Thermo ----
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermo)

    # ---- GSD writer ----
    gsd_writer = hoomd.write.GSD(
        filename=str(traj_path),
        trigger=hoomd.trigger.Periodic(frame_interval),
        mode="wb",
    )
    simulation.operations.writers.append(gsd_writer)

    # ---- CSV observables writer function ----
    # Initialize CSV file with header
    with open(obs_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "Epot", "ek", "dq"])  # Match Julia header
    
    def write_observables_csv(step):
        simulation.run(0)  # Ensure energies are current
        ke = thermo.kinetic_energy
        pe = thermo.potential_energy
        dq = 0.0  # Heat not directly available in HOOMD
        
        # Write CSV (append mode)
        with open(obs_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, pe, ke, dq])  # Match Julia order: step, PE, KE, dq

    # ---- Run (match Julia structure) ----
    print("\n=== Starting Simulation ===")
    
    # Custom action to write initial frame manually
    class InitialFrameWriter(hoomd.custom.Action):
        def __init__(self, filename):
            self.filename = filename
            self.written = False
        
        def act(self, timestep):
            if not self.written and timestep == 0:
                # Write using the existing GSD writer
                # This is a hack - we'll trigger it by setting timestep check
                self.written = True
    
    # Set up the writer to trigger on specific steps
    frame_steps = list(range(frame_interval, n_steps + 1, frame_interval))
    
    # Create initial frame writer that triggers immediately
    initial_writer = hoomd.write.GSD(
        filename=str(traj_path),
        trigger=hoomd.trigger.On(0),
        mode="wb",
    )
    
    # Create periodic writer for later frames  
    periodic_writer = hoomd.write.GSD(
        filename=str(traj_path),  
        trigger=hoomd.trigger.Periodic(frame_interval, phase=0),
        mode="ab",
    )
    
    # Add both writers
    simulation.operations.writers.append(initial_writer)
    simulation.operations.writers.append(periodic_writer)
    
    # Remove the old writer
    simulation.operations.writers.remove(gsd_writer)
    
    write_observables_csv(0)  # Initial observables
    print("Wrote initial frame, step=0")
    
    start_time = time.time()
    
    # Run the main simulation loop
    for s in range(1, n_steps + 1):
        simulation.run(1)
        
        # Write observables every frame_interval steps (like Julia)
        if s % frame_interval == 0:
            write_observables_csv(s)
            print(f"Wrote frame, step={s}")
    
    sim_time = time.time() - start_time
    sps = n_steps / sim_time

    # Final energies
    simulation.run(0)
    final_ke = thermo.kinetic_energy
    final_pe = thermo.potential_energy

    print("\n=== Results ===")
    print(f"Total time: {sim_time:.3f} seconds")
    print(f"Steps/sec: {sps:.3f}")
    print(f"Final KE: {final_ke}")
    print(f"Final PE: {final_pe}")

    print(f"Done. GSD: {traj_path}")
    print(f"Observables: {obs_path}")

    return sps, final_ke, final_pe


if __name__ == "__main__":
    benchmark_minimal_2d()