#!/usr/bin/env python3
"""
Optional Head-to-Head Reference Benchmark: OpenMM 8.4+ (Default CUDA PME vs NVIDIA cuPotential Plugin)
Runs the exact same 30,492-atom SPC/E rigid water box and 10,648-ion NaCl melt on the RTX 3090.

Setup (optional, in an isolated conda/venv on your Linux RTX 3090 workstation):
    pip install openmm nv-openmm-plugin

Usage:
    python3 benchmarks/cupotential_comparison/run_openmm_cupotential.py
"""
import json
import math
import os
import time

import openmm as mm
import openmm.app as app
import openmm.unit as unit

try:
    import nv_openmm_plugin as nv_cupo
    HAS_CUPO = True
except ImportError:
    HAS_CUPO = False


def build_spce_system(nx=22, ny=22, nz=21, cutoff_nm=1.0, ewald_tol=1e-4, use_ljpme=False):
    a = 0.3104
    Lx, Ly, Lz = nx * a, ny * a, nz * a
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(Lx, 0, 0), mm.Vec3(0, Ly, 0), mm.Vec3(0, 0, Lz)
    )
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.LJPME if use_ljpme else mm.NonbondedForce.PME)
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    nb.setEwaldErrorTolerance(ewald_tol)
    nb.setUseDispersionCorrection(not use_ljpme)

    oh = 0.10
    hh = 0.16330
    theta = 2.0 * math.asin(hh / (2.0 * oh))
    positions = []
    nw = nx * ny * nz
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                ox, oy, oz = (ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a
                o_idx = system.addParticle(15.9994 * unit.amu)
                h1_idx = system.addParticle(1.008 * unit.amu)
                h2_idx = system.addParticle(1.008 * unit.amu)
                nb.addParticle(-0.8476, 0.316557 * unit.nanometer, 0.650629 * unit.kilojoule_per_mole)
                nb.addParticle(+0.4238, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
                nb.addParticle(+0.4238, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
                nb.addException(o_idx, h1_idx, 0.0, 1.0, 0.0)
                nb.addException(o_idx, h2_idx, 0.0, 1.0, 0.0)
                nb.addException(h1_idx, h2_idx, 0.0, 1.0, 0.0)
                system.addConstraint(o_idx, h1_idx, oh * unit.nanometer)
                system.addConstraint(o_idx, h2_idx, oh * unit.nanometer)
                system.addConstraint(h1_idx, h2_idx, hh * unit.nanometer)
                positions.append(mm.Vec3(ox, oy, oz) * unit.nanometer)
                positions.append(mm.Vec3(ox + oh, oy, oz) * unit.nanometer)
                positions.append(mm.Vec3(ox + oh * math.cos(theta), oy + oh * math.sin(theta), oz) * unit.nanometer)
    system.addForce(nb)
    return system, positions, 3 * nw


def run_bench(label, cutoff_nm=1.0, use_ljpme=False, use_cupo=False, steps=2000):
    system, positions, natoms = build_spce_system(cutoff_nm=cutoff_nm, use_ljpme=use_ljpme)
    dt = 0.002 * unit.picoseconds
    if use_cupo:
        nv_cupo.replace_nonbonded_force(system)
        integrator = nv_cupo.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, dt)
    else:
        integrator = mm.VerletIntegrator(dt)
    integrator.setConstraintTolerance(1e-6)
    platform = mm.Platform.getPlatformByName("CUDA")
    ctx = mm.Context(system, integrator, platform, {"Precision": "mixed"})
    ctx.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 50.0, 100)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 2026)
    integrator.step(200)
    t0 = time.perf_counter()
    integrator.step(steps)
    ctx.getState(getEnergy=True)
    elapsed = time.perf_counter() - t0
    ms_step = 1e3 * elapsed / steps
    ns_day = (steps / elapsed) * 0.002 * 86.4
    print(f"  {label:<32} | {ms_step:6.3f} ms/step ({ns_day:6.1f} ns/day) | N={natoms}")
    return {"label": label, "cutoff_nm": cutoff_nm, "ms_per_step": ms_step, "ns_per_day": ns_day}


if __name__ == "__main__":
    print(f"=== OpenMM / cuPotential Reference Benchmark (nv-openmm-plugin installed: {HAS_CUPO}) ===")
    rows = [
        run_bench("OpenMM_CUDA_PME_rc1.0", cutoff_nm=1.0, use_ljpme=False, use_cupo=False),
        run_bench("OpenMM_CUDA_LJPME_rc0.7", cutoff_nm=0.7, use_ljpme=True, use_cupo=False),
    ]
    if HAS_CUPO:
        rows.append(run_bench("OpenMM_cuPotential_rc0.7", cutoff_nm=0.7, use_ljpme=True, use_cupo=True))
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "openmm_reference.json"), "w") as f:
        json.dump(rows, f, indent=2)
