"""MACE reference calculation (pure Python) for ParticleDynamics.jl validation.

Builds the Si216 validation system, computes energy + forces with MACE-MP-0
(small, float64, CUDA), and saves everything needed by the Julia side to
validation/reference.npz. This file is the reference for the force-fidelity check:
the Julia provider must reproduce these forces to < 1e-12 eV/Ang.

Run:  python py_reference.py
"""
import os
import time

import numpy as np
from ase.build import bulk
from mace.calculators import mace_mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")
os.makedirs(OUT, exist_ok=True)

# --- Si216: 3x3x3 conventional diamond cells, deterministically rattled ---
atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat((3, 3, 3))
atoms.rattle(stdev=0.05, seed=42)
assert len(atoms) == 216

calc = mace_mp(model="small", device="cuda", default_dtype="float64")
atoms.calc = calc

# warm-up call (JIT/graph build), then timed calls
atoms.get_forces()
times = []
for _ in range(5):
    atoms.rattle(stdev=1e-6)  # invalidate cache so the model actually reruns
    t0 = time.perf_counter()
    F = atoms.get_forces()
    times.append(time.perf_counter() - t0)
E = atoms.get_potential_energy()

fsum = np.abs(F.sum(axis=0)).max()
print(f"N = {len(atoms)}, E = {E:.10f} eV")
print(f"max |sum_i F_i| = {fsum:.3e} eV/Ang (translation invariance)")
print(f"force call: {min(times)*1e3:.1f} ms (best of 5)")
assert np.isfinite(F).all() and np.isfinite(E)
assert fsum < 1e-8, "translation invariance violated"

np.savez(
    os.path.join(OUT, "reference.npz"),
    positions=atoms.get_positions(),      # Ang, ASE domain [0, L) not enforced
    cell=atoms.get_cell()[:],             # 3x3 Ang
    numbers=atoms.get_atomic_numbers(),   # Z
    energy=np.float64(E),                 # eV
    forces=F,                             # eV/Ang, shape (N, 3)
)
print(f"wrote {os.path.join(OUT, 'reference.npz')}")
