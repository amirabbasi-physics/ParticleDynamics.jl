"""V3 reference: 200-step NVE (velocity Verlet) trajectory of Si216 with
MACE-MP-0 in ASE. Initial positions come from validation/reference.npz;
initial velocities are a deterministic Maxwell-Boltzmann draw (numpy seed
2026, 300 K, COM removed) saved alongside the trajectory so the Julia engine
can start from identical conditions.

Run:  python py_md_reference.py
"""
import os

import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from mace.calculators import mace_mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")

KB = 8.617333e-5  # eV/K
TEMP = 300.0
MASS_SI = 28.0855
NSTEPS = 200

ref = np.load(os.path.join(OUT, "reference.npz"))
atoms = Atoms(numbers=ref["numbers"], positions=ref["positions"],
              cell=ref["cell"], pbc=True)
N = len(atoms)

# deterministic MB velocities, ASE-native units (Ang / time-unit, identical
# to the engine's unit system: time = Ang*sqrt(amu/eV))
rng = np.random.RandomState(2026)
sigma = np.sqrt(KB * TEMP / MASS_SI)
v = sigma * rng.standard_normal((N, 3))
v -= v.mean(axis=0)
atoms.set_velocities(v)

atoms.calc = mace_mp(model="small", device="cuda", default_dtype="float64")

traj = np.empty((NSTEPS, N, 3))
k = 0

def record():
    global k
    if k < NSTEPS:
        traj[k] = atoms.get_positions()
    k += 1

dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
dyn.attach(record, interval=1)
dyn.run(NSTEPS)

np.savez(os.path.join(OUT, "md_reference.npz"),
         positions0=ref["positions"], velocities0=v,
         cell=ref["cell"], numbers=ref["numbers"],
         trajectory=traj, dt_fs=np.float64(1.0))
print(f"wrote md_reference.npz: {k-1} recorded steps after t0")
