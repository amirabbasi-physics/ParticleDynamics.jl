"""Build the benzene crystal I (Pbca) starting structures for the head-to-head.

Source structure: Crystallography Open Database entry 7238223 — crystalline
benzene, space group Pbca, Z = 4, T = 150 K, ambient pressure (Nayak,
Sathishkumar & Guru Row, CrystEngComm 12, 3112 (2010)).

The Pbca cell is orthorhombic, which is what the engine's box supports. Note
that the X-ray C-H distance in this entry is 0.93 Å, the usual X-ray
foreshortening artifact (neutron diffraction gives ~1.08 Å); relaxation under
the MLIP is expected to correct it, and that is one of the validation targets.

Writes to validation/:
  benzene_crystal.npz  -- 2x2x2 supercell (384 atoms), MD/scan system
  benzene_cell.npz     -- single unit cell (48 atoms)
  benzene_monomer.npz  -- one molecule in a 20 Å box (gas-phase reference)

Run: python examples/orb/py_benzene_init.py
"""

import os
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")
os.makedirs(OUT, exist_ok=True)
CIF = os.path.join(OUT, "cod_7238223_benzene.cif")

cell0 = read(CIF)
print(f"unit cell: {cell0.get_chemical_formula()}  V = {cell0.get_volume():.3f} A^3")


def save(atoms, name, **extra):
    L = np.diag(atoms.cell.array).astype(np.float64)
    off = np.abs(atoms.cell.array - np.diag(L)).max()
    assert off < 1e-8, f"{name}: cell is not orthorhombic (off-diagonal {off:.2e})"
    pos = atoms.get_positions() % L  # wrap into [0, L)
    np.savez(
        os.path.join(OUT, name),
        positions=pos.astype(np.float64),
        numbers=atoms.get_atomic_numbers().astype(np.int64),
        masses=atoms.get_masses().astype(np.float64),
        cell_lengths=L,
        volume=np.float64(atoms.get_volume()),
        **extra,
    )
    rho = atoms.get_masses().sum() / 6.02214076e23 / (atoms.get_volume() * 1e-24)
    print(f"{name}: N = {len(atoms):4d}  box = "
          f"{L[0]:.4f} x {L[1]:.4f} x {L[2]:.4f} A  rho = {rho:.4f} g/cm3")


# --- unit cell and 2x2x2 supercell ---
save(cell0, "benzene_cell.npz", nmolecules=np.int64(4))
sc = cell0.repeat((2, 2, 2))
save(sc, "benzene_crystal.npz", nmolecules=np.int64(32))

# --- gas-phase monomer: one molecule, extracted whole, in a large box ---
# Take the 12 atoms of one molecule by connectivity from a seed carbon so the
# molecule is not split across the periodic boundary.
from ase import Atoms

d = cell0.get_all_distances(mic=True)
sym = cell0.get_chemical_symbols()
seed = sym.index("C")
group = {seed}
for _ in range(4):  # grow over bonds; benzene needs 3 shells, 4 is safe
    for i in list(group):
        for j in range(len(cell0)):
            if d[i, j] < 1.8 and j != i:
                group.add(j)
group = sorted(group)
assert len(group) == 12, f"molecule extraction found {len(group)} atoms, expected 12"

# Rebuild the molecule with unwrapped geometry relative to the seed atom.
vecs = np.array([cell0.get_distance(seed, j, mic=True, vector=True) for j in group])
mol_pos = vecs - vecs.mean(axis=0)
LBOX = 20.0
mono = Atoms(numbers=cell0.get_atomic_numbers()[group],
             positions=mol_pos + LBOX / 2,
             cell=np.diag([LBOX] * 3), pbc=True)
cc = [mono.get_distance(i, j) for i in range(12) for j in range(i + 1, 12)
      if mono.get_chemical_symbols()[i] == "C"
      and mono.get_chemical_symbols()[j] == "C"
      and mono.get_distance(i, j) < 1.6]
print(f"monomer: {mono.get_chemical_formula()}  "
      f"C-C bonds {len(cc)} mean {np.mean(cc):.4f} A")
assert len(cc) == 6, f"expected 6 C-C ring bonds, got {len(cc)}"
save(mono, "benzene_monomer.npz", nmolecules=np.int64(1))
