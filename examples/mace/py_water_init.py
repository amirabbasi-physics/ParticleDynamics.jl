"""Build a deterministic 64-molecule liquid-water starting configuration at
0.997 g/cm^3 for the MACE-OFF showcase: O atoms on a 4x4x4 grid with random
molecular orientations (seed 7), rigid TIP3P-like geometry (r_OH = 0.9572 A,
HOH = 104.52 deg). Atom order is [O, H, H] per molecule.

Run:  python py_water_init.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation")
os.makedirs(OUT, exist_ok=True)

N_MOL = 64
M_H2O = 18.01528          # amu
RHO = 0.997               # g/cm^3
NA = 6.02214076e23

mass_g = N_MOL * M_H2O / NA
L = (mass_g / RHO) ** (1.0 / 3.0) * 1e8   # cm -> Angstrom
spacing = L / 4

r_oh = 0.9572
half = np.deg2rad(104.52) / 2
# molecule template (O at origin, dipole along +z)
template = np.array([
    [0.0, 0.0, 0.0],
    [r_oh * np.sin(half), 0.0, r_oh * np.cos(half)],
    [-r_oh * np.sin(half), 0.0, r_oh * np.cos(half)],
])

rng = np.random.RandomState(7)
positions = np.empty((3 * N_MOL, 3))
k = 0
for i in range(4):
    for j in range(4):
        for l in range(4):
            center = (np.array([i, j, l]) + 0.5) * spacing
            center += rng.uniform(-0.15, 0.15, 3)   # break lattice symmetry
            # random rotation via QR of a random normal matrix
            Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
            Q *= np.sign(np.diag(R))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            positions[k:k + 3] = center + template @ Q.T
            k += 3

numbers = np.tile([8, 1, 1], N_MOL)
masses = np.tile([15.999, 1.008, 1.008], N_MOL)

print(f"L = {L:.4f} A, {N_MOL} H2O, {len(numbers)} atoms, rho = {RHO} g/cm^3")
np.savez(os.path.join(OUT, "water_init.npz"),
         positions=positions, numbers=numbers, masses=masses,
         L=np.float64(L))
print("wrote water_init.npz")
