# Bonded Terms, Minimization and NPT

All quantities are plain numbers in caller-chosen consistent units (length L,
energy E, time). Angles are radians.

## Typed bonded interactions

Topology sections carry optional labels; each force descriptor applies to the
tuples whose label equals its `type` (`:default` when a section has no labels).
Atom order is kept: the middle atom of an angle is the vertex, and torsions
use the IUPAC signed dihedral `φ = atan2(|b2| b1·(b2×b3), (b1×b2)·(b2×b3))` of
the ordered quartet.

```julia
topology = Topology(bonds=bonds, bond_types=[:backbone, :side, …],
                    angles=angles, dihedrals=dihedrals, impropers=impropers)
forces = Force[
    WCA(epsilon=1.0, sigma=1.0),
    HarmonicBondForce(k=300.0, r0=1.0, type=:backbone),
    FENEBondForce(k=30.0, R0=1.5, type=:side),
    HarmonicAngleForce(k=30.0, theta0=deg2rad(115)),
    PeriodicDihedralForce(k=[1.0, 0.4], n=[1, 3], phase=[0.0, 0.0]),
    HarmonicImproperForce(k=10.0, chi0=0.0),
]
```

| Descriptor | Potential |
|---|---|
| `HarmonicBondForce(k, r0)` | `k(r-r0)²/2` |
| `FENEBondForce(k, R0)` | `-kR0² log(1-r²/R0²)/2`, strictly `r < R0` |
| `HarmonicAngleForce(k, theta0)` | `k(θ-θ0)²/2`; `θ0 = π` is exact at collinear geometry |
| `CosineSquaredAngleForce(k, theta0)` | `k(cosθ-cosθ0)²/2` |
| `UreyBradleyAngleForce(k_theta, theta0, k_r, r13)` | harmonic angle + `k_r(r13-r13⁰)²/2` |
| `PeriodicDihedralForce(k, n, phase)` | `Σ k_m[1+cos(n_mφ-δ_m)]` |
| `OPLSDihedralForce(K1, K2, K3, K4)` | `½[K1(1+cosφ)+K2(1-cos2φ)+K3(1+cos3φ)+K4(1-cos4φ)]` |
| `RyckaertBellemansDihedralForce(C)` | `Σ Cn cos(ψ)^n`, `ψ = φ-π` |
| `HarmonicImproperForce(k, chi0)` | `k·wrap(χ-χ0)²/2` (torsion-style, ordered quartet) |
| `PeriodicImproperForce(k, n, phase)` | Fourier form on impropers |

Every label must be assigned exactly once per section; missing or duplicate
assignments are errors. Torsions and impropers require 3D. Invalid geometry
(zero-length bonds, collinear torsion planes, FENE beyond `R0`) raises
`DomainError` with the topology index. `bonded_energy_components(st)` returns
per-family energies. Exclusion and 1–4 scaling policies beyond the existing
bonded-pair exclusion are not implemented yet.

## Energy minimization

```julia
result = minimize!(sim; method=FIRE(dt_initial=1e-3, dt_max=1e-2),
                   force_tolerance=1e-4, max_displacement=0.05,
                   mobile=Group(:solvent, TypeSelection(:W)))
```

`minimize!` works on a workflow `Simulation` (the integrator only supplies
forces; no thermostat runs) or on a low-level `SimulationState`. The box is
fixed; non-mobile particles keep their interactions but do not move.
`FIRE` (velocity-Verlet FIRE) uses forces only; `SteepestDescent` uses a
backtracking Armijo line search on energies and therefore stops with
`:step_too_small` once energy differences fall below rounding (in practice
about `|F| ≈ √(2 k eps E_term)`). `result.converged` is true only when the
largest mobile force reached `force_tolerance`.

## Isotropic NPT

```julia
barostat = MonteCarloBarostat(pressure=2.0, kT=1.0, interval=25,
                              log_volume_step=0.01, seed=1, tune=true)
# or StochasticCellRescaling(pressure=2.0, kT=1.0, compressibility=0.1,
#                            tau=0.5, interval=10, seed=1)
integrator = Integrator(dt=4e-3, scheme=BAOAB(), forces=forces,
                        methods=[Langevin(all; gamma=1.0, kT=1.0)], barostat=barostat)
```

Pressure is in E/L³, compressibility in L³/E. Both barostats scale all
coordinates with the cell and leave velocities unchanged. They currently
require 3D, `scheme=BAOAB()` with one `Langevin` method on all particles at
the barostat `kT`, no freezing or external potentials. Long-range
electrostatics follow the cell: the mesh influence function, reciprocal
virial and background term are recomputed at every cell change.
Monte Carlo moves use a log-volume proposal with the `(N+1)` Jacobian;
stochastic cell rescaling follows Bernetti & Bussi (2020) with a fresh virial
at every event. `barostat_diagnostics(sim)` reports volume, acceptance and
committed changes. GSD frames record `log/barostat/state`; continue a run
with `restore_barostat!(sim, read_gsd_log(path, "log/barostat/state"))`.

States with rigid water (`attach_rigid_water!`, core API) take Monte Carlo
volume moves only. The moves are molecular: `set_cell_and_molecules!` scales
each water's centre of mass and translates the molecule rigidly, and the
acceptance counts each water once. C-rescale and atom-wise scaling are
rejected for rigid water.

Monte Carlo moves sample the potential energy itself, so the Lennard-Jones
cutoff convention matters (`vdw_modifier`, see [Forces](forces.md)). Use
`:potential_shift` to compare with GROMACS.

Size the neighbor-list `capacity` for the densest volume the run will visit:
an overflow stops the run instead of truncating the list. Each barostat event
currently triggers a neighbor rebuild, so events every 10–25 steps are a
reasonable default.
