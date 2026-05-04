# ParticleDynamics.jl

`ParticleDynamics.jl` is a GPU-first Julia package for non-equilibrium particle simulations (Langevin and Brownian dynamics) on `CUDA.jl`.

```@raw html
<div style="padding:0.6rem 0.8rem; border-left:4px solid #d9534f; background:#fff6f6;">
<strong>GPU-only package:</strong> a functional CUDA environment is required for simulations and test-validated behavior.
</div>
```

## Scope

- 2D/3D particle dynamics with periodic boundaries.
- Langevin integrators: velocity-Verlet (default), BAOAB, BAOA, GSM.
- Brownian integrators: midpoint (Euler-Heun style) and Euler-Maruyama.
- Nonbonded interactions (LJ/WCA/soft-repulsive), bonded interactions (harmonic/FENE), filters/freeze controls, collision counting, and XYZ/CSV/GSD outputs.

## Quick Links

- Manual start: [Getting Started](manual/getting_started.md)
- Legacy notes: [Collision Rate Notes](legacy/collision_rate.md)
- Existing examples: `examples/*.jl` in the repository

## Public API Map (Current Exports)

The table below maps the supported top-level API to where it is intended to be
described. Lower-level helpers that are mainly useful for custom orchestration
stay under qualified paths such as `ParticleDynamics.Simulation`.

| API group | Exported symbols | Planned primary page |
|---|---|---|
| Core simulation | `SimulationState`, `build_simulation`, `step!`, `step_graph!`, `sync_unwrapped!`, `collect_step_observables`, `reset_bath_exchange_accumulators!`, `velocityverlet`, `baoab`, `baoa`, `gsm`, `eulerheun`, `eulermaruyama`, `nosehooverchain`, `csvr` | `manual/simulation_state.md`, `manual/integrators.md` |
| Parameters and physical helpers | `LJParams`, `SoftRepulsiveParams`, `HarmonicBondParams`, `FENEParams`, `BondPotential`, `HarmonicBond`, `FENEBond`, `harmonic_bond`, `fene_bond`, `StokesFrictionCoefficient`, `SphereMass`, `InertialTime`, `DiffusiveTime` | `manual/forces.md` |
| Initialization generators | `box_from_phi_2d`, `box_from_phi_3d`, `hex_random_2d`, `hex_circle_2d`, `hex_circle_plus_random_2d`, `hex_sites_in_box_2d`, `hex_circle_in_box_2d`, `hex_slab_coexistence_2d`, `fcc_sites_in_box_3d`, `fcc_random_3d`, `fcc_slab_coexistence_3d` | `manual/getting_started.md` |
| Writers and I/O | `write_xyz!`, `write_observables_csv!`, `gsd_open`, `gsd_close`, `write_gsd_frame!`, `read_gsd_frame!`, `InMemoryLogger`, `CSVWriter`, `XYZWriter` | `manual/io.md` |
| Bond lists | `BondList`, `build_bondlist` | `manual/forces.md` |
| Collision counting | `enable_collision_counting!`, `disable_collision_counting!`, `collisions_reset_counts!`, `collisions_read_counts!`, `set_collision_pair_cutoffs!` | `manual/collisions.md` |
| Modules | `Filters`, `BondedForces` | `manual/groups_filters_freeze.md`, `manual/forces.md` |

Advanced helpers that are intentionally not part of the default import surface
include `ParticleDynamics.Simulation.zero_forces!`,
`ParticleDynamics.Simulation.accumulate_energies!`,
`ParticleDynamics.Simulation.run_integrator_step!`,
`ParticleDynamics.Simulation.thermostatted_dof`, and
`ParticleDynamics.Simulation.thermostatted_particle_mask`.

## Behavior Baseline (from tests)

Current tests validate:

- Deterministic force/kernel properties (analytic 2-particle checks, Newton symmetry, PBC invariance, backend parity).
- Stochastic behavior at moment level (Brownian MSD slope, Langevin equipartition, OU trend checks, weak-step trend in deterministic limit).
- Regression/IR fixes including `gamma > 0` enforced error behavior on stochastic paths where required.

## Build docs locally

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
