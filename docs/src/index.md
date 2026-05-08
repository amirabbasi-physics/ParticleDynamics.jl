# ParticleDynamics.jl

`ParticleDynamics.jl` is a GPU-first Julia package for non-equilibrium particle
simulations on `CUDA.jl`. The recommended public surface is the workflow API:
`ParticleSystem`, `Group`, `Force`, `Integrator`, `Observable`, `Writer`,
`Stage`, `Simulation`, and `run!`.

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

## Workflow model

- `ParticleSystem` stores positions, types, masses, box, velocities, and topology.
- `Group` selects particles today and is designed to grow into other topology domains later.
- `Force` objects describe physical interactions.
- `Integrator` owns `dt`, forces, methods, and thermostats.
- `Observable` objects compute sampled quantities.
- `Writer` objects own scheduled output.
- `Stage` describes a named block of running.
- `Simulation` assembles the workflow, and `run!` owns the timestep loop.

## Quick Links

- Workflow start: [Quickstart](quickstart.md)
- Expert API: [Low-Level API](manual/getting_started.md)
- State layout: [Simulation State](manual/simulation_state.md)
- Integrators: [Integrators](manual/integrators.md)
- Interactions: [Forces](manual/forces.md)
- Selection and controls: [Groups and Filters](manual/groups_filters.md)
- Thermostat interfaces: [Thermostats](manual/thermostats.md)
- Diagnostics: [Observables](manual/observables.md)
- Output formats: [I/O](manual/io.md)
- Restart reading: [Restarts](manual/restarts.md)
- Legacy notes: [Collision Rate Notes](legacy/collision_rate.md)
- Example inventory: `examples/README.md`

## Public API map

The supported top-level API is intentionally split into a workflow layer and a
low-level expert layer.

| API group | Exported symbols | Primary page |
|---|---|---|
| Workflow | `Simulation`, `ParticleSystem`, `Topology`, `Group`, `Groups`, `Force` objects, `Integrator`, `Observable` objects, `Writer` objects, `Stage`, `prepare!`, `run!`, `reset_step!`, `reset_observables!`, `state` | `quickstart.md` |
| Low-level expert API | `SimulationState`, `build_simulation`, `step!`, `step_graph!`, `sync_unwrapped!`, `collect_step_observables`, `reset_bath_exchange_accumulators!`, `velocityverlet`, `baoab`, `baoa`, `gsm`, `eulerheun`, `eulermaruyama`, `nosehooverchain`, `csvr` | `manual/getting_started.md`, `manual/simulation_state.md`, `manual/integrators.md` |
| Parameters and physical helpers | `LJParams`, `SoftRepulsiveParams`, `HarmonicBondParams`, `FENEParams`, `BondPotential`, `HarmonicBond`, `FENEBond`, `harmonic_bond`, `fene_bond`, `StokesFrictionCoefficient`, `SphereMass`, `InertialTime`, `DiffusiveTime` | `manual/forces.md` |
| Initialization generators | `box_from_phi_2d`, `box_from_phi_3d`, `hex_random_2d`, `hex_circle_2d`, `hex_circle_plus_random_2d`, `hex_sites_in_box_2d`, `hex_circle_in_box_2d`, `hex_slab_coexistence_2d`, `fcc_sites_in_box_3d`, `fcc_random_3d`, `fcc_slab_coexistence_3d` | `manual/getting_started.md` |
| Writers and I/O | `write_xyz!`, `write_observables_csv!`, `gsd_open`, `gsd_close`, `write_gsd_frame!`, `read_gsd_frame!`, `InMemoryLogger`, `CSVWriter`, `XYZWriter` | `manual/io.md` |
| Bond lists | `BondList`, `build_bondlist` | `manual/forces.md` |
| Collision counting | `enable_collision_counting!`, `disable_collision_counting!`, `collisions_reset_counts!`, `collisions_read_counts!`, `set_collision_pair_cutoffs!` | `manual/observables.md` |
| Selection helpers and modules | `ParticleSelection`, `ParticleGroup`, `All`, `TypeIDs`, `Indices`, `materialize`, `apply_scalar!`, `apply_values!`, `gather`, `sum_values`, plus the exported `Filters`/`BondedForces` modules for qualified access | `manual/groups_filters.md`, `manual/forces.md` |

Advanced helpers that are intentionally not part of the default import surface
live under `ParticleDynamics.SimulationCore`, for example
`ParticleDynamics.SimulationCore.zero_forces!`,
`ParticleDynamics.SimulationCore.accumulate_energies!`,
`ParticleDynamics.SimulationCore.run_integrator_step!`,
`ParticleDynamics.SimulationCore.thermostatted_dof`, and
`ParticleDynamics.SimulationCore.thermostatted_particle_mask`.

## Behavior Baseline (from tests)

Current tests validate:

- Deterministic force/kernel properties (analytic 2-particle checks, Newton symmetry, PBC invariance, backend parity).
- Stochastic behavior at moment level (Brownian MSD slope, Langevin equipartition, OU trend checks, weak-step trend in deterministic limit).
- Regression/IR fixes including `gamma > 0` enforced error behavior on stochastic paths where required.

## Current limitations

- CUDA is the primary supported backend.
- Angles, dihedrals, impropers, electrostatics, and broader force-field
  concepts are future-facing vocabulary unless already implemented in the
  existing low-level engine.
- `ForceField` is stable as a container concept, but compiled support remains
  limited to the force families backed by current kernels and wrappers.

## Build docs locally

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
