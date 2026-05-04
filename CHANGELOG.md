# Changelog

All notable user-visible changes to this project will be documented in this file.

## Unreleased

### Changed

- Export `eulerheun` and `read_gsd_frame!` from the top-level `ParticleDynamics` API so the documented restart and Brownian-midpoint workflows use supported imports.
- Stop re-exporting the low-level `Simulation` helpers `zero_forces!`, `accumulate_energies!`, `run_integrator_step!`, `thermostatted_dof`, and `thermostatted_particle_mask`.
- `build_simulation` now accepts an explicit `backend` keyword with `:cuda` as the supported default and an early error for unsupported `:cpu` requests.

Migration notes:
- Replace unqualified imports of those low-level helpers with `ParticleDynamics.Simulation.<name>` if you still need them.
