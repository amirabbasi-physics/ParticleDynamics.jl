# Changelog

All notable user-visible changes to this project will be documented in this file.

## Unreleased

### Performance

- NVE half-kicks now run in the state's native precision and no longer
  maintain `Ekin`/`dU` per step; `st.Ekin` is refreshed lazily when
  observables are sampled. Reading `st.Ekin` directly after `step!` under NVE
  requires `Simulation._refresh_kinetic_buffer!(st)` first.
- `dq`/`dU` are no longer cleared every step for NVE/NHC/CSVR; they are
  cleared once when switching away from an integrator that populates them.
- Neighbor binning uses an O(N) counting sort (histogram + prefix scan +
  scatter) instead of a global `CUDA.sort!` of packed keys plus per-cell
  binary searches. Particle order within a cell is no longer deterministic,
  so Float32 force sums may differ bitwise between identical runs.
- Neighbor rows are stored slot-major (`t*N + i`, "transposed ELL") so force
  kernels read neighbor ids coalesced. `NeighborMatrix` and
  `StencilNeighborMatrix` lost the `packed_keys`/`cell_ids_sorted` fields and
  gained `cell_counts`.
- New `build_simulation(...; spatial_reorder=true)`: at neighbor rebuilds in
  NVE runs, all per-particle state is permuted into cell-sorted order for
  memory locality; `st.tag[k]` maps storage slot `k` back to the build-time
  particle id. GSD output (and therefore restart files) is written in
  build-time order, so trajectories keep stable particle identity. Disabled
  automatically with bonds, freeze controls, collision counting, stencil
  lists, or the workflow layer; pass `spatial_reorder=false` if external
  code holds `Filters.Indices`-style index lists into the state.
- Deterministic drift kernels use a single conditional wrap instead of
  floor-division wrapping.
- Measured on an RTX 3090 (N=10^6 LJ particles, rho=0.3, Float32, steady
  state): 52 -> ~660 steps/s.

### Changed

- Export `eulerheun` and `read_gsd_frame!` from the top-level `ParticleDynamics` API so the documented restart and Brownian-midpoint workflows use supported imports.
- Stop re-exporting the low-level `Simulation` helpers `zero_forces!`, `accumulate_energies!`, `run_integrator_step!`, `thermostatted_dof`, and `thermostatted_particle_mask`.
- `build_simulation` now accepts an explicit `backend` keyword with `:cuda` as the supported default and an early error for unsupported `:cpu` requests.

Migration notes:
- Replace unqualified imports of those low-level helpers with `ParticleDynamics.Simulation.<name>` if you still need them.
