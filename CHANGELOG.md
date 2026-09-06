# Changelog

All notable user-visible changes to this project will be documented in this file.

## Unreleased

### Correctness and maintenance

- Force evaluation checks neighbor displacement after motion and at temporary
  midpoint coordinates, regardless of the step-boundary check interval.
  Spatial reordering stays at safe step boundaries. These checks add a GPU
  reduction/synchronization on neighbor-backed force evaluations.
- Collision rebuilds reconstruct contact history from the last physical sample,
  so entries on rebuild steps are retained and midpoint probes are not counted.
- Cached force validity is independent of the step counter. Nonzero-step
  restarts initialize their first force correctly; `invalidate_forces!(st)` is
  available after manual state/model edits. Provider and freeze transitions and
  workflow state/force setup invalidate automatically. Failed steps invalidate
  the cache without rolling back the trajectory.
- Increased the independent ensembles in the Brownian MSD-slope and OU-decay
  regression tests using their correlated-sample uncertainty; accuracy
  thresholds are unchanged.
- Initial GSD frames requesting forces or virials evaluate them before writing.
  SimulationState gained internal cache flags and optional collision reference
  buffers; positional construction must use the new fields or the public builder.

- Renamed private nonbonded `*_csr_*` kernels to `*_ell_*` and the three
  `*CSRCUDA.jl` files to `*ELLCUDA.jl`. Dense and stencil neighbors use ELL;
  bond adjacency remains CSR. Public force function names are unchanged.
- Collision readers and contact-history buffers now use the same shared
  slot-major ELL index as force kernels and neighbor builders.
- Neighbor builders count all required entries and throw
  `NeighborLists.NeighborCapacityError` on overflow. Failed or unbuilt lists
  cannot be consumed by force/collision entry points. Automatic growth is not
  implemented; callers must select sufficient capacity.
- Neighbor containers gained `valid` and `required_capacity` fields. Code
  using positional struct constructors must migrate to the neighbor builders
  or supply the new fields; exported force signatures remain unchanged.
- Low-level state construction defers its first list build until coordinates
  are initialized, avoiding the quadratic scan of placeholder zero positions.
- Wide periodic stencils visit each cell at most once, preventing duplicate
  neighbors when the stencil spans an axis.
- Workflow replace-mode writers replace output once per simulation writer
  session and append across subsequent stages, including repeated preparation
  of the same writer. Initial-frame failures now restore stage overrides and
  close already-opened writers.
- Added host-only neighbor oracle tests plus GPU completeness, overflow,
  stencil, collision and writer-lifecycle regressions.
- Stabilized statistical validation through an equilibrated CSVR temperature
  average and a larger independent OU particle ensemble, rather than relaxing
  the existing OU accuracy thresholds.

### Features

- GSD output now writes particle masses automatically and infers covalent
  diameters for recognized chemical element type names (for example `"O"`
  and `"H"`). Scalar, per-type, and per-particle overrides are supported for
  diameter, mass, and charge; unknown type names retain the 1.0 fallback.
- **External potential providers (MLIP hook).** New opt-in seam that routes
  all force evaluation through a user-supplied provider:
  `AbstractExternalPotential`, `external_forces!`,
  `attach_external_potential!`, `detach_external_potential!` (exported from
  `SimulationCore` and re-exported at top level). While attached, the
  provider replaces every internal force term (nonbonded and bonded) for all
  integrators — NVE, NHC, CSVR, and the stochastic families — since they all
  evaluate forces through `evaluate_forces_into_f!`. Requires a bond-free
  state and `spatial_reorder=false`. External potentials do not provide a
  virial, so pressure observables are unsupported while attached.
- **MACE foundation-model support** (`examples/mace/`): a `MACEPotential`
  provider drives MACE-MP-0 / MACE-OFF potentials (mace-torch via
  PythonCall) with the engine's native integrators. Validated against the
  ASE reference implementation: forces match to ~1e-14 eV/Å (Si216), and a
  199-step NVE trajectory matches ASE velocity Verlet to 4.3e-5 Å. See
  `examples/mace/README.md` for setup, validation results, and a liquid-water
  (MACE-OFF) structure showcase.
- **Orb foundation-model support** (`examples/orb/`): an `OrbPotential`
  provider drives Orb-v3 potentials (orb-models via PythonCall) through the
  same external-potential interface as `MACEPotential`, covering the
  `conservative` (forces from an energy gradient) and `direct` (forces
  predicted independently, no conserved energy) families and the
  charge/spin-conditioned `omol` models. Forces through the engine path match
  an independent ASE call to 4.1e-15 eV/Å in float64. With two foundation
  models behind one interface, `examples/orb/` also carries a benzene-crystal
  head-to-head against experimental and CCSD(T) reference data: lattice
  energy, equilibrium volume from an energy–volume scan, bond geometry,
  vibrational spectrum, NVE energy conservation, and matched-precision
  throughput — plus a side-by-side Fresnel movie of a heating ramp with live
  throughput and accuracy readouts. `MACEPotential` gained a `dtype` keyword
  (default `"float64"`, existing behavior unchanged) so both providers can be
  timed at matched precision.

- **`build_simulation(...; exclude_bonded_pairs=false)`** lets bonded pairs
  feel the nonbonded potential, enabling canonical Kremer-Grest FENE+WCA
  bonds (the bare FENE bond has no repulsive core, so with the default
  exclusions polymer bonds would collapse). Default behavior is unchanged.

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
