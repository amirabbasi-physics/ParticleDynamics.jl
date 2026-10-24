# Changelog

All notable user-visible changes to this project will be documented in this file.

## 0.6.10 — 2026-10-24

`Project.toml` and `CITATION.cff` share this version; dependency compatibility
bounds are unchanged apart from the new direct `LinearAlgebra` dependency.

### Bonded energies and force evaluation

- Fixed: harmonic bond energies were counted twice in `Epot` (each bond is
  visited from both endpoints of the symmetric bond list). Forces and virials
  were already correct; summed potential and total energies now contain each
  bond once, as FENE already did.
- Fixed: a workflow `Integrator` whose forces contain no pair term (bonds only)
  silently used the low-level default Lennard-Jones potential (ε = σ = 1,
  rc = 1). It now builds an explicit `nonbonded=:none` state, as does an empty
  force list (previously a zero-strength all-pairs WCA kernel).
- `EvaluationRequest`, `evaluate_configuration!`, `total_potential_energy`,
  `total_virial_trace` and `total_configurational_virial`: side-effect-free
  evaluation and Float64 whole-system reductions at the current coordinates.

### Typed bonded interactions

- `Topology` gained `angle_types`, `dihedral_types` and `improper_types`
  (appended fields; the 8-argument positional constructor still works and
  migrates the labels GSD imports used to store in `metadata`).
- New force descriptors on labelled topology sections: `HarmonicAngleForce`,
  `CosineSquaredAngleForce`, `UreyBradleyAngleForce`, `PeriodicDihedralForce`
  (multi-term Fourier), `OPLSDihedralForce`, `RyckaertBellemansDihedralForce`,
  `HarmonicImproperForce` and `PeriodicImproperForce`. Torsions use the IUPAC
  signed dihedral of the ordered quartet.
- Several bond types with their own harmonic/FENE parameters, and any number
  of bonded families at once. A single untyped bond force keeps the previous
  uniform kernel.
- GPU evaluation: one thread per unique interaction plus a per-particle gather
  (no atomics); invalid geometry raises `DomainError` with the topology index.
  `bonded_energy_components(st)` reports per-family energies.

### Energy minimization

- `minimize!(st or sim; method=FIRE(...) | SteepestDescent(...), force_tolerance,
  max_displacement, ...)` at fixed box, with pinned coordinates (`mobile=`),
  domain recovery, deterministic Float64 device reductions and a
  `MinimizationResult` that never reports an exhausted budget as convergence.

### Pressure control

- `set_cell_and_positions!`, `cell_volume`, `cell_supported`: scale the cell
  and all coordinates together. `set_box!` now keeps the dense neighbor
  storage and only recomputes the cell grid.
- `Integrator(...; barostat=StochasticCellRescaling(...))` and
  `MonteCarloBarostat(...)`: isotropic 3D NPT with BAOAB Langevin dynamics,
  `barostat_diagnostics`, `mechanical_pressure`, `total_kinetic_energy`, and
  GSD restart records (`barostat_state_record`, `restore_barostat!`).
- `write_gsd_frame!` accepts `logs=` (extra Float64 log chunks).
- Fixed: with exact (displacement-triggered) neighbor lists, a device-resident
  C-rescale contraction could bring a pair from outside the list radius inside
  the cutoff while no particle had moved more than half the skin, so that
  pair's force was missing until the next rebuild. Exact lists (dense rows and
  cluster lists without `list_intervals`) are now rebuilt after every cell
  change. Buffered cluster lists (`list_intervals`) keep their list across
  scaling, as GROMACS does.

### Long-range electrostatics

- New native GPU Coulomb solvers sharing one particle-mesh engine: `PME`
  (smooth PME, analytic differentiation), `PPPM` (P3M with optimal influence
  functions, `differentiation=:ad` or `:ik`) and `ESP` (Ewald summation with
  prolates). Real space, exceptions, spreading, cuFFT transforms, spectral
  solve and gathering all run on the device; the warmed force path performs no
  device allocations.
- `ParticleSystem(...; charges=...)`, `build_simulation(...; charges=...)`,
  `attach_electrostatics!`, `detach_electrostatics!`, `set_charges!`, `charges`,
  `electrostatic_energy` and `electrostatics_from_gsd`.
- Explicit unit contract (`coulomb_prefactor`, `relative_permittivity`); exact
  per-particle energy decomposition; full reciprocal virial tensor as the
  strain derivative of the discrete mesh energy, stored in the new global
  virial buffer and included in system totals. Subgroup virial requests are
  rejected while electrostatics is attached.
- Exceptions (excluded/scaled pairs), optional uniform neutralising
  background, charge permutation under spatial reordering, and GSD restart
  records for charges and solver settings.
- `build_simulation` accepts `nonbonded=:none` (no vdW term) and
  `neighbor_cutoff` (neighbor radius beyond the vdW cutoff).
- `SimulationState` gained `charge`, `electrostatics`, `virial_global` and
  `virial_global_accum`; `virial_tensor` gained `part=:reciprocal`.
- `tune_electrostatics(st, force; target)` selects the cheapest mesh (PME/PPPM)
  or prolate tolerance (ESP) whose measured force error, against a
  high-accuracy reference at the same cutoff, meets the target. It returns an
  explicit descriptor.
- ESP uses the optimal analytic-differentiation influence function with alias
  sums of the prolate window by default (`influence=:optimal`); the plain
  window deconvolution is available as `influence=:deconvolution`.
- `tune_electrostatics(...; objective=:time)` returns the fastest setting that
  meets the measured target. It searches order, Ewald α and mesh for
  PME/PPPM, and ε and window width for ESP, timing each candidate through the
  state's own force evaluation.
- Mesh spreading accumulates spatially compact blocks of particles in
  shared-memory tiles before writing to the grid (2× faster spreading for
  large systems). Inverse transforms skip cuFFT's defensive input copy.

### Cluster-pair nonbonded backend

- New `ClusterPairs` engine for 3D states, selected with
  `build_simulation(...; nonbonded_backend=:cluster)`,
  `set_nonbonded_backend!(st, :cluster)` or the workflow neighbor policy
  `ClusterPairList(buffer, prune_buffer)`.
- One fused kernel evaluates Lennard-Jones (uniform or pair table) and the
  real-space Coulomb term over a Newton-3 list of 8×8 particle tiles:
  - exclusions and scaled pairs are bit masks;
  - the Ewald or prolate correction is a fitted polynomial in r², with
    reaction field also supported in the pair functor;
  - Coulomb and vdW cutoffs are independent.
- The list is pruned to the interaction cutoff between rebuilds. With a mesh
  solver, the mesh and pair branches can overlap inside the step graph.
- 1,029,000-atom SPC/E water with LJ + PME at Δ = 1e-3 (RTX 3090, Float32):
  37.3 ms → 4.5 ms per force evaluation. Forces match the `:ell` backend to
  round-off.
- `SimulationState` gained `nonbonded_backend`, `cluster` and
  `cluster_config`. Uncharged states default to `:ell`. Attaching long-range
  electrostatics selects `:cluster` by default. The `:ell` backend also supports
  charged 3D states without Coulomb exceptions.
- The cluster backend also runs WCA (rc = 2^(1/6)σᵢⱼ, shifted by −εᵢⱼ),
  per-particle σ (σᵢⱼ = (σᵢ + σⱼ)/2), LJ bond exclusions and collision
  counting.
- Validated against an independent Ewald sum, GROMACS (double-precision PME)
  and LAMMPS (PPPM) snapshots; `scripts/bench_electrostatics.jl` benchmarks
  the solvers at matched measured accuracy.
- `LinearAlgebra` is now a direct dependency.
- Faster list builds: the outer-list search runs one warp per (super-cluster,
  neighbouring column) instead of one warp per super-cluster, which fills the
  GPU on small and medium systems. The resulting lists are bit-for-bit
  identical. 10k SPC/E water NPT (PME, RTX 3090): search 2.7× faster, whole
  build 760–860 → ~400 µs, complete step 0.433 → 0.393 ms (1.10×, 95% CI
  1.09–1.11, five paired runs).
- Faster force and mesh accumulation: the cluster-pair and PME/ESP mesh kernels
  accumulate forces, energies, virials and charge with relaxed atomic additions
  (as CUDA's `atomicAdd`) instead of sequentially consistent atomics, which
  emitted a memory fence and an L1 invalidation around every update. Results
  are unchanged up to floating-point summation order, which was already
  unspecified. RTX 3090, five paired runs: 10k SPC/E water NPT 1.11× faster
  (95% CI 1.105–1.113), 10.6k molten NaCl NVT with PME 1.24× faster (95% CI
  1.23–1.25); the energy/virial pass of the pair kernel is 2.35× faster.
- The pair and pruning kernels take the compensated periodic-image path only
  when a particle pair actually crosses half a box length (decided per warp),
  instead of computing it for every pair. Separation vectors are identical.
  10k SPC/E water NPT: 1.02× faster (95% CI 1.013–1.027).
- With fixed list intervals (`list_intervals`), steps no longer measure particle
  displacements against list references: the rebuild and prune decisions
  follow the step count, and the rigid-water drift only checks positions for
  non-finite values. 10k SPC/E water NPT with buffered lists: 1.01× faster
  (95% CI 1.002–1.018, five paired runs).
- Pruning computes distances only (a tile stays in the working list when any
  pair lies within the prune radius, widened by a rounding bound), and the
  plain force pass then runs on the new working list, instead of a force pass
  over the whole outer list. The working list contains every pair the fused
  pass kept. `ClusterPairs.PRUNE_ONLY[] = false` restores the fused pass for
  newly created engines. 10k SPC/E water NPT (PME, RTX 3090, five paired runs):
  1.03× faster with exact lists (95% CI 1.026–1.032), 1.02× with buffered
  lists (95% CI 1.021–1.027).
- Charged cluster-pair steps gather the mesh forces beside the pair pass
  instead of after it (the gather writes the force slots and the pair scatter
  adds to them). Forces are unchanged up to summation order. 10k SPC/E water
  NPT with buffered lists (PME, RTX 3090, five paired runs): 1.03× faster
  (95% CI 1.025–1.031).
- Rigid-water steps with a CSVR thermostat update the thermostat beside the
  list decision and force work instead of before it (only the final kick uses
  its velocity scale). 10k SPC/E water NPT with buffered lists: 1.02× faster
  (95% CI 1.016–1.024, five paired runs).
- With fixed list intervals (`list_intervals`, nstlist a multiple of
  nstprune), the host schedules the rebuild and prune steps: the steps between
  them launch a graph without list decisions or conditional nodes
  (`DeviceEngine.HOST_SCHEDULE[] = false` restores the device decision).
  The list schedule is unchanged. Buffered lists (skin 0.25 nm, prune buffer
  0.05 nm, intervals 80/20; RTX 3090, five paired runs): 10.6k molten NaCl NVE
  1.10× faster (95% CI 1.100–1.107), NVT with thermostat interval 10 1.10×
  (95% CI 1.089–1.103), 10k SPC/E water NPT 1.03× (95% CI 1.023–1.030).
- Cluster steps advance the device step counter in the final kick kernel
  instead of a separate one-thread launch, and steps between scheduled list
  steps skip the copy of the cluster order for the mesh, which only a list
  build changes (`DeviceEngine.FOLD_TAIL[] = false` restores both). Buffered
  lists (RTX 3090, five paired runs): 10.6k molten NaCl NVE 1.014× faster
  (95% CI 1.007–1.021), NVT with thermostat interval 10 1.012× (95% CI
  1.002–1.021).
- Dense-list (ELL) runs sort their storage in a host-scheduled sort step
  every `reorder_interval` steps (a forced list rebuild that sorts) instead
  of at the first displacement-triggered rebuild after the interval. The
  other steps' rebuilds never sort, so the mesh spread and the CSVR update
  no longer wait for the list check (`DeviceEngine.HOST_SORT[] = false`
  restores the device decision). Exact lists (RTX 3090, five paired runs):
  10.6k molten NaCl NVT 1.016× faster (95% CI 1.016–1.017), Argon 10k NVE
  1.024× (95% CI 1.011–1.037).
- Charged cluster-pair runs with fixed list intervals choose the stream
  priority of their long-range mesh branch: the first steps time the step
  graph with the mesh at high priority and at regular priority, and the run
  keeps the faster one. High priority pays when the mesh is the critical path
  (10.6k molten NaCl, five paired runs: NVE 1.078× faster, 95% CI
  1.074–1.083; NVT with thermostat interval 10 1.080×, 95% CI 1.074–1.086)
  and is not chosen when the pair force is (10k SPC/E water NPT: 1.000×,
  95% CI 0.995–1.005). `DeviceEngine.MESH_PRIORITY[] = true` forces high
  priority; `DeviceEngine.MESH_PRIORITY_TUNE[] = false` turns the choice off.
- Cluster-pair CSVR runs compute the full-step kinetic energy per bath only on
  steps that report energies and at host boundaries; other steps skip that
  reduction and its report kernel, and public observables refresh it on demand.
  The thermostat's half-step reductions, random draws and velocity scale are
  unchanged. This applies to cluster-pair CSVR states without a device barostat
  whose per-particle features are limited to masses and unwrapped coordinates;
  other states keep the previous path. The choice is fixed when a run is
  created (`DeviceEngine.CSVR_LAZY_REPORT[] = false` restores the previous path
  for new runs). 10.6k molten NaCl NVT with thermostat interval 10, buffered
  lists (RTX 3090, three paired runs): 1.029× faster (95% CI 1.015–1.042).

### Device-resident step engine

- Breaking: every `step!` and every public force evaluation runs on a
  device-resident CUDA-graph engine, and the staged step engine is removed.
  There is no fallback. A state the engine cannot run raises `ArgumentError`
  with the reason (`DeviceEngine.engine_unsupported_reason`).
  `step_graph!` is now the same as `step!`.
- The engine covers:
  - all integrators and thermostats;
  - pair models, bonded families, external potentials, freeze, collisions,
    rigid water, barostats and long-range electrostatics;
  - spatially sorted bonded states (topology follows particle ids);
  - small boxes (all-pairs rows up to 32 768 particles);
  - typed stencil lists.
- `evaluate_forces_into_f!(st, energy)` invalidates and re-evaluates on the
  engine. An untethered evaluation of a tethered state (`freeze_spring=false`)
  uses an evaluation engine with the tether off.
- Removed: `run_integrator_step!` and the staged stage protocol; the staged
  pair kernels of `NonBondedForces` (`lj_forces_soa*`, `wca_forces_soa*`,
  `harmonic_rep_forces_soa*`, pair-matrix, stencil and mixed variants); the
  `NonBondedInteractions` module.
- New `vdw_modifier = :potential_shift` (`build_simulation` keyword,
  `set_vdw_modifier!`, `vdw_modifier`). It shifts LJ to zero energy at the
  cutoff, as GROMACS `vdw-modifier = Potential-shift` does, and forces are
  unchanged. The default `:none` keeps plain truncation.
- Dense neighbor lists of dense systems (about 72 or more neighbors within the
  list radius) are built with 16 lanes per row over a cell grid subdivided
  twice per list radius. 10.6k molten NaCl NVT (PME, thermostat every 10
  steps, dense list, RTX 3090): 1.05× faster (95% CI 1.050–1.060, five paired
  runs).

### Correctness and maintenance

- Historical-reference tests use external fixtures configured with `PD_GOLDEN_ROOT`
  or per-dataset overrides. Missing datasets visibly skip their comparison testsets;
  `PD_REQUIRE_GOLDEN=1` requires the complete reference collection. Incomplete or
  corrupt supplied datasets fail. Fixture data is not bundled with the package.

- Mixed per-particle sigma LJ/WCA forces now honor bonded exclusions and support
  all-pairs traversal, including force-only, energy, and virial kernels.
- Stochastic stepping and temperature/OU setters reject timesteps different
  from the spec's cached timestep before changing physical state or coefficients.
  Construct a new spec with the intended `dt`; raw parameter constructors must
  also receive the correct timestep. All steps require finite, positive `dt`.
- FENE evaluations reject nonfinite or overstretched bonds with `DomainError`
  before accumulating bonded outputs. Removed the denominator clamp and improved
  energy accuracy near zero extension. The device engine latches domain errors
  and reports them at host boundaries. Finite nonnegative `k` and a
  finite positive representable `R0²` are required.

- Force evaluation checks neighbor displacement after motion and at temporary
  midpoint coordinates, regardless of the step-boundary check interval.
  Spatial reordering stays at safe step boundaries. Exact lists use device
  displacement checks; buffered lists with fixed intervals follow their schedule.
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

- Dense and stencil neighbor rows use ELL storage; bond adjacency remains CSR.
  The old staged nonbonded kernels were removed with the staged engine.
- Collision readers and contact-history buffers now use the same shared
  slot-major ELL index as force kernels and neighbor builders.
- Neighbor builders count all required entries and throw
  `NeighborLists.NeighborCapacityError` on overflow. Failed or unbuilt lists
  cannot be consumed by force/collision entry points. Low-level list builders
  require sufficient capacity; the device step engine grows overflowing lists
  and resumes the interrupted step at a host boundary.
- Neighbor containers gained `valid` and `required_capacity` fields. Code
  using positional struct constructors must migrate to the neighbor builders
  or supply the new fields.
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
- `build_simulation(...; spatial_reorder=true)` enables spatial sorting for
  memory locality. `st.tag[k]` maps a storage slot back to its build-time
  particle id; bonded topology follows those ids. GSD output and restart files
  preserve build-time particle order. Pass `spatial_reorder=false` if external
  code holds storage-slot indices into the state.
- Deterministic drift kernels use a single conditional wrap instead of
  floor-division wrapping.
- Measured on an RTX 3090 (N=10^6 LJ particles, rho=0.3, Float32, steady
  state): 52 -> ~660 steps/s.

### Changed

- Export `eulerheun` and `read_gsd_frame!` from the top-level `ParticleDynamics` API so the documented restart and Brownian-midpoint workflows use supported imports.
- Stop re-exporting the low-level `Simulation` helpers `zero_forces!`, `accumulate_energies!`, `run_integrator_step!`, `thermostatted_dof`, and `thermostatted_particle_mask`.
- `build_simulation` now accepts an explicit `backend` keyword with `:cuda` as the supported default and an early error for unsupported `:cpu` requests.

Migration notes:
- For retained low-level helpers, use `ParticleDynamics.Simulation.<name>`.
  `run_integrator_step!` has been removed; use the public `step!` API instead.
