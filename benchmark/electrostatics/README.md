# Cross-engine electrostatics benchmark

**Scope:** this directory measures frozen-coordinate forces. The
complete rigid SPC/E NPT comparison is in `../water`.
`run_tuned.jl` writes fresh repeated timings and held-out checks into
`PD_BENCH_RUN` (a new directory, default `PD_BENCH_WORK/runs/<timestamp>`);
cached settings are candidates, never reused performance measurements.

Head-to-head comparison of the ParticleDynamics GPU Coulomb solvers (PME,
PPPM, ESP) with GROMACS and LAMMPS on one GPU, using the systems of Liang et
al., *Accelerating molecular dynamics simulations using fast Ewald summation
with prolates*, Nat. Commun. 17, 6681 (2026). The data release is Zenodo
19559644 (`Data.zip`, MD5 `ce54fbbab32a03ebe012d2288d004675`).

The comparison uses **frozen coordinates**: electrostatic forces on identical
configurations, at matched *measured* accuracy. It needs only charges,
exclusions and 1–4 scaling, so bonded, angular and dihedral terms,
constraints and pressure coupling are never evaluated by ParticleDynamics.
Complete MD time steps of these molecular systems are outside its current
force field.

## Files

| File | Content |
|---|---|
| `Fixtures.jl` | GROMACS and LAMMPS fixture loaders (charges, exclusions, 1–4 pairs via the compiled run input), replication, spatial sorting |
| `Engines.jl` | GROMACS/LAMMPS drivers: TRR reader, εᵣ-trick Coulomb isolation, single points, GPU step timing, LAMMPS frozen-force runs |
| `PDHarness.jl` | ParticleDynamics states, forces and CUDA-event stage timings |
| `Benchmark.jl` | includes the three modules |
| `run_water.jl` | SPC/E water series (3,000 → 1,029,000 atoms), targets Δ = 1e-3 and 1e-4 |
| `run_heterogeneous.jl` | membrane (81,743), lysozyme (1,036,152), LiTFSI (1,011,392) at the paper's targets |
| `run_tuned.jl` | water series with each engine at its own best setting: PD `tune_electrostatics(...; objective=:time)` on the cluster-pair backend, GROMACS in its production configuration |
| `run_openmm.jl`, `openmm_water.py` | OpenMM 8 (CUDA, mixed precision, PME) calibrated to the same measured Δ and timed as GPU-resident Verlet steps |

## Requirements

```bash
export PD_ESP_DATA=/path/to/ESP_Data        # unpacked Data.zip
export PD_BENCH_WORK=/path/to/work          # outside the repository
export PD_GMX_GPU=/path/to/gromacs-gpu/bin/gmx
export PD_GMX_DOUBLE=gmx_d                  # double-precision GROMACS (references)
export PD_LMP_KOKKOS=/path/to/lmp           # LAMMPS with KOKKOS/CUDA, KOKKOS_PREC=mixed, FFT_SINGLE
export PD_OPENMM_PYTHON=/path/to/python     # environment with openmm (CUDA platform)
export PD_GMX_TOPDIR=/path/to/gromacs/share/gromacs/top
julia --project=. benchmark/electrostatics/run_water.jl [max_atoms]
julia --project=. benchmark/electrostatics/run_heterogeneous.jl
```

Results are cached in `PD_BENCH_WORK` (`*.jls`) and summarised in
`water_report.md` and `heterogeneous_report.md`.

## Method

- **Error metric.** Δ = (Σ‖F − F_ref‖² / Σ‖F_ref‖²)^½ on the electrostatic
  forces only. References are ParticleDynamics Float64 ESP solves at
  ε = 1e-10, validated against double-precision GROMACS: forces 9e-8,
  energy 1e-11 and virial 3e-7 relative on 3000 waters.
- **Isolating other engines' Coulomb forces.** GROMACS: two single-point
  evaluations at εᵣ = 1 and 2 (`X_coul = 2(X₁ − X₂)`, exact for forces,
  energy and virial). LAMMPS: `coul/long` with bonds encoding the exclusions
  and `special_bonds lj/coul 0 1 1`.
- **Water inputs are lattices.** The archived water boxes are periodic
  replicas of a 0.31 nm single-molecule cell. Their Bragg peaks alias onto
  low mesh modes, so the force error swings by 30× between neighbouring grid
  sizes (40³: 1.2e-4, 48³: 4.0e-3, 80³: 3.9e-5 at the same α, identically in
  GROMACS and ParticleDynamics). Every size is therefore equilibrated
  (GPU NVT, 300 K, 20 ps) into liquid water. Frame A is used for tuning and
  frame B, 2 ps later, for held-out evaluation; these are derived fixtures.
- **Tuning.** For each engine and target the knob is tightened until the
  engine's own measured Δ meets the target: grid spacing for GROMACS PME
  (order 4, ewald-rtol 1e-5) and ParticleDynamics PME/PPPM; the accuracy
  parameter for LAMMPS PPPM (order 5); the tolerance ε for ESP. Each engine
  runs in its production precision (GROMACS mixed, LAMMPS-KOKKOS mixed,
  ParticleDynamics Float32).
- **Timing.**
  - ParticleDynamics: median of 5 batches of 30 CUDA-event timings of the
    electrostatic force call, after warm-up, with stage breakdown.
  - LAMMPS: frozen run (no integrator), `timer sync full`, Pair + Kspace per
    step.
  - GROMACS: Nsight Systems cannot run in the sandboxed environment and the
    built-in GPU timers of GROMACS 2026.2 fail. GROMACS is therefore timed by
    its GPU-resident MD step (`-nb gpu -pme gpu -update gpu`, PME tuning off).
    Its electrostatics cost is estimated as that step minus the same step
    with zero charges.
- **GROMACS 2026.2 crash.** The GPU build segfaults with rvdw ≪ rcoulomb
  (the paper's "LJ off" trick), so all GROMACS runs use rvdw = r_c with the
  SPC/E O–O Lennard-Jones term. The W2 workload (LJ + Coulomb) compares that
  against ParticleDynamics and LAMMPS evaluating the same LJ term.
- **ParticleDynamics backends.**
  * `run_water.jl` times both nonbonded backends: `:ell`, the per-particle
    neighbor rows, and `:cluster`, the fused cluster-pair kernel with the
    mesh overlapped.
  * The cluster backend runs with an exact prune radius (`prune_buffer=0`).
    For fixed coordinates that is the equivalent of GROMACS's unbuffered
    `rlist = r_c` runs.
- **GROMACS GPU-side time.** The md.log cycle accounting gives the CPU pair
  search per step. The GPU waits for it, so step − search estimates the
  GPU-resident part: nonbonded, PME, update and SETTLE. Both numbers are
  reported.
- **GROMACS production configuration** (`run_tuned.jl`): automatic Verlet
  buffer (rlist 1.032 nm). Its PME load balancing (`-tunepme`) does not
  complete on this single-GPU, GPU-update setup: with a fixed rlist it never
  ends, and with the automatic buffer it ends with final = initial. The
  production timing therefore uses the calibrated grid.
