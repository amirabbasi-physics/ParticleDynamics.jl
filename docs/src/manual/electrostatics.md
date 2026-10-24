# Electrostatics

ParticleDynamics computes long-range Coulomb interactions natively on the GPU
with three particle-mesh solvers that share one mesh engine:

| Descriptor | Method | Split | Assignment | Differentiation | Transforms per force call |
|---|---|---|---|---|---|
| [`PME`](@ref) | smooth particle-mesh Ewald | Gaussian | cardinal B-spline | analytic | 1 forward, 1 inverse |
| [`PPPM`](@ref)`(differentiation=:ad)` | P3M, optimal influence function | Gaussian | B-spline | analytic | 1 forward, 1 inverse |
| [`PPPM`](@ref)`(differentiation=:ik)` | P3M, optimal influence function | Gaussian | B-spline | reciprocal-space field | 1 forward, 3 inverse |
| [`ESP`](@ref) | Ewald summation with prolates | prolate (PSWF) | prolate window | analytic | 1 forward, 1 inverse |

Every step of a force evaluation runs on the device: real-space pairs,
exception corrections, charge spreading, cuFFT R2C/C2R transforms with
preallocated plans, the spectral solve (with Float64 energy and virial
reductions), and the force gather. Host code validates the input, generates
coefficients and plans transforms at setup only. A warmed force call performs
no device allocations.

Supported scope: 3D orthorhombic periodic boxes, conducting (tin-foil)
boundary conditions, uniform relative permittivity, fixed charges, a single
GPU, Float32 or Float64. 2D, slab, triclinic and open-boundary electrostatics
are rejected at construction.

## Quick example

```julia
using ParticleDynamics

n, a = 16, 2.82                                  # Å
L = n * a
pos = [(a * (i - 0.5) - L / 2, a * (j - 0.5) - L / 2, a * (k - 0.5) - L / 2)
       for i in 1:n for j in 1:n for k in 1:n]
typeids = Int32[isodd(i + j + k) ? 1 : 2 for i in 1:n for j in 1:n for k in 1:n]
system = ParticleSystem(pos; box=(L, L, L), types=[:Na, :Cl], typeids=typeids,
                        masses=Dict(:Na => 22.99, :Cl => 35.45),
                        charges=Dict(:Na => 1.0, :Cl => -1.0))
all_ions = Group(:all, AllSelection())

sim = Simulation(system;
    groups=Groups(all_ions),
    integrator=Integrator(
        dt=0.1,                                   # in √(amu Å²/eV) ≈ 10.18 fs units
        forces=[
            LennardJones(epsilon=0.01, sigma=2.8, cutoff=9.0, pairs=:neighborlist,
                         neighborlist=CellList(capacity=512)),
            PME(coulomb_prefactor=14.399645,      # eV·Å/e²
                real_cutoff=9.0, real_space_tolerance=1e-5,
                mesh_spacing=1.0, order=5),
        ],
        methods=[ConstantVolume(all_ions)],
    ),
    observables=[ThermodynamicObservable(all_ions)],
    precision=Float32,
)
run!(sim, Stage(:nve, steps=1000; progress=false))
```

Replace the `PME(...)` entry by
`PPPM(...; differentiation=:ik)` or
`ESP(coulomb_prefactor=14.399645, real_cutoff=9.0, tolerance=1e-5)` to switch
methods. The short-range vdW force keeps its own cutoff; the shared neighbor
list covers the larger of the two cutoffs.

## Units and conventions

The engine has no built-in unit system. The pair energy is

```math
U_{ij} = C\,\frac{q_i q_j}{r_{ij}},\qquad
C = \frac{\texttt{coulomb\_prefactor}}{\varepsilon_r},
```

and `C` enters every term (real space, reciprocal space, self energy,
exceptions, background, virial) exactly once.

| Quantity | Unit |
|---|---|
| positions, box, cutoffs, `mesh_spacing` | L |
| `alpha` (Gaussian split) | 1/L |
| charges | caller's Q |
| `coulomb_prefactor` | E·L/Q² |
| energy, virial | E |
| forces | E/L |

Common choices: ≈138.935458 for nm, kJ/mol and elementary charges;
≈14.399645 for Å, eV and elementary charges; 1 for reduced units. Thermostat
inputs stay `kT` in energy units. ESP shape parameters (`split_shape`,
`window_shape`) are dimensionless and distinct from `alpha`.

### Ewald split

For a split kernel χ (dimensionless, `∫χ = 1`) with `a(t) = ∫₀ᵗ χ`, `t = r/r_c`:

```math
\frac{1}{r} = \frac{1-a(t)}{r} + \frac{a(t)}{r},\qquad
\hat g(k) = \frac{4\pi}{k^2}\int \chi(x)\cos(k r_c x)\,dx .
```

The Gaussian choice gives `erfc(αr)/r`, `ĝ = 4π e^{-k²/4α²}/k²` and the self
energy `−Cα/√π Σqᵢ²`. The ESP choice uses the order-zero prolate spheroidal
wave function ψ₀ᶜ on `[0, 1]`, so the real-space part vanishes exactly at
`r_c`. Its self energy is `−(C/2) χ(0)/r_c Σqᵢ²`.

### Mesh

Charges are assigned per node with the window `W` (B-spline of order `P`, or
the prolate window `ψ₀ᶜ¹(2x/P)`), transformed with an unnormalised forward
FFT, multiplied by the influence function `G` (which contains `C/V`) and
transformed back with an unnormalised inverse. The mesh energy is
`½ Σ G|b̂|²`; analytic-differentiation forces are the exact gradient of this
discrete energy. Influence functions:

- SPME: `G = (C/V) ĝ(k) Π_d D_d`, with SPME's exact discrete B-spline factor `D`;
- P3M-AD/ik: Hockney–Eastwood optimal alias sums for the selected derivative;
- ESP: by default the optimal AD influence function with alias sums of the
  prolate window transform (`influence=:optimal`). With
  `influence=:deconvolution` it is the plain `ĝ/φ̂²` of the paper's eq. (9).

## Energies, virials and observables

- **Per-particle energies.** Real-space and exception pair energies are split
  ½/½. The mesh contributes `½ qᵢ uᵢ` (`uᵢ` = interpolated mesh potential),
  which sums exactly to the mesh energy. The self energy is local, and a
  neutralising background is distributed in proportion to `qᵢ Q`. System
  totals (`sum(st.Epot)`, thermodynamic observables, NHC/CSVR extended
  Hamiltonians) therefore include electrostatics automatically. Subgroup
  energies follow this documented convention.
- **Virial.** Real-space and exception virials are per-particle (in
  `virial_nonbonded`). The reciprocal-space tensor is the strain derivative of
  the discrete mesh energy at fixed fractional coordinates. It has no
  per-particle decomposition and is stored globally in `st.virial_global`.
  [`virial_tensor`](@ref)`(st)` and full-system observables include it;
  `virial_tensor(st; part=:reciprocal)` returns it alone. Group observables
  that request a virial for a strict subgroup raise an error while
  electrostatics is attached. Per-particle GSD virial output contains the
  local terms only.
- **Diagnostics.** [`electrostatic_energy`](@ref) returns the real, self,
  reciprocal and background components plus the reciprocal virial at the
  current coordinates.

Energies and virials are refreshed on force evaluations with
`compute_energy=true`, like every other term in the engine.

## Exceptions

Scaled or excluded pairs (e.g. 1–2/1–3 exclusions, 1–4 scaling) are handled
by subtracting `(1 − s) C qᵢqⱼ/r` at the pair's minimum-image distance,
independently of the cutoff. Periodic images keep the full interaction. The
real-space sum itself never skips pairs, and Lennard-Jones exclusions never
delete Coulomb pairs. Workflow descriptors collect exceptions from
`exclusions=:topology` (`Topology.exclusions`, scale 0), `exclude_bonded=true`,
and `scaled_pairs=[(i, j, s), …]`.

## Charged systems

`net_charge=:error` (default) rejects a nonzero total charge. With
`net_charge=:background` a uniform neutralising background adds
`−C π Q²/(2α²V)` (Gaussian) or `−C π r_c² m₂ Q²/V` (prolate). It contributes
the same amount to each diagonal virial component and adds no force.

## Lifecycle, reordering and restarts

- The solver holds no particle-array pointers; positions, charges and force
  slots are passed on every call. Force-slot swaps, Brownian midpoint
  evaluations and spatial reordering are therefore safe. Charges are permuted
  with their particles. Reordering is suspended while exception rows are
  attached.
- [`set_charges!`](@ref) replaces charges and re-checks the net-charge
  policy. [`detach_electrostatics!`](@ref) removes the solver.
- GSD frames written while a solver is attached store `particles/charge` and
  `log/electrostatics/settings` (method, prefactor, εᵣ, cutoff, α or shapes,
  order, mesh, alias order, net-charge policy, precision), plus
  `log/electrostatics/exceptions`. `ParticleSystem.from_gsd(path)` restores
  the charges, and [`electrostatics_from_gsd`](@ref) returns the exact
  descriptor. FFT plans and workspaces are rebuilt on load.

## Choosing parameters

Parameters are explicit; nothing is tuned behind the user's back.

- **PME/PPPM.** Choose `real_cutoff`, then `alpha`, or `real_space_tolerance`
  (`erfc(α r_c) = tol`, a real-space potential-tail criterion rather than a
  force error). Then refine `mesh`/`mesh_spacing` and `order` until the
  measured force error meets the target.
- **ESP.** `tolerance = ε` sets the split shape (`ψ₀ᶜ(1) = ε` for the
  L²-normalised prolate), the window shape (from `ε/2`), the window width
  (Table 2 of Liang et al. 2026: 5, 5, 6, 7, 8 for ε = 1e-3 … 1e-5) and
  `mesh_spacing = π r_c / c`. The measured force error is typically 1–2 ε and
  does **not** decrease further when only the mesh is refined: the prolate
  window's alias level is fixed in grid units. To reach a target Δ, tighten ε
  (e.g. ε ≈ Δ/2) instead of the mesh.
- **Measured targeting.** [`tune_electrostatics`](@ref)`(st, force; target)`
  finds the cheapest mesh (PME/PPPM) or tolerance (ESP) whose *measured*
  error against a high-accuracy ESP reference at the same cutoff meets
  `target`. It returns an explicit descriptor, so nothing is retuned silently
  on restart.
- **Fastest setting.** `tune_electrostatics(st, force; target,
  objective=:time)` searches more knobs at the same real-space cutoff:
  - for PME/PPPM, the order (4–6 / 5–7), the Ewald α (real-space tail
    `target × 0.01 … 0.3`) and the mesh;
  - for ESP, ε and the window width.

  It measures each candidate through the state's own force evaluation
  (vdW term and [nonbonded backend](forces.md) included) and returns the
  fastest one that meets `target`, with the full candidate table.

  The Float32 reference is an order-6 PME on a fine grid. Float32 ESP
  references at very tight ε are not accurate enough, because wide prolate
  windows amplify single-precision rounding.

Measured accuracy against an independent converged Ewald sum (Float64, 120
random charges, `r_c = 4`):

| Method | Settings | relative force RMS |
|---|---|---|
| PME | α = 0.9, P = 6, h ≈ 0.30 | 1.3e-5 |
| PME | α = 0.9, P = 8, h ≈ 0.22 | < 1e-6 |
| PPPM-AD | α = 0.9, P = 6, h ≈ 0.30 | 1.3e-5 |
| PPPM-ik | α = 0.9, P = 6, h ≈ 0.30 | 5.9e-7 |
| ESP | ε = 1e-6, P = 8, h ≈ 0.45 | 3.4e-6 |

Float32 adds roughly 1e-6 to 1e-5 on top of the discretisation error.

Cross-engine agreement on a 1000-ion snapshot (r_c = 1 nm, same Coulomb
constant): against GROMACS 2025 double-precision PME (order 10, 72³ grid) the
PME energy agrees to 6e-13 relative, PPPM-AD/ik to 8e-11 and ESP to 4e-10.
Forces agree to the 6-digit precision of the GROMACS trajectory dump. Against
LAMMPS `pppm 1e-9`, all three methods differ by the same 1.6e-7 (energy) and
6e-7 (forces), which is LAMMPS's own discretisation error at that setting.

## Performance

`scripts/bench_electrostatics.jl` times the solvers at matched *measured*
force error. It uses a dense ±1 e ion melt in Å/eV units, Coulomb only,
r_c = 9 Å, Float32, an RTX 3090 and spatially ordered storage; each entry is
one force call:

| N | Δ | SPME | P3M-AD | P3M-ik | ESP |
|---|---|---|---|---|---|
| 64 000 | 1e-4 | 1.14 ms (150³) | 1.09 ms (150³) | 1.15 ms (120³) | 0.74 ms (50³) |
| 64 000 | 1e-5 | 12.5 ms (375³) | 12.5 ms (375³) | 6.7 ms (270³) | 1.6 ms (63³) |
| 512 000 | 1e-4 | 13.5 ms (300³) | 13.5 ms (300³) | 14.0 ms (240³) | 7.6 ms (100³) |
| 512 000 | 1e-5 | mesh > 400³ | mesh > 400³ | mesh > 400³ | 13.5 ms (125³) |

ESP's advantage comes from its much smaller FFT grid and grows with the
accuracy target. Spreading and gathering walk particles in the cell list's
spatial order, so they stay efficient for any storage order. The real-space
kernel profits from spatial reordering (NVE), like the vdW kernels.

### Historical component timings with other engines

The test is equilibrated SPC/E water (the series of Liang et al. 2026) with
O–O Lennard-Jones and Coulomb at r_c = 1 nm, on one RTX 3090. Every engine
runs in its production precision and at a *measured* accuracy target:
relative RMS Coulomb force error Δ against a Float64 reference, checked on a
held-out configuration. ParticleDynamics uses the [cluster-pair
backend](forces.md) with the setting returned by
`tune_electrostatics(...; objective=:time)`. Times are milliseconds.

* ParticleDynamics: one complete force evaluation (LJ + Coulomb).
* GROMACS 2026.2: one GPU-resident MD step (`-update gpu`), production
  Verlet buffer. In brackets, the step minus its CPU pair search.
* OpenMM 8.5.1: one CUDA Verlet step (force + integration).

| N | Δ | GROMACS | OpenMM | PD PME | PD PPPM-AD | PD ESP |
|---|---|---|---|---|---|---|
| 24 000 | 1e-3 | 0.160 (0.115) | 0.215 | 0.191 | 0.229 | 0.216 |
| 24 000 | 1e-4 | 0.196 (0.137) | 0.272 | 0.230 | 0.238 | 0.326 |
| 192 000 | 1e-3 | 1.14 (0.81) | 1.54 | 0.98 | 1.04 | 0.93 |
| 192 000 | 1e-4 | 1.27 (0.96) | 2.24 | 1.09 | 1.17 | 1.06 |
| 648 000 | 1e-3 | 3.92 (2.72) | 5.14 | 2.69 | 2.87 | 2.33 |
| 648 000 | 1e-4 | 4.38 (3.20) | 10.1 | 3.51 | 3.27 | 3.03 |
| 1 029 000 | 1e-3 | 6.24 (4.37) | 9.58 | 4.21 | 4.48 | 3.69 |
| 1 029 000 | 1e-4 | 7.31 (5.44) | 15.5 | 6.07 | 6.01 | 4.86 |

LAMMPS 2026 (KOKKOS/CUDA, mixed precision, `pppm`) takes 25.2 ms (Δ = 1e-3)
and 52.7 ms (1e-4) at 1M atoms, against 5.5 and 11.2 ms for PD PPPM-ik.

These historical rows measure different amounts of work. The PD values exclude
integration, coupling, list rebuilds and re-pruning, while the other engines time
simulation steps. Subtracting GROMACS's CPU pair-search time does not make the work
equivalent. The table describes component costs; it does not establish a complete
simulation speedup or equilibrium accuracy. Use the complete-step benchmark suites
with matched observation schedules, validated list policies and ensemble checks
for those comparisons.

`benchmark/electrostatics/` holds the harness (`run_water.jl`,
`run_tuned.jl`, `run_openmm.jl`) and its method notes.

## Low-level API

```julia
st = build_simulation(N=N, box=(L, L, L), nonbonded=:none, cutoff=rc,
                      precision=:f64, charges=q)
copyto!(st.rx, x); copyto!(st.ry, y); copyto!(st.rz, z)
attach_electrostatics!(st, PME(coulomb_prefactor=1.0, real_cutoff=rc, alpha=1.0,
                               mesh=(32, 32, 32)); exceptions=[(1, 2, 0.0)])
SimulationCore.evaluate_forces_into_f!(st, true)
electrostatic_energy(st)
```

`build_simulation(...; neighbor_cutoff=rc)` widens the neighbor list of a vdW
state to the Coulomb cutoff; `attach_electrostatics!` also widens a dense list
automatically.

```@docs
ParticleDynamics.PME
ParticleDynamics.PPPM
ParticleDynamics.ESP
ParticleDynamics.attach_electrostatics!
ParticleDynamics.detach_electrostatics!
ParticleDynamics.set_charges!
ParticleDynamics.charges
ParticleDynamics.electrostatic_energy
ParticleDynamics.electrostatics_from_gsd
ParticleDynamics.tune_electrostatics
ParticleDynamics.Electrostatics.ElectrostaticsSpec
```

## Provenance

The ESP implementation follows the mathematics published by Liang, Lu,
Barnett, Greengard and Jiang, *Accelerating molecular dynamics simulations
using fast Ewald summation with prolates*, Nat. Commun. 17, 6681 (2026). The
prolate functions are computed from the Legendre expansion of the prolate
differential operator. The split and window shapes follow the paper's rule
`ψ₀ᶜ(1) = ε`, and Table 2 is reproduced as a test. No source code or
coefficient tables from the published reference implementations were used.
References: Essmann et al., J. Chem. Phys. 103, 8577 (1995) (SPME); Hockney
and Eastwood (1988) and Deserno and Holm, J. Chem. Phys. 109, 7678 (1998)
(P3M); Ballenegger, Cerdà and Holm, J. Chem. Theory Comput. 8, 936 (2012)
(influence functions for AD/ik).
