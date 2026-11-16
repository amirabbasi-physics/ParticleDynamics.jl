# Electrostatics

ParticleDynamics computes long-range Coulomb interactions natively on the GPU
with three particle-mesh solvers that share one mesh engine:

| Descriptor | Method | Split | Assignment | Differentiation | Transforms per force call |
|---|---|---|---|---|---|
| [`PME`](@ref) | smooth particle-mesh Ewald | Gaussian | cardinal B-spline | analytic | 1 forward, 1 inverse |
| [`PPPM`](@ref)`(differentiation=:ad)` | P3M, optimal influence function | Gaussian | B-spline | analytic | 1 forward, 1 inverse |
| [`PPPM`](@ref)`(differentiation=:ik)` | P3M, optimal influence function | Gaussian | B-spline | reciprocal-space field | 1 forward, 3 inverse |
| [`ESP`](@ref) | Ewald summation with prolates | prolate (PSWF) | prolate window | analytic | 1 forward, 1 inverse |

The solvers run on the GPU and combine with the short-range interactions
in your integrator.

Supported scope: 3D orthorhombic periodic boxes, conducting (tin-foil)
boundary conditions, uniform relative permittivity, fixed charges, a single
GPU, Float32 or Float64. 2D, slab, triclinic and open-boundary electrostatics
are rejected at construction.

## Choose a solver

PME and PPPM use B-spline assignment; ESP uses a prolate window and can use a
smaller FFT grid for a given measured accuracy. The best choice depends on
system size, charge distribution and accuracy target. A smaller mesh alone
does not guarantee a faster complete simulation: spreading, pair forces and
integration also cost time.

Use representative configurations to check force, energy and pressure errors.
For ESP, `tolerance` controls a parameter rule rather than a guaranteed bound
on measured force error; the [parameter guidance](#Choosing-parameters) below
explains its precision limits.

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
the prolate window `ψ₀ᶜ¹(2x/P) − ψ₀ᶜ¹(1)`), transformed with an unnormalised forward
FFT, multiplied by the influence function `G` (which contains `C/V`) and
transformed back with an unnormalised inverse. The mesh energy is
`½ Σ G|b̂|²`; analytic-differentiation forces are the exact gradient of this
discrete energy. The constant `ψ₀ᶜ¹(1)` makes the prolate weights vanish at
the edge of the support of the window, so that the mesh energy is continuous
when a particle moves into the next mesh cell, as it is for B-splines.
Influence functions:

- SPME: `G = (C/V) ĝ(k) Π_d D_d`, with SPME's exact discrete B-spline factor `D`;
- P3M-AD/ik: Hockney–Eastwood optimal alias sums for the selected derivative;
- ESP: by default the optimal AD influence function with alias sums of the
  prolate window transform (`influence=:optimal`). With
  `influence=:deconvolution` it is the plain `ĝ/φ̂²` of the paper's eq. (9).

## Energies and measurements

Thermodynamic observables include electrostatic energy automatically. Pair
energies are shared equally between the two particles; the mesh assigns energy
through each particle's interpolated potential. Subgroup energies follow this
convention.

The reciprocal-space virial belongs to the whole system and has no per-particle
decomposition. Request virial measurements on the full particle group when
electrostatics is present. Per-particle GSD virials contain the local terms only.

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

## Save and continue

GSD output records charges, electrostatics settings and exceptions.
`ParticleSystem.from_gsd(path)` restores the particle configuration;
`electrostatics_from_gsd(path)` returns the saved force descriptor for your
new integrator. See [continuing a simulation](restarts.md).

## Choosing parameters

Parameters are explicit; nothing is tuned behind the user's back.

- **PME/PPPM.** Choose `real_cutoff`, then `alpha`, or `real_space_tolerance`
  (`erfc(α r_c) = tol`, a real-space potential-tail criterion rather than a
  force error). Then refine `mesh`/`mesh_spacing` and `order` until the
  measured force error meets the target.
- **ESP.** `tolerance = ε` sets the split shape (`ψ₀ᶜ(1) = ε` for the
  L²-normalised prolate), the window shape (from `ε/2`), the window width
  (Table 2 of Liang et al. 2026: 5, 5, 6, 7, 8 for ε = 1e-3 … 1e-5) and
  `mesh_spacing = π r_c / c`. The measured force error is 0.4–2.5 ε in our tests and
  does **not** decrease further when only the mesh is refined: the prolate
  window's alias level is fixed in grid units. To reach a target Δ, tighten ε
  (e.g. ε ≈ Δ/2) instead of the mesh.

  The window also has to be resolved by the precision `T` of the state. The
  influence function divides by the squared window transform, so a window
  that decays too far inside the mesh band amplifies round-off. The bound is
  `(φ̂(π)/φ̂(0))² ≥ 0.02 eps(T)^(2/3)` at the mesh Nyquist frequency: 4.8e-7
  in Float32, 7.3e-13 in Float64. Where the `ε/2` shape falls below it
  (Float32: ε ≈ 3.3e-5–4.5e-5 and ε below about 6e-6; Float64: ε below about
  3e-9), the default window shape is the smallest one that meets the bound,
  at the same width. An explicit `window_shape` below the bound is refused
  when the solver is built. Float32 forces reach about 2e-6 relative accuracy
  at best, whatever ε; tighter targets need a Float64 state.
Test accuracy on representative configurations before production. Compare
forces against a more accurate calculation, then check energy conservation and
the observables that matter to your system. A requested tolerance alone is
not a measured force-error bound.

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

## Solver reference

```@docs
ParticleDynamics.PME
ParticleDynamics.PPPM
ParticleDynamics.ESP
ParticleDynamics.ElectrostaticsForce
ParticleDynamics.electrostatics_from_gsd
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
