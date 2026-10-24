# Forces

ParticleDynamics exposes high-level parameter and topology objects for bonded
and nonbonded interactions. Force kernels themselves remain internal and
CUDA-specific.

## Public parameter and bond API

```@docs
ParticleDynamics.LJParams
ParticleDynamics.SoftRepulsiveParams
ParticleDynamics.HarmonicBondParams
ParticleDynamics.FENEParams
ParticleDynamics.BondPotential
ParticleDynamics.HarmonicBond
ParticleDynamics.FENEBond
ParticleDynamics.harmonic_bond
ParticleDynamics.fene_bond
ParticleDynamics.BondList
ParticleDynamics.build_bondlist
```

## Current interaction families

- nonbonded Lennard-Jones
- nonbonded WCA
- nonbonded soft-repulsive
- harmonic bonds
- FENE bonds
- long-range Coulomb: smooth PME, PPPM (AD/ik) and Ewald summation with
  prolates, see [Electrostatics](electrostatics.md)

The stable user surface is the parameter/topology layer plus
`build_simulation(...; nonbonded=..., bonding=..., bonds=...)`.
`nonbonded=:none` builds a state without a short-range vdW term (for purely
charged or purely bonded systems).

### Lennard-Jones energy at the cutoff

`vdw_modifier` sets the Lennard-Jones energy convention at the cutoff (the
GROMACS `vdw-modifier` names):

- `:none` (default): plain truncation, `u(r) = 4ε[(σ/r)¹² − (σ/r)⁶]` for
  `r < r_c`;
- `:potential_shift`: `u(r) − u(r_c)` for every pair inside its cutoff (per pair
  with pair matrices), the GROMACS default.

Forces and virials are identical for both, so molecular dynamics samples the
same ensemble. Energies differ by `u(r_c)` per pair inside the cutoff. This
matters wherever the energy itself drives or is compared:
- Monte Carlo volume moves sample the energy they are given;
- energy totals compared against another code.

Plain truncation with a 1 nm cutoff makes SPC/E water about 0.5% denser under
Monte Carlo volume moves than under MD. The missing shift acts like roughly
100 bar of extra pressure.

```julia
st = build_simulation(...; nonbonded=:lj, vdw_modifier=:potential_shift)
set_vdw_modifier!(st, :none)    # switch later; energies are re-evaluated
```

The modifier applies on both nonbonded backends.

```@docs
ParticleDynamics.set_vdw_modifier!
```

## Nonbonded backends

3D states choose how short-range pairs are traversed:

- `:ell` (default): per-particle neighbor rows, one thread per particle.
  The vdW kernel and the real-space Coulomb kernel each walk the rows. This
  is the fastest choice for short rows, e.g. Lennard-Jones fluids at
  r_c = 2.5σ (tens of neighbors per particle).
- `:cluster`: the cluster-pair engine. Particles are grouped into spatial
  clusters of 8. One kernel evaluates Lennard-Jones and the real-space
  Coulomb term over a Newton-3 list of 8×8 tiles, with exclusions and scaled
  pairs as bit masks, and the Ewald or prolate correction as a polynomial in
  r². With a mesh solver, the mesh transform runs concurrently with the pair
  kernel and the list maintenance. This is the
  fast choice for long rows, e.g. molecular systems with a 1 nm cutoff
  (~500 neighbors per atom). On 1,029,000 SPC/E atoms with LJ + PME at
  Δ = 1e-3 (RTX 3090, Float32), a complete force evaluation takes 37.3 ms
  with `:ell` and 4.5 ms with `:cluster`.

**Default choice:** attaching electrostatics (`attach_electrostatics!`, or an
electrostatics force in the workflow) switches the state to `:cluster`, and
detaching it returns to `:ell`. Uncharged systems stay on `:ell`.

Charged states with soft repulsion, and 2D charged states, are not supported:
the cluster backend has no soft-repulsion kernel, and electrostatics are 3D.
In small boxes the cluster skin is capped so that the list radius stays
within half the box.

```julia
st = build_simulation(...; nonbonded_backend=:cluster)   # or later:
set_nonbonded_backend!(st, :cluster; prune_buffer=0.05)
nonbonded_backend(st)                                      # :cluster
```

In the workflow, pass `neighborlist=ClusterPairList(buffer=0.1)` to the
nonbonded force (or to the electrostatics force when there is no vdW term).

```@docs
ParticleDynamics.set_nonbonded_backend!
ParticleDynamics.nonbonded_backend
ParticleDynamics.ClusterPairList
```

The list radius is the largest interaction cutoff plus the skin (`buffer`).
Between rebuilds the list is pruned to the cutoff plus `prune_buffer`
(default half the skin). It is re-pruned when a particle has moved more than
half the prune buffer and rebuilt when one has moved more than half the
skin. The Coulomb and vdW cutoffs are independent: a short ESP real-space
cutoff inside a longer LJ cutoff costs no extra traversal.

The cluster backend supports:
- no vdW term, LJ and WCA (uniform, pair table or per-particle σ);
- bond exclusions of the vdW term and collision counting;
- with or without an electrostatics solver.

Use `:ell` for soft repulsion and 2D.
Both backends give the same forces to round-off: Float64 agrees to 1e-11
and Float32 to ~2e-6 relative, the latter from the polynomial Coulomb
correction and the summation order.

## Mixed sizes and exclusions

The low-level `sigma_particle` LJ/WCA path uses Lorentz mixing and honors
`exclude_bonded_pairs` for dense ELL, stencil ELL, and all-pairs neighbors.
After manual size changes, configure neighbor cutoffs to cover the largest
pair interaction and rebuild the list; `invalidate_forces!` alone does not
resize cutoff ranges. All-pairs traversal avoids neighbor truncation but has
quadratic cost in particle count.

## FENE domain

FENE requires finite `k ≥ 0`, positive finite `R0` with a representable positive
finite square, and finite minimum-image bond lengths strictly below `R0`.
Evaluation throws `DomainError` identifying an incident particle if a bond
reaches or exceeds the limit. The potential and force are not clamped or
continued beyond the physical domain. Near the limit they diverge; choose the
timestep and initial configuration accordingly.

A GPU precheck completes before bonded outputs are accumulated. It adds a
kernel, a small status allocation, and host synchronization to every FENE force
evaluation. A failed simulation step invalidates its force cache but does not
roll back positions, velocities, or earlier nonbonded output. Restore a valid
configuration before resuming.

## Example scripts

- `examples/2D_soft_repulsive.jl`: soft-repulsive neighbor-list path
- `examples/3D_stencil_two_sizes.jl`: mixed-size cutoff workflow
- bonded polymer examples under `examples/` remain GPU-first today and should
  only be treated as future CPU-compatible candidates after a real CPU backend lands

## Related pages

- [Simulation State](simulation_state.md)
- [Observables](observables.md)
