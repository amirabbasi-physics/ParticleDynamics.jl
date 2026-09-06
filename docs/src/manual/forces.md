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

The stable user surface is the parameter/topology layer plus
`build_simulation(...; nonbonded=..., bonding=..., bonds=...)`.

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
