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

## Example scripts

- `examples/2D_softrep_nl_check.jl`: soft-repulsive neighbor-list path
- `examples/3D_stencil_two_sizes.jl`: mixed-size cutoff workflow
- bonded polymer examples under `examples/` remain GPU-first today and should
  only be treated as future CPU-compatible candidates after a real CPU backend lands

## Related pages

- [Simulation State](simulation_state.md)
- [Observables](observables.md)
