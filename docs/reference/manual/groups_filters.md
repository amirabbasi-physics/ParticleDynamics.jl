# Groups and Filters

Particle groups are the stable public selection API. The exported `Filters`
module layers higher-level workflows on top of those selections.

## Stable selection API

```@docs
ParticleDynamics.ParticleSelection
ParticleDynamics.ParticleGroup
ParticleDynamics.All
ParticleDynamics.TypeIDs
ParticleDynamics.Indices
ParticleDynamics.materialize
ParticleDynamics.apply_scalar!
ParticleDynamics.apply_values!
ParticleDynamics.gather
ParticleDynamics.sum_values
ParticleDynamics.ParticleGroups
ParticleDynamics.Filters
```

## Typical uses

- define thermostat baths by particle subset
- assign temperatures or frictions to selected particles
- gather subset observables without copying the whole state to host
- freeze/tether subsets through the higher-level `Filters` workflows

## Notes

- `ParticleGroup` is the reusable materialized form.
- The top-level stable API is the group layer above raw implementation buffers.
- The `Filters` module remains the place for freeze controls and stochastic
  selector-based convenience operations.

## Example scripts

- `examples/TwoT_LJ2D_LD.jl`: two particle groups, each with its own Langevin bath
  and its own thermodynamic observables
