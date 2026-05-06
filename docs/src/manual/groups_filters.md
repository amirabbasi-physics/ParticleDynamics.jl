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

- `examples/TwoT_2D_LD_BAOAB.jl`
- `examples/TwoT_SR2D_MD_CSVR_slab.jl`
- `examples/TwoT_2D_LD_freeze_hold.jl`
- `examples/TwoT_2D_LD_freeze_spring.jl`

These examples are GPU-first today. Small subset-control cases should become
good CPU-compatible smoke candidates only after a CPU execution backend exists.
