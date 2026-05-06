# Observables

Observables are accumulated on device and exposed through a small public API for
step-level diagnostics, virial accumulation, and thermostat exchange tracking.

## Public observables API

```@docs
ParticleDynamics.collect_step_observables
ParticleDynamics.reset_bath_exchange_accumulators!
ParticleDynamics.accumulate_virial!
ParticleDynamics.virial_components
ParticleDynamics.virial_tensor
```

## What is available

- total potential and kinetic energy
- scalar virial and tensor virial
- heat/work accumulators used by stochastic and thermostat paths
- integrator-specific metadata merged into the returned `NamedTuple`

## Collision counting

Collision counting is also part of the current top-level API:

```@docs
ParticleDynamics.enable_collision_counting!
ParticleDynamics.disable_collision_counting!
ParticleDynamics.collisions_reset_counts!
ParticleDynamics.collisions_read_counts!
ParticleDynamics.set_collision_pair_cutoffs!
```

## Example scripts

- `examples/3D_quicktest.jl`: writes energies while stepping
- `examples/` collision-rate workflows: production-style diagnostics, still GPU-first today
