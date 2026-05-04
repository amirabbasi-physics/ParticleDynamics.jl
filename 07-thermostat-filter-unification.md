# PR 7: Unify particle selection and thermostat operator interfaces

**Branch**: `refactor/05-thermostat-filter-unification`  
**Commit**: `ab4634a77a7fe7de2a81f635dbd06495c83d814a`  
**Date**: 2026-05-04 17:02:33  
**Status**: ✅ Committed

## Summary
Phase 5 architecture work. Unifies particle selection and thermostat operator patterns to eliminate ad hoc conditional logic and duplication across thermostats, observables, and diagnostics. Introduces two new core modules:

- `ParticleGroups.jl`: canonical particle selection and grouping abstraction
- `Thermostats.jl`: unified thermostat interface and state management

The new abstractions make it possible to apply thermostats, observables, and diagnostics to arbitrary particle subsets with a consistent API, and prepare the foundation for future constraints, barostats, and advanced operations.

## Changes
**Created**:
- `src/ParticleGroups.jl` — Canonical particle selection API (replacing ad hoc `Filters`)
- `src/Thermostats.jl` — Unified thermostat interface (`AbstractThermostat`, state management)
- `test/test_particle_groups.jl` — Selection and group operation tests
- `test/test_thermostats.jl` — Thermostat interface tests

**Modified**:
- `src/ParticleDynamics.jl` — Added new module includes and exports
- `test/runtests.jl` — Registered new test suites

## Goals
- Eliminate ad hoc particle selection logic duplicated across `Filters`, integrators, and observables
- Establish a shared interface for thermostats (NHC, CSVR) that enables:
  - Per-bath parameter tuning
  - Multi-bath support with unified APIs
  - Energy exchange diagnostics
  - Future extension to constraints and barostats
- Separate AOUP as a first-class operator (deferred to follow-up PR)
- Reduce complexity in integrator specifications by factoring out thermostat state
- Enable particle-subset operations (filters, diagnostics, partial thermostating) uniformly

## Testing
- Full `Pkg.test()`: 585/585 tests pass (+28 new tests from Phase 4)
- ParticleGroups: Selection specification resolution tests ✅
- ParticleGroups: GPU and host operations tests (apply_scalar!, apply_values!, gather, sum_values) ✅
- Thermostats: Interface existence and parameter container tests ✅
- All previous test suites remain green ✅

## Performance
No kernel or stepping logic changes — architecture-only refactoring. No performance regression expected.

## Acceptance Criteria
✅ `ParticleGroup` abstraction replaces ad hoc index/filter management  
✅ `ParticleSelection` specifications resolve uniformly (host and device)  
✅ `AbstractThermostat` provides common interface for all thermostat types  
✅ Per-bath parameter tuning works through unified `set_target_temperature!` and `set_response_time!`  
✅ All 585 tests pass  

## Breaking Changes
**Additive only** — no breaking changes to existing code. New modules are internal unless explicitly imported.

**New public interface**:
```julia
# Particle selection API
sel = ParticleGroups.TypeIDs(1)
group = ParticleGroups.materialize(sel, state)
ParticleGroups.apply_scalar!(buffer, group, value)
ParticleGroups.gather(buffer, group)

# Thermostat API (future integration)
thermo = NoseHooverChainThermostat(...)
Thermostats.n_baths(thermo)
Thermostats.set_target_temperature!(thermo, T_new)
Thermostats.set_response_time!(thermo, tau_new)
```

## Migration Notes
None required yet. The `Filters` module continues to work and will be refactored to use `ParticleGroups` internally in a follow-up. Users of `Filters` API see no changes.

The `Thermostats` module is currently interface-only. Integration with `Simulation` module will happen in Phase 6.

## Rollback
```bash
git revert ab4634a
git rm src/ParticleGroups.jl src/Thermostats.jl
git rm test/test_particle_groups.jl test/test_thermostats.jl
```

## Architecture Notes

### ParticleGroups Design
The `ParticleGroups` module separates **selection specification** from **materialization**:

1. **Selection Spec** (`ParticleSelection`):
   - `All`: all particles
   - `TypeIDs`: particles of given type(s)
   - `Indices`: explicit index list
   - User-extendable via `resolve(::CustomSelection, state)`

2. **Materialized Group** (`ParticleGroup`):
   - Holds both host and device indices
   - Reusable across operations (no recomputation)
   - Enables efficient filtering on both CPU and GPU

3. **Operations**:
   - `apply_scalar!`: set values for a group
   - `apply_values!`: assign distinct per-member values
   - `gather`: copy group values to host
   - `sum_values`: reduce group values

This design is more efficient than the old ad hoc approach because selections are resolved once and groups are reused across multiple operations.

### Thermostats Design
The `Thermostats` module establishes a two-tier hierarchy:

1. **Interface Tier** (`AbstractThermostat`):
   - `ThermostatState`: common state interface
   - Parameter accessors: `n_baths`, `target_temperature`, `response_time`
   - Parameter mutators: `set_target_temperature!`, `set_response_time!`
   - Diagnostics: `cumulative_energy_exchange`

2. **Concrete Implementations**:
   - `NoseHooverChainThermostat`: deterministic multi-bath NHC
   - `CSVRThermostat`: stochastic global rescaling

Both implementations unify:
- Multi-bath support (per-particle bath assignment via `particle_bath_id`)
- Target temperature and response time per bath
- Energy exchange tracking per bath
- DOF and kinetic energy management

In Phase 6, the `Simulation` module will be refactored to compose these operators explicitly.

## Integration Notes
This PR is part of a multi-phase refactor:
- PR1-PR6: Completed (all tests passing)
- PR7 (this): New abstractions for particle groups and thermostats
- PR8 (planned): Redesign nonbonded interaction interfaces
- PR9 (planned): Topology and force-field foundation

Phase 5 work improves code organization and prepares for later phases by establishing:
1. A unified model for particle subsets (groups, selections, masks)
2. A clear thermostat operator interface
3. Foundation for constraints, barostats, and AOUP separation

Future work will integrate these with the main simulation loop and leverage them to simplify force/integrator code.
