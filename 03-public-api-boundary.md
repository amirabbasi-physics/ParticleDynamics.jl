# PR 3: Define the supported public API and de-export internals

**Branch**: `refactor/02-public-api-boundary`  
**Commit**: `b241b42f9963fd775a29134afb774bec9673135a`  
**Date**: 2026-05-04 15:17:55  
**Status**: ✅ Committed

## Summary
Phase 2 public/private API boundary work. Freezes what ParticleDynamics actually promises at the top level by curating exports in `src/ParticleDynamics.jl`, updating documentation, and ensuring curated public examples use only supported top-level API. De-exports internal helpers that should not be part of the public contract, and documents that they remain available as `ParticleDynamics.Simulation.<name>` for users who need them.

## Changes
**Modified**:
- `src/ParticleDynamics.jl` — Curated top-level exports
- `README.md` — Updated public API documentation
- `docs/src/index.md` — Clarified public API boundary
- `docs/src/manual/getting_started.md` — Updated for public API
- `examples/2D_allpairs_quicktest.jl` — Uses only public API
- `examples/3D_quicktest.jl` — Uses only public API
- `test/test_api.jl` — Added/updated export smoke tests
- `test/test_ir_phase2.jl` — Updated for new exports
- `CHANGELOG.md` — Documented API changes

## Goals
- Freeze the public API boundary
- De-export internal helpers from top level
- Document the full public contract
- Update curated examples to use only public API
- Add regression tests for API stability

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Targeted API smoke tests: ✅
- Example scripts smoke test: ✅
- Docs build smoke test: ✅

## Performance
None (API/docs only, no hot paths touched)

## Acceptance Criteria
✅ Export list is explicit and documented  
✅ Curated public examples use only supported top-level API  
✅ All 557 tests pass  
✅ API smoke tests pass  

## Breaking Changes
**YES** — Low-level helpers are no longer exported at top level:
- `zero_forces!` — now `ParticleDynamics.Simulation.zero_forces!`
- `accumulate_energies!` — now `ParticleDynamics.Simulation.accumulate_energies!`
- `run_integrator_step!` — now `ParticleDynamics.Simulation.run_integrator_step!`
- `thermostatted_dof` — now `ParticleDynamics.Simulation.thermostatted_dof`
- `thermostatted_particle_mask` — now `ParticleDynamics.Simulation.thermostatted_particle_mask`

**New top-level exports**:
- `eulerheun` — Now available at top level
- `read_gsd_frame!` — Now available at top level

## Migration Notes
If your scripts use any of the de-exported helpers, update them:
```julia
# Old (no longer works):
# zero_forces!(sim)

# New (required):
ParticleDynamics.Simulation.zero_forces!(sim)
```

For most users, this change is transparent because they were using documented public APIs. Only advanced users working with low-level internals are affected.

## Rollback
```bash
git revert b241b42
```

## Notes
This PR is reviewable because it stays at the module/documentation boundary. It does not touch hot GPU paths or integrator math. The de-exports improve clarity about what is officially supported vs. what is internal implementation detail. The change is small enough to be merged incrementally and documented clearly.
