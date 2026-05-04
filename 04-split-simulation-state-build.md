# PR 4: Extract state and build responsibilities out of Simulation.jl

**Branch**: `refactor/03-split-simulation-state-build`  
**Commit**: `06096e09bec66f49c2218ad6edebc4bd2a59fb1f`  
**Date**: 2026-05-04 15:29:19  
**Status**: ✅ Committed

## Summary
Phase 3 architectural work. Takes the first safe slice out of `src/Simulation.jl` by extracting state management and simulation construction concerns into separate modules. Creates `SimulationState.jl` for state layout and validation, and `SimulationBuild.jl` for construction/builder logic. Leaves the step engine in place in `Simulation.jl` to keep the refactor incremental and reviewable.

## Changes
**Created**:
- `src/SimulationBuild.jl` — Simulation construction and builder logic
- `src/SimulationState.jl` — Simulation state layout and validation

**Modified**:
- `src/Simulation.jl` — Removed state/build code, kept step engine
- `test/test_build.jl` — Added/updated build and state wiring regression checks

## Goals
- Split concerns: state layout → `SimulationState.jl`, construction → `SimulationBuild.jl`, stepping → `Simulation.jl`
- Keep changes incremental and safe (no stepping/kernel changes)
- Establish clean architectural boundaries
- Prepare for backend abstraction in next phase
- Catch wiring drift with regression tests

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Targeted build tests: ✅
- Targeted state tests: ✅
- Example scripts smoke test: ✅
- Virial smoke tests: ✅

## Performance
Reduced-step `benchmarks/Argon_3D_NVT_NHC_basic.jl` verification — no regression expected (refactoring only, no algorithm changes)

## Acceptance Criteria
✅ `SimulationState` and build helpers live outside `Simulation.jl`  
✅ Shared step engine remains behaviorally equivalent on deterministic cases  
✅ All 557 tests pass  
✅ Build wiring is verified by regression tests  

## Breaking Changes
None — internal refactoring only. Public API remains unchanged.

## Migration Notes
None — this is purely internal restructuring. Users of `Simulation` see no changes.

## Rollback
```bash
git revert 06096e0
git rm src/SimulationBuild.jl src/SimulationState.jl
```

## Notes
This is a structural refactoring that improves code organization without changing behavior. The risk is wiring drift (incorrect inclusion/export of new modules), which is mitigated by regression tests. It touches `Simulation.jl` (a hot-path-owning file) but not the kernels themselves, keeping it safe and reviewable.
