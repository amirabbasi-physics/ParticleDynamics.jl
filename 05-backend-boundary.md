# PR 5: Introduce backend traits and storage adaptation at the build boundary

**Branch**: `refactor/04-backend-boundary`  
**Commit**: `aa7c0bc5824f2f1702a43712dfb8dcfdfd6359e0`  
**Date**: 2026-05-04 15:46:04  
**Status**: ✅ Committed

## Summary
Phase 4 backend architecture work. Introduces a thin internal backend layer without abstracting kernels. Creates `src/Backends.jl` with backend marker types and adds storage-adaptation/builder logic to handle CUDA-specific initialization. The default CUDA path remains unchanged; this work is purely internal isolation at the build boundary.

## Changes
**Created**:
- `src/Backends.jl` — Backend trait types and selection logic

**Modified**:
- `src/ParticleDynamics.jl` — Updated exports and module includes
- `src/Simulation.jl` — References backend traits where needed
- `src/SimulationBuild.jl` — Backend-aware storage adaptation logic
- `src/Filters.jl` — Minor updates for backend compatibility
- `test/test_build.jl` — Backend selection construction tests
- `CHANGELOG.md` — Documented backend addition

## Goals
- Introduce backend marker types without full abstraction
- Enable storage adaptation at the build boundary
- Keep hot kernels unchanged
- Prepare for future CPU backend or CUDA variant support
- Maintain backward compatibility by making backend selection optional

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Targeted backend selection tests: ✅
- Targeted `test_build.jl` (backend construction): ✅
- Targeted `test_api.jl` (backend in public API): ✅
- Example scripts smoke test: ✅
- `CUDA.allowscalar(false)` enforcement smoke test: ✅

## Performance
Reduced-step `benchmarks/Argon_3D_NVT_NHC_basic.jl` verification (N=512, warmup=100, prod=200, write_gsd=false):
- Elapsed time: 0:32.46
- No regression from baseline (refactoring only)

## Acceptance Criteria
✅ Default CUDA path is unchanged  
✅ Builders choose backend concretely  
✅ No hot-kernel rewrites  
✅ Performance stays within agreed threshold  
✅ All 557 tests pass  

## Breaking Changes
**Additive only** — no breaking changes to existing code.

**New feature**:
- `build_simulation(...; backend=:cuda)` — Optional keyword to specify backend (defaults to `:cuda`)
- `build_simulation(...; backend=:cpu)` — Fails early with informative error (CPU backend not yet implemented)

## Migration Notes
None required. Existing code continues to work:
```julia
# Old style (still works):
sim = build_simulation(...)

# New style (optional):
sim = build_simulation(...; backend=:cuda)
```

## Rollback
```bash
git revert aa7c0bc
git rm src/Backends.jl
```

## Notes
This PR introduces infrastructure for backend selection but does not implement multiple backends or change kernel behavior. The backend trait layer is purely internal and sits at the build boundary. The default CUDA path is identical to pre-PR behavior. This keeps the change safe and reviewable while enabling future backend support.
