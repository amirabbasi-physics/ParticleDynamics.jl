# PR 6: Expand force correctness and GPU-residency regression coverage

**Branch**: `test/05-force-gpu-harness`  
**Commit**: `e25c9db209cf070f1e5a6461cfdda20b9e08a26d`  
**Date**: 2026-05-04 15:54:44  
**Status**: ✅ Committed

## Summary
Phase 0 validation-first suite expansion. Adds explicit force correctness and GPU-residency regression checks to protect all the refactoring work done in previous PRs. Creates `test/test_gpu_residency.jl` for CUDA scalar indexing and workspace smoke tests, and expands `test/test_phase4a_forces.jl` with analytic pair-energy verification. These tests ensure refactored code paths maintain GPU efficiency and numerical correctness.

## Changes
**Created**:
- `test/test_gpu_residency.jl` — New GPU residency and CUDA scalar indexing regression suite

**Modified**:
- `test/test_phase4a_forces.jl` — Added analytic pair-energy checks and pairwise regression coverage
- `test/runtests.jl` — Includes new test suites

## Goals
- Protect refactored force and storage paths with explicit regression coverage
- Verify GPU residency and prevent accidental scalar indexing
- Add analytic verification for force calculations
- Establish harness for future force/storage regression checks
- Keep tests focused (no redesign of `NonBondedForces.jl`, only hooks)

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Targeted `test_phase4a_forces.jl` tests: ✅
- Targeted `test_gpu_residency.jl` tests: ✅
- Pairwise/regression coverage for touched force paths: ✅
- `CUDA.allowscalar(false)` enforcement: ✅ (fails loudly on scalar indexing)

## Performance
None beyond `CUDA.allowscalar(false)` enforcement in test suite (ensures no accidental scalar indexing performance regression)

## Acceptance Criteria
✅ Refactor-relevant force/storage paths have explicit regression coverage  
✅ CUDA scalar indexing still fails loudly (caught by tests)  
✅ Analytic pair-energy verification passes  
✅ All 557 tests pass  

## Breaking Changes
None

## Migration Notes
None — this is test-only. No user-facing API or behavior changes.

## Rollback
```bash
git revert e25c9db
git rm test/test_gpu_residency.jl
```

## Notes
This is a test-only PR that adds observability hooks to verify correctness of refactored paths. It does not redesign `NonBondedForces.jl` or change hot kernel code. The new tests use analytic verification and GPU residency checks to ensure that the refactoring PRs did not introduce silent correctness or performance regressions. The `CUDA.allowscalar(false)` enforcement is particularly valuable for catching accidental scalar indexing that would otherwise hide as silent slowdowns.

## Integration Notes
This is the final PR in the 6-PR edit phase. All previous PRs (1-5) are prerequisites:
1. PR1: Stabilize baseline
2. PR2: Release hygiene
3. PR3: Public API boundary
4. PR4: Split simulation state/build
5. PR5: Backend boundary
6. PR6: Force/residency harness ← You are here

All 557 tests pass with all 6 PRs applied.
