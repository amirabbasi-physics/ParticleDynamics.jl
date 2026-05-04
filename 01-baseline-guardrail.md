# PR 1: Characterize and stabilize the current GPU baseline

**Branch**: `test/00-baseline-guardrail`  
**Commit**: `849aebd01e5c502020dd955302f67598316ca5b4`  
**Date**: 2026-05-04 14:47:05  
**Status**: ✅ Committed

## Summary
Phase 0 safety and characterization work to establish a trustworthy performance baseline before proceeding with refactors. The current `Pkg.test()` contained a fragile slowdown guardrail in `test_bonded_exclusions.jl` that was noisy and unreliable. This PR stabilizes that guardrail with a paired-median timing helper.

## Changes
- **Modified**: `test/test_bonded_exclusions.jl` (32 insertions, 11 deletions)

## Goals
- Make the baseline trustworthy: either the guardrail passes reliably or it is replaced with a stable regression check that still catches real slowdowns
- Provide stable characterization helper for the soft-repulsive bonded path
- Establish clean slate for subsequent refactors

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Paired-median bonded-exclusion timing guardrail is now stable

## Performance
- Paired-median bonded-exclusion timing guardrail verification (baseline characterization)
- No kernel changes, no performance regression expected

## Acceptance Criteria
✅ Baseline is trustworthy  
✅ Guardrail passes consistently  
✅ All 557 tests pass  

## Breaking Changes
None

## Migration Notes
None

## Rollback
```bash
git revert 849aebd
```

## Notes
This is a test/benchmark-centered PR that does not touch hot GPU code or user-facing API. The main risk was discovering a real hotspot in bonded exclusions, but stabilization of the guardrail was achieved without requiring kernel changes.
