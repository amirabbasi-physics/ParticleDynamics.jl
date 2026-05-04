# PR 2: Finish ParticleDynamics release-facing rename and repo hygiene

**Branch**: `repo/01-release-hygiene`  
**Commit**: `aca580164c37a73144105223dd69b67d821df24c`  
**Date**: 2026-05-04 15:02:16  
**Status**: ✅ Committed

## Summary
Phase 1 repository cleanup and release preparation work. Finishes the package rename residue from `NonEqSimGPU` to `ParticleDynamics`. This includes aligning release metadata, CI/workflows, and documentation to reflect the new package identity. Removes obsolete CI configurations (.travis.yml, .appveyor.yml) and creates a comprehensive CHANGELOG.

## Changes
**Deleted**:
- `.appveyor.yml`
- `.travis.yml`

**Modified**:
- `.github/workflows/ci-cpu.yml` — Updated for new branch and package naming
- `.github/workflows/ci-gpu-selfhosted.yml` — Updated for new branch and package naming
- `README.md` — Removed legacy naming references
- `CONTRIBUTING.md` — Updated for new repo state
- `CHANGELOG.md` — Created new comprehensive changelog

## Goals
- Complete the repo-facing rename cleanup
- Remove stale CI configurations
- Establish consistent release metadata
- Ensure no tracked release-facing file points to `NonEqSimGPU`
- Align CI targets with chosen base branch

## Testing
- Full `Pkg.test()`: 557/557 tests pass
- Import smoke test: ✅
- Docs build smoke test: ✅
- Example scripts smoke test: ✅

## Performance
None (no hot paths touched)

## Acceptance Criteria
✅ No tracked release-facing file contains `NonEqSimGPU`  
✅ CI targets the correct base branch  
✅ CHANGELOG.md exists and is comprehensive  
✅ All 557 tests pass  

## Breaking Changes
None

## Migration Notes
None — this is purely external/metadata cleanup.

## Rollback
```bash
git revert aca5801
```

## Notes
This PR is small and safe because it does not touch hot code paths or user-facing API. It is purely release hygiene and repository cleanup. All changes are at the documentation, workflow, and configuration level.
