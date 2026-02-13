# Release Guide

This repository publishes a **GPU-only** Julia package (`CUDA.jl` required).

## 1) Pre-release checks

Run the local GPU CI bundle:

```bash
bash scripts/ci_gpu_local.sh
```

This runs:

1. `Pkg.instantiate()`
2. `Pkg.test()` (GPU suite)
3. docs build (`docs/make.jl`)
4. examples smoke runner

## 2) Versioning

- Update `Project.toml` version.
- Add release notes in `NEWS.md`.
- Update `CITATION.cff` version and date.

## 3) Commit and tag

```bash
git add Project.toml NEWS.md README.md CITATION.cff
git add .github/workflows scripts/ci_gpu_local.sh scripts/examples_smoke.jl
git add RELEASE.md RELEASE_CHECKLIST.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "NonEqSimGPU vX.Y.Z"
git push origin main --tags
```

## 4) GitHub release

1. Open `Releases` on GitHub.
2. Draft new release from tag `vX.Y.Z`.
3. Copy top section from `NEWS.md`.
4. Mark breaking changes and GPU-only requirements clearly.
5. Publish release.

## 5) Optional DOI / Zenodo

- Ensure Zenodo GitHub integration is enabled.
- Create release tag first; Zenodo archives the tagged state.
- Add DOI badge to README once minted.
