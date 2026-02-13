# Release Checklist

## Validation

- [ ] `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` passes on CUDA machine.
- [ ] `julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'` succeeds.
- [ ] `julia --project scripts/examples_smoke.jl` smoke subset passes.

## Versioning and notes

- [ ] `Project.toml` version bumped.
- [ ] `NEWS.md` updated (highlights, breaking changes, limitations).
- [ ] `CITATION.cff` version/date updated.

## Repository metadata

- [ ] License confirmed (MIT in `LICENSE`).
- [ ] `README.md` updated with GPU-only requirements and docs/test commands.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` present.

## CI/workflows

- [ ] CPU workflow validates docs/static checks and handles no-GPU test skip.
- [ ] Self-hosted GPU workflow runs full test suite.
- [ ] Local GPU CI script (`scripts/ci_gpu_local.sh`) works.

## Publish

- [ ] Merge release PR to `main`.
- [ ] Create annotated tag `vX.Y.Z`.
- [ ] Publish GitHub release notes.
- [ ] (Optional) Zenodo DOI verified.
