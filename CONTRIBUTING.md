# Contributing

Thanks for contributing to `ParticleDynamics.jl`.

## Before opening a PR

1. Open an issue for bugs, regressions, API changes, or design changes.
2. Describe GPU environment details for runtime issues:
   - Julia version
   - `CUDA.versioninfo()` output
   - GPU model and driver/runtime versions
3. Keep changes focused and minimal.

## Development setup

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

GPU is required for full simulation/test coverage.

## Test requirements

Run tests on a CUDA-functional machine:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

For release-style local validation:

```bash
bash scripts/ci_gpu_local.sh
```

## Pull request guidelines

- Do not change runtime behavior unless the issue and fix are clearly documented.
- Do not add CPU fallback implementations for simulation kernels.
- Keep API changes explicit and documented in `NEWS.md`.
- Add/adjust tests for behavior changes.
- Update docs when user-visible behavior changes.

## Coding/style notes

- Prefer small, reviewable commits.
- Keep GPU arrays on device in performance-sensitive code paths.
- Avoid formatting-only churn in unrelated files.

## Documentation

Build docs locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
