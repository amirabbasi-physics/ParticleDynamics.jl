# NEWS

## v0.4.0 (2026-02-13)

### Highlights

- GPU-first verification and release hardening for `NonEqSimGPU.jl`.
- Expanded deterministic and stochastic validation coverage on CUDA paths.
- Documenter scaffold and manual quickstart added.
- Release metadata/community files and GPU-aware CI workflows added.

### Breaking changes

- Removed stale/unbound exports from the public API surface:
  - `read_last_gsd`
  - `step_fused!`
  - `ObservableCSVWriter`
- `eulerheun` remains intentionally unexported at top level.
  Use exported paths (for public API) such as `eulermaruyama`.

### Behavioral safety updates

- Stochastic integrator paths now enforce `gamma > 0` with informative errors:
  - BAOAB path
  - Brownian midpoint path
  - Euler-Maruyama path

### Known limitations

- Package is GPU-only (`CUDA.functional() == true` required for simulation).
- Bitwise-identical trajectories across devices/toolchains are not guaranteed.
- CPU-only GitHub-hosted runners can run docs/static checks and skip GPU tests.
