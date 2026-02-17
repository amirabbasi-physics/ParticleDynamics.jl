# Phase 2 Ambiguity Notes

Date: 2026-02-13

## 1) Export visibility policy (`eulerheun`)
- Current behavior: `NonEqSimGPU.eulerheun` is bound but not exported from the top-level module, while the sibling `eulermaruyama` is exported.
- Test policy in Phase 2: treat this as an API inconsistency and fail until made explicit.
- Recommended resolution in Phase 3: either export `eulerheun` at top-level or document it as intentionally internal and remove user-facing expectations.

## 2) Zero-friction (`gamma = 0`) behavior in stochastic integrators
- A physically and numerically safe policy should be explicit.
- Phase 2 tests accept either:
  - an informative error (message mentions gamma positivity), or
  - finite outputs from a mathematically valid `gamma -> 0` limit implementation.
- Current code path produces non-finite outputs for BAOAB, Brownian midpoint, and EM in minimal GPU runs.

## 3) EM collision rebuild omission test design
- The EM path currently rebuilds neighbor lists but omits collision-state reinitialization.
- To make failure deterministic and fast:
  - force a rebuild on the next EM step,
  - prefill `coll_prev` with a sentinel byte (`0x07`),
  - assert sentinel removal after the step.
- If reinitialization is called, sentinel values are fully overwritten. Without it, stale sentinel values remain.
