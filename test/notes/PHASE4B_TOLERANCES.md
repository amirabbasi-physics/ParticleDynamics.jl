# Phase 4B Tolerances and Sampling

Date: 2026-02-13

## 1) Example parsing inputs
- Parser file: `test/params_from_examples.jl`
- Scanned files: `37` Julia scripts under `examples/` (recursive).
- Read failures: none.
- Parse-fallback files (no simple scalar assignment matched by conservative regex):  
  `examples/2D_allpairs_quicktest.jl`, `examples/2D_example_forces.jl`, `examples/2D_polymer_bonded.jl`, `examples/2D_soft_repulsive.jl`, `examples/3D_quicktest.jl`, `examples/restart_jobfile/alpha_50_phi_0.4_in.jl`.

## 2) Extracted recommendations (raw medians/modes)
- Brownian group: `dt=2.0e-7`, `gamma=615.985`, `temperature=1.0`, `tau=100.0`, `boxL=125.0`, integrator mode `:eulerheun`.
- Langevin group: `dt=2.0e-6`, `gamma=615.985`, `temperature=1000.0`, `tau=0.5`, `boxL=125.0`, integrator mode `:velocityverlet`.
- OU group: `dt=1.0e-6`, `gamma=10.0`, `temperature=1.0`, `tau=100.0`, `boxL=200.0`, integrator mode `:eulerheun`.

## 3) Test parameter stabilization rules
Raw extracted values are often production-scale (very small `dt`, very large `T`/`tau`) and would produce slow/flaky CI tests. Phase 4B clamps parameters to stay in physically consistent but test-stable ranges:
- Brownian MSD test clamps to: `dt∈[2e-4, 5e-4]`, `gamma∈[5, 20]`, `T∈[1, 5]`.
- Langevin equipartition clamps to: `dt∈[5e-6, 2e-5]`, `gamma∈[200, 800]`, `T∈[10, 100]`.
- OU test clamps to: `dt∈[1e-3, 2e-3]`, `gamma∈[5, 20]`, `T∈[1, 5]`, `tau∈[0.2, 0.5]`.

These ranges are chosen to keep dynamics in the same regime as examples while reducing estimator variance and total runtime.

## 4) Sampling sizes used
- 4B-1 Brownian MSD slope:
  - `N=256`, `steps=2400`, sample every `40` steps.
  - Fit window: middle `20%..80%` of sampled times.
- 4B-2 Langevin equipartition (VV + BAOAB):
  - `N=256`, burn-in `1500` steps, sampling run `2500` steps, sample every `10` steps.
- 4B-3 OU autocorrelation:
  - `N=512`, burn-in `400` steps, `300` lag samples.
  - Fit window for OU state ACF: lags `20..220`.
- 4B-4 weak-convergence trend (deterministic limit):
  - `N=256`, `dt={4e-3, 2e-3, 1e-3}`, fixed physical time `0.5`.

## 5) Assertion tolerances and rationale
- Brownian MSD slope relative error: `<= 15%`, with additional linearity check `R^2 >= 0.985`.
  - Chosen from calibration runs showing ~5–9% slope error with these sample sizes.
- Equipartition relative error (`⟨vx^2+vy^2⟩` vs `2T/m`): `<= 10%` for both VV and BAOAB.
  - Calibration runs were typically around 1–2%; 10% leaves margin across GPU hardware.
- OU decay:
  - Velocity ACF: finite, bounded in `[-1.2, 1.2]`, and net decay from first to last lag.
  - OU-state fitted `tau` relative error: `<= 30%`.
  - This balances robustness with the finite-lag linear-fit sensitivity.
- Weak convergence trend:
  - Monotone error reduction against analytic deterministic limit:  
    `|err(dt/2)| < |err(dt)|` and `|err(dt/4)| < |err(dt/2)|`.
  - Deterministic (`T=0`) setup is intentional to remove Monte Carlo noise and make monotonicity non-flaky.
