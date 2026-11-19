# `cuPotential`-Class Accuracy, Drift & Speedup Benchmark Suite

Reproducible RTX 3090 benchmark and validation harness comparing:
1. **`ParticleDynamics` `v0.7.0` Baseline** (`PME` and C⁰ `ESP` + truncated `LennardJones` at `r_c = 1.0 nm` and `0.7 nm`).
2. **`ParticleDynamics` Native `cuPotential`-Class Solvers** (`ESP(continuity=:c2)` + `DispersionESP` at `r_c = 0.70 nm` / `0.65 nm`, fused dual-mesh `(q, √c6)`, and 64-bit fixed-point force accumulation).
3. **(Optional) OpenMM 8.4 + `nv-openmm-plugin` (`cuPotential`)** on the identical 30,492-atom SPC/E rigid water box.

## Running on Your RTX 3090 Workstations

1. **Capture frozen `v0.7.0` baseline** (works on both `master` and `feat/cupotential-native`):
   ```bash
   julia --project --startup-file=no benchmarks/cupotential_comparison/run_baseline.jl
   ```
   *(Set `PD_BENCH_QUICK=1` for a ~2-minute smoke run.)*

2. **Run full comparison** (C² splits/windows, C² lattice dispersion, fused dual-mesh, and `fixed64` accumulation vs. `v0.7.0`):
   ```bash
   julia --project --startup-file=no benchmarks/cupotential_comparison/run_comparison.jl
   ```

3. **(Optional) Run OpenMM / `cuPotential` reference**:
   ```bash
   python3 benchmarks/cupotential_comparison/run_openmm_cupotential.py
   ```

Outputs are written to `benchmarks/cupotential_comparison/results/` as JSON and CSV tables.
