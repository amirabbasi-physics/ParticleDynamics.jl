# NonEqSimGPU.jl

`NonEqSimGPU.jl` is a GPU-only Julia package for non-equilibrium particle simulations
using `CUDA.jl`. It provides Langevin and Brownian dynamics integrators,
neighbor-list backends, nonbonded/bonded force models, collision counting, and
XYZ/CSV/GSD writers.

The package is designed for production GPU workflows and verification-oriented
testing. Core simulation behavior is validated with deterministic force checks,
backend parity checks, and stochastic moment-based physics tests.

## GPU requirement

This package requires a functional CUDA environment.

```julia
using CUDA
CUDA.functional() || error("NonEqSimGPU requires CUDA.functional() == true")
```

No CPU fallback simulation path is provided.

## Installation

### Add from GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/ORG_OR_USER/NonEqSimGPU.jl")
```

### Develop locally

```julia
using Pkg
Pkg.develop(path="/path/to/NonEqSimGPU")
Pkg.instantiate()
```

## Minimal quickstart (GPU)

```julia
using NonEqSimGPU
using CUDA

N = 64
dt = 2.0f-4
sigma = 1.0f0
rcut = Float32(2^(1/6)) * sigma

st = build_simulation(
    N=N,
    box=(40.0f0, 40.0f0),
    cutoff=rcut,
    skin=0.4f0,
    cap=Int32(32),
    neigh_interval=10,
    epsilon=10.0f0,
    sigma=sigma,
    gamma=50.0f0,
    temperature=1.0f0,
    nonbonded=:wca,
    precision=:f32,
)

vv = velocityverlet(st; gamma=50.0f0, temperature=1.0f0, dt=dt)

for _ in 1:200
    step!(st, vv, dt; compute_energy=false)
end

@show st.step
```

## Running tests (GPU machine)

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Documentation

Build docs locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

Docs site placeholder: `https://ORG_OR_USER.github.io/NonEqSimGPU.jl/`

## Example smoke run

```bash
julia --project scripts/examples_smoke.jl
```

Optional heavier smoke case:

```bash
NEQSIM_SMOKE_HEAVY=1 julia --project scripts/examples_smoke.jl
```

For full local GPU CI bundle:

```bash
bash scripts/ci_gpu_local.sh
```

## Known limitations

- GPU-only package; simulation requires CUDA.
- Bitwise-identical trajectories are not guaranteed across GPUs/toolchains.
- Statistical reproducibility should be evaluated via moments/summary statistics.
- `gamma > 0` is required for stochastic integrator paths that divide by friction
  (BAOAB, Brownian midpoint, Euler-Maruyama).
- Supported simulation dimensions are 2D and 3D.

## Citation

Please cite this software using metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
