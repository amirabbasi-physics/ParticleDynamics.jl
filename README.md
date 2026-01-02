# NonEqSimGPU

[![Build Status](https://travis-ci.com/abbaa90/NonEqSimGPU.jl.svg?branch=master)](https://travis-ci.com/abbaa90/NonEqSimGPU.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/abbaa90/NonEqSimGPU.jl?svg=true)](https://ci.appveyor.com/project/abbaa90/NonEqSimGPU-jl)
[![Coverage](https://codecov.io/gh/abbaa90/NonEqSimGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/abbaa90/NonEqSimGPU.jl)
[![Coverage](https://coveralls.io/repos/github/abbaa90/NonEqSimGPU.jl/badge.svg?branch=master)](https://coveralls.io/github/abbaa90/NonEqSimGPU.jl?branch=master)

## Overview

NonEqSimGPU is a CUDA-accelerated toolkit for non-equilibrium particle
simulations. It provides:

- Structure-of-arrays state storage (`SimulationState`);
- GPU neighbor lists (dense and stencil);
- Nonbonded pair kernels (LJ, WCA, soft repulsive) and bonded chains
  (harmonic/FENE);
- Langevin (VV, BAOAB, BAOA, GSM) and Brownian (midpoint, Euler–Maruyama)
  integrators;
- Collision counting, filters for per-type parameters, and writers for XYZ/CSV/GSD.

Every API is exercised in the `examples/` folder; docstrings reference the
scripts they were derived from so you can copy parameter sets verbatim.

## Installation

The package is currently developed from source. In the Julia package manager:

```julia
pkg> dev https://github.com/abbaa90/NonEqSimGPU.jl
pkg> activate NonEqSimGPU
pkg> instantiate
```

Make sure a recent CUDA toolkit is available. Legacy Pascal GPUs can pin the
runtime to CUDA 12.4.1 by setting `NEQSIMGPU_CUDA_COMPAT=force`.

## Quick Start

The snippet below mirrors `examples/2D_allpairs_quicktest.jl` (reduced to 256
particles) and demonstrates a short WCA run:

```julia
using NonEqSimGPU

N = 256
box = (80.0f0, 80.0f0)
dt = 2.0f-4
rcut = Float32(2^(1/6))

st = build_simulation(N=N, box=box, dt=dt,
                      cutoff=rcut, skin=0.4f0, cap=Int32(64),
                      neigh_interval=10, use_neighborlist=false,
                      epsilon=10f0, sigma=1f0,
                      gamma=50f0, temperature=1f0,
                      nonbonded=:wca, precision=:f32)

for _ in 1:50
    step!(st, dt; compute_energy=true)
end

write_observables_csv!("obs.csv", st.step; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
```

See `examples/TwoT_2D_LD_VV.jl`, `examples/3D_filters.jl`, and the Brownian
demos for richer setups (filters, collision counting, GSD output).

## Documentation

- Collision event rate (GPU) design and usage: `docs/CollisionRate.md`

## Testing and Examples

Run the unit tests from the package root:

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

The examples are ready-to-run scripts; use `julia --project examples/2D_allpairs_quicktest.jl`.

## Legacy CUDA GPUs

Pascal-generation GPUs (e.g. GTX 1080 Ti, cc 6.1) are blocked by CUDA 13+/driver 580+. To keep them working without downgrading the driver, the package will try to pin CUDA to a legacy runtime (default `12.4.1`) when it detects cc < 7.5 and a 13.x runtime. You can control this via:

- `NEQSIMGPU_CUDA_COMPAT=off` to skip the downgrade logic.
- `NEQSIMGPU_CUDA_COMPAT=force` to force pinning even if detection does not trigger.
- `NEQSIMGPU_CUDA_LEGACY_VERSION=12.3.0` (or similar) to choose a specific runtime version.

If you still see capability errors on a legacy GPU, set `NEQSIMGPU_CUDA_COMPAT=force` and retry so the runtime is pinned before compilation.
