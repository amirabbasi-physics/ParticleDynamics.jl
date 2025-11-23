# NonEqSimGPU

[![Build Status](https://travis-ci.com/abbaa90/NonEqSimGPU.jl.svg?branch=master)](https://travis-ci.com/abbaa90/NonEqSimGPU.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/abbaa90/NonEqSimGPU.jl?svg=true)](https://ci.appveyor.com/project/abbaa90/NonEqSimGPU-jl)
[![Coverage](https://codecov.io/gh/abbaa90/NonEqSimGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/abbaa90/NonEqSimGPU.jl)
[![Coverage](https://coveralls.io/repos/github/abbaa90/NonEqSimGPU.jl/badge.svg?branch=master)](https://coveralls.io/github/abbaa90/NonEqSimGPU.jl?branch=master)

## Documentation

- Collision event rate (GPU) design and usage: `docs/CollisionRate.md`

## Legacy CUDA GPUs

Pascal-generation GPUs (e.g. GTX 1080 Ti, cc 6.1) are blocked by CUDA 13+/driver 580+. To keep them working without downgrading the driver, the package will try to pin CUDA to a legacy runtime (default `12.4.1`) when it detects cc < 7.5 and a 13.x runtime. You can control this via:

- `NEQSIMGPU_CUDA_COMPAT=off` to skip the downgrade logic.
- `NEQSIMGPU_CUDA_COMPAT=force` to force pinning even if detection does not trigger.
- `NEQSIMGPU_CUDA_LEGACY_VERSION=12.3.0` (or similar) to choose a specific runtime version.

If you still see capability errors on a legacy GPU, set `NEQSIMGPU_CUDA_COMPAT=force` and retry so the runtime is pinned before compilation.
