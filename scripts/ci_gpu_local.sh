#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[ci_gpu_local] repo: $ROOT_DIR"
echo "[ci_gpu_local] checking CUDA availability"
julia --project -e 'using CUDA; @assert CUDA.functional() "CUDA.functional() must be true for GPU local CI"; println("CUDA.functional() = ", CUDA.functional())'

echo "[ci_gpu_local] instantiate package"
julia --project -e 'using Pkg; Pkg.instantiate()'

echo "[ci_gpu_local] run test suite"
julia --project -e 'using Pkg; Pkg.test()'

echo "[ci_gpu_local] build docs"
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'

echo "[ci_gpu_local] run examples smoke"
NEQSIM_SMOKE_FAST=1 julia --project scripts/examples_smoke.jl

echo "[ci_gpu_local] success"
