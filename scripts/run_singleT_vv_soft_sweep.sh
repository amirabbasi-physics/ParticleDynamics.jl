#!/usr/bin/env bash
set -euo pipefail

# Sweep parameters
phis=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
dts=(2e-6)
epsilons=(1.0e7 1.0e8 5.0e8 1e9)

# Resolve repo root (directory of this script, up one level if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "${REPO_ROOT}"

JOB="examples/single_T_collision_calc/SingleT_2D_LD_VV_soft.jl"

if [[ ! -f "$JOB" ]]; then
  echo "Job file not found: $JOB" >&2
  exit 1
fi

echo "Running SingleT VV soft-repulsive sweep (sequential)"
for phi in "${phis[@]}"; do
  for dt in "${dts[@]}"; do
    for eps in "${epsilons[@]}"; do
      echo "--- Starting run: PHI=$phi, DT=$dt, EPS=$eps ---"
      PHI="$phi" DT="$dt" EPS="$eps" julia --project=. "$JOB"
      echo "--- Finished run: PHI=$phi, DT=$dt, EPS=$eps ---"
    done
  done
done

echo "All runs completed."
