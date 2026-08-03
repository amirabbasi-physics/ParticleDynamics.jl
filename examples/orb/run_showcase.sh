#!/usr/bin/env bash
# Full benzene-crystal head-to-head, in order. Each MD/throughput model runs in
# its own process: loading an Orb checkpoint calls torch.set_default_dtype
# process-wide, and the MACE work here runs float64.
#
# Usage: bash examples/orb/run_showcase.sh
# Expects PARTICLEDYNAMICS_PYTHON to point at a python with mace-torch and
# orb-models installed (see requirements.txt).

set -u
cd "$(dirname "$0")/../.." || exit 1
: "${PARTICLEDYNAMICS_PYTHON:?set PARTICLEDYNAMICS_PYTHON to the venv python}"
JL="julia --project=examples/orb"
VAL=examples/orb/validation

echo "=== 0. structures ==="
"$PARTICLEDYNAMICS_PYTHON" examples/orb/py_benzene_init.py || exit 1

echo "=== 1. bridge fidelity (float64, criterion 1e-10) ==="
$JL examples/orb/orb_bridge_smoke.jl || exit 1

echo "=== 2. lattice energy / E(V) scan ==="
$JL examples/orb/benzene_lattice_energy.jl || exit 1

echo "=== 3. finite-T MD + VDOS + NVE conservation ==="
rm -f "$VAL/benzene_md_summary.txt"
for m in mace-off orb-cons orb-omol orb-direct; do
    echo "--- $m ---"
    $JL examples/orb/benzene_md_vdos.jl "$m" || exit 1
done

echo "=== 4. throughput (matched precision + float64 penalty) ==="
rm -f "$VAL/benzene_throughput.txt"
$JL examples/orb/benzene_throughput.jl mace-off  float32       || exit 1
$JL examples/orb/benzene_throughput.jl mace-off  float64       || exit 1
$JL examples/orb/benzene_throughput.jl orb-cons  float32-high  || exit 1
$JL examples/orb/benzene_throughput.jl orb-direct float32-high || exit 1
$JL examples/orb/benzene_throughput.jl orb-omol  float32-high  || exit 1
# Orb in float64 costs ~30x on a consumer GPU; enable if you want the row
# measured at 384 atoms rather than quoted from a single-point timing.
# $JL examples/orb/benzene_throughput.jl orb-cons float64 || exit 1

echo "=== 5. figures ==="
"$PARTICLEDYNAMICS_PYTHON" examples/orb/py_benzene_plots.py || exit 1

echo "=== 6. head-to-head movie ==="
# Heating-ramp trajectories (GPU), then Fresnel rendering (CPU). The renderer
# needs fresnel + gsd, which are not in the MLIP venv; create the environment
# from examples/kg_fresnel_environment.yml and point PD_FRESNEL_PYTHON at it:
#   micromamba create -y -p ~/.venvs/pd-fresnel -f examples/kg_fresnel_environment.yml
FRESNEL_PY="${PD_FRESNEL_PYTHON:-$HOME/.venvs/pd-fresnel/bin/python}"
for m in mace-off orb-cons; do
    echo "--- melt ramp $m ---"
    $JL examples/orb/benzene_melt_movie_gsd.jl "$m" || exit 1
done
if [ -x "$FRESNEL_PY" ]; then
    "$FRESNEL_PY" examples/orb/py_benzene_movie.py --preview || exit 1
    "$FRESNEL_PY" examples/orb/py_benzene_movie.py --stride 2 || exit 1
else
    echo "skipping render: no fresnel python at $FRESNEL_PY"
fi

echo
echo "=== RESULTS ==="
cat "$VAL/benzene_lattice_energy.txt"
echo
cat "$VAL/benzene_md_summary.txt"
echo
cat "$VAL/benzene_throughput.txt"
