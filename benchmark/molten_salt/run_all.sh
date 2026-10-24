#!/usr/bin/env bash
# Molten NaCl benchmark matrix: GROMACS and ParticleDynamics, NVE/NVT/NPT, one run at a time,
# GPU utilization sampled (200 ms) around each run. PD list settings from PD_SKIN/PD_PRUNE.
# Usage: run_all.sh STATES OUT "22 46" "nve nvt npt" [ENGINES="gromacs pd"]
set -uo pipefail
failed=0
source ~/.bashrc >/dev/null 2>&1 || true
STATES=$(realpath $1); OUT=$2; SIZES=$3; ENS=$4; ENGINES=${ENGINES:-"gromacs pd_ell pd_esp"}
HERE=$(dirname $(realpath $0)); PD=$(dirname $(dirname $HERE))
julia_bin=${JULIA:-julia}
STEPS=${STEPS:-10000}; PD_SKIN=${PD_SKIN:-0.2}; PD_PRUNE=${PD_PRUNE:-0.1}
mkdir -- "$OUT" || exit 2; OUT=$(realpath "$OUT")
sampled() {
    local dir=$1; shift
    nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 200 > $dir/gpu_util.csv &
    local smi=$!; local t0=$(date +%s.%N)
    "$@"; local rc=$?
    local t1=$(date +%s.%N); kill $smi 2>/dev/null; wait $smi 2>/dev/null
    python3 -c "print($t1-$t0)" > $dir/wall_s.txt; echo $rc > $dir/exit.txt; return $rc
}
# ESP settings meeting GROMACS PME's Coulomb force error at each size (Δ 1.444e-3 at 10k with its
# 64³ grid, 1.035e-3 at 97k with its 144³ grid): ESP order 4, Δ 1.281e-3 (36³) and 1.020e-3 (80³)
esp_settings() { case $1 in 22) echo "tolerance=1e-3 esp_order=4 esp_mesh=36,36,36" ;;
                            46) echo "tolerance=6e-4 esp_order=4 esp_mesh=80,80,80" ;; esac; }
# NSTTCOUPLE (default 1): thermostat coupling interval of both engines (GROMACS nsttcouple, PD CSVR interval)
for n in $SIZES; do
    gro=$STATES/MS_$n.gro; top=$STATES/MS_$n.top
    for e in $ENS; do
        for E in $ENGINES; do
            d=$OUT/$n/$e/$E; mkdir -p $d
            echo "=== $n $e $E ($(date))"
            case $E in
            gromacs) sampled $d $HERE/gromacs_run.sh $gro $top $d/run $e $STEPS ${WARM:-6000} 16 > $d/run.log 2>&1
                     [ "$(cat "$d/exit.txt")" = 0 ] || failed=1
                     cp $d/run/perf.txt $d/perf.txt 2>/dev/null; cat $d/perf.txt ;;
            pd|pd_ell|pd_esp)   # pd: cluster-pair backend; pd_ell: the dense list (charged ELL path), PME;
                     # pd_esp: the dense list with ESP at GROMACS PME's force error (match_accuracy.jl, esp_scan.jl)
                     be=cluster; [ $E != pd ] && be=ell
                     # A failed/missing upstream run cannot supply a default PME grid.
                     if ! mesh=$(python3 "$HERE/../common/audit.py" mesh --settings "$OUT/$n/$e/gromacs/run/settings.txt" --status "$OUT/$n/$e/gromacs/exit.txt"); then
                         printf 'Upstream mesh unavailable; downstream case failed before Julia launch.\n' > "$d/run.log"
                         echo 2 > "$d/exit.txt"; failed=1; continue
                     fi
                     esp=""; [ $E = pd_esp ] && esp="method=esp $(esp_settings $n)"
                     sampled $d "$julia_bin" --project=$PD --startup-file=no $HERE/pd_run.jl gro=$gro top=$top ensemble=$e backend=$be \
                         steps=$STEPS warm=${WARM:-6000} out=$d/record.json skin=$PD_SKIN prune=$PD_PRUNE mesh=$mesh $esp \
                         ${NSTTCOUPLE:+nsttcouple=$NSTTCOUPLE} > $d/run.log 2>&1
                     [ "$(cat "$d/exit.txt")" = 0 ] || failed=1
                     python3 -c "import json; r=json.load(open('$d/record.json')); print('ms_per_step', r['ms_per_step'], 'builds', r['builds'], 'prunes', r['prunes'], 'density', round(r['density_kg_m3'],2))" ;;
            esac
            echo "    exit $(cat $d/exit.txt) wall $(cat $d/wall_s.txt)"
        done
    done
done
echo "matrix finished ($(date)); failures=$failed"
exit "$failed"
