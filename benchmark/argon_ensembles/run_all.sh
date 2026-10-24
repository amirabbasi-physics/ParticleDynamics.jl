#!/usr/bin/env bash
# Argon NVE/NVT/NPT matrix with matched settings: GROMACS (nstlist 20/50/100) and ParticleDynamics
# (dense ELL list for NVE/NVT; cluster pairs for all three, required by the device barostat),
# one run at a time. Usage: run_all.sh STATES OUT "10000 100000" "nve nvt npt" [NSTLISTS=, PD_BACKENDS= env]
set -uo pipefail
failed=0
source ~/.bashrc >/dev/null 2>&1 || true
STATES=$(realpath $1); OUT=$2; SIZES=$3; ENS=$4; STEPS=${STEPS:-10000}
HERE=$(dirname $(realpath $0)); PD=$(dirname $(dirname $HERE))
mkdir -- "$OUT" || exit 2; OUT=$(realpath "$OUT")
for n in $SIZES; do
    gro=$STATES/AR_$n.gro; top=$STATES/AR_$n.top; p=$(cat $STATES/AR_$n.pressure)
    for e in $ENS; do
        for nl in ${NSTLISTS:-20 50 100}; do
            d=$OUT/$n/$e/gromacs_nstlist$nl; mkdir -p $d
            printf "%s %s gromacs nstlist %s: " $n $e $nl
            $HERE/gromacs_run.sh $gro $top $p $d/run $e $STEPS $nl ${WARM:-6000} 8 > $d/run.log 2>&1 && cat $d/run/perf.txt || { echo FAILED; failed=1; }
        done
        for be in ${PD_BACKENDS:-ell cluster}; do

            d=$OUT/$n/$e/pd_$be; mkdir -p $d
            printf "%s %s pd %s: " $n $e $be
            julia --project=$PD --startup-file=no $HERE/pd_run.jl gro=$gro top=$top pressure=$p ensemble=$e backend=$be \
                steps=$STEPS warm=${WARM:-6000} out=$d/record.json > $d/run.log 2>&1 && \
                python3 -c "import json; r=json.load(open('$d/record.json')); print('ms_per_step', r['ms_per_step'], 'density', round(r['density_nm3'],4), {k:r[k] for k in ('builds','prunes') if k in r})" || { echo FAILED; failed=1; }
        done
    done
done
echo "argon matrix finished ($(date)); failures=$failed"
exit "$failed"
