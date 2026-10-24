#!/usr/bin/env bash
# Argon accuracy comparison: GROMACS (OBS=1: energies every 100 steps, final frame) and
# ParticleDynamics (pd_run.jl obs=1), NVT and NPT, matched settings. Usage: obs_all.sh STATES OUT "10000 100000"
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1 || true
STATES=$(realpath $1); OUT=$2; SIZES=$3; STEPS=${STEPS:-10000}
HERE=$(dirname $(realpath $0)); PD=$(dirname $(dirname $HERE)); G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
mkdir -p $OUT; OUT=$(realpath $OUT)
for n in $SIZES; do
    gro=$STATES/AR_$n.gro; top=$STATES/AR_$n.top; p=$(cat $STATES/AR_$n.pressure)
    for e in nvt npt; do
        d=$OUT/$n/$e; mkdir -p $d
        OBS=1 $HERE/gromacs_run.sh $gro $top $p $d/gromacs $e $STEPS 100 0 8 > $d/gromacs.log 2>&1 || echo "gromacs $n $e FAILED"
        for t in Potential Temperature Pressure Density; do
            echo $t | HOME=$d/gromacs/.home $G energy -f $d/gromacs/md.edr -o $d/gromacs/$t.xvg -xvg none > /dev/null 2>&1
        done
        be=ell; [ $e = npt ] && be=cluster
        julia --project=$PD --startup-file=no $HERE/pd_run.jl gro=$gro top=$top pressure=$p ensemble=$e backend=$be \
            steps=$STEPS obs=1 out=$d/pd > $d/pd.log 2>&1 || echo "pd $n $e FAILED"
        echo "done $n $e"
    done
done
