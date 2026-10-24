#!/usr/bin/env bash
# Nsight Systems kernel summary of a command (CUDA trace only, graph nodes
# traced individually). With CAPTURE=api only the cudaProfilerStart/Stop range is recorded
# (prof_pd.jl). Usage: prof_nsys.sh OUT_PREFIX command...   -> OUT_PREFIX.nsys-rep, OUT_PREFIX_kern.csv
set -uo pipefail
out=$1; shift
S=${NSYS_SCRATCH:-${TMPDIR:-/tmp}/pd-nsys}
mkdir -p $S/home $S/tmp
export HOME=$S/home TMPDIR=$S/tmp XDG_CONFIG_HOME=$S/home/.config
cap=""
[ "${CAPTURE:-}" = api ] && cap="--capture-range=cudaProfilerApi --capture-range-end=stop"
nsys profile -o $out --force-overwrite true -t cuda --cuda-graph-trace=node --cuda-event-trace=false \
    --sample=none --cpuctxsw=none $cap "$@" > $out.nsys.log 2>&1
echo "nsys exit $?"
nsys stats --report cuda_gpu_kern_sum --format csv --output $out --force-export=true $out.nsys-rep > $out.stats.log 2>&1
ls $out*kern* 2>/dev/null
