#!/usr/bin/env bash
# Head-to-head: rigid SPC/E water NPT, 298 K / 1 atm, 10,000 steps,
# I/O at the last step only, every engine started from the shared GROMACS state.
#   GPU, single precision:  GROMACS PME, ParticleDynamics PME (GROMACS' grid), OpenMM PME,
#                           LAMMPS KOKKOS PPPM, ParticleDynamics ESP
#   CPU (16 cores):         GROMACS-ESP, LAMMPS-ESP (the authors' forks; ESP is CPU-only there)
# Knobs come from the accuracy matching (match_accuracy.jl, match_accuracy_esp.jl). Runs are
# sequential (the GPU is never shared). GPU utilization is sampled every 200 ms during
# every run (gpu_util.csv) as the GPU-bound check. Each run directory gets its final frame
# (final.gro), the engine's own log/record and the RDF/density analysis (rdf.toml).
#
# Usage: run_all.sh STATES_DIR OUT_DIR MATCH.toml MATCH_ESP.toml [sizes...]
#   ENGINES="gromacs pd_pme openmm lammps pd_esp gromacs_esp lammps_esp" (default: all)
#   STEPS=10000; LMP_PK, LMP_SKIN, LMP_EVERY (LAMMPS KOKKOS run settings); PD_SKIN (PD list buffer)
set -uo pipefail
failed=0
source ~/.bashrc >/dev/null 2>&1 || true
STATES=$(realpath $1); OUT=$2; MATCH=$(realpath $3); MATCH_ESP=$(realpath $4); shift 4
SIZES=${*:-"10000 100000 500000"}
ENGINES=${ENGINES:-"gromacs pd_pme openmm lammps pd_esp gromacs_esp lammps_esp"}
HERE=$(dirname $(realpath $0))
PD=$(dirname $(dirname $HERE))
GMX_GPU=${GMX_GPU:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
GMX_ESP=${GMX_ESP:-/net/storage/abbaa90/gromacs-esp-cpu/bin/gmx}
LMP_KK=${LMP_KK:-/net/storage/abbaa90/lammps-kokkos-single/bin/lmp}
LMP_ESP=${LMP_ESP:-/net/storage/abbaa90/lammps-esp-cpu/bin/lmp}
OMM_PY=${OMM_PY:-/net/storage/abbaa90/openmm-gpu-env/bin/python}
STEPS=${STEPS:-10000}
# LAMMPS run settings from tune_lammps.sh (tuned at 10k and 100k: half lists with Newton on,
# skin 2 A, check every 5 steps; skin 1 A with checks every 5 steps had dangerous builds)
LMP_PK=${LMP_PK:-"neigh half newton on"}; LMP_SKIN=${LMP_SKIN:-2.0}; LMP_EVERY=${LMP_EVERY:-5}
mkdir -p $OUT; OUT=$(realpath $OUT)
key() { python3 -c "import tomllib; d=tomllib.load(open('$1','rb')); print(d['$2']['$3'])" 2>/dev/null; }
ALPHA=$(key $MATCH "PD PME" alpha)
OMM_TOL=$(key $MATCH "OpenMM PME" tolerance)
LMP_SPACING=$(key $MATCH "LAMMPS PPPM fixed" spacing_nm); LMP_GEWALD=$(key $MATCH "LAMMPS PPPM fixed" gewald)
PD_TUNED=${PD_TUNED:-$(dirname $MATCH)/pd_tuned.toml}   # tune_pd_electrostatics.jl
PD_PME_TUNED="$(key $PD_TUNED "PD PME" alpha),$(key $PD_TUNED "PD PME" order),$(key $PD_TUNED "PD PME" spacing_nm)"
PD_ESP_TUNED="$(key $PD_TUNED "PD ESP" split_shape),$(key $PD_TUNED "PD ESP" window_shape),$(key $PD_TUNED "PD ESP" order),$(key $PD_TUNED "PD ESP" spacing_nm)"
GMX_ESP_TOL=$(key $MATCH_ESP "GROMACS-ESP" tolerance)
LMP_ESP_ACC=$(key $MATCH_ESP "LAMMPS-ESP" accuracy)
echo "knobs: pd_pme_tuned=$PD_PME_TUNED pd_esp_tuned=$PD_ESP_TUNED openmm_tol=$OMM_TOL lammps_spacing=$LMP_SPACING lammps_gewald=$LMP_GEWALD" \
     "gmx_esp_tol=$GMX_ESP_TOL lmp_esp_acc=$LMP_ESP_ACC" | tee $OUT/knobs.txt

# sampled DIR cmd...: GPU sampling around the command, wall time and exit status
sampled() {
    local dir=$1; shift
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 200 > $dir/gpu_util.csv &
    local smi=$!
    local t0=$(date +%s.%N)
    "$@"; local rc=$?
    local t1=$(date +%s.%N)
    kill $smi 2>/dev/null; wait $smi 2>/dev/null
    python3 -c "print($t1-$t0)" > $dir/wall_s.txt
    echo $rc > $dir/exit.txt
    return $rc
}
analyse() {   # final frame -> partial RDFs and density
    local dir=$1
    [ -f $dir/final.gro ] && julia --startup-file=no -t 16 $HERE/rdf.jl $dir/final.gro $dir/rdf.toml > $dir/rdf.log 2>&1
}
lammps_pppm() {   # "gewald,nx,ny,nz,5": the smallest 2,3,5-smooth grid with spacing <= LMP_SPACING
    python3 - "$1" $LMP_SPACING $LMP_GEWALD <<'PY'
import sys, math
gro, h, g = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(gro) as f:
    lines = f.read().splitlines()
L = [float(t) for t in lines[-1].split()[:3]]
def smooth(n):
    for p in (2, 3, 5):
        while n % p == 0: n //= p
    return n == 1
def grid(l):
    n = math.ceil(l / h - 1e-9)
    while not smooth(n): n += 1
    return n
print("%s,%d,%d,%d,5" % (g, *[grid(l) for l in L]))
PY
}
lammps_final() {   # LAMMPS dump (A, A/fs; id order) -> .gro (nm, nm/ps)
    python3 - "$1" "$2" <<'PY'
import sys
dump, gro = sys.argv[1], sys.argv[2]
L = open(dump).read().splitlines()
i = [k for k, l in enumerate(L) if l.startswith("ITEM: BOX BOUNDS")][0]
lo_hi = [tuple(map(float, L[i + 1 + k].split()[:2])) for k in range(3)]
box = [(b - a) / 10 for a, b in lo_hi]
j = [k for k, l in enumerate(L) if l.startswith("ITEM: ATOMS")][0]
rows = sorted((l.split() for l in L[j + 1:] if l.strip()), key=lambda r: int(r[0]))
names = ("OW", "HW1", "HW2")
with open(gro, "w") as f:
    f.write("LAMMPS final frame\n%d\n" % len(rows))
    for r in rows:
        i = int(r[0]) - 1
        x = [(float(r[3 + k]) - lo_hi[k][0]) / 10 for k in range(3)]
        v = [float(r[6 + k]) * 100 for k in range(3)]
        f.write("%5d%-5s%5s%5d%11.6f%11.6f%11.6f%12.7f%12.7f%12.7f\n" % ((i // 3 + 1) % 100000, "SOL", names[i % 3],
                (i + 1) % 100000, *x, *v))
    f.write("%12.7f%12.7f%12.7f\n" % tuple(box))
PY
}

for N in $SIZES; do
    gro=$STATES/water_$N.gro; top=$STATES/water_$N.top
    for E in $ENGINES; do
        d=$OUT/$N/$E; mkdir -p $d
        [ -f $d/exit.txt ] && [ "$(cat $d/exit.txt)" = 0 ] && { echo "$N $E done"; continue; }
        echo "=== $N $E ($(date))"
        case $E in
        gromacs)
            sampled $d env GMX=$GMX_GPU GMX_MODE=gpu $HERE/gromacs_npt.sh $gro $top $d/run 1e-5 0.12 4 $STEPS 16 > $d/run.log 2>&1
            cp $d/run/final.gro $d/final.gro 2>/dev/null; cp $d/run/grid.txt $d/grid.txt 2>/dev/null ;;
        pd_pme)   # SPME at PD's own fastest setting meeting the target (tune_pd_electrostatics.jl)
            sampled $d julia --startup-file=no --project=$PD $HERE/pd_npt.jl gro=$gro out=$d/final.gro json=$d/record.json \
                steps=$STEPS warm=${WARM:-6000} nsttcouple=${NSTTCOUPLE:-1} method=pme pme_tuned=$PD_PME_TUNED nstpcouple=100 skin=${PD_SKIN:-0.2} prune=${PD_PRUNE:-0.1} > $d/run.log 2>&1 ;;
        pd_esp)
            sampled $d julia --startup-file=no --project=$PD $HERE/pd_npt.jl gro=$gro out=$d/final.gro json=$d/record.json \
                steps=$STEPS warm=${WARM:-6000} nsttcouple=${NSTTCOUPLE:-1} method=esp esp_tuned=$PD_ESP_TUNED nstpcouple=100 skin=${PD_SKIN:-0.2} prune=${PD_PRUNE:-0.1} > $d/run.log 2>&1 ;;
        openmm)
            sampled $d $OMM_PY $HERE/openmm_npt.py --gro $gro --out $d/final.gro --json $d/record.json --steps $STEPS \
                --ewald-tol $OMM_TOL --mc-frequency 100 > $d/run.log 2>&1 ;;
        lammps)
            python3 $HERE/lammps_npt.py --gro $gro --dir $d/run --pppm $(lammps_pppm $gro) --steps $STEPS \
                --skin ${LMP_SKIN:-2.0} --neigh-every ${LMP_EVERY:-1} > $d/gen.log 2>&1
            sampled $d env OMP_NUM_THREADS=1 $LMP_KK -k on g 1 -sf kk -pk kokkos $LMP_PK \
                -in $d/run/in.water -log $d/run/log.lammps -screen none > $d/run.log 2>&1
            lammps_final $d/run/final.dump $d/final.gro ;;
        gromacs_esp)
            [ -n "$GMX_ESP_TOL" ] || { echo "    no GROMACS-ESP knob"; continue; }
            sampled $d env GMX=$GMX_ESP GMX_MODE=cpu $HERE/gromacs_npt.sh $gro $top $d/run $GMX_ESP_TOL 0.12 5 $STEPS 16 > $d/run.log 2>&1
            cp $d/run/final.gro $d/final.gro 2>/dev/null ;;
        lammps_esp)
            [ -n "$LMP_ESP_ACC" ] || { echo "    no LAMMPS-ESP knob"; continue; }
            python3 $HERE/lammps_npt.py --gro $gro --dir $d/run --esp-accuracy $LMP_ESP_ACC --steps $STEPS > $d/gen.log 2>&1
            sampled $d env OMP_NUM_THREADS=1 mpirun --oversubscribe -np 16 $LMP_ESP -in $d/run/in.water \
                -log $d/run/log.lammps -screen none > $d/run.log 2>&1
            lammps_final $d/run/final.dump $d/final.gro ;;
        esac
        echo "    exit $(cat $d/exit.txt) wall $(cat $d/wall_s.txt) s"
        [ -f "$d/exit.txt" ] && [ "$(cat "$d/exit.txt")" = 0 ] || failed=1
        analyse $d
    done
done
echo "Water runs finished ($(date)); failures=$failed"
exit "$failed"
