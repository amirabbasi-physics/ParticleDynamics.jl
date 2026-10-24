#!/usr/bin/env bash
# Molten NaCl benchmark: one GROMACS GPU-resident run (nonbonded, PME and update on the GPU).
# ENSEMBLE nve (no coupling), nvt (v-rescale τ_t 0.1 ps, 1500 K, applied every step: nsttcouple 1,
# as ParticleDynamics' CSVR) or npt (+ C-rescale τ_p 5 ps, nstpcouple 100, 1 atm). No centre-of-mass
# removal (comm-mode None; PD has none). Settings matched to pd_run.jl (user: apples to apples).
# PME fixed (-notunepme). No output except the final energy frame and
# log every 2000 steps (energies computed and written: nstcalcenergy = nstenergy = nstlog = 2000, as PD's
# I/O; user); timing counters reset after WARM steps (-resetstep), so the reported performance covers
# the STEPS timed steps. Usage: gromacs_run.sh STATE.gro TOP RUNDIR ENSEMBLE STEPS [WARM=6000] [NTOMP=16]
set -euo pipefail
gro=$(realpath $1); top=$(realpath $2); dir=$3; ens=$4; steps=$5; warm=${6:-6000}; ntomp=${7:-16}
G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
export GMXLIB=$(dirname $(dirname $G))/share/gromacs/top
mkdir -p $dir; cd $dir
case ${GMX_PREPARE_ONLY:-0} in 0|1) ;; *) echo 'GMX_PREPARE_ONLY must be 0 or 1' >&2; exit 2 ;; esac
case $ens in
    nve) coup="tcoupl = no
pcoupl = no" ;;
    nvt) coup="tcoupl = v-rescale
tc-grps = System
tau-t = 0.1
ref-t = 1500
nsttcouple = ${NSTTCOUPLE:-1}
pcoupl = no" ;;
    npt) coup="tcoupl = v-rescale
tc-grps = System
tau-t = 0.1
ref-t = 1500
nsttcouple = ${NSTTCOUPLE:-1}
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 5.0
ref-p = 1.01325
compressibility = 4.5e-5
nstpcouple = 100" ;;
    *) echo "unknown ensemble $ens"; exit 2 ;;
esac
total=$((steps + warm))
cat > md.mdp <<MDP
integrator = md
dt = 0.002
nsteps = $total
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
vdw-modifier = Potential-shift
DispCorr = no
comm-mode = None
ewald-rtol = ${RTOL:-1e-5}
fourierspacing = ${FSPACING:-0.12}
pme-order = ${PMEORDER:-4}
$coup
gen-vel = no
continuation = yes
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = 0
nstcalcenergy = 2000
nstenergy = 2000
nstlog = 2000
MDP
if [ -n "${NSTLIST:-}${RLIST:-}" ]; then
    python3 $(dirname $(realpath $0))/../common/explicit_list.py md.mdp
fi
python3 $(dirname $(realpath $0))/../water/gro_to_g96.py $gro start.g96
$G grompp -f md.mdp -c start.g96 -p $top -o md.tpr -maxwarn 2 > grompp.log 2>&1
if [ "${GMX_PREPARE_ONLY:-0}" = 1 ]; then
    $G dump -s md.tpr > settings.txt 2> dump.log
    echo 'Prepared TPR and actual settings; dynamics not run.'
    exit 0
fi
# GMX_NO_FINAL_OUTPUT=1: no final coordinates or checkpoint inside the timed run (PD's timed loop
# ends before its final output); otherwise the final frame and checkpoint are written as before
fin=$([ "${GMX_NO_FINAL_OUTPUT:-0}" = 1 ] && echo "-noconfout -cpt -1" || echo "-cpt 1e9")
OMP_NUM_THREADS=$ntomp $G mdrun -s md.tpr -deffnm md -nb gpu -pme gpu -update gpu -notunepme $fin \
    -ntmpi 1 -ntomp $ntomp -pin on -resetstep $warm > mdrun.log 2>&1
$G dump -s md.tpr > settings.txt 2> dump.log
nsday=$(grep -E "^Performance:" md.log | awk '{print $2}')
python3 -c "print('ms_per_step', 0.002 * 86400e3 / ($nsday * 1e3))" | tee perf.txt
grep -iE "outer list|inner list|nstlist from" md.log | head -4 || true
