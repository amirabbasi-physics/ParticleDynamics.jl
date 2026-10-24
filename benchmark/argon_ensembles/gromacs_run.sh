#!/usr/bin/env bash
# Argon NVE/NVT/NPT comparison, GROMACS side, with every setting matched to ParticleDynamics
# (user: apples to apples): plain LJ cutoff 2.5σ (no modifier), no dispersion correction,
# dt 0.00539087 ps, no centre-of-mass removal (comm-mode None), energies computed and written every
# 2000 steps (nstcalcenergy = nstenergy = nstlog = 2000; user: I/O every 2000 steps for both),
# v-rescale τ_t 0.5 ps at 101.8 K applied EVERY step (nsttcouple 1, as PD's CSVR), C-rescale
# τ_p 5 ps, compressibility 4.5e-5 bar⁻¹, every 10 steps, at the state's mean NVT pressure.
# GPU-resident (nonbonded and update on the GPU), timing counters reset after WARM steps.
# The pair list (nstlist NSTLIST, Verlet buffer tolerance 0.005) is GROMACS's own algorithm.
# Usage: gromacs_run.sh STATE.gro TOP PRESSURE_BAR RUNDIR ENSEMBLE STEPS NSTLIST [WARM=6000] [NTOMP=8]
set -euo pipefail
gro=$(realpath $1); top=$(realpath $2); pbar=$3; dir=$4; ens=$5; steps=$6; nstlist=$7; warm=${8:-6000}; ntomp=${9:-8}
G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
mkdir -p $dir; cd $dir
case ${GMX_PREPARE_ONLY:-0} in 0|1) ;; *) echo 'GMX_PREPARE_ONLY must be 0 or 1' >&2; exit 2 ;; esac
tc="tcoupl = v-rescale
tc-grps = System
tau-t = 0.5
ref-t = 101.8
nsttcouple = ${NSTTCOUPLE:-1}"
case $ens in
    nve) coup="tcoupl = no
pcoupl = no" ;;
    nvt) coup="$tc
pcoupl = no" ;;
    npt) coup="$tc
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 5.0
ref-p = $pbar
compressibility = 4.5e-5
nstpcouple = 10" ;;
    *) echo "unknown ensemble $ens"; exit 2 ;;
esac
total=$((steps + warm))
io=2000; [ "${OBS:-0}" = 1 ] && io=100   # OBS=1: energies every 100 steps (accuracy comparison, not timed)
cat > md.mdp <<MDP
integrator = md
dt = 0.00539087
nsteps = $total
cutoff-scheme = Verlet
nstlist = $nstlist
verlet-buffer-tolerance = 0.005
coulombtype = Cut-off
rcoulomb = 0.851250
vdwtype = Cut-off
vdw-modifier = None
rvdw = 0.851250
DispCorr = no
comm-mode = None
$coup
gen-vel = no
continuation = yes
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = 0
nstcalcenergy = $io
nstenergy = $io
nstlog = $io
MDP
if [ -n "${NSTLIST:-}${RLIST:-}" ]; then
    python3 $(dirname $(realpath $0))/../common/explicit_list.py md.mdp
fi
python3 $(dirname $(realpath $0))/../water/gro_to_g96.py $gro start.g96
$G grompp -f md.mdp -c start.g96 -p $top -o md.tpr -maxwarn 3 > grompp.log 2>&1
if [ "${GMX_PREPARE_ONLY:-0}" = 1 ]; then
    $G dump -s md.tpr > settings.txt 2> dump.log
    echo 'Prepared TPR and actual settings; dynamics not run.'
    exit 0
fi
# GMX_NO_FINAL_OUTPUT=1: no final coordinates or checkpoint inside the timed run (as molten_salt/gromacs_run.sh)
fin=$([ "${GMX_NO_FINAL_OUTPUT:-0}" = 1 ] && echo "-noconfout -cpt -1" || echo "-cpt 1e9")
OMP_NUM_THREADS=$ntomp $G mdrun -s md.tpr -deffnm md -nb gpu -update gpu $fin -ntmpi 1 -ntomp $ntomp -pin on \
    -resetstep $warm > mdrun.log 2>&1
$G dump -s md.tpr > settings.txt 2> dump.log
nsday=$(grep -E "^Performance:" md.log | awk '{print $2}')
python3 -c "print('ms_per_step', 0.00539087 * 86400e3 / ($nsday * 1e3))" | tee perf.txt
grep -iE "nstlist from|outer list|inner list" md.log | head -3 || true
