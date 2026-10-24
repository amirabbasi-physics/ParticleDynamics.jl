#!/usr/bin/env bash
# Rigid SPC/E water NPT with GROMACS: GPU-resident run (nonbonded,
# PME and update on the GPU; rigid water has no bonded terms), PME parameters fixed
# (-notunepme), SETTLE, LJ potential shift, no dispersion correction, no centre-of-mass removal
# (comm-mode None, as ParticleDynamics), v-rescale (CSVR) every step (nsttcouple 1, as PD) +
# C-rescale (nstpcouple = 100, tau-p = 5 ps: GROMACS requires tau-p >= 25 nstpcouple dt). Requested energy/log output every 2000 steps;
# WARM=6000 untimed steps precede the measured interval via -resetstep. Final coordinates/checkpoint follow.
# Usage: gromacs_npt.sh STATE.gro TOP RUNDIR EWALD_RTOL FOURIERSPACING ORDER [STEPS] [NTOMP]
#   GMX: the binary; GMX_MODE=gpu (default: nonbonded, PME and update on the GPU) or cpu
#   (the CPU-only GROMACS-ESP build: ewald-rtol is its target force accuracy; NTOMP cores as
#   OpenMP threads of one rank, or GMX_CPU_NPME=1 for one PP and one PME rank).
#   The grid GROMACS chose is written to grid.txt (ParticleDynamics PME uses the same).
set -euo pipefail
gro=$(realpath $1); top=$(realpath $2); dir=$3; rtol=$4; spacing=$5; order=$6
steps=${7:-10000}; ntomp=${8:-16}; mode=${GMX_MODE:-gpu}
warm=${WARM:-6000}; total=$((steps + warm))
G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
export GMXLIB=$(dirname $(dirname $G))/share/gromacs/top
mkdir -p $dir; cd $dir
case ${GMX_PREPARE_ONLY:-0} in 0|1) ;; *) echo 'GMX_PREPARE_ONLY must be 0 or 1' >&2; exit 2 ;; esac
cat > npt.mdp <<MDP
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
ewald-rtol = $rtol
fourierspacing = $spacing
pme-order = $order
tcoupl = v-rescale
tc-grps = System
tau-t = 0.1
ref-t = 298
nsttcouple = ${NSTTCOUPLE:-1}
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 5.0
ref-p = 1.01325
compressibility = 4.5e-5
nstpcouple = 100
gen-vel = no
continuation = yes
constraints = h-bonds
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = 0
nstcalcenergy = 2000
nstenergy = 2000
nstlog = 2000
MDP
# GROMACS misreads the velocity columns of a 6-decimal .gro: start from the same numbers via .g96
python3 $(dirname $(realpath $0))/gro_to_g96.py $gro start.g96
$G grompp -f npt.mdp -c start.g96 -p $top -o npt.tpr -maxwarn 2 > grompp.log 2>&1
if [ "${GMX_PREPARE_ONLY:-0}" = 1 ]; then
    $G dump -s npt.tpr > settings.txt 2> dump.log
    echo 'Prepared TPR and actual settings; dynamics not run.'
    exit 0
fi
# GMX_NO_FINAL_OUTPUT=1: no final coordinates or checkpoint inside the timed run (PD's timed loop
# ends before its final output); final.gro is then not produced
fin=$([ "${GMX_NO_FINAL_OUTPUT:-0}" = 1 ] && echo "-noconfout -cpt -1" || echo "-cpt 1e9")
# mdrun refuses an inherited OMP_NUM_THREADS that differs from -ntomp: set it to match
if [ "$mode" = gpu ]; then
    OMP_NUM_THREADS=$ntomp $G mdrun -s npt.tpr -deffnm npt -nb gpu -pme gpu -update gpu -notunepme $fin \
        -ntmpi 1 -ntomp $ntomp -pin on -resetstep $warm > mdrun.log 2>&1
else
    # the ESP fork fails under domain decomposition (2 PP ranks: SETTLE error at step 0), so its
    # cores go to OpenMP threads of one PP rank, or (GMX_CPU_NPME=1) to one PP and one PME rank
    if [ "${GMX_CPU_NPME:-0}" = 1 ]; then
        OMP_NUM_THREADS=$((ntomp / 2)) $G mdrun -s npt.tpr -deffnm npt -notunepme -cpt 1e9 -ntmpi 2 -npme 1 \
            -ntomp $((ntomp / 2)) -pin on -resetstep $warm > mdrun.log 2>&1
    else
        OMP_NUM_THREADS=$ntomp $G mdrun -s npt.tpr -deffnm npt -notunepme -cpt 1e9 -ntmpi 1 -ntomp $ntomp -pin on -resetstep $warm > mdrun.log 2>&1
    fi
fi
grep -m1 -oE "fourier-nx *= *[0-9]+|Using a fourier grid of [0-9x]+" npt.log > grid.txt || true
$G dump -s npt.tpr > settings.txt 2> dump.log
cat settings.txt | grep -E "^ *fourier-n[xyz]" | awk '{print $3}' | tr '\n' ' ' > grid.txt
if [ -f npt.cpt ]; then
    echo 0 | $G trjconv -s npt.tpr -f npt.cpt -o final.g96 -vel > trjconv.log 2>&1   # full precision
    python3 $(dirname $(realpath $0))/g96_to_gro.py final.g96 final.gro
fi
printf 'Density\nVolume\nTemperature\nPressure\n' | $G energy -f npt.edr -o averages.xvg > energy.log 2>&1 || true
grep -A3 "Core t (s)" npt.log | tail -3
