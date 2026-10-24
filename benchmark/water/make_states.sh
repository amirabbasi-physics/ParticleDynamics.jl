#!/usr/bin/env bash
# Shared initial states: rigid SPC/E water (amber99sb-ildn.ff/spce.itp,
# SETTLE), 10k / 100k / 500k molecules, relaxed in GROMACS at 298 K and 1 atm:
#   solvate (box slightly larger than the target density, -maxsol N) -> steepest descent ->
#   NVT 20 ps (v-rescale) -> NPT 100 ps (v-rescale + C-rescale, tau_p 1 ps).
# The final frame (positions and velocities) is exported via .g96 and written with 6 decimals so the rigid
# geometry survives the export: water_<size>.gro, the shared start of every engine.
# Usage: make_states.sh OUTDIR [sizes...]   (GMX overrides the GROMACS binary)
set -euo pipefail
OUT=${1:?outdir}; shift; SIZES=${*:-"10000 100000 500000"}
G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
TOP=$(dirname $(dirname $G))/share/gromacs/top
mkdir -p $OUT; cd $OUT
export HOME=$OUT/.home GMXLIB=$TOP OMP_NUM_THREADS=8; mkdir -p $HOME
ulimit -s unlimited 2>/dev/null || ulimit -s 65536
common='cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
vdw-modifier = Potential-shift
DispCorr = no
fourierspacing = 0.12
pme-order = 4
ewald-rtol = 1e-5'
for N in $SIZES; do
    name=water_$N
    [ -f $name.gro ] && { echo "$name exists"; continue; }
    L=$(python3 -c "print(round(($N/32.6)**(1/3), 4))")   # a little below 33.4 /nm^3: no overlaps
    $G solvate -cs spc216.gro -box $L $L $L -maxsol $N -o $name.start.gro > $name.solvate.log 2>&1
    n=$(grep -c " OW" $name.start.gro); [ "$n" -eq "$N" ] || { echo "$name: solvate gave $n waters"; exit 1; }
    printf '#include "amber99sb-ildn.ff/forcefield.itp"\n#include "amber99sb-ildn.ff/spce.itp"\n\n[ system ]\n%s\n\n[ molecules ]\nSOL %d\n' $name $N > $name.top
    printf 'integrator = steep\nnsteps = 1000\nemtol = 500\n%s\n' "$common" > $name.em.mdp
    printf 'integrator = md\ndt = 0.002\nnsteps = 10000\n%s\ntcoupl = v-rescale\ntc-grps = System\ntau-t = 0.1\nref-t = 298\ngen-vel = yes\ngen-temp = 298\ngen-seed = 298\nconstraints = h-bonds\nnstxout-compressed = 0\nnstenergy = 1000\nnstlog = 1000\n' "$common" > $name.nvt.mdp
    printf 'integrator = md\ndt = 0.002\nnsteps = 50000\n%s\ntcoupl = v-rescale\ntc-grps = System\ntau-t = 0.1\nref-t = 298\npcoupl = C-rescale\npcoupltype = isotropic\ntau-p = 1.0\nref-p = 1.01325\ncompressibility = 4.5e-5\ngen-vel = no\ncontinuation = yes\nconstraints = h-bonds\nnstxout-compressed = 0\nnstenergy = 500\nnstcalcenergy = 100\nnstlog = 5000\n' "$common" > $name.npt.mdp
    $G grompp -f $name.em.mdp -c $name.start.gro -p $name.top -o $name.em.tpr -maxwarn 2 > $name.em.grompp.log 2>&1
    $G mdrun -s $name.em.tpr -deffnm $name.em -nb gpu -ntmpi 1 -ntomp 8 > $name.em.mdrun.log 2>&1
    $G grompp -f $name.nvt.mdp -c $name.em.gro -p $name.top -o $name.nvt.tpr -maxwarn 2 > $name.nvt.grompp.log 2>&1
    $G mdrun -s $name.nvt.tpr -deffnm $name.nvt -nb gpu -pme gpu -update ${UPDATE:-gpu} -ntmpi 1 -ntomp 8 > $name.nvt.mdrun.log 2>&1
    $G grompp -f $name.npt.mdp -c $name.nvt.gro -t $name.nvt.cpt -p $name.top -o $name.npt.tpr -maxwarn 2 > $name.npt.grompp.log 2>&1
    $G mdrun -s $name.npt.tpr -deffnm $name.npt -nb gpu -pme gpu -update ${UPDATE:-gpu} -ntmpi 1 -ntomp 8 > $name.npt.mdrun.log 2>&1
    # from the checkpoint (full precision) via .g96 (9 decimals); .gro output of GROMACS has 3
    # decimals (trjconv -ndec applies to .xtc only), so the 6-decimal .gro is written by g96_to_gro.py
    echo 0 | $G trjconv -s $name.npt.tpr -f $name.npt.cpt -o $name.g96 -vel > $name.trjconv.log 2>&1
    python3 $(dirname $(realpath $0))/g96_to_gro.py $name.g96 $name.gro
    echo Density | $G energy -f $name.npt.edr -o $name.density.xvg -b 50 > $name.density.log 2>&1 || true
    echo "$name: $(grep -E '^Density' $name.density.log | awk '{print $2, $3}') kg/m3 (50-100 ps); $(grep -A2 Performance $name.npt.log | tail -1)"
done
