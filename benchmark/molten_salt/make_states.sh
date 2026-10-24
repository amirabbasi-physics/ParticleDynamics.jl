#!/usr/bin/env bash
# Molten NaCl benchmark: shared initial states.
# Joung–Cheatham (SPC/E) NaCl (σ+ 0.215954 nm, ε+ 1.47544 kJ/mol, σ− 0.483045 nm,
# ε− 0.0534924 kJ/mol, Lorentz–Berthelot, ±1 e, masses 22.99/35.45), LJ potential shift at
# 1.0 nm, no dispersion correction, PME (ewald-rtol 1e-5, spacing 0.12 nm, order 4).
# (The OPLS/Åqvist set of the PD tests, σ+ 0.333 nm, melts to only 1165 kg/m³ at 1400 K.)
# Rock-salt lattice of n³ ions (n even) at 32.07 ions/nm³ → steepest descent → NVT 3000 K 40 ps
# (melt; 1500 K left a superheated lattice intact) → NPT 1500 K (above the model's melting
# point), 1 atm, 100 ps (v-rescale + C-rescale). The final frame (positions and
# velocities, from the checkpoint via .g96) is MS_<n>.gro (6 decimals), plus MS_<n>.top.
# Usage: make_states.sh OUTDIR n [n ...]     (GMX overrides the GROMACS binary)
set -euo pipefail
OUT=${1:?outdir}; shift; SIZES=${*:-"22 46"}
G=${GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
HERE=$(dirname $(realpath $0)); WATER=$(dirname $HERE)/water
mkdir -p $OUT; cd $OUT
export HOME=$OUT/.home OMP_NUM_THREADS=8; mkdir -p $HOME
common='cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
vdw-modifier = Potential-shift
DispCorr = no
fourierspacing = 0.12
pme-order = 4
ewald-rtol = 1e-5'
for n in $SIZES; do
    name=MS_$n
    [ -f $name.gro ] && { echo "$name exists"; continue; }
    python3 - $n $name.start.gro <<'PY'
import sys
n = int(sys.argv[1]); out = sys.argv[2]
a = (1 / 32.07) ** (1 / 3)
na, cl = [], []
for i in range(n):
    for j in range(n):
        for k in range(n):
            p = ((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a)
            (na if (i + j + k) % 2 == 0 else cl).append(p)
L = n * a
with open(out, "w") as f:
    f.write("NaCl rock salt %d^3\n%d\n" % (n, n ** 3))
    k = 0
    for name, ps in (("NA", na), ("CL", cl)):
        for p in ps:
            k += 1
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (k % 100000, name, name, k % 100000, *p))
    f.write("%10.5f%10.5f%10.5f\n" % (L, L, L))
print(n ** 3 // 2, "Na,", n ** 3 // 2, "Cl, L =", L)
PY
    nh=$(( n * n * n / 2 ))
    cat > $name.top <<TOP
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
1 2 no 1.0 1.0

[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
NA 11 22.99 1.0 A 0.215954 1.47544
CL 17 35.45 -1.0 A 0.483045 0.0534924

[ moleculetype ]
NA 1
[ atoms ]
1 NA 1 NA NA 1 1.0 22.99

[ moleculetype ]
CL 1
[ atoms ]
1 CL 1 CL CL 1 -1.0 35.45

[ system ]
$name

[ molecules ]
NA $nh
CL $nh
TOP
    printf 'integrator = steep\nnsteps = 500\nemtol = 1000\n%s\n' "$common" > $name.em.mdp
    printf 'integrator = md\ndt = 0.002\nnsteps = 20000\n%s\ntcoupl = v-rescale\ntc-grps = System\ntau-t = 0.1\nref-t = 3000\ngen-vel = yes\ngen-temp = 3000\ngen-seed = 3000\nnstxout-compressed = 0\nnstenergy = 1000\nnstlog = 5000\n' "$common" > $name.nvt.mdp
    printf 'integrator = md\ndt = 0.002\nnsteps = 50000\n%s\ntcoupl = v-rescale\ntc-grps = System\ntau-t = 0.1\nref-t = 1500\npcoupl = C-rescale\npcoupltype = isotropic\ntau-p = 1.0\nref-p = 1.01325\ncompressibility = 4.5e-5\ngen-vel = no\ncontinuation = yes\nnstxout-compressed = 0\nnstenergy = 500\nnstcalcenergy = 100\nnstlog = 5000\n' "$common" > $name.npt.mdp
    $G grompp -f $name.em.mdp -c $name.start.gro -p $name.top -o $name.em.tpr -maxwarn 2 > $name.em.grompp.log 2>&1
    $G mdrun -s $name.em.tpr -deffnm $name.em -nb gpu -ntmpi 1 -ntomp 8 > $name.em.mdrun.log 2>&1
    $G grompp -f $name.nvt.mdp -c $name.em.gro -p $name.top -o $name.nvt.tpr -maxwarn 2 > $name.nvt.grompp.log 2>&1
    $G mdrun -s $name.nvt.tpr -deffnm $name.nvt -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 8 > $name.nvt.mdrun.log 2>&1
    $G grompp -f $name.npt.mdp -c $name.nvt.gro -t $name.nvt.cpt -p $name.top -o $name.npt.tpr -maxwarn 2 > $name.npt.grompp.log 2>&1
    $G mdrun -s $name.npt.tpr -deffnm $name.npt -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 8 > $name.npt.mdrun.log 2>&1
    echo 0 | $G trjconv -s $name.npt.tpr -f $name.npt.cpt -o $name.g96 -vel > $name.trjconv.log 2>&1
    python3 $HERE/g96_to_gro.py $name.g96 $name.gro $name.npt.gro
    echo Density | $G energy -f $name.npt.edr -o $name.density.xvg -b 50 > $name.density.log 2>&1 || true
    echo "$name: $(grep -E '^Density' $name.density.log | awk '{print $2, $3}') kg/m3 (50-100 ps); $(grep -A2 Performance $name.npt.log | tail -1)"
done
