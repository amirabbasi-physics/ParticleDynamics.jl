#!/usr/bin/env bash
# Reference for water_mc_density.jl: SPC/E water NPT density from GROMACS (C-rescale, 300 K,
# 1 bar) from the same equilibrated water box the PD Monte Carlo run starts from, with
# the same interaction settings (PME, rc = 1.0 nm, potential-shift, SETTLE, no
# dispersion correction). Usage: water_gromacs_npt.sh SYSTEMS_DIR NAME OUTDIR [NSTEPS] [REF_P bar]
# PD's cluster LJ is plainly truncated: Monte Carlo samples the truncated energy, MD
# the shifted one (it never sees the cutoff step). The missing shift acts as an
# internal pressure ρ²V_c e(r_c)/2 ≈ −100 bar for SPC/E at 1 nm, so PD MC at 1 bar is
# compared with GROMACS at REF_P = 1 + 99.7 bar (mean field, g(r_c) ≈ 1).
set -euo pipefail
sysdir=$1; name=$2; out=$3; nsteps=${4:-250000}; refp=${5:-1.0}
GMX=${PD_GMX:-/net/storage/abbaa90/gromacs-gpu/bin/gmx}
mkdir -p "$out"; cd "$out"
cat >npt.mdp <<MDP
integrator = md
dt = 0.002
nsteps = $nsteps
continuation = yes
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
fourierspacing = 0.12
tcoupl = v-rescale
tc-grps = System
tau-t = 0.1
ref-t = 300
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 1.0
ref-p = $refp
compressibility = 4.5e-5
gen-vel = no
constraints = h-bonds
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = 0
nstenergy = 500
nstlog = 5000
MDP
"$GMX" grompp -f npt.mdp -c "$sysdir/$name.gro" -p "$sysdir/$name.top" -o npt.tpr -maxwarn 1 >grompp.log 2>&1
OMP_NUM_THREADS=8 "$GMX" mdrun -s npt.tpr -deffnm npt -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 8 >mdrun.log 2>&1
printf 'Density\n0\n' | "$GMX" energy -f npt.edr -o density.xvg >energy.log 2>&1
grep -v '^[#@]' density.xvg >density.dat
echo done >status
