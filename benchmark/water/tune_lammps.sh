#!/usr/bin/env bash
# LAMMPS KOKKOS run-time settings scan (package options, neighbor skin and check
# interval) on a shared state at the matched PPPM accuracy; 1000 NPT steps each.
# Usage: tune_lammps.sh STATE.gro PPPM OUTDIR   (PPPM = "gewald,nx,ny,nz,order")
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1 || true
gro=$1; pppm=$2; out=$3; mkdir -p $out
HERE=$(dirname $(realpath $0)); LMP=${LMP_KK:-/net/storage/abbaa90/lammps-kokkos-single/bin/lmp}
printf "%-26s %5s %5s %10s %10s\n" package skin every ms/step dangerous | tee $out/summary.txt
for pk in "neigh full newton off" "neigh half newton on" "neigh half newton off"; do
  for skin in 2.0 1.0; do
    for every in 1 5; do
      tag=$(echo "$pk" | tr ' ' '_')_s${skin}_e${every}; d=$out/$tag
      python3 $HERE/lammps_npt.py --gro $gro --dir $d --pppm $pppm --steps 1000 --skin $skin --neigh-every $every > /dev/null
      OMP_NUM_THREADS=1 $LMP -k on g 1 -sf kk -pk kokkos $pk -in $d/in.water -log $d/log.lammps -screen none > $d/run.log 2>&1
      t=$(grep -oE "Loop time of [0-9.e+-]+" $d/log.lammps | awk '{print $4}')
      ms=$(python3 -c "print('%.3f' % (1000*float('${t:-nan}')/1000))" 2>/dev/null || echo fail)
      dang=$(grep -oE "Dangerous builds = [0-9]+" $d/log.lammps | awk '{print $4}')
      printf "%-26s %5s %5s %10s %10s\n" "$pk" $skin $every $ms "${dang:-?}" | tee -a $out/summary.txt
    done
  done
done
