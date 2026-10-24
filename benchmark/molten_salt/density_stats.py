#!/usr/bin/env python3
"""NPT density statistics: GROMACS (gmx energy xvg) and PD (pd_run.jl io.csv: box length per sample).
Discards the first `skip` fraction, block-averages (`nblocks`) for the mean and its standard error.
Usage: density_stats.py DIR [skip=0.2] [nblocks=10]   (DIR/{gromacs,pd_pme,pd_esp})"""
import sys, os, json, subprocess, math
d = sys.argv[1]; skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 10
def stats(x):
    x = x[int(len(x) * skip):]; n = len(x) // nb
    b = [sum(x[i*n:(i+1)*n]) / n for i in range(nb)]
    m = sum(b) / nb; se = math.sqrt(sum((v - m) ** 2 for v in b) / (nb - 1) / nb)
    return m, se, len(x)
out = {}
g = os.path.join(d, "gromacs", "run")
if os.path.exists(os.path.join(g, "md.edr")):
    G = "/net/storage/abbaa90/gromacs-gpu/bin/gmx"; env = dict(os.environ, GMXLIB="/net/storage/abbaa90/gromacs-gpu/share/gromacs/top", HOME=g)
    xvg = os.path.join(os.environ.get("TMPDIR", "/tmp"), "dens_%d.xvg" % os.getpid())
    subprocess.run([G, "energy", "-f", os.path.join(g, "md.edr"), "-o", xvg], input=b"Density\n", env=env, capture_output=True)
    x = [float(l.split()[1]) for l in open(xvg) if l.strip() and l[0] not in "#@"]
    out["gromacs"] = stats(x)
for e in ("pd_pme", "pd_esp", "pd"):
    f = os.path.join(d, e, "record.json")
    if not os.path.exists(f): continue
    r = json.load(open(f)); L = r["final_box"]; mass_rho = r["density_kg_m3"] * L[0] * L[1] * L[2]   # ρ·V = const
    x = []
    for l in open(f + ".io.csv").readlines()[1:]:
        Lx = float(l.strip().split(",")[4]); x.append(mass_rho / (Lx ** 3))
    out[e] = stats(x)
for k, (m, se, n) in out.items():
    print(f"{k:8s} density {m:9.3f} ± {se:.3f} kg/m^3  ({n} samples)")
