#!/usr/bin/env python3
"""Shared water state (.gro, any precision) -> GROMACS .g96 with full POSITION and VELOCITY
blocks (fixed 15.9f fields). GROMACS misreads the velocity columns of a 6-decimal .gro,
so the GROMACS runs start from this file: the same numbers as every other engine.
Usage: gro_to_g96.py IN.gro OUT.g96"""
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    title = f.readline().rstrip("\n"); n = int(f.readline())
    lines = [f.readline() for _ in range(n)]
    box = [float(b) for b in f.readline().split()[:3]]
t = lines[0][20:]; d1 = t.index("."); w = t.index(".", d1 + 1) - d1
with open(dst, "w") as f:
    f.write("TITLE\n%s\nEND\nPOSITION\n" % title)
    vel = []
    for i, l in enumerate(lines):
        resnr = int(l[0:5]); resname = l[5:10].strip(); atom = l[10:15].strip()
        s = l[20:]
        x = [float(s[k * w:(k + 1) * w]) for k in range(3)]
        r = s[3 * w:]
        v = [float(r[k * (w + 1):(k + 1) * (w + 1)]) for k in range(3)] if r.strip() else [0.0, 0.0, 0.0]
        f.write("%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n" % (resnr, resname, atom, i + 1, *x))
        vel.append((resnr, resname, atom, i + 1, v))
    f.write("END\nVELOCITY\n")
    for resnr, resname, atom, k, v in vel:
        f.write("%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n" % (resnr, resname, atom, k, *v))
    f.write("END\nBOX\n%15.9f%15.9f%15.9f\nEND\n" % tuple(box))
