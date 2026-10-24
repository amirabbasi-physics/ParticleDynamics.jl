#!/usr/bin/env python3
"""GROMACS .g96 (9 decimals) -> .gro with 6 decimals for positions and 7 for velocities (field
widths read back from the decimal points). Reduced blocks (POSITIONRED/VELOCITYRED, from a
checkpoint) carry no names: they are taken from NAMES.gro (same atom order).
Usage: g96_to_gro.py IN.g96 OUT.gro [NAMES.gro]"""
import sys
src, dst = sys.argv[1], sys.argv[2]
blocks = {}; key = None
for line in open(src):
    s = line.rstrip("\n")
    if s in ("TITLE", "POSITION", "VELOCITY", "POSITIONRED", "VELOCITYRED", "BOX", "TIMESTEP"):
        key = s; blocks[key] = []; continue
    if s == "END":
        key = None; continue
    if key is not None:
        blocks[key].append(s)
if "POSITION" in blocks:
    pos = [p.split() for p in blocks["POSITION"]]
    vel = [v.split() for v in blocks.get("VELOCITY", [])]
else:
    ref = open(sys.argv[3]).read().splitlines()[2:]
    names = [(l[0:5].strip(), l[5:10].strip(), l[10:15].strip()) for l in ref]
    pos = [[*names[i], str(i + 1), *p.split()] for i, p in enumerate(blocks["POSITIONRED"])]
    vel = [[*names[i], str(i + 1), *v.split()] for i, v in enumerate(blocks.get("VELOCITYRED", []))]
box = [float(t) for t in blocks["BOX"][0].split()[:3]]
with open(dst, "w") as f:
    f.write((blocks["TITLE"][0] if blocks.get("TITLE") else "molten salt") + "\n%d\n" % len(pos))
    for i, p in enumerate(pos):
        x = [float(t) for t in p[4:7]]
        v = [float(t) for t in vel[i][4:7]] if vel else [0.0, 0.0, 0.0]
        f.write("%5d%-5s%5s%5d%11.6f%11.6f%11.6f%12.7f%12.7f%12.7f\n" % (int(p[0]) % 100000, p[1], p[2],
                (i + 1) % 100000, *x, *v))
    f.write("%12.7f%12.7f%12.7f\n" % tuple(box))
