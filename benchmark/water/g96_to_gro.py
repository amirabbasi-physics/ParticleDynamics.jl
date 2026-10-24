#!/usr/bin/env python3
"""GROMACS .g96 (9-decimal positions and velocities) -> .gro with 6 decimals for
positions and 7 for velocities (the shared-state layout; gro readers in this
directory take the field width from the decimal points). Usage: g96_to_gro.py IN.g96 OUT.gro"""
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
names = ("OW", "HW1", "HW2")   # SPC/E waters, O-H-H order (the reduced blocks carry no names)
if "POSITION" in blocks:
    pos = [[float(t) for t in p[24:].split()] for p in blocks["POSITION"]]
    vel = [[float(t) for t in v[24:].split()] for v in blocks.get("VELOCITY", [])]
else:
    pos = [[float(t) for t in p.split()] for p in blocks["POSITIONRED"]]
    vel = [[float(t) for t in v.split()] for v in blocks.get("VELOCITYRED", [])]
box = [float(t) for t in blocks["BOX"][0].split()[:3]]
with open(dst, "w") as f:
    f.write((blocks["TITLE"][0] if blocks.get("TITLE") else "water state") + "\n%d\n" % len(pos))
    for i, x in enumerate(pos):
        v = vel[i] if vel else [0.0, 0.0, 0.0]
        f.write("%5d%-5s%5s%5d%11.6f%11.6f%11.6f%12.7f%12.7f%12.7f\n" % ((i // 3 + 1) % 100000, "SOL", names[i % 3],
                (i + 1) % 100000, *x, *v))
    f.write("%12.7f%12.7f%12.7f\n" % tuple(box))
