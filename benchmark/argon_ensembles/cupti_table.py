#!/usr/bin/env python3
"""Per-kernel time per step from a CUDA.jl profiler table (device-side activity).
Usage: cupti_table.py FILE.cupti.txt STEPS [top=40]"""
import re, sys
txt = open(sys.argv[1]).read(); steps = int(sys.argv[2]); top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
dev = txt.split("Device-side activity")[1]
rows = []
for line in dev.splitlines():
    if not line.startswith("│"): continue
    cells = [c.strip() for c in line.strip("│").split("│")]
    if len(cells) < 5 or cells[0].startswith("Time"): continue
    m = re.match(r"([\d.]+) (ms|µs|ns|s)", cells[1])
    if not m: continue
    ms = float(m.group(1)) * {"s": 1e3, "ms": 1, "µs": 1e-3, "ns": 1e-6}[m.group(2)]
    name = re.sub(r"\(.*", "", cells[4]).replace("ParticleDynamics.", "")
    rows.append((ms, cells[2], name[:100]))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"total kernel time {tot:.2f} ms = {1e3*tot/steps:.2f} us/step over {steps} steps")
for ms, c, n in rows[:top]:
    print(f"{1e3*ms/steps:9.2f} us/step  {c:>8}  {n}")
