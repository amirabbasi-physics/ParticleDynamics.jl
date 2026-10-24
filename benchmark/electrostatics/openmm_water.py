"""OpenMM (CUDA, mixed precision) side of the electrostatics head-to-head.

usage: openmm_water.py GRO TOP INCLUDE_DIR MODE TOL OUT [NSTEPS]

MODE   forces : write the PME Coulomb forces (kJ/mol/nm, Float64, 3×N column
                major, .gro atom order) to OUT; LJ switched off
       w1     : time GPU-resident Verlet steps with LJ switched off
       w2     : time GPU-resident Verlet steps with LJ on
TOL    ewaldErrorTolerance (OpenMM picks α and the grid from it)
Prints one line: "alpha <α> grid <nx> <ny> <nz> ms_per_step <t>" (t = 0 for forces).
Water is flexible (-DFLEXIBLE), r_c = 1.0 nm, no dispersion correction.
"""
import sys
import time

import openmm as mm
from openmm import app, unit


class _Gro:
    """Variable-precision .gro (field width = distance between decimal points): OpenMM's
    GromacsGroFile rejects the 6-decimal files of the shared water states."""
    def __init__(self, path):
        with open(path) as f:
            f.readline(); n = int(f.readline())
            lines = [f.readline() for _ in range(n)]
            box = [float(b) for b in f.readline().split()[:3]]
        t = lines[0][20:]; d1 = t.index("."); w = t.index(".", d1 + 1) - d1
        x = [[float(l[20 + k * w:20 + (k + 1) * w]) for k in range(3)] for l in lines]
        names = [l[10:15].strip() for l in lines]
        # waters whole (H imaged next to O): OpenMM corrects excluded pairs without the
        # minimum image, so a water split across the boundary gives wrong PME forces
        if n % 3 == 0 and all(names[i:i + 3] == ["OW", "HW1", "HW2"] for i in range(0, n, 3)):
            for o in range(0, n, 3):
                for h in (o + 1, o + 2):
                    for k in range(3):
                        d = x[h][k] - x[o][k]
                        x[h][k] = x[o][k] + d - box[k] * round(d / box[k])
        self._x = [mm.Vec3(*r) for r in x]
        self._box = (mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2]))
    def getPositions(self):
        return unit.Quantity(self._x, unit.nanometer)
    def getPeriodicBoxVectors(self):
        return unit.Quantity(self._box, unit.nanometer)


def _read_gro(path):
    # always the own reader: any precision, and waters made whole (see _Gro)
    return _Gro(path)


def build(gro, top, incdir, tol, lj):
    g = _read_gro(gro)
    t = app.GromacsTopFile(top, periodicBoxVectors=g.getPeriodicBoxVectors(), includeDir=incdir,
                           defines={"FLEXIBLE": True})
    system = t.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                            ewaldErrorTolerance=tol, constraints=None, rigidWater=False)
    # force groups by index (getForces() returns fresh proxy objects, so
    # identity comparisons never match)
    inb = [k for k in range(system.getNumForces()) if isinstance(system.getForce(k), mm.NonbondedForce)][0]
    nb = system.getForce(inb)
    nb.setUseDispersionCorrection(False)
    for k in range(system.getNumForces()):
        system.getForce(k).setForceGroup(0 if k == inb else 1)
    if not lj:
        for i in range(nb.getNumParticles()):
            q, s, e = nb.getParticleParameters(i)
            nb.setParticleParameters(i, q, s, 0.0)
        for k in range(nb.getNumExceptions()):
            i, j, qq, s, e = nb.getExceptionParameters(k)
            nb.setExceptionParameters(k, i, j, qq, s, 0.0)
    return g, system, nb


def main():
    gro, top, incdir, mode, tol, out = sys.argv[1:7]
    nsteps = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
    tol = float(tol)
    g, system, nb = build(gro, top, incdir, tol, lj=(mode == "w2"))
    integ = mm.VerletIntegrator(0.0005 * unit.picoseconds)
    plat = mm.Platform.getPlatformByName("CUDA")
    ctx = mm.Context(system, integ, plat, {"Precision": "mixed"})
    ctx.setPositions(g.getPositions())
    alpha, nx, ny, nz = nb.getPMEParametersInContext(ctx)
    ms = 0.0
    if mode == "forces":
        st = ctx.getState(getForces=True, groups={0})
        F = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        import numpy as np
        np.asarray(F, dtype=np.float64).tofile(out)
    else:
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)
        integ.step(200)
        ctx.getState(getEnergy=False)          # synchronise
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            integ.step(nsteps)
            ctx.getState(getEnergy=False)
            dt = (time.perf_counter() - t0) / nsteps * 1e3
            best = dt if best is None else min(best, dt)
        ms = best
    print(f"alpha {alpha} grid {nx} {ny} {nz} ms_per_step {ms}")


if __name__ == "__main__":
    main()
