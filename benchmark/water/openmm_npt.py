#!/usr/bin/env python3
"""Rigid SPC/E water NPT with OpenMM.

CUDA platform, single precision; rigid SPC/E (SETTLE via constraints), PME with
explicit parameters, CSVR thermostat as a CustomIntegrator (velocity Verlet +
constraints, Bussi rescale each step), Monte Carlo barostat with molecular scaling.
LJ is potential-shifted like the other engines: the shift -u(rc) per O-O pair inside
rc lives in force group 1. Its force is zero, so the integrator evaluates group 0 only,
while the barostat's energies (all integration groups) include it.

Input: a GROMACS .gro with velocities (the shared water state). Output: the final frame
(.gro with velocities) and a JSON record. I/O happens only after the last step.
"""
import argparse, json, math, sys, time
import numpy as np
import openmm as mm
from openmm import unit as u

KB = 0.0083144626  # kJ/mol/K
QO, QH = -0.8476, 0.4238
SIG, EPS = 0.316557, 0.650629
MO, MH = 15.9994, 1.008
OH, HH = 0.1, 0.16330


def read_gro(path):
    """GROMACS .gro with any precision (field width = distance between decimal points;
    velocities are one character wider)."""
    with open(path) as f:
        f.readline()
        n = int(f.readline())
        lines = [f.readline() for _ in range(n)]
        box = [float(b) for b in f.readline().split()[:3]]
    tail = lines[0][20:]
    d1 = tail.index("."); w = tail.index(".", d1 + 1) - d1
    x = np.empty((n, 3)); v = np.zeros((n, 3)); names = []
    for i, line in enumerate(lines):
        names.append(line[10:15].strip())
        t = line[20:]
        x[i] = [float(t[k * w:(k + 1) * w]) for k in range(3)]
        rest = t[3 * w:]
        if rest.strip():
            v[i] = [float(rest[k * (w + 1):(k + 1) * (w + 1)]) for k in range(3)]
    return names, x, v, box


def write_gro(path, names, x, v, box, title):
    """6 decimals for positions, 7 for velocities (GROMACS -ndec 6 layout)."""
    with open(path, "w") as f:
        f.write(title + "\n%d\n" % len(names))
        for i, nm in enumerate(names):
            r = i // 3 + 1
            f.write("%5d%-5s%5s%5d%11.6f%11.6f%11.6f%12.7f%12.7f%12.7f\n" % (
                r % 100000, "SOL", nm, (i + 1) % 100000, *x[i], *v[i]))
        f.write("%12.7f%12.7f%12.7f\n" % tuple(box))


def whole_molecules(x, box):
    """Hydrogens imaged next to their oxygen: OpenMM applies constraints (SETTLE) to
    absolute coordinates, so a water split across the boundary (states are wrapped atom
    by atom) would be torn apart."""
    L = np.asarray(box)
    for o in range(0, len(x), 3):
        for k in (1, 2):
            d = x[o + k] - x[o]
            x[o + k] = x[o] + d - L * np.round(d / L)
    return x


def lj(r):
    return 4 * EPS * ((SIG / r) ** 12 - (SIG / r) ** 6)


def build_system(n, box, rc, pme, mc_frequency, temperature, pressure, seed):
    nw = n // 3
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2]))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(rc)
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(False)
    if pme[0] == "tol":                    # OpenMM chooses alpha and the grid from the tolerance
        nb.setEwaldErrorTolerance(pme[1])
    else:
        alpha, nx, ny, nz = pme
        nb.setPMEParameters(alpha, nx, ny, nz)
    for w in range(nw):
        o = system.addParticle(MO); h1 = system.addParticle(MH); h2 = system.addParticle(MH)
        nb.addParticle(QO, SIG, EPS); nb.addParticle(QH, 1.0, 0.0); nb.addParticle(QH, 1.0, 0.0)
        system.addConstraint(o, h1, OH); system.addConstraint(o, h2, OH); system.addConstraint(h1, h2, HH)
        for a, b in ((o, h1), (o, h2), (h1, h2)):
            nb.addException(a, b, 0.0, 1.0, 0.0)
    nb.setForceGroup(0)
    system.addForce(nb)
    # potential shift of the O-O LJ term: energy only (zero force), group 1. A per-particle
    # oxygen flag on the ordinary cutoff list (interaction groups of all oxygens overflow at
    # 100k waters: "Error uploading array interactionGroupData").
    shift = mm.CustomNonbondedForce("-shift*o1*o2")
    shift.addGlobalParameter("shift", lj(rc))
    shift.addPerParticleParameter("o")
    shift.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    shift.setCutoffDistance(rc)
    shift.setUseLongRangeCorrection(False)
    for i in range(n):
        shift.addParticle([1.0 if i % 3 == 0 else 0.0])
    for o in range(0, n, 3):   # CUDA requires the NonbondedForce's exclusions on every cutoff force
        shift.addExclusion(o, o + 1); shift.addExclusion(o, o + 2); shift.addExclusion(o + 1, o + 2)
    shift.setForceGroup(1)
    system.addForce(shift)
    baro = mm.MonteCarloBarostat(pressure, temperature, mc_frequency)
    baro.setRandomNumberSeed(seed)
    system.addForce(baro)
    return system


def csvr_integrator(dt, temperature, tau, nf, seed, gamma_draw=True):
    """Velocity Verlet with constraints, then CSVR (Bussi et al. 2007) every step.
    nf: kinetic degrees of freedom; nmask: per-DOF mask selecting nf - 1 Gaussians."""
    integ = mm.CustomIntegrator(dt)
    integ.setRandomNumberSeed(seed)
    integ.addGlobalVariable("kt", KB * temperature / 2)   # target kinetic energy per DOF
    integ.addGlobalVariable("nf", nf)
    integ.addGlobalVariable("c", math.exp(-dt / tau))
    integ.addGlobalVariable("ke", 0.0)
    integ.addGlobalVariable("r1", 0.0)
    integ.addGlobalVariable("s", 0.0)
    integ.addGlobalVariable("alpha2", 0.0)
    integ.addGlobalVariable("alpha", 1.0)
    integ.addPerDofVariable("x1", 0.0)
    integ.addPerDofVariable("mask", 0.0)
    integ.addPerDofVariable("g", 0.0)
    # chi-square(nf - 1) = 2 Gamma((nf - 1)/2), Gamma by Marsaglia-Tsang (exact; ~1 iteration)
    a = (nf - 1) / 2.0
    integ.addGlobalVariable("gd", a - 1.0 / 3.0)
    integ.addGlobalVariable("gc", 1.0 / math.sqrt(9.0 * (a - 1.0 / 3.0)))
    for v in ("gx", "gv", "gu", "acc"):
        integ.addGlobalVariable(v, 0.0)
    integ.addUpdateContextState()
    integ.addComputePerDof("v", "v+0.5*dt*f0/m")
    integ.addComputePerDof("x", "x+dt*v")
    integ.addComputePerDof("x1", "x")
    integ.addConstrainPositions()
    integ.addComputePerDof("v", "v+(x-x1)/dt")
    integ.addComputePerDof("v", "v+0.5*dt*f0/m")
    integ.addConstrainVelocities()
    # CSVR: K' = alpha^2 K with alpha^2 = c + (1-c)(S + R1^2) Kt/(nf K) + 2 R1 sqrt(c(1-c) Kt/(nf K))
    integ.addComputeSum("ke", "0.5*m*v*v")
    integ.addComputeGlobal("r1", "gaussian")
    if gamma_draw:
        integ.addComputeGlobal("acc", "0")
        integ.beginWhileBlock("acc < 0.5")
        integ.addComputeGlobal("gx", "gaussian")
        integ.addComputeGlobal("gv", "(1 + gc*gx)^3")
        integ.addComputeGlobal("gu", "uniform")
        integ.addComputeGlobal("acc", "step(gv)*step(0.5*gx*gx + gd - gd*gv + gd*log(max(gv, 1e-30)) - log(gu))")
        integ.endBlock()
        integ.addComputeGlobal("s", "2*gd*gv")
    else:
        # each "gaussian" in an expression is a fresh draw: square a stored one
        integ.addComputePerDof("g", "gaussian")
        integ.addComputeSum("s", "mask*g*g")
    integ.addComputeGlobal("alpha2", "c + (1-c)*(s+r1*r1)*kt/ke + 2*r1*sqrt(c*(1-c)*kt/ke)")
    integ.addComputeGlobal("alpha", "select(step(r1 + sqrt(c*ke/((1-c)*kt))), 1, -1)*sqrt(alpha2)")
    integ.addComputePerDof("v", "v*alpha")
    return integ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gro", required=True); p.add_argument("--out", required=True); p.add_argument("--json", required=True)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--temperature", type=float, default=298.0); p.add_argument("--pressure", type=float, default=1.01325)
    p.add_argument("--tau-t", type=float, default=0.1); p.add_argument("--mc-frequency", type=int, default=100)
    p.add_argument("--rc", type=float, default=1.0)
    p.add_argument("--pme", help="alpha,nx,ny,nz (alpha in 1/nm)")
    p.add_argument("--ewald-tol", type=float, help="ewaldErrorTolerance (instead of --pme)")
    p.add_argument("--precision", default="single"); p.add_argument("--seed", type=int, default=29)
    p.add_argument("--chi2-sum", action="store_true", help="CSVR chi-square as a sum of per-DOF Gaussians (slow reference)")
    p.add_argument("--integrator", default="csvr", choices=("csvr", "langevin", "verlet"),
                   help="csvr: the CSVR CustomIntegrator; langevin/verlet: native integrators (overhead reference)")
    a = p.parse_args()
    names, x, v, box = read_gro(a.gro)
    x = whole_molecules(x, box)
    n = len(names); nw = n // 3
    if a.ewald_tol is not None:
        pme = ("tol", a.ewald_tol)
    else:
        pme = [float(t) for t in a.pme.split(",")]; pme = (pme[0], int(pme[1]), int(pme[2]), int(pme[3]))
    t0 = time.time()
    system = build_system(n, box, a.rc, pme, a.mc_frequency, a.temperature, a.pressure, a.seed)
    nf = 6 * nw                      # rigid waters, no centre-of-mass removal
    if a.integrator == "csvr":
        integ = csvr_integrator(a.dt, a.temperature, a.tau_t, nf, a.seed, gamma_draw=not a.chi2_sum)
    elif a.integrator == "langevin":
        integ = mm.LangevinMiddleIntegrator(a.temperature, 1.0 / a.tau_t, a.dt); integ.setRandomNumberSeed(a.seed)
    else:
        integ = mm.VerletIntegrator(a.dt)
    platform = mm.Platform.getPlatformByName("CUDA")
    ctx = mm.Context(system, integ, platform, {"Precision": a.precision, "DeviceIndex": "0"})
    ctx.setPositions(x); ctx.setVelocities(v)
    if a.integrator == "csvr":
        mask = np.zeros((n, 3)); mask.reshape(-1)[: nf - 1] = 1.0
        integ.setPerDofVariableByName("mask", [mm.Vec3(*row) for row in mask])
    ctx.applyConstraints(1e-6); ctx.applyVelocityConstraints(1e-6)
    setup = time.time() - t0
    t1 = time.time()
    integ.step(a.steps)
    st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    loop = time.time() - t1
    nbf = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    pme_used = list(nbf.getPMEParametersInContext(ctx))
    pos = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    vel = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
    L = [bv[0][0], bv[1][1], bv[2][2]]
    write_gro(a.out, names, pos, vel, L, "OpenMM final frame step %d" % a.steps)
    vol = L[0] * L[1] * L[2]
    rec = {"engine": "OpenMM", "version": mm.__version__, "precision": a.precision, "integrator": a.integrator, "n_atoms": n, "n_waters": nw,
           "steps": a.steps, "dt": a.dt, "pme": list(pme), "pme_in_context": pme_used, "mc_frequency": a.mc_frequency,
           "setup_s": setup, "loop_s": loop, "ms_per_step": 1e3 * loop / a.steps,
           "ns_per_day": a.steps * a.dt * 1e-3 / (loop / 86400.0),
           "final_box": L, "final_density_kg_m3": nw * (MO + 2 * MH) / 6.02214076e23 * 1e-3 / (vol * 1e-27),
           "potential_energy": st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)}
    json.dump(rec, open(a.json, "w"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
