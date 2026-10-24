#!/usr/bin/env python3
"""Argon accuracy comparison: block averages (after SKIP ps) of U/N, T, P, density for GROMACS
(edr series) and ParticleDynamics (obs.csv), and the Ar-Ar RDF of each final frame (periodic
k-d tree), with the RMS difference in units of the counting noise.
Usage: compare_obs.py OBSDIR [skip_ps=10]"""
import sys, os
import numpy as np
from scipy.spatial import cKDTree
D = sys.argv[1]; skip = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

def block(x, nb=10):
    x = np.asarray(x); m = len(x) // nb
    if m == 0: return x.mean(), float("nan")
    b = np.array([x[k*m:(k+1)*m].mean() for k in range(nb)])
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)

def xvg(path):
    a = np.loadtxt(path, comments=("#", "@")); a = a[None, :] if a.ndim == 1 else a
    return a[:, 0], a[:, 1]

def read_gro(path):
    L = open(path).read().splitlines(); n = int(L[1])
    t = L[2][20:]; d1 = t.index("."); w = t.index(".", d1 + 1) - d1
    X = np.array([[float(l[20 + k*w:20 + (k+1)*w]) for k in range(3)] for l in L[2:2+n]])
    box = np.array([float(b) for b in L[2+n].split()[:3]])
    return X, box

def rdf(X, box, rmax=1.2, dr=0.005):
    X = np.mod(X, box)
    tree = cKDTree(X, boxsize=box)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    d = X[pairs[:, 0]] - X[pairs[:, 1]]; d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    edges = np.arange(0, rmax + dr, dr); h, _ = np.histogram(r, edges)
    n = len(X); rho = n / np.prod(box)
    shell = 4/3*np.pi*(edges[1:]**3 - edges[:-1]**3)
    g = 2*h / (n * rho * shell); err = 2*np.sqrt(np.maximum(h, 1)) / (n * rho * shell)
    return 0.5*(edges[1:]+edges[:-1]), g, err

for n in sorted(os.listdir(D), key=lambda s: int(s) if s.isdigit() else 0):
    for e in ("nvt", "npt"):
        d = os.path.join(D, n, e)
        if not os.path.isdir(d): continue
        print(f"== {n} atoms, {e.upper()}")
        g = {}
        for name, key in (("Potential", "U"), ("Temperature", "T"), ("Pressure", "P"), ("Density", "rho")):
            p = os.path.join(d, "gromacs", name + ".xvg")
            if os.path.exists(p):
                t, x = xvg(p); keep = t > skip
                if key == "U": x = x / int(n)
                if key == "rho": x = x / (39.948 * 1.66053906660) # kg/m3 -> atoms/nm3
                g[key] = block(x[keep])
        a = np.genfromtxt(os.path.join(d, "pd.obs.csv"), delimiter=",", names=True)
        keep = a["t_ps"] > skip
        pd_ = {"U": block(a["U_per_atom"][keep]), "T": block(a["T_K"][keep]), "P": block(a["P_bar"][keep]),
               "rho": block(a["density_nm3"][keep])}
        for key, unit in (("U", "kJ/mol per atom"), ("T", "K"), ("P", "bar"), ("rho", "atoms/nm^3")):
            if key in g:
                print(f"   {key:4s} GROMACS {g[key][0]:10.4f} ± {g[key][1]:.4f}   PD {pd_[key][0]:10.4f} ± {pd_[key][1]:.4f}   ({unit})")
        Xg, bg = read_gro(os.path.join(d, "gromacs", "md.gro")); Xp, bp = read_gro(os.path.join(d, "pd.final.gro"))
        r, gg, eg = rdf(Xg, bg); _, gp, ep = rdf(Xp, bp)
        m = r > 0.3
        z = np.sqrt(np.mean(((gg[m] - gp[m]) / np.sqrt(eg[m]**2 + ep[m]**2))**2))
        i1 = np.argmax(gg); j1 = np.argmax(gp)
        print(f"   RDF first peak: GROMACS r={r[i1]:.3f} g={gg[i1]:.3f}   PD r={r[j1]:.3f} g={gp[j1]:.3f};  RMS z of g(r) difference = {z:.2f}")
        np.savetxt(os.path.join(d, "rdf.csv"), np.c_[r, gg, eg, gp, ep], delimiter=",", header="r_nm,g_gromacs,err_gromacs,g_pd,err_pd")
