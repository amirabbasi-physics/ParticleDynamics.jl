"""How large a water box can a foundation model actually drive, and how does the
cost scale?

Replicates the 64-molecule water box from examples/mace/validation/water_init.npz
into k x k x k supercells and times single-point force calls for MACE-OFF23-small
and Orb-v3, recording peak GPU memory and stopping each model at the first
out-of-memory failure.

Single-point force cost is the right thing to measure: the engine's own
integration and host staging add ~2% at these sizes (78.2 ms per NVE step against
76.8 ms for a bare force call at 384 atoms), so the model call *is* the step.

Run: python examples/orb/py_scaling_study.py
"""

import time
import warnings
import numpy as np
import ase
import torch

warnings.filterwarnings("ignore")

WATER = ("/net/storage/abbaa90/.julia/dev/ParticleDynamics/examples/mace/"
         "validation/water_init.npz")
MACE_OFF = "/home/abbaa90/.cache/mace/MACE-OFF23_small.model"
REPS = [1, 2, 3, 4, 5, 6, 7, 8]      # k -> 192 ... 98304 atoms
NTIME = 3
DT_FS = 0.5                           # for the ns/day figure


def build(k):
    d = np.load(WATER)
    pos, Z, L = d["positions"], d["numbers"], float(d["L"])
    a = ase.Atoms(numbers=Z, positions=pos, cell=np.diag([L] * 3), pbc=True)
    return a.repeat((k, k, k))


def timeit(atoms, n=NTIME, warm=3):
    # Three warm-up calls, not one: MACE runs through TorchScript and the first
    # call after model load pays JIT compilation, which otherwise contaminates
    # the smallest system in each sweep.
    rs = np.random.RandomState(0)
    base = atoms.get_positions()
    for _ in range(warm):
        atoms.set_positions(base + 0.002 * rs.standard_normal(base.shape))
        atoms.get_forces()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        atoms.set_positions(base + 0.002 * rs.standard_normal(base.shape))
        atoms.get_forces()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def run(label, make_calc):
    print(f"\n=== {label} ===", flush=True)
    print(f"{'atoms':>8} {'H2O':>7} {'box(Å)':>9} {'ms/call':>10} "
          f"{'µs/atom':>9} {'peakGB':>7} {'ns/day':>9}")
    calc = make_calc()
    for k in REPS:
        atoms = build(k)
        n = len(atoms)
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            atoms.calc = calc
            s = timeit(atoms)
            gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            ns_day = (DT_FS * 1e-6) * (86400.0 / s)
            print(f"{n:8d} {n // 3:7d} {atoms.cell[0,0]:9.2f} {s*1000:10.1f} "
                  f"{s*1e6/n:9.2f} {gb:7.2f} {ns_day:9.4f}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).split("\n")[0][:70]
            print(f"{n:8d} {n // 3:7d} {atoms.cell[0,0]:9.2f}   FAILED: {msg}",
                  flush=True)
            torch.cuda.empty_cache()
            break
    del calc
    torch.cuda.empty_cache()


def mace_f32():
    from mace.calculators import mace_off
    return mace_off(model=MACE_OFF, device="cuda", default_dtype="float32")


def orb(name="orb_v3_conservative_inf_omat"):
    def _make():
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        torch.set_default_dtype(torch.float32)
        model, adapter = getattr(pretrained, name)(
            device="cuda", precision="float32-high")
        return ORBCalculator(model, adapter, device="cuda")
    return _make


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    run("MACE-OFF23-small, float32", mace_f32)
    run("Orb-v3-conservative-inf-omat, float32-high",
        orb("orb_v3_conservative_inf_omat"))
