# MACE foundation models in ParticleDynamics.jl

This directory connects the engine to **MACE machine-learned interatomic
potentials** (MACE-MP-0 for materials, MACE-OFF for organic systems) through
the package's external-potential interface. The MLIP replaces every classical
force term — no bonds, angles, dihedrals, or electrostatics are configured;
the trained potential carries the chemistry.

```julia
include("MACEPotential.jl")   # provider (needs PythonCall + mace-torch)

pot = MACEPotential(atomic_numbers, (L, L, L); variant=:mp, model="small")
attach_external_potential!(st, pot)
step!(st, nve(st; dt), dt)    # NVE/NHC/CSVR/Langevin all work
```

## Units and conventions

Everything runs in MACE-native units: **Å, eV, amu**. The derived time unit
is t* = Å·√(amu/eV) ≈ 10.18 fs, so dt = 1 fs = 0.098226 t*;
kB = 8.617333e-5 eV/K. Engine positions live in **[-L/2, L/2)** (ASE accepts
them as-is with pbc=true). Float64 end to end.

## Environment setup (once)

```bash
python3 -m venv ~/.venvs/pd-mace
~/.venvs/pd-mace/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/.venvs/pd-mace/bin/pip install mace-torch ase
source ~/.venvs/pd-mace/bin/activate
# MACE-OFF checkpoint (the in-process auto-download can fail):
curl -sL -o ~/.cache/mace/MACE-OFF23_small.model \
  https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model
julia --project=examples/mace -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Pick the torch CUDA wheel matching your driver (cu128 for CUDA 12.8/12.9
drivers). Exact package versions used here: `requirements.txt`. The Julia
scripts use the active `python3`; set `PARTICLEDYNAMICS_PYTHON` to override it.

## Validation suite (Si216, MACE-MP-0 small, float64)

Run order: `py_reference.py` → `bridge_smoke.jl` → `force_fidelity_check.jl` →
`py_md_reference.py` → `v3_trajectory_check.jl` → `validate_mace_nve.jl`.

| check | result | criterion |
|---|---|---|
| V0 bridge fidelity (`bridge_smoke.jl`) | max\|ΔF\| = 1.6e-15 eV/Å | < 1e-12 |
| V1 forces through the engine path (`force_fidelity_check.jl`) | max\|ΔF\| = 1.1e-14 eV/Å, \|ΔE\| = 2.3e-13 eV | < 1e-12 |
| V2 NVE energy conservation, 10 ps (`validate_mace_nve.jl`) | rel. drift 4.4e-7 (max \|dev\| 1.4e-6); −2.3e-10 eV/step/atom | < 1e-5 relative |
| V3 trajectory equivalence vs ASE velocity Verlet, 199 steps (`v3_trajectory_check.jl`) | max dev = 4.3e-5 Å | ≪ bond length |

Plots: `py_make_plots.py` renders `validation/v2_energy.png` and
`validation/water_rdf.png`.

## Showcase: liquid water with MACE-OFF

`water_rdf_showcase.jl` — 64 H₂O (192 atoms) at 0.997 g/cm³, a system this
engine cannot describe classically (it has no water force field): gentle
start (0.1 fs, strong friction) → 2 ps BAOAB Langevin equilibration at
300 K → 10 ps NVE production, O–O radial distribution function vs the
experimental x-ray landmarks of Skinner et al., J. Chem. Phys. 138, 074506
(2013).

Result (10 ps, 1000 sampled frames): **first peak g_OO = 2.62 at
r = 2.83 Å**, against the experimental 2.57 at 2.80 Å — structure agreement
within ~2% from a foundation model with zero system-specific tuning. NVE
production energy drift: 4.2e-8 relative over 20,000 steps. Plot:
`validation/water_rdf.png`.

`water_movie_gsd.jl` continues from the saved final state and writes a
HOOMD-schema GSD trajectory for visualization in OVITO.

## Driver benchmark

`mlip_benchmark.jl` compares ParticleDynamics and ASE while holding the MACE
model, configuration, precision, GPU, warm-up, and number of timed steps
fixed. On an RTX 3090 (Float64, 300 timed steps):

| system and model | ParticleDynamics | ASE VelocityVerlet | ratio |
|---|---:|---:|---:|
| Si216, MACE-MP-0 small | 12.28 steps/s | 12.09 steps/s | 1.02x |
| 64 H₂O (192 atoms), MACE-OFF small | 17.46 steps/s | 17.55 steps/s | 0.99x |

The nearly identical rates show that MACE inference dominates runtime in
this correctness-first implementation; the Julia driver and host staging add
no measurable overhead at these system sizes.

## Limitations (deliberate, documented)

- External potentials replace **all** internal force terms; bond-free states
  and `spatial_reorder=false` are required (enforced at attach time).
- No virial from the provider → pressure observables unsupported.
- Positions round-trip through the host each step (numpy staging). At these
  system sizes model inference dominates; the copy is irrelevant. Zero-copy
  (DLPack) and batched-graph paths are future work.
- All PythonCall usage stays on the main Julia thread (GIL).

## Roadmap

Package extension (`ext/`) promoting `MACEPotential` into the package proper;
ACEpotentials.jl as a second, Julia-native backend behind the same interface
(via AtomsCalculators.jl); zero-copy tensor exchange; NPT once a virial
contract exists.
