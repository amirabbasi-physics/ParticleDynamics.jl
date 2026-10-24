# Benchmarks

Head-to-head comparisons of ParticleDynamics.jl with GROMACS (and, for water and
electrostatics, LAMMPS and OpenMM) on a single GPU. The suites provide shared starting
states and controls for matching models and settings. Reserve the GPU for the entire
comparison and reject any sample that overlaps another GPU job, including one that
starts after the sample begins.

| Directory | What it compares | Entry points |
|---|---|---|
| `argon_ensembles/` | Lennard-Jones argon, NVE/NVT/NPT, 10k and 100k atoms | `make_states.jl` → `run_all.sh` (speed), `obs_all.sh` + `compare_obs.py` (accuracy) |
| `molten_salt/` | Molten NaCl (Joung–Cheatham) with PME or ESP, NVE/NVT/NPT | `make_states.sh` → `match_accuracy.jl` → `run_all.sh` |
| `water/` | Rigid SPC/E water NPT, 10k–500k molecules, at matched Coulomb force accuracy | `make_states.sh` → `match_accuracy.jl`, `tune_*.jl/sh` → `run_all.sh` → `analyze.py`, `rdf.jl` |
| `electrostatics/` | Frozen-coordinate Coulomb forces (PME, PPPM, ESP) | see `electrostatics/README.md` |
| `gromacs/` | Protocol and runtime helpers; force references used by the GPU tests | see `gromacs/README.md` |
| `common/` | Run provenance, settings audit and GROMACS readers shared by the suites | see `common/README.md` |

External engine binaries are set through environment variables (`GMX`, `GMX_GPU`, `GMX_ESP`,
`LMP_ESP`, …); their defaults, at the top of each script, are local install paths to override. The profiling
drivers (`water/prof_*.{jl,sh}`, `electrostatics/profile_engine_*.jl`) wrap the same runs for
Nsight Systems.

A single timing is one measurement. Speed claims should use repeated, alternating runs of
the compared variants on the same GPU, with a confidence interval. Keep at least five
valid paired rounds after exclusions, and report the actual list policy and resolved
solver settings for each engine. Select settings in separate tuning runs before the
final comparison.

Force evaluation timings and complete simulation steps measure different work. For
simulation throughput, include list maintenance, integration, coupling and the same
requested observations and output in each timed interval. A same-coordinate force comparison
checks discretization and list errors; it does not establish NVE energy drift or NPT
density agreement. Validate those properties separately for the production settings,
using predefined error margins and uncertainty estimates.
