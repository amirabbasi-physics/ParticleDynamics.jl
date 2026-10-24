# Shared benchmark helpers

Helpers used by the benchmark suites in `../argon_ensembles`, `../molten_salt`,
`../water` and `../electrostatics`. They run no simulation physics of their own.

| File | Purpose |
|---|---|
| `metadata.jl` | `BenchmarkMetadata`: run provenance (source and input hashes, GPU and runtime information), resolved settings and the shared observation loop used by the ParticleDynamics drivers |
| `phase_metadata.jl` | Optional device phase counters read by `metadata.jl` (diagnostics only) |
| `explicit_list.py` | Opt-in `NSTLIST`/`RLIST`/`VERLET_BUFFER_TOLERANCE` edits of a GROMACS `.mdp`; without these variables the file is left unchanged |
| `audit.py` | Fail-closed comparison of benchmark result records (settings, cadences, provenance) |
| `gromacs_work.py`, `gromacs_topology.py` | Readers for GROMACS cycle counters and `gmx dump` topology metadata, used by `audit.py` |

A timing produced by these suites is a single measurement. Speed claims should use repeated,
alternating runs of the compared variants on the same GPU, with a confidence interval.
