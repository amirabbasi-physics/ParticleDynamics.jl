# GROMACS comparison support

| File | Purpose |
|---|---|
| `Protocol.jl` | Input generation and result checks for the GROMACS comparisons (used by `../argon_ensembles/make_states.jl` and `../electrostatics/run_tuned.jl`) |
| `RuntimeInfo.jl` | GPU and runtime information recorded with benchmark results |
| `DeviceArgonNVE.jl` | Compatibility name for the device engine (`ParticleDynamics.DeviceEngine`), used by tests |
| `GPUForceReference.jl` | Independent CUDA force/energy reference used by GPU tests, outside any timing |
