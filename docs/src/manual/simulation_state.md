# Simulation State

`SimulationState` is the long-lived, GPU-resident container returned by
`build_simulation`. It owns particle arrays, neighbor data, observables
buffers, and optional restart/output metadata for a single simulation.

## Core API

```@docs
ParticleDynamics.SimulationState
ParticleDynamics.build_simulation
ParticleDynamics.sync_unwrapped!
ParticleDynamics.step!
ParticleDynamics.step_graph!
```

## What stays in the state

- wrapped positions, velocities, and forces
- optional unwrapped positions
- neighbor-list storage
- bonded and nonbonded parameter storage
- per-particle observables and virial buffers

The current implementation remains GPU-first: state arrays are `CuArray` based,
and the execution path remains CUDA-specific even though build-time backend
selection is now routed through the storage layer.

## Minimal pattern

```julia
using ParticleDynamics

st = build_simulation(
    N=64,
    box=(40.0f0, 40.0f0),
    cutoff=Float32(2^(1/6)),
    skin=0.4f0,
    cap=Int32(32),
    epsilon=10.0f0,
    sigma=1.0f0,
    gamma=50.0f0,
    temperature=1.0f0,
    nonbonded=:wca,
)
```

## Example scripts

- `examples/2D_allpairs_quicktest.jl`: smallest force-path smoke example, and a likely future CPU-compatible smoke candidate once a CPU backend exists.
- `examples/3D_quicktest.jl`: small 3D stepping/output example, also a likely future CPU-compatible smoke candidate.
- `examples/SingleT_2D_LD_VV.jl`: larger production-style Langevin setup.
