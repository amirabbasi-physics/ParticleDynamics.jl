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

## Neighbor initialization and capacity

The low-level constructor allocates a dense neighbor container but leaves
`st.nbh.valid == false` until positions have been assigned and a rebuild
succeeds. The first simulation force evaluation/step builds the list; callers
using standalone force kernels must first call
`NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2)`
(include `st.rz` and use `st.box3` in 3D). Workflow preparation initializes the
coordinates and builds the list automatically.

Dense and stencil rows use **slot-major ELL** storage: zero-based neighbor slot
`t` for one-based particle `i` is stored at `t*N + i`. The `neighbors_index`
field remains for compatibility but is not an ELL row offset. Bond adjacency
uses a separate CSR representation.

`cap` is a hard capacity. A build that needs more space throws
`NeighborLists.NeighborCapacityError` and leaves the list invalid, with
`required_capacity` recording the largest required row. Incomplete lists are
rejected by force and collision entry points. Rebuild a new container with
sufficient capacity, or correct the configuration and rebuild successfully.
Rows are never silently truncated. Direct mutation of internal buffers does
not maintain these invariants and is unsupported.

`valid` records successful construction and capacity checking, not freshness
after arbitrary coordinate changes. Low-level callers that move particles
must still rebuild or use the neighbor displacement-check protocol.

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
- `examples/TwoT_2D_LD_VV.jl`: larger production-style Langevin setup on the workflow API.
