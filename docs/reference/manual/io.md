# I/O

Writers and trajectory helpers are kept at the top level so user scripts can
write diagnostics without importing internal modules.

## Public writer API

```@docs
ParticleDynamics.CSVWriter
ParticleDynamics.XYZWriter
ParticleDynamics.write_xyz!
ParticleDynamics.write_observables_csv!
ParticleDynamics.gsd_open
ParticleDynamics.gsd_close
ParticleDynamics.write_gsd_frame!
ParticleDynamics.read_gsd_log
```

## In-memory workflow output

`InMemoryWriter` is a workflow writer like `TableWriter` and `GSDWriter`, but
it keeps frames in host memory, for analysis or debugging in the same Julia
session without file I/O:

```julia
mem = InMemoryWriter(every=100, fields=[:positions, :forces, :energy, :virial])
sim = Simulation(system; integrator=integrator, writers=[mem])
run!(sim, 10_000)
data = recorded_data(mem)
data.step                  # step of each frame
data.positions[:, :, end]  # N × D, rows in particle-id order
data.potential_energy      # one value per frame
empty!(mem)                # drop the recorded frames
```

Each recorded frame copies the requested fields from the device, so record
only what you need at the interval you need.

```@docs
ParticleDynamics.InMemoryWriter
ParticleDynamics.recorded_data
```

## GSD metadata types

```@docs
ParticleDynamics.Writers.GSDFrameData
ParticleDynamics.Writers.GSDTopology
```

## Notes

- Workflow `TableWriter` and `GSDWriter` use `mode=:replace` to replace a file
  once when their simulation writer session first opens it. Later stages append
  to that session, even though handles are closed at stage boundaries. Calling
  `prepare!` again with the same writer preserves the session. A new `Simulation`
  starts a new session; use `mode=:append` to retain an existing file across
  separate simulations.
- XYZ and CSV helpers stage data to host for output.
- GSD writing stays outside the simulation hot path and is the preferred
  trajectory format for richer restart data.

## Example scripts

- `examples/3D_LJ_NVE.jl`: `TableWriter` and `GSDWriter`
- `examples/2D_example_virial.jl`: per-particle virials in GSD frames, read back
  with `read_gsd_frame!`
