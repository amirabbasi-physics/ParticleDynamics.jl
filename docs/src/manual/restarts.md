# Restarts

Restart support currently centers on reading GSD frames back into a structured
host-side container that can be used to rebuild a simulation.

## Public restart API

```@docs
ParticleDynamics.read_gsd_frame!
```

## What `read_gsd_frame!` returns

`read_gsd_frame!` returns a `GSDFrameData` object that:

- preserves legacy tuple-style destructuring for older scripts
- exposes particle arrays, topology, and metadata through named fields
- keeps restart parsing outside the hot CUDA stepping path

The metadata container types themselves are documented on the [I/O](io.md)
page:

- `ParticleDynamics.Writers.GSDFrameData`
- `ParticleDynamics.Writers.GSDTopology`

## Minimal pattern

```julia
frame = read_gsd_frame!("traj.gsd")
@show frame.step frame.N frame.box
```

## Related scripts

- `examples/3D_quicktest.jl` for write-side GSD usage
- `examples/restart_from_gsd.jl` for the high-level restart workflow wrapper
- restart-oriented user workflows should treat current examples as GPU-first;
  later CPU-compatible restart smoke tests can be added after a CPU backend exists
