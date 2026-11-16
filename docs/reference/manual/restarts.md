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

- `examples/3D_LJ_NVE.jl` writes a GSD trajectory
- `examples/2D_example_virial.jl` writes GSD frames and reads one back with
  `read_gsd_frame!`
