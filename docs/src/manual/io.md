# I/O

Writers and trajectory helpers are kept at the top level so user scripts can
write diagnostics without importing internal modules.

## Public writer API

```@docs
ParticleDynamics.InMemoryLogger
ParticleDynamics.CSVWriter
ParticleDynamics.XYZWriter
ParticleDynamics.write_xyz!
ParticleDynamics.write_observables_csv!
ParticleDynamics.gsd_open
ParticleDynamics.gsd_close
ParticleDynamics.write_gsd_frame!
```

## GSD metadata types

```@docs
ParticleDynamics.Writers.GSDFrameData
ParticleDynamics.Writers.GSDTopology
```

## Notes

- XYZ and CSV helpers stage data to host for output.
- GSD writing stays outside the simulation hot path and is the preferred
  trajectory format for richer restart data.

## Example scripts

- `examples/3D_quicktest.jl`
- `examples/2D_allpairs_quicktest.jl`

These remain GPU-first examples. The small quicktests are the best future
CPU-compatible candidates once a CPU backend becomes real.
