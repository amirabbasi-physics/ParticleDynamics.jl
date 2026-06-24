# Integrators

The package exposes explicit integrator constructors that allocate and own the
workspace needed for repeated stepping. Reuse a spec across many `step!` calls.

## Integrator constructors

```@docs
ParticleDynamics.nve
ParticleDynamics.velocityverlet
ParticleDynamics.baoab
ParticleDynamics.baoa
ParticleDynamics.gsm
ParticleDynamics.eulerheun
ParticleDynamics.eulermaruyama
ParticleDynamics.nosehooverchain
ParticleDynamics.csvr
```

## Public spec types

```@docs
ParticleDynamics.AbstractIntegratorSpec
ParticleDynamics.VVSpec
ParticleDynamics.BAOABSpec
ParticleDynamics.BAOASpec
ParticleDynamics.GSMSpec
ParticleDynamics.BrownianSpec
ParticleDynamics.EMSpec
ParticleDynamics.NVESpec
ParticleDynamics.NHCSpec
ParticleDynamics.CSVRSpec
```

## Usage pattern

```julia
nve_spec = nve(st; dt=2.0f-4)
for _ in 1:100
    step!(st, nve_spec, 2.0f-4; compute_energy=false)
end
```

## Notes

- `nve` is the deterministic microcanonical MD path.
- Langevin and Brownian constructors own stochastic buffers.
- `gamma > 0` is required on stochastic paths that divide by friction.
- Thermostat-driven MD paths (`nosehooverchain`, `csvr`) allocate device
  workspaces per bath and remain CUDA-specific internally.

## Example scripts

- `examples/2D_example.jl`
- `examples/TwoT_2D_LD_BAOAB.jl`
- `examples/TwoT_2D_LD_GSM.jl`
- `examples/TwoT_2D_BD_EH.jl`
- `examples/3D_BD.jl`
- `examples/3D_LJ_NVE.jl`
