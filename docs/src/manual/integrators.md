# Integrators

The package exposes explicit integrator constructors that allocate and own the
workspace needed for repeated stepping. Reuse a spec across many `step!` calls.

## Integrator constructors

```@docs
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
ParticleDynamics.NHCSpec
ParticleDynamics.CSVRSpec
```

## Usage pattern

```julia
vv = velocityverlet(st; gamma=50.0f0, temperature=1.0f0, dt=2.0f-4)
for _ in 1:100
    step!(st, vv, 2.0f-4; compute_energy=false)
end
```

## Notes

- Langevin and Brownian constructors own stochastic buffers.
- `gamma > 0` is required on stochastic paths that divide by friction.
- Thermostat-driven MD paths (`nosehooverchain`, `csvr`) allocate device
  workspaces per bath and remain CUDA-specific internally.

## Example scripts

- `examples/SingleT_2D_LD_VV.jl`
- `examples/TwoT_2D_LD_BAOAB.jl`
- `examples/TwoT_2D_LD_GSM.jl`
- `examples/TwoT_2D_BD_EH.jl`
- `examples/3D_BD.jl`
