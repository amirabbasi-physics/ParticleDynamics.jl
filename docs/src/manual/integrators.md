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

## Timestep consistency

`step!` and `step_graph!` require a finite positive timestep. For Langevin and
Brownian specs, it must equal the timestep supplied to their constructor (after
conversion to the simulation precision). Noise amplitudes and OU coefficients
are cached for that timestep. A mismatch throws `ArgumentError` before stepping
changes state or consumes random numbers. The low-level EM and Langevin
wrappers that consume cached stochastic parameters enforce the same contract.
For static forces and virials, use
`ParticleDynamics.SimulationCore.evaluate_forces_into_f!(st, true)`; zero-duration
steps are rejected.

```julia
dt = 0.001f0
spec = eulermaruyama(st; gamma=1, temperature=1, dt=dt)
step!(st, spec, dt)
```

To change the timestep, construct a new spec with all desired friction,
temperature, and OU settings. This starts a new stochastic workspace; it does
not preserve the previous OU realization. Temperature and OU setters require
the existing spec's timestep, including filtered updates. Raw parameter
constructors that omit `dt` default to one; pass `dt` explicitly when wrapping
these parameters in a spec or passing them to `step!`.

`Filters.set_friction!` controls friction independently of the noise amplitude.
To maintain a chosen thermal temperature after changing friction, also call
`Filters.set_temperature!` for the same particles and the spec's timestep.

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
