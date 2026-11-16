# Thermostats

Thermostat objects describe bath structure and expose state/parameter accessors
that can be used from diagnostics or higher-level workflows.

## Public thermostat API

```@docs
ParticleDynamics.AbstractThermostat
ParticleDynamics.ThermostatState
ParticleDynamics.NoseHooverChainThermostat
ParticleDynamics.CSVRThermostat
ParticleDynamics.n_baths
ParticleDynamics.target_temperature
ParticleDynamics.response_time
ParticleDynamics.set_target_temperature!
ParticleDynamics.set_response_time!
ParticleDynamics.cumulative_energy_exchange
ParticleDynamics.NHCParams
ParticleDynamics.CSVRParams
ParticleDynamics.Thermostats
```

## Runtime construction

Thermostat-backed integrator specs are built through:

- `nosehooverchain(st; ...)`
- `csvr(st; ...)`

Those constructors remain the recommended entrypoints for stepping.

## Example scripts

- `examples/SingleT_2D_MD_CSVR.jl`: one CSVR bath
- `examples/mace/asi_melt_quench.jl`: Nosé–Hoover chain with a machine-learned
  potential
