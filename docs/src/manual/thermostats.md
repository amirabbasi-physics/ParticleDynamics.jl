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
```

## Runtime construction

Thermostat-backed integrator specs are built through:

- `nosehooverchain(st; ...)`
- `csvr(st; ...)`

Those constructors remain the recommended entrypoints for stepping.

## Example scripts

- `examples/TwoT_SR2D_MD_CSVR_slab.jl`
- `examples/SingleT_2D_MD_NHC.jl` if present in your local tree

The thermostat examples remain GPU-first. Bath-control logic is a likely later
candidate for CPU-compatible validation only after kernel portability work.
