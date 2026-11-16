# Temperature control

Use a thermostat to sample at a target temperature, or a Langevin method to
model friction and stochastic forcing. Temperature inputs are expressed as
`kT` in energy units.

## Stochastic velocity rescaling

```julia
all = Group(:all, AllSelection())
integrator = Integrator(
    dt=0.002,
    forces=[LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5)],
    methods=[ConstantVolume(all; thermostat=CSVR(kT=1.0, tau=0.1))],
)
```

`tau` sets the thermostat response time. The
[single-temperature example](../examples.md#Driven-mixtures) combines CSVR
with thermodynamic measurements and collision counting.

## Nosé–Hoover chains

To use a deterministic chain thermostat, replace the method above with:

```julia
ConstantVolume(all; thermostat=NoseHooverChain(
    kT=1.0, tau=0.1, chain_length=5, substeps=4,
))
```

Monitor equilibration and temperature fluctuations before collecting production
averages. A target temperature does not make every instantaneous measurement equal
to that target.

## Langevin baths

For Langevin dynamics, specify friction and temperature on the method itself:

```julia
Langevin(all; gamma=10.0, kT=1.0)
```

Select a compatible [integration scheme](integrators.md). Different
[particle groups](groups_filters.md) can have different bath parameters;
see the [two-temperature mixture](../examples.md#Driven-mixtures).

## Equilibrate, then run NVE

After equilibration, replace the integrator's methods with
`ConstantVolume(all)` and call `prepare!(sim)`. The
[first simulation](../quickstart.md#NVE-production-after-equilibration)
shows the complete pattern.

The [workflow reference](workflow.md#Integrator,-schemes-and-methods) documents
`CSVR`, `NoseHooverChain` and `Langevin`.
