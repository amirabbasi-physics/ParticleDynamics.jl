# Particle groups

A group selects particles for a dynamical method, a measurement or trajectory
output. Name the group once and reuse it throughout the simulation.

## Select all particles or a subset

```julia
all = Group(:all, AllSelection())
cold = Group(:cold, TypeSelection(:C))
hot = Group(:hot, TypeSelection(:H))
tracers = Group(:tracers, IndexSelection([1, 5, 9]))
groups = Groups(all, cold, hot, tracers)
```

Type names refer to the `types` supplied to `ParticleSystem`. Indices identify
particles in the input ordering. Add the groups to `Simulation(...; groups=groups)`.

## Give each type a bath

```julia
integrator = Integrator(
    dt=0.0002,
    scheme=VelocityVerlet(),
    forces=[WCA(epsilon=10.0, sigma=1.0)],
    methods=[
        Langevin(cold; gamma=50.0, kT=5.0),
        Langevin(hot; gamma=50.0, kT=100.0),
    ],
)
```

The [two-temperature example](../examples.md#Driven-mixtures) contains a
complete configuration and separate measurements for each bath.

## Measure a subset

```julia
cold_thermo = ThermodynamicObservable(cold; name=:cold)
hot_thermo = ThermodynamicObservable(hot; name=:hot)
```

Pass these observables to a `TableWriter` to compare the groups. For charged
systems, virial measurements require the whole system because the reciprocal
virial has no per-particle decomposition; see [electrostatics](electrostatics.md).

Group constructors are listed in the [workflow reference](workflow.md#Groups).
