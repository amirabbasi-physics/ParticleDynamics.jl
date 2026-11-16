# Workflow API

The workflow layer describes a simulation with objects: a particle system,
groups, forces, an integrator with methods, observables and writers.
`run!` owns the time loop. The [Quickstart](../../src/quickstart.md) shows a complete
script; this page is the reference for each object.

## Simulation and system

```@docs
ParticleDynamics.Simulation
ParticleDynamics.ParticleSystem
ParticleDynamics.Particles
ParticleDynamics.PeriodicBox
ParticleDynamics.Topology
ParticleDynamics.prepare!
ParticleDynamics.state
```

## Groups

```@docs
ParticleDynamics.Group
ParticleDynamics.Groups
ParticleDynamics.AllSelection
ParticleDynamics.TypeSelection
ParticleDynamics.IndexSelection
```

## Nonbonded forces

```@docs
ParticleDynamics.Force
ParticleDynamics.ForceField
ParticleDynamics.LennardJones
ParticleDynamics.WCA
ParticleDynamics.SoftRepulsive
ParticleDynamics.PairTable
ParticleDynamics.CellList
```

`ClusterPairList` is described under [Forces](forces.md). Bonded force
descriptors are listed under
[Bonded Terms, Minimization and NPT](bonded_minimization_npt.md), and
long-range Coulomb forces under [Electrostatics](electrostatics.md).

## Integrator, schemes and methods

```@docs
ParticleDynamics.Integrator
ParticleDynamics.VelocityVerlet
ParticleDynamics.BAOAB
ParticleDynamics.BAOA
ParticleDynamics.GSM
ParticleDynamics.EulerHeun
ParticleDynamics.EulerMaruyama
ParticleDynamics.Method
ParticleDynamics.ConstantVolume
ParticleDynamics.Langevin
ParticleDynamics.Brownian
ParticleDynamics.ActiveOrnsteinUhlenbeck
ParticleDynamics.OUSpectrum
ParticleDynamics.Thermostat
ParticleDynamics.CSVR
ParticleDynamics.NoseHooverChain
```

Barostats are passed as `Integrator(...; barostat=...)`, see
[Bonded Terms, Minimization and NPT](bonded_minimization_npt.md).

## Running

```@docs
ParticleDynamics.run!
ParticleDynamics.Stage
ParticleDynamics.reset_step!
ParticleDynamics.reset_observables!
```

## Schedules

```@docs
ParticleDynamics.Every
ParticleDynamics.AtSteps
ParticleDynamics.Between
```

## Observables

```@docs
ParticleDynamics.Observable
ParticleDynamics.ThermodynamicObservable
ParticleDynamics.VirialObservable
ParticleDynamics.MSDObservable
ParticleDynamics.VACFObservable
ParticleDynamics.CollisionObservable
ParticleDynamics.BathExchangeObservable
```

## Writers

```@docs
ParticleDynamics.Writer
ParticleDynamics.TableWriter
ParticleDynamics.GSDWriter
```

`InMemoryWriter` and the low-level writers are described under [I/O](io.md).
