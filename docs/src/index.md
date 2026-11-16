# ParticleDynamics

```@raw html
<div class="pd-hero">
<p class="pd-eyebrow">MOLECULAR DYNAMICS · SOFT MATTER · ACTIVE PARTICLES</p>
<p class="pd-headline">Small particles.<br>Rich dynamics.</p>
<p class="pd-intro">Explore fluids, polymers and driven systems with GPU particle
simulations in Julia. Define the physics, choose what to measure, and follow your
system from its initial configuration to a trajectory you can analyse.</p>
</div>
<div class="pd-paths">
```

- **[Run your first simulation →](quickstart.md)**
  Build a small fluid and save its temperature, energy and trajectory.
- **[Explore the examples →](examples.md)**
  Start from a working script for molecular, Brownian or active dynamics.
- **[Find a workflow object →](manual/workflow.md)**
  Look up the arguments for systems, forces, integrators and writers.

```@raw html
</div>
```

## What will you simulate?

| A system to explore | A place to start |
|:--|:--|
| **Fluids** — conservative, thermostatted or overdamped dynamics | [Lennard-Jones and Brownian examples](examples.md#Fluids) |
| **Polymers** — bonded chains, melts and relaxation | [Polymer examples](examples.md#Polymers) |
| **Active matter** — correlated forcing, thermal noise and diffusion | [Active-particle examples](examples.md#Active-particles) |
| **Charged systems** — periodic Coulomb interactions with PME, PPPM or ESP | [Electrostatics guide](manual/electrostatics.md) |
| **Driven mixtures** — particle groups coupled to different baths | [Two-temperature example](examples.md#Driven-mixtures) |

## Install and begin

Simulations run on a **CUDA-capable NVIDIA GPU**. Use Julia 1.10 or later with a
working CUDA installation, then install the registered package:

```julia
using Pkg
Pkg.add("ParticleDynamics")
```

These are the development docs for **0.7.0**; the registered release may differ.
The [first simulation](quickstart.md) walks through the setup and a complete run.
All physical quantities use a consistent set of units chosen by you.

## From a model to a measurement

A `ParticleSystem` holds your initial configuration. Forces describe its
interactions; an `Integrator` sets the dynamics and timestep. Add observables
and writers, assemble a `Simulation`, and run named stages for equilibration
and production.

```julia
# With your system, integrator and writers defined:
sim = Simulation(system; integrator=integrator, writers=writers)
run!(sim, Stage(:equilibrate, steps=1_000))
reset_observables!(sim)
run!(sim, Stage(:production, steps=10_000))
```

Learn how to [choose interactions](manual/forces.md),
[control temperature](manual/thermostats.md), and
[record measurements](manual/observables.md).

## Supported systems

ParticleDynamics supports 2D and 3D periodic systems on a single GPU, including
Lennard-Jones, WCA and soft-repulsive interactions, bonded models, Langevin and
Brownian dynamics, and active Ornstein–Uhlenbeck forcing. Energy minimization
and isotropic pressure control are also available.

Long-range electrostatics requires a 3D orthorhombic periodic box. Pressure
control has specific integrator requirements, described in the
[bonds, minimization and pressure guide](manual/bonded_minimization_npt.md).
General holonomic constraints beyond rigid three-site water, triclinic
electrostatics and multi-GPU execution are not implemented.
