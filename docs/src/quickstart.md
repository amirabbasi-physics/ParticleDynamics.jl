# Quickstart

This page shows the recommended public workflow API. It is the interface normal
examples and user scripts should use.

`ParticleDynamics.jl` is currently CUDA-first, so the workflow still builds a
GPU-resident low-level simulation under the hood. The workflow layer exists to
hide manual `build_simulation(...)`, manual stepping loops, manual GSD open/close
calls, and manual observable plumbing.

## Conceptual model

- `ParticleSystem`: positions, box, types, masses, velocities, topology
- `Group` / `Groups`: particle selections
- `Force`: physical interactions
- `Integrator`: `dt`, scheme, forces, methods, thermostats
- `Observable`: sampled quantities
- `Writer`: scheduled output
- `Stage`: a named block of running
- `Simulation`: assembled workflow object

## Minimal workflow

```julia
using ParticleDynamics

N = 128
cfg = hex_random_2d(N, 1.0f0, 0.25f0; T=Float32)

system = ParticleSystem(
    cfg;
    types=[:A],
    typeids=fill(Int32(1), N),
    masses=Dict(:A => 1.0f0),
)

all = Group(:all, AllSelection())
groups = Groups(all)

force = WCA(
    epsilon=10.0f0,
    sigma=1.0f0,
    cutoff=Float32(2^(1 / 6)),
    pairs=:all,
)

integrator = Integrator(
    dt=2.0f-4,
    scheme=VelocityVerlet(),
    forces=[force],
    methods=[Langevin(all; gamma=50.0f0, kT=1.0f0)],
)

thermo = ThermodynamicObservable(all; name=:thermo)

sim = Simulation(
    system;
    groups=groups,
    integrator=integrator,
    observables=[thermo],
    writers=[
        TableWriter(
            "obs.csv";
            every=50,
            observables=[thermo => [:temperature, :kinetic_energy, :potential_energy]],
            mode=:replace,
        ),
        GSDWriter("traj.gsd"; every=200, group=all, write_start=true, mode=:replace),
    ],
    precision=Float32,
    seed=0xC9A319,
)

run!(sim, Stage(:warmup, steps=200; dt=1.0f-4, neighbor_rebuild_interval=1))
reset_observables!(sim)
reset_step!(sim, 0)
run!(sim, Stage(:production, steps=1_000))
```

## NVE production after equilibration

Pure NVE on the workflow API is expressed as `ConstantVolume(group)` with no
thermostat. A common pattern is to equilibrate with a thermostat, then swap the
integrator and re-prepare before production:

```julia
all = Group(:all, AllSelection())

sim = Simulation(
    system;
    groups=Groups(all),
    integrator=Integrator(
        dt=2.0f-4,
        forces=[force],
        methods=[ConstantVolume(all; thermostat=CSVR(kT=1.0f0, tau=0.1f0))],
    ),
)

run!(sim, Stage(:equil, steps=5_000))

sim.integrator = Integrator(
    dt=2.0f-4,
    forces=[force],
    methods=[ConstantVolume(all)],
)
prepare!(sim)
reset_observables!(sim)
reset_step!(sim, 0)
run!(sim, Stage(:production, steps=20_000))
```

`examples/3D_LJ_NVE.jl` is stricter than this pattern: it runs pure NVE from
step 0 and sets the initial velocities explicitly on the `ParticleSystem`.

## Groups apply methods and observables

Groups are currently particle selections. They are designed so the same
high-level vocabulary can later grow to bonds, angles, or other topology
domains without changing how users think about selection.

```julia
cold = Group(:cold, TypeSelection(:C))
hot = Group(:hot, TypeSelection(:H))
groups = Groups(cold, hot, Group(:all, AllSelection()))
```

Methods and observables attach to those groups:

```julia
Integrator(
    dt=2.0f-4,
    scheme=VelocityVerlet(),
    forces=[force],
    methods=[
        Langevin(cold; gamma=50.0f0, kT=5.0f0),
        Langevin(hot; gamma=50.0f0, kT=100.0f0),
    ],
)
```

## Output is scheduled, not hand-written

Writers own file management and run on schedules:

- `Every(n)`
- `AtSteps([0, 100, 1_000])`
- `Between(first, last; every=...)`

This keeps examples free of manual `gsd_open`, `gsd_close`, `write_gsd_frame!`,
manual CSV header logic, and direct `collect_step_observables(...)` calls.

## Low-level expert API

The older expert API is still supported:

- `build_simulation`
- `step!`
- `step_graph!`
- `nve`, `velocityverlet`, `baoab`, `baoa`, `gsm`, `eulerheun`, `eulermaruyama`, `nosehooverchain`, `csvr`
- `ParticleDynamics.SimulationCore`

Use that surface if you need direct control over the GPU-resident
`SimulationState`, debugging of stepping internals, or custom orchestration that
the workflow layer intentionally does not expose.

## Current limitations

- CUDA is the main supported backend.
- Angles, dihedrals, impropers, electrostatics, and similar force-field
  features are future-ready in the vocabulary but not generally implemented.
- `ForceField` is a stable container concept, but compiled support is limited to
  the existing low-level kernels and wrappers.
