# Your first simulation

Start with 128 repulsive particles in a periodic 2D box. A Langevin bath
sets the temperature, and scheduled writers save measurements and a trajectory.
The script uses reduced units with particle mass and diameter equal to one.

## Before you start

These development docs describe version 0.7.0. Use a repository checkout so
this guide and the installed API match:

```sh
git clone https://github.com/amirabbasi-physics/ParticleDynamics.jl.git
cd ParticleDynamics.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

From that directory, check that Julia can use your NVIDIA GPU:

```sh
julia --project -e 'using CUDA; CUDA.functional() || error("A working CUDA installation is required")'
```

For a registered release, use `Pkg.add("ParticleDynamics")` and documentation
that matches that version.

## Run a small fluid

Save the following as `first_simulation.jl` in the checkout and run
`julia --project first_simulation.jl`. The first run includes compilation;
later runs in the same Julia session are faster.

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

## Inspect the output

The script writes two files in your working directory:

| File | Contents | What to look for |
|:--|:--|:--|
| `obs.csv` | Temperature, kinetic energy and potential energy every 50 steps | The kinetic temperature fluctuates around the bath target after equilibration |
| `traj.gsd` | Particle configurations every 200 steps | Motion in the periodic box; open the trajectory with a GSD-compatible viewer |

The warmup uses a smaller timestep, then production uses the integrator's
configured timestep. `reset_observables!` starts fresh measurement intervals;
`reset_step!` restarts step numbering. Writers keep the same output session
across the stages, so the files also contain the warmup records.

This short run checks the workflow. Increase the equilibration and production
lengths before drawing physical conclusions.

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

Use particle types to give different groups their own dynamics or measurements.

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

Choose a schedule for each writer; file opening, sampling and closing are handled
by the simulation. See [trajectories and tables](manual/io.md).

## Try another system

Explore [runnable examples](examples.md) for a 3D fluid, bonded polymers,
active particles and two-temperature mixtures. The
[workflow reference](manual/workflow.md) lists the available objects and arguments.

## Current limitations

- CUDA is the only backend; a CUDA-capable NVIDIA GPU is required.
- Bonded terms cover bonds, angles, dihedrals and impropers. CMAP terms and
  general holonomic constraints beyond rigid three-site water are not
  implemented.
- Long-range electrostatics covers 3D orthorhombic periodic boxes with
  tin-foil boundary conditions on a single GPU. Slab, triclinic and multi-GPU
  electrostatics are not implemented.
- Barostats are isotropic and 3D, see
  [Bonded Terms, Minimization and NPT](manual/bonded_minimization_npt.md) for the
  supported integrator combinations.
