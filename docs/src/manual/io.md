# Trajectories and tables

Attach writers to a `Simulation` to save measurements and configurations on
independent schedules.

## Save a table and a trajectory

```julia
all = Group(:all, AllSelection())
thermo = ThermodynamicObservable(all; name=:thermo)
writers = [
    TableWriter("thermo.csv"; every=100,
        observables=[thermo => [:temperature, :total_energy]], mode=:replace),
    GSDWriter("trajectory.gsd"; every=1_000, group=all,
        write_start=true, mode=:replace),
]
sim = Simulation(system; groups=Groups(all), integrator=integrator,
                 observables=[thermo], writers=writers)
run!(sim, 10_000)
```

A table is convenient for plotting time series. GSD stores particle configurations
and metadata for visualization and [continuing a simulation](restarts.md).

## Choose when to write

Use `every=100` for a regular interval, or pass an explicit schedule:

```julia
TableWriter("selected.csv";
    schedule=AtSteps([0, 100, 1_000]),
    observables=[thermo => [:temperature]],
)
GSDWriter("production.gsd"; schedule=Between(1_000, 10_000; every=500))
```

## In-memory workflow output

`InMemoryWriter` is a workflow writer like `TableWriter` and `GSDWriter`, but
it keeps frames in host memory, for analysis or debugging in the same Julia
session without file I/O:

```julia
mem = InMemoryWriter(every=100, fields=[:positions, :forces, :energy, :virial])
sim = Simulation(system; integrator=integrator, writers=[mem])
run!(sim, 10_000)
data = recorded_data(mem)
data.step                  # step of each frame
data.positions[:, :, end]  # N × D, rows in particle-id order
data.potential_energy      # one value per frame
empty!(mem)                # drop the recorded frames
```

Each recorded frame copies the requested fields from the device, so record
only what you need at the interval you need.

```@docs
ParticleDynamics.InMemoryWriter
ParticleDynamics.recorded_data
```

## File sessions

- Workflow `TableWriter` and `GSDWriter` use `mode=:replace` to replace a file
  once when their simulation writer session first opens it. Later stages append
  to that session, even though handles are closed at stage boundaries. Calling
  `prepare!` again with the same writer preserves the session. A new `Simulation`
  starts a new session; use `mode=:append` to retain an existing file across
  separate simulations.
- GSD writing stays outside the simulation hot path and is the preferred
  trajectory format for richer restart data.

## Next steps

The [NVE example](../examples.md#Fluids) writes a trajectory and thermodynamic
table together. The [workflow reference](workflow.md#Writers) lists writer
arguments, including GSD force and virial output.
