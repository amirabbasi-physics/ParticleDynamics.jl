# Continue a simulation

Load a saved GSD configuration into a `ParticleSystem`, then supply the forces,
dynamics and output schedule for the next run.

```julia
system = ParticleSystem.from_gsd("trajectory.gsd")  # last saved frame
all = Group(:all, AllSelection())
sim = Simulation(system;
    groups=Groups(all),
    integrator=Integrator(
        dt=0.002,
        forces=[LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5)],
        methods=[ConstantVolume(all)],
    ),
    writers=[GSDWriter("continued.gsd"; every=1_000, mode=:replace)],
)
run!(sim, Stage(:continued, steps=10_000))
```

The Lennard-Jones settings above are an example: use the interactions, units,
precision and timestep of your original model. Loading a configuration does
not reconstruct every simulation setting or an uninterrupted stochastic history.

## What is restored

`ParticleSystem.from_gsd` reads particle data and topology stored in the frame,
including velocities and charges when available. For charged systems,
`electrostatics_from_gsd("trajectory.gsd")` returns the saved electrostatics
descriptor, or `nothing` when none was recorded. Add that descriptor to the
new integrator's forces.

For pressure-controlled runs, prepare the simulation with the same barostat
settings and restore its recorded state:

```julia
prepare!(sim)
restore_barostat!(sim, read_gsd_log("trajectory.gsd", "log/barostat/state"))
```

See [pressure control](bonded_minimization_npt.md) for the supported combinations.

## Keep or extend existing output

A new output filename keeps the continuation separate and makes the two runs
easy to inspect. To append to an existing file across separate `Simulation`
objects, use `mode=:append`. Step numbering is managed separately through
`reset_step!`; choose it explicitly if you need a continuous output timeline.
