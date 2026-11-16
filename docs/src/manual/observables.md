# Observables

Choose the quantities you want to measure, then give a writer the sampling
schedule. You can measure the whole system or a named particle group.

## Temperature and energy

```julia
all = Group(:all, AllSelection())
thermo = ThermodynamicObservable(all; name=:thermo)
writer = TableWriter("thermo.csv";
    every=100,
    observables=[thermo => [:temperature, :kinetic_energy,
                           :potential_energy, :total_energy]],
    mode=:replace,
)
```

Pass `observables=[thermo]` and `writers=[writer]` to your `Simulation`.
Use total energy to assess an NVE run; for thermostatted dynamics, energy
exchange with the bath is part of the model.

## Pick a measurement

| Observable | Typical fields | Use it for |
|:--|:--|:--|
| `ThermodynamicObservable(group)` | `:temperature`, `:kinetic_energy`, `:potential_energy`, `:total_energy`, `:virial` | Equilibration and energy checks |
| `VirialObservable(group)` | `:virial`, `:virial_accumulated` | Mechanical diagnostics |
| `MSDObservable(group)` | `:msd`, `:elapsed_time` | Displacement and diffusion |
| `VACFObservable(group)` | `:vacf`, `:elapsed_time` | Velocity correlations |
| `CollisionObservable()` | `:counts`, `:rate`, `:pair_counts`, `:pair_rates` | Encounters between particle types |
| `BathExchangeObservable()` | `:heat`, `:entropy`, `:entropy_production_rate` | Exchange with stochastic baths |

Available bath diagnostics depend on the chosen dynamics. For collision
measurements, follow the interaction and cutoff setup in the
[two-temperature example](../examples.md#Driven-mixtures).

## Start a production interval

```julia
run!(sim, Stage(:equilibrate, steps=1_000))
reset_observables!(sim)
run!(sim, Stage(:production, steps=10_000))
```

`reset_observables!` clears accumulated measurements and sets a new displacement
and correlation reference at the current configuration. It does not erase
output already written to disk. Use [writers](io.md) to choose what to save
and how often.

See the [workflow reference](workflow.md#Observables) for constructor arguments.
