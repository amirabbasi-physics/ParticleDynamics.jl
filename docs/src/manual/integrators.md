# Dynamics

Choose a method for the physics and a scheme for the numerical integration.
The `Integrator` combines them with your forces and timestep.

| Dynamics | Method | Scheme |
|:--|:--|:--|
| Microcanonical MD (NVE) | `ConstantVolume(all)` | `VelocityVerlet()` |
| Thermostatted MD (NVT) | `ConstantVolume(all; thermostat=CSVR(kT=1.0, tau=0.1))` | `VelocityVerlet()` |
| Langevin dynamics | `Langevin(all; gamma=1.0, kT=1.0)` | `VelocityVerlet()`, `BAOAB()`, `BAOA()` or `GSM()` |
| Overdamped Brownian dynamics | `Brownian(all; gamma=1.0, kT=1.0)` | `EulerHeun()` or `EulerMaruyama()` |

Here `all = Group(:all, AllSelection())`; `kT` is an energy in your chosen units.

## Molecular dynamics

For NVE, provide initial velocities and leave the system without a thermostat:

```julia
integrator = Integrator(
    dt=0.002,
    scheme=VelocityVerlet(),
    forces=[LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5)],
    methods=[ConstantVolume(all)],
)
```

The [NVE example](../examples.md#Fluids) includes velocity initialization and
energy output. For NVT, add a [thermostat](thermostats.md). For NPT, see
[pressure control](bonded_minimization_npt.md#Isotropic-NPT).

## Brownian dynamics

Use an overdamped method when inertia is not part of your model:

```julia
integrator = Integrator(
    dt=0.00025,
    scheme=EulerMaruyama(),
    forces=[WCA(epsilon=10.0, sigma=1.0)],
    methods=[Brownian(all; gamma=10.0, kT=1.0)],
)
```

Friction must be positive on stochastic paths that divide by it. A timestep
that works for an inertial system need not work for an overdamped one;
check displacement and the observables relevant to your model.

## Active forcing

`ActiveOrnsteinUhlenbeck` adds a correlated stochastic process. Thermal noise
is specified separately through `Brownian` or `Langevin`, so both methods
can act on the same group:

```julia
methods = [
    Brownian(all; gamma=10.0, kT=1.0),
    ActiveOrnsteinUhlenbeck(all; gamma=10.0, tau=1.0, noise_scale=1.0),
]
```

`noise_scale` is the stationary standard deviation of the OU impulse; it is
not a bath temperature. For multiple correlation times, supply an `OUSpectrum`.
The [active-particle examples](../examples.md#Active-particles) include
free-particle displacement checks.

## Choose and change a timestep

Timesteps must be finite and positive. Start conservatively, then compare
runs at smaller timesteps using energy conservation, structural observables
or transport coefficients appropriate to your system.

A stage can override the configured timestep, for example during warmup:

```julia
run!(sim, Stage(:warmup, steps=1_000; dt=0.0001))
run!(sim, Stage(:production, steps=10_000))
```

If you replace `sim.integrator`, call `prepare!(sim)` before continuing.
Changing stochastic settings can rebuild stochastic workspaces; a continued
run is not a promise of bitwise replay of an uninterrupted noise history.

See the [workflow reference](workflow.md#Integrator,-schemes-and-methods) for
constructor arguments.
