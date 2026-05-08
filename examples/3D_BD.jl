using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

N = maybe_override_int(40_000, "SIM_NPARTICLES")
box = (50.0, 50.0, 50.0)
r_cut = 2^(1 / 6)
sigma = 1.0
epsilon = 10.0
gamma = 10.0
temperature = 1.0
dt = 2.5e-4
nsteps = maybe_override_int(10_000, "SIM_MAX_STEPS")
log_interval = maybe_override_interval(1_000, nsteps)

system = ParticleSystem(
    simple_cubic_positions(N, box);
    box=PeriodicBox(box),
    types=[:C],
    typeids=fill(Int32(1), N),
    masses=Dict(:C => 1.0),
)

all_particles = Group(:all, AllSelection())
groups = Groups(all_particles)
thermo = ThermodynamicObservable(all_particles; name=:all)
bath = BathExchangeObservable(name=:bath)

sim = Simulation(
    system;
    groups=groups,
    integrator=Integrator(
        dt=dt,
        scheme=EulerMaruyama(),
        forces=[WCA(epsilon=epsilon, sigma=sigma, pairs=:neighborlist,
                    neighborlist=CellList(buffer=0.4, capacity=100, rebuild_interval=10))],
        methods=[Brownian(all_particles; gamma=gamma, kT=temperature)],
    ),
    observables=[thermo, bath],
    writers=[
        TableWriter(
            joinpath(@__DIR__, "obs3d_bd.csv");
            every=log_interval,
            observables=[
                thermo => [:kinetic_energy, :potential_energy, :total_energy],
                bath => [:heat],
            ],
            mode=:replace,
        ),
        GSDWriter(
            joinpath(@__DIR__, "traj3d_bd.gsd");
            every=log_interval,
            group=all_particles,
            write_start=true,
            mode=:replace,
            diameter=sigma,
        ),
    ],
    precision=Float32,
)

run!(sim, nsteps)
println("Done. GSD (bd): ", joinpath(@__DIR__, "traj3d_bd.gsd"))
