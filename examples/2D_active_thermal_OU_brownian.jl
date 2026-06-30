using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

# Thermal active Ornstein-Uhlenbeck particles in 2D (overdamped limit).
# Brownian provides the thermal white noise; ActiveOrnsteinUhlenbeck provides
# the separate correlated active forcing on the same particles.
function main()
    N = maybe_override_int(10_000, "SIM_NPARTICLES")
    L = 125.0
    dt = 2.0e-7
    nsteps = maybe_override_int(100_000, "SIM_MAX_STEPS")
    write_interval = maybe_override_interval(10_000, nsteps)

    gamma = 8057.06
    temperature = 1.0
    corr_time = 10.0
    active_impulse = 5.0
    epsilon = 1.0e9
    sigma = 1.0

    system = ParticleSystem(
        square_lattice_positions(N, (L, L));
        box=PeriodicBox((L, L)),
        types=[:C],
        typeids=fill(Int32(1), N),
        masses=Dict(:C => 0.0),
    )

    all_particles = Group(:all, AllSelection())
    sim = Simulation(
        system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=dt,
            scheme=EulerHeun(),
            forces=[SoftRepulsive(epsilon=epsilon, sigma=sigma, cutoff=sigma, pairs=:neighborlist,
                                neighborlist=CellList(buffer=0.5, capacity=256, rebuild_interval=20))],
            methods=[
                Brownian(all_particles; gamma=gamma, kT=temperature),
                ActiveOrnsteinUhlenbeck(all_particles; gamma=gamma, tau=corr_time, noise_scale=active_impulse),
            ],
        ),
        writers=[
            GSDWriter(
                joinpath(@__DIR__, "traj2d_active_thermal_OU_bd.gsd");
                every=write_interval,
                group=all_particles,
                write_start=true,
                mode=:replace,
                diameter=1.0,
            ),
        ],
        precision=Float64,
    )

    run!(sim, nsteps)
    println("Finished $(nsteps) BD steps with thermal white noise (kT=$(temperature)) and active OU noise (τ=$(corr_time)). GSD: ", joinpath(@__DIR__, "traj2d_active_thermal_OU_bd.gsd"))
end

main()