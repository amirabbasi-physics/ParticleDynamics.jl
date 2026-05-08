using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

function main()
    n = maybe_override_int(10_000, "SIM_NPARTICLES")
    ϕ = 0.7
    sigma = 1.0
    cfg = hex_random_2d(n, sigma, ϕ; T=Float64)
    epsilon = 10.0
    rcut = 2.5 * sigma
    dt = 1e-4
    gamma = 1 / (2 * 100 * dt)
    warmup_steps = maybe_override_int(1_000, "SIM_WARMUP_STEPS"; lower=0)
    init_steps = maybe_override_int(1_000, "SIM_INIT_STEPS"; lower=0)
    relax_steps = maybe_override_int(0, "SIM_RELAX_STEPS"; lower=0)
    nsteps = maybe_override_int(10_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_interval(1_000, nsteps)
    t_cold = 10000.0 / 500
    t_hot = 10000.0

    typeids = binary_typeids(n; cold_fraction=0.5, seed=0x5A17)
    system = ParticleSystem(cfg; types=[:C, :H], typeids=typeids, masses=Dict(:C => 1.0, :H => 1.0))

    cold, hot, all_particles, groups = two_type_particle_groups()
    thermo_all = ThermodynamicObservable(all_particles; name=:all)
    thermo_cold = ThermodynamicObservable(cold; name=:cold)
    thermo_hot = ThermodynamicObservable(hot; name=:hot)
    bath = BathExchangeObservable(name=:bath)
    collisions = CollisionObservable(name=:collisions)

    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=dt,
            scheme=VelocityVerlet(),
            forces=[LennardJones(epsilon=epsilon, sigma=sigma, cutoff=rcut, pairs=:neighborlist,
                                 neighborlist=CellList(buffer=0.5, capacity=100, rebuild_interval=10))],
            methods=[
                Langevin(cold; gamma=gamma, kT=t_cold),
                Langevin(hot; gamma=gamma, kT=t_hot),
            ],
        ),
        observables=[thermo_all, thermo_cold, thermo_hot, bath, collisions],
        writers=[
            TableWriter(joinpath(@__DIR__, "traj2d_lj_ld.log"); every=log_interval,
                        observables=[thermo_all => [:temperature, :kinetic_energy, :potential_energy, :total_energy, :virial],
                                     thermo_cold => [:kinetic_energy, :potential_energy, :virial],
                                     thermo_hot => [:kinetic_energy, :potential_energy, :virial],
                                     bath => [:heat, :entropy_production_rate],
                                     collisions => [:pair_rates]],
                        mode=:replace),
            GSDWriter(joinpath(@__DIR__, "traj2d_lj_ld.gsd"); every=log_interval, group=all_particles,
                      write_start=true, mode=:replace, diameter=sigma),
        ],
        precision=Float64,
    )

    run_equilibration!(sim;
                       warmup_steps=warmup_steps,
                       warmup_dt=dt * 0.1,
                       warmup_neighbor_rebuild_interval=1,
                       init_steps=init_steps,
                       relax_steps=relax_steps,
                       progress=false)
    prepare_production!(sim)
    run!(sim, Stage(:production, steps=nsteps; progress=false, max_seconds=maybe_override_runtime()))
end

main()
