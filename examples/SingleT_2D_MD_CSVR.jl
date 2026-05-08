using ParticleDynamics
using Printf

include(joinpath(@__DIR__, "_example_utils.jl"))

function main(phi::Float64, temperature::Float64)
    n = maybe_override_int(10_000, "SIM_NPARTICLES")
    sigma = 1.0

    cfg = hex_random_2d(n, sigma, phi; T=Float64)
    epsilon = 1e9
    rcut = sigma

    dt = 1e-5
    nsteps = maybe_override_int(20_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_interval(5_000, nsteps)

    warmup_steps = maybe_override_int(10_000, "SIM_WARMUP_STEPS"; lower=0)
    warmup_dt = dt * 0.1
    warmup_neigh_interval = 5
    tau_csvr = 10 * dt

    system = ParticleSystem(
        cfg;
        types=[:C],
        typeids=fill(Int32(1), n),
        masses=Dict(:C => 1.0),
    )
    all_particles, groups = single_particle_groups()

    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=dt,
            forces=[SoftRepulsive(epsilon=epsilon, sigma=sigma, cutoff=rcut, pairs=:neighborlist,
                                  neighborlist=CellList(buffer=0.55, capacity=250, rebuild_interval=20))],
            methods=[ConstantVolume(all_particles; thermostat=CSVR(kT=temperature, tau=tau_csvr))],
        ),
        precision=Float64,
        seed=0xC9A319,
    )

    println("Integrator: molecular dynamics with CSVR thermostat.")
    println(" - Thermostat: canonical stochastic velocity rescaling")
    println(" - Coupling time tau = $(tau_csvr)")

    run_equilibration!(sim;
                       warmup_steps=warmup_steps,
                       warmup_dt=warmup_dt,
                       warmup_neighbor_rebuild_interval=warmup_neigh_interval,
                       progress=false)

    thermo = ThermodynamicObservable(all_particles; name=:all)
    bath = BathExchangeObservable(name=:bath)
    collisions = CollisionObservable(name=:collisions)

    gsd_path = joinpath(@__DIR__, "traj2d_csvr_alpha_$(temperature)_fraction_$(phi).gsd")
    log_path = replace(gsd_path, ".gsd" => ".log")
    rm(gsd_path; force=true)
    rm(log_path; force=true)

    sim.observables = Observable[thermo, bath, collisions]
    sim.writers = Writer[
        TableWriter(
            log_path;
            every=log_interval,
            observables=[
                thermo => [:temperature, :potential_energy, :virial],
                bath => [:heat, :entropy_production_rate],
                collisions => [:counts],
            ],
            mode=:replace,
        ),
        GSDWriter(
            gsd_path;
            every=log_interval,
            group=all_particles,
            write_start=true,
            mode=:replace,
            sync_on_write=true,
            diameter=sigma,
        ),
    ]

    prepare_production!(sim)
    run!(sim, Stage(:production, steps=nsteps; progress=true, max_seconds=maybe_override_runtime()))

    println("Wrote trajectory to $(gsd_path)")
    println("Wrote log to $(log_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 2
        phi = parse(Float64, ARGS[1])
        temperature = parse(Float64, ARGS[2])
        println("Running with phi=$(phi), temperature=$(temperature)")
        main(phi, temperature)
    else
        println("Usage: julia SingleT_2D_MD_CSVR.jl <phi> <temperature>")
        println("  Example: julia SingleT_2D_MD_CSVR.jl 0.85 10000.0")
        exit(1)
    end
end
