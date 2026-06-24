using ParticleDynamics
using Random

include(joinpath(@__DIR__, "_example_utils.jl"))

function initial_velocities_3d(n::Integer, initial_temperature::Real; mass::Real=1.0, rng=Random.default_rng())
    vx = randn(rng, n)
    vy = randn(rng, n)
    vz = randn(rng, n)

    vx .-= sum(vx) / n
    vy .-= sum(vy) / n
    vz .-= sum(vz) / n

    current_temperature = mass * sum(vx .^ 2 .+ vy .^ 2 .+ vz .^ 2) / (3 * n)
    scale = sqrt(initial_temperature / current_temperature)
    vx .*= scale
    vy .*= scale
    vz .*= scale

    return collect(zip(vx, vy, vz))
end

function main()
    seed = 0x3D110E
    rng = Random.MersenneTwister(seed)

    n = maybe_override_int(4_000, "SIM_NPARTICLES")
    rho = maybe_override_float(0.80, "SIM_DENSITY"; lower=1.0e-6)
    initial_temperature = maybe_override_float(1.0, "SIM_INITIAL_TEMPERATURE"; lower=1.0e-6)
    sigma = 1.0
    epsilon = 1.0
    mass = 1.0
    rcut = 2.5 * sigma
    neighbor_buffer = 0.5
    neighbor_capacity = 256
    neighbor_rebuild_interval = 10
    dt = maybe_override_float(2.0e-3, "SIM_DT"; lower=1.0e-8)

    nsteps = maybe_override_int(20_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_interval(1_000, nsteps)

    ϕ = π * rho / 6
    cfg = fcc_random_3d(n, sigma, ϕ; T=Float64, rng=rng)
    minimum(cfg.box) >= 3 * (rcut + neighbor_buffer) ||
        error("3D_LJ_NVE requires min(box) >= 3*(cutoff + neighbor buffer) = $(3 * (rcut + neighbor_buffer)) so the dense cell-list grid has at least 3 cells per dimension. Increase SIM_NPARTICLES or lower SIM_DENSITY for small smoke runs.")

    system = ParticleSystem(
        cfg;
        velocities=initial_velocities_3d(n, initial_temperature; mass=mass, rng=rng),
        types=[:A],
        typeids=fill(Int32(1), n),
        masses=Dict(:A => mass),
    )

    all_particles, groups = single_particle_groups()
    thermo = ThermodynamicObservable(all_particles; name=:all)
    lj_force = LennardJones(
        epsilon=epsilon,
        sigma=sigma,
        cutoff=rcut,
        pairs=:neighborlist,
        neighborlist=CellList(buffer=neighbor_buffer, capacity=neighbor_capacity, rebuild_interval=neighbor_rebuild_interval),
    )

    traj_path = joinpath(@__DIR__, "traj3d_lj_nve.gsd")
    log_path = replace(traj_path, ".gsd" => ".log")
    rm(traj_path; force=true)
    rm(log_path; force=true)

    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=dt,
            forces=[lj_force],
            methods=[ConstantVolume(all_particles)],
        ),
        observables=[thermo],
        writers=[
            TableWriter(
                log_path;
                every=log_interval,
                observables=[thermo => [:temperature, :kinetic_energy, :potential_energy, :total_energy, :virial]],
                mode=:replace,
            ),
            GSDWriter(
                traj_path;
                every=log_interval,
                group=all_particles,
                write_start=true,
                mode=:replace,
                sync_on_write=true,
                diameter=sigma,
            ),
        ],
        precision=Float64,
    )

    println("3D Lennard-Jones pure NVE example")
    println(" - density rho = $(rho)")
    println(" - initial kinetic temperature = $(initial_temperature)")
    println(" - ensemble: NVE from step 0")

    run!(sim, Stage(:production, steps=nsteps; progress=false, max_seconds=maybe_override_runtime()))

    println("Wrote NVE trajectory to $(traj_path)")
    println("Wrote production log to $(log_path)")
end

main()
