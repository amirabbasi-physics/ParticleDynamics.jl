using ParticleDynamics
using Printf

include(joinpath(@__DIR__, "_example_utils.jl"))

function make_melt_bonds(n_chains::Int, chain_length::Int)
    bonds = Tuple{Int32,Int32}[]
    sizehint!(bonds, n_chains * max(chain_length - 1, 0))
    for chain in 0:(n_chains - 1)
        offset = chain * chain_length
        for bead in 1:(chain_length - 1)
            push!(bonds, (Int32(offset + bead), Int32(offset + bead + 1)))
        end
    end
    return bonds
end

function polymer_melt_positions_3d(n_chains::Int, chain_length::Int;
                                   bond_length::Real=1.05,
                                   chain_spacing::Real=1.25)
    n_chains > 0 || throw(ArgumentError("n_chains must be positive."))
    chain_length > 1 || throw(ArgumentError("chain_length must be at least 2."))

    T = promote_type(typeof(float(bond_length)), typeof(float(chain_spacing)))
    n_rows = ceil(Int, sqrt(Float64(n_chains)))
    n_layers = cld(n_chains, n_rows)

    box = (
        T(chain_length) * T(bond_length),
        T(n_rows) * T(chain_spacing),
        T(n_layers) * T(chain_spacing),
    )

    x_left = -T(0.5) * box[1] + T(0.5) * T(bond_length)
    x_right = T(0.5) * box[1] - T(0.5) * T(bond_length)
    positions = Vector{NTuple{3,T}}(undef, n_chains * chain_length)

    idx = 1
    for chain in 0:(n_chains - 1)
        row = mod(chain, n_rows)
        layer = div(chain, n_rows)
        y = -T(0.5) * box[2] + (T(row) + T(0.5)) * T(chain_spacing)
        z = -T(0.5) * box[3] + (T(layer) + T(0.5)) * T(chain_spacing)
        reverse_x = isodd(row + layer)

        for bead in 0:(chain_length - 1)
            x = reverse_x ?
                (x_right - T(bead) * T(bond_length)) :
                (x_left + T(bead) * T(bond_length))
            positions[idx] = (x, y, z)
            idx += 1
        end
    end

    return positions, box
end

function main()
    n_chains = maybe_override_int(64, "SIM_NCHAINS")
    chain_length = maybe_override_int(25, "SIM_CHAIN_LENGTH")
    total_beads = n_chains * chain_length

    sigma = maybe_override_float(1.0, "SIM_SIGMA"; lower=1.0e-6)
    epsilon = maybe_override_float(1.0, "SIM_EPSILON"; lower=0.0)
    bond_length = maybe_override_float(1.05 * sigma, "SIM_BOND_LENGTH"; lower=1.0e-6)
    chain_spacing = maybe_override_float(1.25 * sigma, "SIM_CHAIN_SPACING"; lower=1.0e-6)
    gamma = maybe_override_float(5.0, "SIM_GAMMA"; lower=1.0e-6)
    temperature = maybe_override_float(1.0, "SIM_TEMPERATURE"; lower=0.0)
    dt = maybe_override_float(2.0e-5, "SIM_DT"; lower=1.0e-8)

    warmup_steps = maybe_override_int(5_000, "SIM_WARMUP_STEPS"; lower=0)
    relax_steps = maybe_override_int(10_000, "SIM_RELAX_STEPS"; lower=0)
    nsteps = maybe_override_int(50_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_interval(2_500, nsteps)

    buffer = maybe_override_float(0.4, "SIM_NEIGHBOR_BUFFER"; lower=0.0)
    capacity = maybe_override_int(192, "SIM_NEIGHBOR_CAPACITY")
    rebuild_interval = maybe_override_int(10, "SIM_NEIGHBOR_REBUILD")
    use_fene = maybe_override_bool(false, "SIM_USE_FENE")

    positions, box = polymer_melt_positions_3d(
        n_chains,
        chain_length;
        bond_length=bond_length,
        chain_spacing=chain_spacing,
    )

    bonds = make_melt_bonds(n_chains, chain_length)
    system = ParticleSystem(
        positions;
        box=PeriodicBox(box),
        types=[:C],
        typeids=fill(Int32(1), total_beads),
        masses=Dict(:C => 1.0),
        topology=Topology(bonds=bonds),
    )

    all_particles = Group(:all, AllSelection())
    thermo = ThermodynamicObservable(all_particles; name=:all)
    bath = BathExchangeObservable(name=:bath)

    # Harmonic bonds with r0=bond_length are the most robust default for a dense start.
    bond_force = use_fene ?
        FENEBondForce(k=300.0, R0=1.5 * sigma) :
        HarmonicBondForce(k=300.0, r0=bond_length)

    run_tag = @sprintf("nc%d_len%d", n_chains, chain_length)
    obs_path = joinpath(@__DIR__, "obs3d_polymer_melt_$(run_tag).csv")
    gsd_path = joinpath(@__DIR__, "traj3d_polymer_melt_$(run_tag).gsd")

    sim = Simulation(
        system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=dt,
            scheme=VelocityVerlet(),
            forces=[
                WCA(
                    epsilon=epsilon,
                    sigma=sigma,
                    cutoff=2^(1 / 6) * sigma,
                    pairs=:neighborlist,
                    neighborlist=CellList(
                        buffer=buffer,
                        capacity=capacity,
                        rebuild_interval=rebuild_interval,
                    ),
                ),
                bond_force,
            ],
            methods=[Langevin(all_particles; gamma=gamma, kT=temperature)],
        ),
        observables=[thermo, bath],
        writers=[
            TableWriter(
                obs_path;
                every=log_interval,
                observables=[
                    thermo => [:temperature, :kinetic_energy, :potential_energy, :total_energy, :virial],
                    bath => [:heat],
                ],
                mode=:replace,
            ),
            GSDWriter(
                gsd_path;
                every=log_interval,
                group=all_particles,
                write_start=true,
                mode=:replace,
                diameter=sigma,
                write_unwrapped=true,
            ),
        ],
        precision=Float64,
        seed=0x3D501A,
    )

    monomer_volume_fraction = total_beads * (pi / 6) * sigma^3 / prod(box)
    println("3D polymer melt example")
    println(" - chains = $(n_chains), beads/chain = $(chain_length), total beads = $(total_beads)")
    println(" - box = $(box)")
    @printf(" - approximate monomer volume fraction = %.4f\n", monomer_volume_fraction)
    println(" - bond potential = ", use_fene ? "FENE" : "harmonic")

    run_stage_sequence!(
        sim;
        warmup_steps=warmup_steps,
        warmup_dt=0.25 * dt,
        warmup_neighbor_rebuild_interval=1,
        relax_steps=relax_steps,
        production_steps=nsteps,
        progress=false,
        max_seconds=maybe_override_runtime(),
    )

    println("Wrote trajectory to ", gsd_path)
    println("Wrote observables to ", obs_path)
end

main()
