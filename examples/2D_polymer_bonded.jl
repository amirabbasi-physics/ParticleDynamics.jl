using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

N = maybe_override_int(200, "SIM_NPARTICLES")
box = (200.0, 50.0)
r_cut = 2^(1 / 6)
sigma = 1.0
epsilon = 1.0
gamma = 1.0
temperature = 1.0
dt = 1.0e-5
nsteps = maybe_override_int(1_000_000, "SIM_MAX_STEPS")
log_interval = maybe_override_interval(10_000, nsteps)

bonds = [(Int32(i), Int32(i + 1)) for i in 1:(N - 1)]
use_fene = maybe_override_bool(true, "SIM_USE_FENE")

system = ParticleSystem(
    linear_chain_positions(N; spacing=0.97, origin=(-0.5 * box[1] + 1.0, 0.0));
    box=PeriodicBox(box),
    types=[:C],
    typeids=fill(Int32(1), N),
    masses=Dict(:C => 1.0),
    topology=Topology(bonds=bonds),
)

all_particles, groups = single_particle_groups()
thermo = ThermodynamicObservable(all_particles; name=:all)

forces = Force[
    WCA(epsilon=epsilon, sigma=sigma, pairs=:neighborlist,
        neighborlist=CellList(buffer=r_cut / 2, capacity=96, rebuild_interval=100)),
    use_fene ? FENEBondForce(k=300.0, R0=1.5) : HarmonicBondForce(k=300.0, r0=1.0),
]

sim = Simulation(
    system;
    groups=groups,
    integrator=Integrator(
        dt=dt,
        scheme=VelocityVerlet(),
        forces=forces,
        methods=[Langevin(all_particles; gamma=gamma, kT=temperature)],
    ),
    observables=[thermo],
    writers=[
        TableWriter(
            joinpath(@__DIR__, "obs2d_polymer.csv");
            every=log_interval,
            observables=[thermo => [:kinetic_energy, :potential_energy, :total_energy]],
            mode=:replace,
        ),
        GSDWriter(
            joinpath(@__DIR__, "polymer2d.gsd");
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
println("Done. GSD: ", joinpath(@__DIR__, "polymer2d.gsd"))
