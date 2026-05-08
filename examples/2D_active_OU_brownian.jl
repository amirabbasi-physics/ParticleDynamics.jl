using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

# Active Ornstein-Uhlenbeck particles in 2D (overdamped limit, Fodor et al. PRL 117, 038103)
# Parameters follow the paper: N=10000, box 250×250, D=100, τ=20, μ=1 (γ=1).

function square_lattice_positions(N::Integer, box::NTuple{2,<:Real})
    n_side = ceil(Int, sqrt(N))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    return [
        (
            (mod(i - 1, n_side) + 0.5) * spacing_x - box[1] / 2,
            (div(i - 1, n_side) + 0.5) * spacing_y - box[2] / 2,
        )
        for i in 1:N
    ]
end

N = maybe_override_int(10_000, "SIM_NPARTICLES")
L = 125.0
dt = 2e-4
nsteps = maybe_override_int(100_000, "SIM_MAX_STEPS")
write_interval = maybe_override_interval(10_000, nsteps)

gamma = 1.0
temperature = 0.0
corr_time = 100.0
active_impulse = 10.0
epsilon = 1.0e6
sigma = 1.0

system = ParticleSystem(
    square_lattice_positions(N, (L, L));
    box=PeriodicBox((L, L)),
    types=[:C],
    typeids=fill(Int32(1), N),
    masses=Dict(:C => 1.0),
)

all_particles = Group(:all, AllSelection())
sim = Simulation(
    system;
    groups=Groups(all_particles),
    integrator=Integrator(
        dt=dt,
        scheme=EulerMaruyama(),
        forces=[SoftRepulsive(epsilon=epsilon, sigma=sigma, cutoff=sigma, pairs=:neighborlist,
                              neighborlist=CellList(buffer=0.5, capacity=256, rebuild_interval=20))],
        methods=[ActiveOrnsteinUhlenbeck(all_particles; gamma=gamma, kT=temperature, tau=corr_time, noise_scale=active_impulse)],
    ),
    writers=[
        GSDWriter(
            joinpath(@__DIR__, "traj2d_active_OU_bd.gsd");
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
println("Finished $(nsteps) BD steps with correlated OU noise (τ=$(corr_time)). GSD: ", joinpath(@__DIR__, "traj2d_active_OU_bd.gsd"))
