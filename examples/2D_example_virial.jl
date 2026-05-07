using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))

"""
Tiny 2D example that writes the total per-particle configurational virial tensor
into a GSD trajectory.

This uses the `write_virial=true` flag of `GSDWriter`, which stores custom
virial chunks. The component order matches `virial_components(state(sim))`, so
in 2D each row stores `(xx, yy, xy)`.
"""

N = maybe_override_int(256, "SIM_NPARTICLES")
box = (40.0f0, 40.0f0)
r_cut = Float32(2^(1 / 6))
sigma = 1.0f0
epsilon = 10.0f0
dt = 2.0f-4
gamma = 50.0f0
temperature = 1.0f0
nsteps = maybe_override_int(1_000, "SIM_MAX_STEPS")
log_interval = maybe_override_interval(100, nsteps)

logged_steps = sort!(unique(vcat(1, collect(log_interval:log_interval:nsteps))))

system = ParticleSystem(
    square_lattice_positions(N, box);
    box=PeriodicBox(box),
    types=[:A],
    typeids=fill(Int32(1), N),
    masses=Dict(:A => 1.0f0),
)

all_particles = Group(:all, AllSelection())

sim = Simulation(
    system;
    groups=Groups(all_particles),
    integrator=Integrator(
        dt=dt,
        scheme=VelocityVerlet(),
        forces=[WCA(epsilon=epsilon, sigma=sigma, pairs=:neighborlist,
                    neighborlist=CellList(buffer=0.4, capacity=64, rebuild_interval=25))],
        methods=[Langevin(all_particles; gamma=gamma, kT=temperature)],
    ),
    writers=[
        GSDWriter(
            joinpath(@__DIR__, "traj2d_virial.gsd");
            schedule=AtSteps(logged_steps),
            group=all_particles,
            write_start=false,
            mode=:replace,
            diameter=sigma,
            write_virial=true,
        ),
    ],
    precision=Float32,
)

run!(sim, Stage(:production, steps=nsteps; progress=false, max_seconds=maybe_override_runtime()))

frame = ParticleDynamics.read_gsd_frame!(joinpath(@__DIR__, "traj2d_virial.gsd"))
virial = frame.particle_properties[:virial]
println("Wrote trajectory with virial tensors to ", joinpath(@__DIR__, "traj2d_virial.gsd"))
println("Virial component order: ", virial_components(state(sim)))
println("Read back virial matrix with size ", size(virial), " from the last frame")
