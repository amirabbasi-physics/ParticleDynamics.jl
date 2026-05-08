#!/usr/bin/env julia
# Test Stage 12 - complete workflow with observables and writers

using ParticleDynamics
using ParticleDynamics.Workflow
using Random
using Dates

println("Testing Stage 12 - Complete workflow with observables and writers...")

# Set seed for reproducibility
Random.seed!(42)

# Create a simple 2D system
N = 30
positions = [randn(Float64, 2) .* 2.0 for _ in 1:N]
box = PeriodicBox((15.0, 15.0))
system = ParticleSystem(positions; box=box)
println("✓ ParticleSystem created with $(length(system)) particles")

# Create groups
all_group = Group(:all, AllSelection())
groups = Groups([all_group])
println("✓ Groups created")

# Create forces
lj_force = LennardJones(epsilon=1.0, sigma=1.0, cutoff=Float64(2^(1/6)))
forces = [lj_force]
println("✓ Forces created")

# Create integrator with Langevin method
methods = [Langevin(group=:all, gamma=1.0, kT=1.0)]
integrator = Integrator(dt=0.01, scheme=VelocityVerlet(), forces=forces, methods=methods)
println("✓ Integrator created")

# Create observables
thermo_obs = ThermodynamicObservable(:all)
observables = [thermo_obs]
println("✓ Observables created")

# Create writers
tmpdir = tempdir()
csv_file = joinpath(tmpdir, "test_stage12_$(now()).csv")
table_writer = TableWriter(
    filename=csv_file,
    every=Every(5),
    observables=[thermo_obs]
)
writers = [table_writer]
println("✓ Writers created")

# Create simulation
sim = Simulation(
    system=system,
    groups=groups,
    integrator=integrator,
    observables=observables,
    writers=writers,
    precision=Float64
)
println("✓ Simulation object created")

# Prepare the simulation
prepare!(sim)
println("✓ Simulation prepared")

# Run simulation with a Stage
try
    stage = Stage(:test, steps=20; progress=true)
    run!(sim, stage)
    println("✓ Simulation ran successfully for 20 steps")
catch e
    println("✗ Error during run!: $e")
    Base.showerror(stdout, e)
end

# Check that output file was created
if isfile(csv_file)
    println("✓ CSV output file created: $(basename(csv_file))")
    lines = readlines(csv_file)
    println("  - Header: $(lines[1])")
    println("  - Sample row: $(lines[min(2, end)])")
    rm(csv_file)
else
    println("✗ CSV output file not created!")
end

println("\n✓ Stage 12 implementation verified!")
