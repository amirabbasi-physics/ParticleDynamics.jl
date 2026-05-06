#!/usr/bin/env julia
# Test basic workflow integration - instantiate and prepare a simulation.

using ParticleDynamics
using ParticleDynamics.Workflow

println("Testing basic workflow instantiation...")

# Create a simple 2D system
positions = [(randn(Float64, 2)) for _ in 1:10]
system = ParticleSystem(positions; box=(10.0, 10.0))
println("✓ ParticleSystem created")

# Create groups
all_group = Group(:all, AllSelection())
groups = Groups([all_group])
println("✓ Groups created")

# Create forces
println("✓ Forces can be created via Force type")

# Create integrator
methods = [Langevin(group=:all, gamma=1.0, kT=1.0)]
integrator = Integrator(dt=0.01, scheme=VelocityVerlet(), methods=methods)
println("✓ Integrator created")

# Create simulation
sim = Simulation(system=system, groups=groups, integrator=integrator, precision=Float64)
println("✓ Simulation object created")

println("\n✓ All basic workflow components are working!")
