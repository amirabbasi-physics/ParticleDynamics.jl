using Test
using Random
using CUDA
using ParticleDynamics
using ParticleDynamics: SimulationCore, BrownianIntegrators, LangevinIntegrators, Filters

CUDA.allowscalar(false)

include("utils.jl")
using .TestUtils
include("params_from_examples.jl")
using .ParamsFromExamples

@testset "ParticleDynamics.jl (GPU)" begin
    if !gpu_required()
        @testset "CUDA unavailable" begin
            @test_skip false
        end
    else
        seed_all!(0xBADC0DE)
        include("test_api.jl")
        include("test_workflow_system.jl")
        include("test_workflow_groups.jl")
        include("test_workflow_forces.jl")
        include("test_workflow_integrators.jl")
        include("test_workflow_observables.jl")
        include("test_workflow_writers.jl")
        include("test_workflow_runloop.jl")
        include("test_workflow_api.jl")
        include("test_workflow_examples_smoke.jl")
        include("test_build.jl")
        include("test_particle_groups.jl")
        include("test_thermostats.jl")
        include("test_forces.jl")
        include("test_gpu_residency.jl")
        include("test_nonbonded_interface.jl")
        include("test_bonded_exclusions.jl")
        include("test_phase4a_forces.jl")
        include("test_phase4a_pair_pbc.jl")
        include("test_phase4a_neighbors.jl")
        include("test_phase4b_stochastic.jl")
        include("test_neighbors.jl")
        include("test_integrators_langevin.jl")
        include("test_integrators_brownian.jl")
        include("test_integrators_nve.jl")
        include("test_integrators_nhc.jl")
        include("test_integrators_csvr.jl")
        include("test_entropy_observables.jl")
        include("test_virial.jl")
        include("test_collisions.jl")
        include("test_io_gsd.jl")
        include("test_ir_phase2.jl")
    end
end
