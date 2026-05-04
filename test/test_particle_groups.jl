using ParticleDynamics
using ParticleDynamics.ParticleGroups
using CUDA

@testset "ParticleGroups: Selections" begin
    # Create a simple test state
    N = 100
    st = build_simulation(N=N, box=(50.0f0, 50.0f0, 50.0f0),
                         dt=1.0f-3, precision=:f32)
    
    # Test All selection
    sel_all = ParticleGroups.All()
    group_all = ParticleGroups.materialize(sel_all, st)
    @test ParticleGroups.count(group_all) == N
    @test length(group_all.host) == N
    @test length(group_all.device) == N
    
    # Test Indices selection
    indices = [1, 5, 10, 50, 99]
    sel_idx = ParticleGroups.Indices(indices)
    group_idx = ParticleGroups.materialize(sel_idx, st)
    @test ParticleGroups.count(group_idx) == length(indices)
    @test group_idx.host == sort(indices)
    
    # Test TypeIDs selection (all particles should have type 1 by default)
    sel_type = ParticleGroups.TypeIDs(1)
    group_type = ParticleGroups.materialize(sel_type, st)
    @test ParticleGroups.count(group_type) == N
end

@testset "ParticleGroups: Operations" begin
    N = 50
    st = build_simulation(N=N, box=(40.0f0, 40.0f0, 40.0f0),
                         dt=1.0f-3, precision=:f32)
    
    sel = ParticleGroups.Indices([1, 5, 10])
    group = ParticleGroups.materialize(sel, st)
    
    # Test apply_scalar! on device array
    test_array = CUDA.zeros(Float32, N)
    ParticleGroups.apply_scalar!(test_array, group, 2.5f0)
    result = Array(test_array)
    @test result[1] ≈ 2.5f0
    @test result[5] ≈ 2.5f0
    @test result[10] ≈ 2.5f0
    @test result[2] ≈ 0.0f0
    
    # Test apply_values! on device array
    test_array2 = CUDA.zeros(Float32, N)
    values = [1.0f0, 2.0f0, 3.0f0]
    ParticleGroups.apply_values!(test_array2, group, values)
    result2 = Array(test_array2)
    @test result2[1] ≈ 1.0f0
    @test result2[5] ≈ 2.0f0
    @test result2[10] ≈ 3.0f0
    
    # Test gather
    gathered = ParticleGroups.gather(test_array2, group)
    @test gathered ≈ values
    
    # Test sum_values
    total = ParticleGroups.sum_values(test_array2, group)
    @test total ≈ sum(values)
end

println("ParticleGroups tests passed.")
