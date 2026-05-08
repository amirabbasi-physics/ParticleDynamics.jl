using ParticleDynamics
using ParticleDynamics: Filters
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

@testset "Filters compatibility via ParticleGroups" begin
    N = 12
    st = build_simulation(N=N, box=(30.0f0, 30.0f0, 30.0f0),
                          dt=1.0f-3, precision=:f32)
    st.typeid .= CuArray(Int32[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

    f_type = Filters.TypeIDs(1)
    pg_type = ParticleGroups.TypeIDs(1)

    filter_host = Filters.resolve(f_type, st)
    group_host = ParticleGroups.resolve(pg_type, st)
    @test filter_host == group_host

    filter_dev = Array(Filters.resolve_gpu(f_type, st))
    group_dev = Array(ParticleGroups.resolve_gpu(pg_type, st))
    @test filter_dev == group_dev

    filter_sel = Filters.selection(st, f_type)
    group_sel = ParticleGroups.materialize(pg_type, st)
    @test filter_sel isa Filters.Selection
    @test filter_sel.host == group_sel.host
    @test Array(filter_sel.device) == Array(group_sel.device)
    @test Filters.count(f_type, st) == ParticleGroups.count(pg_type, st)
    @test Filters.count(filter_sel) == ParticleGroups.count(group_sel)

    dest_filters = CUDA.zeros(Float32, N)
    dest_groups = CUDA.zeros(Float32, N)
    Filters.assign_scalar!(dest_filters, st, f_type, 2.5f0)
    ParticleGroups.apply_scalar!(dest_groups, group_sel, 2.5f0)
    @test Array(dest_filters) == Array(dest_groups)

    vals = Float32.(1:ParticleGroups.count(group_sel))
    dest_filters_vals = CUDA.zeros(Float32, N)
    dest_groups_vals = CUDA.zeros(Float32, N)
    Filters.assign_values!(dest_filters_vals, st, f_type, vals)
    ParticleGroups.apply_values!(dest_groups_vals, group_sel, vals)
    @test Array(dest_filters_vals) == Array(dest_groups_vals)
    @test Filters.gather(dest_filters_vals, st, f_type) == ParticleGroups.gather(dest_groups_vals, group_sel)
    @test Filters.sum(dest_filters_vals, st, f_type) ≈ ParticleGroups.sum_values(dest_groups_vals, group_sel)

    f_idx = Filters.Indices([2, 4, 8, 10])
    pg_idx = ParticleGroups.Indices([2, 4, 8, 10])
    @test Filters.resolve(f_idx, st) == ParticleGroups.resolve(pg_idx, st)
    @test Array(Filters.resolve_gpu(f_idx, st)) == Array(ParticleGroups.resolve_gpu(pg_idx, st))
end
