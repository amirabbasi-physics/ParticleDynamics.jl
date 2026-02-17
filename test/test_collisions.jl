@testset "Collision Counting API" begin
    seed_all!(0xC0111)

    st = build_tiny2d(
        N=8, T=Float32, box=(24f0, 24f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
        neigh_interval=2, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0.5f0
    )
    st.typeid .= CuArray(Int32[1, 1, 1, 1, 2, 2, 2, 2])

    NonEqSimGPU.enable_collision_counting!(st; ntypes=2, bins=:all_pairs)
    @test st.coll_enabled
    @test st.coll_prev !== nothing
    @test st.coll_counts !== nothing
    @test length(st.coll_counts) == 3

    for _ in 1:4
        Simulation.step!(st, Simulation.velocityverlet(st), 1f-3; compute_energy=false)
    end
    counts = NonEqSimGPU.collisions_read_counts!(st)
    @test length(counts) == 3
    @test all(c -> c >= 0, counts)

    NonEqSimGPU.collisions_reset_counts!(st)
    counts_reset = NonEqSimGPU.collisions_read_counts!(st)
    @test all(c -> c == 0, counts_reset)

    rcut_pair = Float32[1.0 1.1; 1.1 1.2]
    NonEqSimGPU.set_collision_pair_cutoffs!(st, rcut_pair)
    @test st.rcut_pair !== nothing
    @test size(st.rcut_pair) == (2, 2)

    NonEqSimGPU.disable_collision_counting!(st)
    @test st.coll_enabled == false
    @test st.coll_prev === nothing
    @test st.coll_counts === nothing
    @test st.coll_bins === nothing
end
