@testset "Build and State Layout" begin
    seed_all!(0xB1001)

    st2 = build_tiny2d(N=9, T=Float32, nonbonded=:wca, unwrapped_positions=true)
    @test st2.rz === nothing
    @test st2.vz === nothing
    @test st2.fz === nothing
    @test st2.box2 !== nothing
    @test st2.box3 === nothing
    @test st2.rx_unwrap !== nothing
    @test st2.ry_unwrap !== nothing
    Simulation.sync_unwrapped!(st2)
    @test msd_2d(st2.rx, st2.ry, st2.rx_unwrap, st2.ry_unwrap) <= 1e-12

    st3 = build_tiny3d(N=8, T=Float32, nonbonded=:lj, unwrapped_positions=true)
    @test st3.rz !== nothing
    @test st3.vz !== nothing
    @test st3.fz !== nothing
    @test st3.box2 === nothing
    @test st3.box3 !== nothing
    @test st3.rz_unwrap !== nothing
    Simulation.sync_unwrapped!(st3)
    @test msd_3d(st3.rx, st3.ry, st3.rz, st3.rx_unwrap, st3.ry_unwrap, st3.rz_unwrap) <= 1e-12

    st_dense = build_tiny2d(N=8, T=Float32, use_neighborlist=true)
    @test st_dense.nbh isa ParticleDynamics.NeighborLists.NeighborMatrix{Float32}

    st_allpairs = build_tiny2d(N=8, T=Float32, use_neighborlist=false)
    @test st_allpairs.nbh isa ParticleDynamics.NeighborLists.AllPairsNeighborMatrix{Float32}
end

@testset "WCA cutoff override" begin
    seed_all!(0xB1002)

    sigma = 1.3f0
    st = Simulation.build_simulation(
        N=32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32), neigh_interval=5,
        epsilon=0.5f0, sigma=sigma, gamma=1f0, temperature=1f0, nonbonded=:wca
    )
    factor = Float32(1.122462048309373)
    expected = sigma * factor
    @test isapprox(st.pair_lj.rcut, expected; rtol=1f-6)
    @test isapprox(st.nbh.cutoff, expected; rtol=1f-6)
    @test isapprox(st.rcut_factor * sigma, expected; rtol=1f-6)
end
