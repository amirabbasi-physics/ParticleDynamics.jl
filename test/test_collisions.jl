@testset "Collision Counting API" begin
    seed_all!(0xC0111)

    st = build_tiny2d(
        N=8, T=Float32, box=(24f0, 24f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
        neigh_interval=2, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0.5f0
    )
    st.typeid .= CuArray(Int32[1, 1, 1, 1, 2, 2, 2, 2])

    ParticleDynamics.enable_collision_counting!(st; ntypes=2, bins=:all_pairs)
    @test st.coll_enabled
    @test st.coll_prev !== nothing
    @test st.coll_counts !== nothing
    @test length(st.coll_counts) == 3
    vv = Simulation.velocityverlet(st; gamma=1f0, temperature=0.5f0, dt=1f-3)

    for _ in 1:4
        Simulation.step!(st, vv, 1f-3; compute_energy=false)
    end
    counts = ParticleDynamics.collisions_read_counts!(st)
    @test length(counts) == 3
    @test all(c -> c >= 0, counts)

    ParticleDynamics.collisions_reset_counts!(st)
    counts_reset = ParticleDynamics.collisions_read_counts!(st)
    @test all(c -> c == 0, counts_reset)

    rcut_pair = Float32[1.0 1.1; 1.1 1.2]
    ParticleDynamics.set_collision_pair_cutoffs!(st, rcut_pair)
    @test st.rcut_pair !== nothing
    @test size(st.rcut_pair) == (2, 2)

    ParticleDynamics.disable_collision_counting!(st)
    @test st.coll_enabled == false
    @test st.coll_prev === nothing
    @test st.coll_counts === nothing
    @test st.coll_bins === nothing
end

@testset "Collision Counting excludes directly bonded pairs" begin
    seed_all!(0xC0112)
    T = Float32

    function _make_state(; bonds=nothing)
        st = Simulation.build_simulation(
            N=2,
            box=(T(10), T(10)),
            cutoff=T(1),
            skin=T(1),
            cap=Int32(8),
            neigh_interval=50,
            use_neighborlist=true,
            epsilon=T(1),
            sigma=T(1),
            gamma=T(0),
            temperature=T(0),
            nonbonded=:wca,
            bonds=bonds,
            bonding=(bonds === nothing ? nothing : harmonic_bond(k=T(300), r0=T(1))),
            precision=:f32,
        )
        st.typeid .= CuArray(Int32[1, 1])

        # Start as neighbors but outside collision cutoff.
        copyto!(st.rx, T[0, 1.5])
        copyto!(st.ry, T[0, 0])
        ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
        ParticleDynamics.enable_collision_counting!(st; ntypes=1, bins=:all_pairs)
        return st
    end

    st_unbonded = _make_state()
    st_bonded = _make_state(bonds=Tuple{Int32,Int32}[(Int32(1), Int32(2))])

    # Move into contact without rebuilding NL so this is an entry event.
    copyto!(st_unbonded.rx, T[0, 0.5])
    copyto!(st_bonded.rx, T[0, 0.5])
    ParticleDynamics.Collisions._collisions_update_after_positions!(st_unbonded)
    ParticleDynamics.Collisions._collisions_update_after_positions!(st_bonded)

    cu = ParticleDynamics.collisions_read_counts!(st_unbonded)
    cb = ParticleDynamics.collisions_read_counts!(st_bonded)
    @test cu == Int64[1]
    @test cb == Int64[0]
end

@testset "Collision Counting 3D pair cutoffs" begin
    seed_all!(0xC0113)
    T = Float64

    st = build_tiny3d(
        N=8, T=T, box=(T(12), T(12), T(12)), cutoff=T(2.5), skin=T(0.3), cap=Int32(32),
        neigh_interval=2, use_neighborlist=true, nonbonded=:lj, gamma=T(1), temperature=T(0.5)
    )

    st.typeid .= CuArray(Int32[1, 1, 1, 1, 2, 2, 2, 2])
    ParticleDynamics.enable_collision_counting!(st; ntypes=2, bins=:all_pairs)
    ParticleDynamics.set_collision_pair_cutoffs!(st, T[1.0 1.1; 1.1 1.2])
    vv = Simulation.velocityverlet(st; gamma=T(1), temperature=T(0.5), dt=T(1e-3))

    for _ in 1:2
        Simulation.step!(st, vv, T(1e-3); compute_energy=false)
    end

    counts = ParticleDynamics.collisions_read_counts!(st)
    @test length(counts) == 3
    @test all(c -> c >= 0, counts)
end
