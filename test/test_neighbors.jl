@testset "Neighbor Lists" begin
    seed_all!(0xD1001)

    st = build_tiny2d(
        N=16, T=Float32, box=(30f0, 30f0), cutoff=2.5f0, skin=0.4f0,
        use_neighborlist=true, nonbonded=:wca
    )
    @test st.nbh isa ParticleDynamics.NeighborLists.NeighborMatrix{Float32}

    needed0 = ParticleDynamics.NeighborLists.update_needed!(
        st.nbh, st.rx, st.ry;
        skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=st.step
    )
    @test needed0 == false

    rx = Array(st.rx)
    rx[1] += 0.8f0
    copyto!(st.rx, rx)
    needed1 = ParticleDynamics.NeighborLists.update_needed!(
        st.nbh, st.rx, st.ry;
        skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=st.step + 1
    )
    @test needed1 == true

    ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step + 1)
    @test st.nbh.last_build_step == st.step + 1

    st_ap = build_tiny2d(N=8, T=Float32, use_neighborlist=false, nonbonded=:wca)
    @test st_ap.nbh isa ParticleDynamics.NeighborLists.AllPairsNeighborMatrix{Float32}
    @test ParticleDynamics.NeighborLists.update_needed!(
        st_ap.nbh, st_ap.rx, st_ap.ry;
        skin=0.4f0, Lx=st_ap.box2[1], Ly=st_ap.box2[2], step=0
    ) == false
end
