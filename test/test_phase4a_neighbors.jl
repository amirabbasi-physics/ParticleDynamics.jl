@testset "Phase 4A: Neighbor Backend Parity and Rebuild Trigger" begin
    seed_all!(0x4A0303)

    @testset "AllPairs vs Dense parity (LJ, small N)" begin
        N = 16
        cutoff = 2.8
        skin = 0.4
        box = (32.0, 32.0)

        st_dense = build_tiny2d(
            N=N, T=Float64, box=box, cutoff=cutoff, skin=skin, cap=Int32(64),
            use_neighborlist=true, nonbonded=:lj, epsilon=1.3, sigma=1.0, gamma=1.0, temperature=0.0
        )
        st_allpairs = build_tiny2d(
            N=N, T=Float64, box=box, cutoff=cutoff, skin=skin, cap=Int32(64),
            use_neighborlist=false, nonbonded=:lj, epsilon=1.3, sigma=1.0, gamma=1.0, temperature=0.0
        )

        rx = Array(st_dense.rx)
        ry = Array(st_dense.ry)
        set_positions_2d!(st_allpairs, rx, ry)

        CUDA.fill!(st_dense.fx, 0.0); CUDA.fill!(st_dense.fy, 0.0); CUDA.fill!(st_dense.Epot, 0.0)
        CUDA.fill!(st_allpairs.fx, 0.0); CUDA.fill!(st_allpairs.fy, 0.0); CUDA.fill!(st_allpairs.Epot, 0.0)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st_dense.rx, st_dense.ry, st_dense.fx, st_dense.fy, st_dense.Epot, st_dense.nbh, st_dense.box2, st_dense.pair_lj)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st_allpairs.rx, st_allpairs.ry, st_allpairs.fx, st_allpairs.fy, st_allpairs.Epot, st_allpairs.nbh, st_allpairs.box2, st_allpairs.pair_lj)

        fdx = Array(st_dense.fx) .- Array(st_allpairs.fx)
        fdy = Array(st_dense.fy) .- Array(st_allpairs.fy)
        @test maximum(abs.(fdx)) <= 1e-11
        @test maximum(abs.(fdy)) <= 1e-11
        @test isapprox(Float64(CUDA.sum(st_dense.Epot)), Float64(CUDA.sum(st_allpairs.Epot)); rtol=1e-12, atol=1e-12)
    end

    @testset "Dense vs Stencil parity (uniform cutoff)" begin
        N = 16
        cutoff = 2.6
        skin = 0.4
        box = (30.0, 30.0)

        st_dense = build_tiny2d(
            N=N, T=Float64, box=box, cutoff=cutoff, skin=skin, cap=Int32(64),
            use_neighborlist=true, nonbonded=:lj, epsilon=1.0, sigma=1.0, gamma=1.0, temperature=0.0
        )
        st_stencil = build_tiny2d(
            N=N, T=Float64, box=box, cutoff=cutoff, skin=skin, cap=Int32(64),
            use_neighborlist=true, nonbonded=:lj, epsilon=1.0, sigma=1.0, gamma=1.0, temperature=0.0
        )
        set_positions_2d!(st_stencil, Array(st_dense.rx), Array(st_dense.ry))

        rcut_particle = fill(Float64(cutoff), N)
        st_stencil.nbh = NonEqSimGPU.NeighborLists.build_neighbors_stencil!(
            st_stencil.rx, st_stencil.ry; box=st_stencil.box2, rcut_particle=rcut_particle, cap=Int32(64), skin=skin
        )

        CUDA.fill!(st_dense.fx, 0.0); CUDA.fill!(st_dense.fy, 0.0); CUDA.fill!(st_dense.Epot, 0.0)
        CUDA.fill!(st_stencil.fx, 0.0); CUDA.fill!(st_stencil.fy, 0.0); CUDA.fill!(st_stencil.Epot, 0.0)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st_dense.rx, st_dense.ry, st_dense.fx, st_dense.fy, st_dense.Epot, st_dense.nbh, st_dense.box2, st_dense.pair_lj)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st_stencil.rx, st_stencil.ry, st_stencil.fx, st_stencil.fy, st_stencil.Epot, st_stencil.nbh, st_stencil.box2, st_stencil.pair_lj)

        fdx = Array(st_dense.fx) .- Array(st_stencil.fx)
        fdy = Array(st_dense.fy) .- Array(st_stencil.fy)
        @test maximum(abs.(fdx)) <= 1e-10
        @test maximum(abs.(fdy)) <= 1e-10
        @test isapprox(Float64(CUDA.sum(st_dense.Epot)), Float64(CUDA.sum(st_stencil.Epot)); rtol=1e-11, atol=1e-11)
    end

    @testset "Rebuild trigger threshold (controlled displacement)" begin
        skin = 0.4
        st = build_tiny2d(
            N=8, T=Float64, box=(24.0, 24.0), cutoff=2.5, skin=skin, cap=Int32(32),
            use_neighborlist=true, nonbonded=:wca, epsilon=1.0, sigma=1.0, gamma=1.0, temperature=0.0
        )

        rx0 = Array(st.rx)
        ry0 = Array(st.ry)
        step_base = 1
        @test NonEqSimGPU.NeighborLists.update_needed!(
            st.nbh, st.rx, st.ry; skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=step_base
        ) == false

        rx_small = copy(rx0)
        rx_small[1] += 0.49 * skin
        set_positions_2d!(st, rx_small, ry0)
        @test NonEqSimGPU.NeighborLists.update_needed!(
            st.nbh, st.rx, st.ry; skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=step_base + 1
        ) == false

        rx_large = copy(rx0)
        rx_large[1] += 0.51 * skin
        set_positions_2d!(st, rx_large, ry0)
        @test NonEqSimGPU.NeighborLists.update_needed!(
            st.nbh, st.rx, st.ry; skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=step_base + 2
        ) == true
    end
end
