@testset "Phase 4A: Pair Matrix and PBC Invariance" begin
    seed_all!(0x4A0202)

    lj_force_x(dx, r, eps, sig) = 24 * eps * (2 * (sig / r)^12 - (sig / r)^6) * (dx / (r * r))

    @testset "Pair-matrix parameter selection (type-dependent WCA)" begin
        st = build_tiny2d(
            N=2, T=Float64, box=(20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
            use_neighborlist=true, nonbonded=:wca, epsilon=1.0, sigma=1.0, gamma=1.0, temperature=0.0
        )
        st.typeid .= CuArray(Int32[1, 2])
        set_positions_2d!(st, [-0.5, 0.5], [0.0, 0.0]) # r = 1.0
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)

        sigma_pair_h = Float64[
            0.90 1.10;
            1.10 0.95
        ]
        epsilon_pair_h = Float64[
            1.00 2.50;
            2.50 1.20
        ]
        rcut_pair_h = Float64[
            1.01 1.60;
            1.60 1.01
        ]
        sigma_pair = CuArray(sigma_pair_h)
        epsilon_pair = CuArray(epsilon_pair_h)
        rcut_pair = CuArray(rcut_pair_h)

        CUDA.fill!(st.fx, 0.0)
        CUDA.fill!(st.fy, 0.0)
        CUDA.fill!(st.Epot, 0.0)
        NonEqSimGPU.NonBondedForces.wca_forces_soa_pairs!(
            st.rx, st.ry, st.fx, st.fy, st.Epot,
            st.nbh, st.box2, st.typeid, sigma_pair, epsilon_pair, rcut_pair
        )

        fx = Array(st.fx)
        dx = -1.0
        r = 1.0
        fref = lj_force_x(dx, r, epsilon_pair_h[1, 2], sigma_pair_h[1, 2])
        @test isapprox(fx[1], fref; rtol=1e-11, atol=1e-11)
        @test isapprox(fx[1], -fx[2]; rtol=1e-12, atol=1e-12)
        @test abs(Float64(CUDA.sum(st.fx))) <= 1e-12
    end

    @testset "Periodic translation invariance (forces + energy)" begin
        st = build_tiny2d(
            N=12, T=Float64, box=(24.0, 24.0), cutoff=2.5, skin=0.4, cap=Int32(32),
            use_neighborlist=true, nonbonded=:lj, epsilon=1.8, sigma=1.1, gamma=1.0, temperature=0.0
        )

        CUDA.fill!(st.fx, 0.0)
        CUDA.fill!(st.fy, 0.0)
        CUDA.fill!(st.Epot, 0.0)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2, st.pair_lj)
        fx0 = Array(st.fx)
        fy0 = Array(st.fy)
        e0 = Float64(CUDA.sum(st.Epot))

        Lx, Ly = st.box2
        st.rx .+= Lx
        st.ry .-= Ly
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step + 1)

        CUDA.fill!(st.fx, 0.0)
        CUDA.fill!(st.fy, 0.0)
        CUDA.fill!(st.Epot, 0.0)
        NonEqSimGPU.NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2, st.pair_lj)
        fx1 = Array(st.fx)
        fy1 = Array(st.fy)
        e1 = Float64(CUDA.sum(st.Epot))

        @test all(isapprox.(fx0, fx1; rtol=1e-12, atol=1e-12))
        @test all(isapprox.(fy0, fy1; rtol=1e-12, atol=1e-12))
        @test isapprox(e0, e1; rtol=1e-12, atol=1e-12)
    end
end
