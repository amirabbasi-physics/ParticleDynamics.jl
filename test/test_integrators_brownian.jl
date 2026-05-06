@testset "Brownian Integrators" begin
    @testset "Brownian correlated noise path" begin
        seed_all!(0xF1001)
        dt = 0.001f0
        st = build_tiny2d(
            N=8, T=Float32, box=(10f0, 10f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(16),
            neigh_interval=5, gamma=1f0, temperature=1f0, noise_corr_time=0.05f0
        )
        spec = SimulationCore.eulerheun(st; gamma=1f0, temperature=1f0, noise_corr_time=0.05f0, dt=dt)
        SimulationCore.ensure_integrator_workspace!(spec, st)
        bp = spec.params
        @test bp.corr_time !== nothing
        spec.params = Filters.set_corr_time!(bp, 0.2f0)
        bp = spec.params
        @test bp.corr_time !== nothing
        @test all(abs.(Array(bp.corr_time) .- 0.2f0) .< 1f-6)

        BrownianIntegrators.bd_prepare_noise_2d!(
            spec.workspace.rf_x, spec.workspace.rf_y;
            noise_scale=bp.noise_scale,
            ou=bp.ou,
            state_x=spec.workspace.ou_x, state_y=spec.workspace.ou_y,
        )
        CUDA.synchronize()
        @test gpu_allfinite(spec.workspace.rf_x)
        @test gpu_allfinite(spec.workspace.rf_y)
    end

    @testset "Euler-Heun and Euler-Maruyama smoke" begin
        seed_all!(0xF1002)
        dt = 1f-3
        st = build_tiny2d(
            N=12, T=Float32, box=(18f0, 18f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(24),
            neigh_interval=4, gamma=1f0, temperature=1f0, nonbonded=:wca
        )

        rx0 = copy(st.rx)
        ry0 = copy(st.ry)
        eh = SimulationCore.eulerheun(st; gamma=1f0, temperature=1f0, dt=dt)
        SimulationCore.step!(st, eh, dt; compute_energy=true)
        @test state_allfinite(st)
        @test msd_2d(rx0, ry0, st.rx, st.ry) > 0.0

        rx1 = copy(st.rx)
        ry1 = copy(st.ry)
        em = SimulationCore.eulermaruyama(st; gamma=1f0, temperature=1f0, dt=dt)
        SimulationCore.step!(st, em, dt; compute_energy=true)
        @test state_allfinite(st)
        @test msd_2d(rx1, ry1, st.rx, st.ry) > 0.0
    end

    @testset "Freeze spring" begin
        seed_all!(0xF1003)
        N = 2
        dt = 0.001f0
        st = build_tiny2d(
            N=N, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(4),
            neigh_interval=5, epsilon=0f0, sigma=1f0, gamma=1f0, temperature=0f0,
            nonbonded=:lj
        )
        set_positions_2d!(st, Float32[-2, 2], Float32[0, 0])

        st.typeid .= CuArray(Int32[1, 2])
        Filters.freeze_particles!(st; filter=Filters.TypeIDs(2), mode=:spring, k=100f0, steps=1, include_energy=true)

        rx_shift = Float32[-2, 2.25]
        copyto!(st.rx, rx_shift)

        em = SimulationCore.eulermaruyama(st; gamma=1f0, temperature=0f0, dt=dt)
        SimulationCore.step!(st, em, dt; compute_energy=true)

        rx_now = Array(st.rx)
        ry_now = Array(st.ry)
        rx_anchor = Array(st.freeze_rx)
        ry_anchor = Array(st.freeze_ry)
        Epot = Array(st.Epot)
        k = 100f0
        dx = rx_now[2] - rx_anchor[2]
        dy = ry_now[2] - ry_anchor[2]
        expected = 0.5f0 * k * (dx * dx + dy * dy)
        @test isapprox(Epot[2], expected; rtol=1f-4, atol=1f-6)
        @test isapprox(Epot[1], 0f0; atol=1f-6)
    end
end
