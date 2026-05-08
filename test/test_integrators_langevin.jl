@testset "Langevin Integrators" begin
    @testset "Correlated noise path" begin
        seed_all!(0xE1001)
        dt = 0.001f0
        st = build_tiny2d(
            N=8, T=Float32, box=(10f0, 10f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(16),
            neigh_interval=5, gamma=1f0, temperature=1f0, noise_corr_time=0.05f0
        )
        spec = SimulationCore.velocityverlet(st; gamma=1f0, temperature=1f0, noise_corr_time=0.05f0, dt=dt)
        SimulationCore.ensure_integrator_workspace!(spec, st)
        @test spec.params.corr_time !== nothing
        @test spec.workspace.ou_x !== nothing
        @test spec.workspace.ou_y !== nothing

        LangevinIntegrators.vv_prepare_noise!(
            spec.workspace.rf_x, spec.workspace.rf_y, spec.params.noise_scale;
            beta_z=nothing,
            ou=spec.params.ou,
            state_x=spec.workspace.ou_x, state_y=spec.workspace.ou_y, state_z=nothing,
        )
        CUDA.synchronize()
        @test gpu_allfinite(spec.workspace.rf_x)
        @test gpu_allfinite(spec.workspace.rf_y)

        Filters.set_corr_time!(spec, 0.1f0)
        CUDA.synchronize()
        @test all(abs.(Array(spec.params.corr_time) .- 0.1f0) .< 1f-6)
    end

    @testset "BAOAB/BAOA/GSM smoke" begin
        seed_all!(0xE1002)
        dt = 1f-3
        st = build_tiny2d(
            N=12, T=Float32, box=(18f0, 18f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(24),
            neigh_interval=4, gamma=1f0, temperature=1f0, nonbonded=:wca
        )
        for spec in (SimulationCore.baoab(st; gamma=1f0, temperature=1f0, dt=dt),
                     SimulationCore.baoa(st; gamma=1f0, temperature=1f0, dt=dt),
                     SimulationCore.gsm(st; gamma=1f0, temperature=1f0, dt=dt))
            SimulationCore.step!(st, spec, dt; compute_energy=true)
            @test state_allfinite(st)
        end
        km = kinetic_moments(st.vx, st.vy; mass=eltype(st.vx)(st.mass))
        @test isfinite(km.mean_v2)
        @test isfinite(km.mean_kinetic)
    end

    @testset "Freeze hold" begin
        seed_all!(0xE1003)
        N = 4
        dt = 0.01f0
        st = build_tiny2d(
            N=N, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(8),
            neigh_interval=5, epsilon=0f0, sigma=1f0, gamma=0f0, temperature=0f0,
            nonbonded=:lj
        )
        set_positions_2d!(st, Float32[-1, 1, -1, 1], Float32[-1, -1, 1, 1])
        set_velocities_2d!(st, fill(0.1f0, N), fill(-0.05f0, N))

        st.typeid .= CuArray(Int32[1, 2, 1, 2])
        frozen = Filters.selection(st, Filters.TypeIDs(2))
        Filters.freeze_particles!(st; filter=Filters.TypeIDs(2), mode=:hold, steps=2)
        vv = SimulationCore.velocityverlet(st; gamma=0f0, temperature=0f0, dt=dt)
        rx0 = Filters.gather(st.rx, frozen)
        ry0 = Filters.gather(st.ry, frozen)

        for _ in 1:2
            SimulationCore.step!(st, vv, dt; compute_energy=false)
            rx_now = Filters.gather(st.rx, frozen)
            ry_now = Filters.gather(st.ry, frozen)
            @test all(isapprox.(rx_now, rx0; atol=1f-6))
            @test all(isapprox.(ry_now, ry0; atol=1f-6))
        end

        SimulationCore.step!(st, vv, dt; compute_energy=false)
        rx_after = Filters.gather(st.rx, frozen)
        ry_after = Filters.gather(st.ry, frozen)
        @test any(abs.(rx_after .- rx0) .> 1f-5)
        @test any(abs.(ry_after .- ry0) .> 1f-5)

        vx_after = Filters.gather(st.vx, frozen)
        vy_after = Filters.gather(st.vy, frozen)
        @test all(isapprox.(vx_after, fill(0.1f0, length(vx_after)); atol=1f-6))
        @test all(isapprox.(vy_after, fill(-0.05f0, length(vy_after)); atol=1f-6))
    end
end
