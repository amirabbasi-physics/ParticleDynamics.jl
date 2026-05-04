@testset "Bath Entropy Observables" begin
    function _ld_manual_totals(spec, st, dt)
        invT = let s = Array(spec.params.noise_scale), g = Array(spec.params.gamma)
            (2 .* g .* eltype(s)(dt)) ./ (s .^ 2)
        end
        sign = spec isa Simulation.VVSpec ? 1.0 : -1.0
        heat_total = sign * sum(Array(st.dq)) * dt
        entropy_total = sign * sum(Array(st.dq) .* invT) * dt
        return heat_total, entropy_total
    end

    @testset "Langevin family totals and reset helper" begin
        for (seed, builder) in (
            (0xE5001, st -> Simulation.velocityverlet(st; gamma=1f0, temperature=1f0, dt=1f-3)),
            (0xE5002, st -> Simulation.baoab(st; gamma=1f0, temperature=1f0, dt=1f-3)),
            (0xE5003, st -> Simulation.baoa(st; gamma=1f0, temperature=1f0, dt=1f-3)),
            (0xE5004, st -> Simulation.gsm(st; gamma=1f0, temperature=1f0, dt=1f-3)),
        )
            seed_all!(seed)
            dt = 1f-3
            N = 48
            st = build_tiny2d(
                N=N, T=Float32, box=(45f0, 45f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(64),
                neigh_interval=8, epsilon=0f0, sigma=1f0, gamma=1f0, temperature=0f0, nonbonded=:lj
            )

            vx0 = randn(Float32, N) .* 3f0
            vy0 = randn(Float32, N) .* 3f0
            set_velocities_2d!(st, vx0, vy0)

            tid = fill(Int32(1), N)
            tid[(N ÷ 2 + 1):end] .= Int32(2)
            st.typeid .= CuArray(tid)
            cold = Filters.TypeIDs(1)
            hot = Filters.TypeIDs(2)

            spec = builder(st)
            Filters.set_temperature!(spec, st, dt, cold => 0.8f0, hot => 1.4f0)

            for _ in 1:400
                Simulation.step!(st, spec, dt; compute_energy=false)
            end

            obs = Simulation.collect_step_observables(st, spec)
            manual_heat, manual_entropy = _ld_manual_totals(spec, st, dt)

            @test hasproperty(obs, :bath_heat_total)
            @test hasproperty(obs, :bath_entropy_total)
            @test isfinite(obs.bath_heat_total)
            @test isfinite(obs.bath_entropy_total)
            @test obs.bath_heat_total ≈ manual_heat atol=5f-3 rtol=5f-5
            @test obs.bath_entropy_total ≈ manual_entropy atol=5f-3 rtol=5f-5

            ParticleDynamics.reset_bath_exchange_accumulators!(st, spec)
            obs_reset = Simulation.collect_step_observables(st, spec)
            @test obs_reset.bath_heat_total == 0f0
            @test obs_reset.bath_entropy_total == 0f0
            @test obs_reset.dq_total == 0f0
            @test obs_reset.dU_total == 0f0
        end
    end

    @testset "NHC totals, per-bath breakdown, and reset helper" begin
        seed_all!(0xE5005)
        dt = 1f-3
        N = 96
        st = build_tiny3d(
            N=N, T=Float32, box=(45f0, 45f0, 45f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(64),
            neigh_interval=8, epsilon=0f0, sigma=1f0, gamma=nothing, temperature=0f0, nonbonded=:lj
        )

        vx0 = randn(Float32, N) .* 3f0
        vy0 = randn(Float32, N) .* 3f0
        vz0 = randn(Float32, N) .* 3f0
        set_velocities_3d!(st, vx0, vy0, vz0)

        tid = fill(Int32(1), N)
        tid[(N ÷ 2 + 1):end] .= Int32(2)
        st.typeid .= CuArray(tid)
        cold = Filters.TypeIDs(1)
        hot = Filters.TypeIDs(2)

        spec = Simulation.nosehooverchain(st; temperature=1.0f0, tau=0.2f0, chain_length=5, substeps=6)
        Filters.set_temperature!(spec, st, dt, cold => 0.8f0, hot => 1.4f0)

        for _ in 1:120
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        obs = Simulation.collect_step_observables(st, spec)
        heat_b = Array(spec.workspace.cumulative_energy_exchange_per_bath)
        entropy_b = heat_b ./ spec.params.target_temperature

        @test hasproperty(obs, :bath_heat_total)
        @test hasproperty(obs, :bath_entropy_total)
        @test hasproperty(obs, :bath_heat_per_bath)
        @test hasproperty(obs, :bath_entropy_per_bath)
        @test hasproperty(obs, :bath_temperature_per_bath)
        @test length(obs.bath_heat_per_bath) == 2
        @test length(obs.bath_entropy_per_bath) == 2
        @test length(obs.bath_temperature_per_bath) == 2
        @test obs.bath_heat_per_bath ≈ heat_b atol=5f-4 rtol=5f-5
        @test obs.bath_entropy_per_bath ≈ entropy_b atol=5f-4 rtol=5f-5
        @test obs.bath_heat_total ≈ sum(heat_b) atol=5f-4 rtol=5f-5
        @test obs.bath_entropy_total ≈ sum(entropy_b) atol=5f-4 rtol=5f-5

        ParticleDynamics.reset_bath_exchange_accumulators!(st, spec)
        obs_reset = Simulation.collect_step_observables(st, spec)
        @test obs_reset.bath_heat_total == 0f0
        @test obs_reset.bath_entropy_total == 0f0
        @test all(iszero, obs_reset.bath_heat_per_bath)
        @test all(iszero, obs_reset.bath_entropy_per_bath)
    end

    @testset "CSVR totals, per-bath breakdown, and reset helper" begin
        seed_all!(0xE5006)
        dt = 1f-3
        N = 96
        st = build_tiny3d(
            N=N, T=Float32, box=(45f0, 45f0, 45f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(64),
            neigh_interval=8, epsilon=0f0, sigma=1f0, gamma=nothing, temperature=0f0, nonbonded=:lj
        )

        vx0 = randn(Float32, N) .* 3f0
        vy0 = randn(Float32, N) .* 3f0
        vz0 = randn(Float32, N) .* 3f0
        set_velocities_3d!(st, vx0, vy0, vz0)

        tid = fill(Int32(1), N)
        tid[(N ÷ 2 + 1):end] .= Int32(2)
        st.typeid .= CuArray(tid)
        cold = Filters.TypeIDs(1)
        hot = Filters.TypeIDs(2)

        spec = Simulation.csvr(st; temperature=1.0f0, tau=0.2f0)
        Filters.set_temperature!(spec, st, dt, cold => 0.8f0, hot => 1.4f0)

        for _ in 1:120
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        obs = Simulation.collect_step_observables(st, spec)
        heat_b = Array(spec.workspace.cumulative_energy_exchange_per_bath)
        entropy_b = heat_b ./ spec.params.target_temperature

        @test hasproperty(obs, :bath_heat_total)
        @test hasproperty(obs, :bath_entropy_total)
        @test hasproperty(obs, :bath_heat_per_bath)
        @test hasproperty(obs, :bath_entropy_per_bath)
        @test hasproperty(obs, :bath_temperature_per_bath)
        @test length(obs.bath_heat_per_bath) == 2
        @test length(obs.bath_entropy_per_bath) == 2
        @test length(obs.bath_temperature_per_bath) == 2
        @test obs.bath_heat_per_bath ≈ heat_b atol=5f-4 rtol=5f-5
        @test obs.bath_entropy_per_bath ≈ entropy_b atol=5f-4 rtol=5f-5
        @test obs.bath_heat_total ≈ sum(heat_b) atol=5f-4 rtol=5f-5
        @test obs.bath_entropy_total ≈ sum(entropy_b) atol=5f-4 rtol=5f-5

        ParticleDynamics.reset_bath_exchange_accumulators!(st, spec)
        obs_reset = Simulation.collect_step_observables(st, spec)
        @test obs_reset.bath_heat_total == 0f0
        @test obs_reset.bath_entropy_total == 0f0
        @test all(iszero, obs_reset.bath_heat_per_bath)
        @test all(iszero, obs_reset.bath_entropy_per_bath)
    end
end
