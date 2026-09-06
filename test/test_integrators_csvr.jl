@testset "CSVR Integrator" begin
    @testset "Protocol smoke and observables" begin
        seed_all!(0xE4001)
        dt = 5f-4
        st = build_tiny3d(
            N=24, T=Float32, box=(20f0, 20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=4, gamma=nothing, temperature=1f0, nonbonded=:wca
        )
        vv = SimulationCore.velocityverlet(st; gamma=1f0, temperature=1f0, dt=dt)
        @test ParticleDynamics.IntegratorInterfaces.integrator_name(vv) == :velocity_verlet
        spec = SimulationCore.csvr(st; temperature=1.0f0, tau=0.5f0)

        @test ParticleDynamics.IntegratorInterfaces.integrator_name(spec) == :csvr
        @test ParticleDynamics.IntegratorInterfaces.integrator_id(spec) == UInt8(4)
        @test ParticleDynamics.IntegratorInterfaces.stage_sequence(spec) ==
              (:kick1, :drift, :force, :kick2, :thermostat)

        vx0 = randn(Float32, length(st.rx))
        vy0 = randn(Float32, length(st.rx))
        vz0 = randn(Float32, length(st.rx))
        set_velocities_3d!(st, vx0, vy0, vz0)

        SimulationCore.step!(st, spec, dt; compute_energy=true)
        @test state_allfinite(st)
        @test st.last_integrator == UInt8(4)

        obs = SimulationCore.collect_step_observables(st, spec)
        @test hasproperty(obs, :extended_hamiltonian)
        @test hasproperty(obs, :thermostat_energy)
        @test hasproperty(obs, :thermostat_temperature_error)
        @test isfinite(obs.extended_hamiltonian)
        @test isfinite(obs.thermostat_energy)
        @test isfinite(obs.thermostat_temperature_error)
    end

    @testset "Temperature regulation" begin
        seed_all!(0xE4002)
        dt = 1f-3
        N = 64
        st = build_tiny3d(
            N=N, T=Float32, box=(40f0, 40f0, 40f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(64),
            neigh_interval=8, epsilon=0f0, sigma=1f0, gamma=nothing, temperature=0f0, nonbonded=:lj
        )

        vx0 = randn(Float32, N) .* 3f0
        vy0 = randn(Float32, N) .* 3f0
        vz0 = randn(Float32, N) .* 3f0
        set_velocities_3d!(st, vx0, vy0, vz0)

        spec = SimulationCore.csvr(st; temperature=1.0f0, tau=0.2f0)
        # Equilibrate for 10 thermostat times. A single final temperature after
        # only 4 tau still has both relaxation bias and canonical fluctuations.
        for _ in 1:2000
            SimulationCore.step!(st, spec, dt; compute_energy=false)
        end

        dof = 3 * N
        temperatures = Float64[]
        # Forty samples separated by tau: canonical variance is 2/dof and
        # residual correlation gives a mean standard error of about 0.024.
        for _ in 1:40
            for _ in 1:200
                SimulationCore.step!(st, spec, dt; compute_energy=false)
            end
            push!(temperatures, Float64(2f0 * CUDA.sum(st.Ekin) / Float32(dof)))
        end
        @test all(isfinite, temperatures)
        @test abs(sum(temperatures) / length(temperatures) - 1.0) < 0.10
    end

    @testset "Two-bath filtered temperature control" begin
        seed_all!(0xE4003)
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

        spec = SimulationCore.csvr(st; temperature=1.0f0, tau=0.2f0)
        Filters.set_temperature!(spec, st, dt, cold => 0.8f0, hot => 1.4f0)

        for _ in 1:1200
            SimulationCore.step!(st, spec, dt; compute_energy=false)
        end

        Nc = Filters.count(st, cold)
        Nh = Filters.count(st, hot)
        Tc = Float64(2f0 * Filters.sum(st.Ekin, st, cold) / Float32(3 * Nc))
        Th = Float64(2f0 * Filters.sum(st.Ekin, st, hot) / Float32(3 * Nh))

        @test isfinite(Tc)
        @test isfinite(Th)
        @test Tc < Th
        @test abs(Tc - 0.8) < 0.45
        @test abs(Th - 1.4) < 0.45
    end

    @testset "Effective energy accounting" begin
        seed_all!(0xE4004)
        dt = 5e-4
        N = 48
        st = build_tiny3d(
            N=N, T=Float64, box=(40.0, 40.0, 40.0), cutoff=2.5, skin=0.3, cap=Int32(64),
            neigh_interval=8, epsilon=0.0, sigma=1.0, gamma=nothing, temperature=0.0, nonbonded=:lj
        )

        vx0 = randn(Float64, N) .* 2.0
        vy0 = randn(Float64, N) .* 2.0
        vz0 = randn(Float64, N) .* 2.0
        set_velocities_3d!(st, vx0, vy0, vz0)

        spec = SimulationCore.csvr(st; temperature=1.0, tau=0.1)
        hvals = Float64[]
        for step in 1:400
            SimulationCore.step!(st, spec, dt; compute_energy=true)
            if step % 10 == 0
                obs = SimulationCore.collect_step_observables(st, spec)
                push!(hvals, obs.extended_hamiltonian)
            end
        end

        @test !isempty(hvals)
        drift = maximum(hvals) - minimum(hvals)
        @test isfinite(drift)
        @test drift < 1e-8
    end
end
