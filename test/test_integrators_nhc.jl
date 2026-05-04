@testset "Nose-Hoover Chain Integrator" begin
    @testset "Protocol smoke and observables" begin
        seed_all!(0xE3001)
        dt = 5f-4
        st = build_tiny3d(
            N=24, T=Float32, box=(20f0, 20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=4, gamma=nothing, temperature=1f0, nonbonded=:wca
        )
        @test_throws UndefKeywordError Simulation.velocityverlet(st)
        vv = Simulation.velocityverlet(st; gamma=1f0, temperature=1f0, dt=dt)
        @test ParticleDynamics.IntegratorInterfaces.integrator_name(vv) == :velocity_verlet
        spec = Simulation.nosehooverchain(st; temperature=1.0f0, tau=0.5f0, chain_length=5, substeps=4)

        @test ParticleDynamics.IntegratorInterfaces.integrator_name(spec) == :nose_hoover_chain
        @test ParticleDynamics.IntegratorInterfaces.integrator_id(spec) == UInt8(3)
        @test ParticleDynamics.IntegratorInterfaces.stage_sequence(spec) ==
              (:thermostat_pre, :kick1, :drift, :force, :kick2, :thermostat_post)

        Simulation.step!(st, spec, dt; compute_energy=true)
        @test state_allfinite(st)
        @test st.last_integrator == UInt8(3)

        obs = Simulation.collect_step_observables(st, spec)
        @test hasproperty(obs, :extended_hamiltonian)
        @test hasproperty(obs, :thermostat_kinetic)
        @test hasproperty(obs, :thermostat_potential)
        @test hasproperty(obs, :thermostat_temperature_error)
        @test isfinite(obs.extended_hamiltonian)
        @test isfinite(obs.thermostat_kinetic)
        @test isfinite(obs.thermostat_potential)
        @test isfinite(obs.thermostat_temperature_error)
    end

    @testset "Deterministic temperature regulation" begin
        seed_all!(0xE3002)
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

        spec = Simulation.nosehooverchain(st; temperature=1.0f0, tau=0.2f0, chain_length=5, substeps=6)
        for _ in 1:800
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        dof = 3 * N
        Tinst = Float64(2f0 * CUDA.sum(st.Ekin) / Float32(dof))
        @test isfinite(Tinst)
        @test abs(Tinst - 1.0) < 0.35
    end

    @testset "Two-bath filtered temperature control" begin
        seed_all!(0xE3003)
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

        for _ in 1:1200
            Simulation.step!(st, spec, dt; compute_energy=false)
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

    @testset "GROMACS propagator smoke and control" begin
        seed_all!(0xE3004)
        dt = 1f-3
        N = 64
        st = build_tiny3d(
            N=N, T=Float64, box=(40.0, 40.0, 40.0), cutoff=2.5, skin=0.3, cap=Int32(64),
            neigh_interval=8, epsilon=0.0, sigma=1.0, gamma=nothing, temperature=0.0, nonbonded=:lj
        )

        vx0 = randn(Float64, N) .* 3.0
        vy0 = randn(Float64, N) .* 3.0
        vz0 = randn(Float64, N) .* 3.0
        set_velocities_3d!(st, vx0, vy0, vz0)

        spec = Simulation.nosehooverchain(
            st; temperature=1.0, tau=0.2, chain_length=5, substeps=5, propagator=:gromacs
        )
        Simulation.step!(st, spec, dt; compute_energy=true)
        @test state_allfinite(st)

        obs = Simulation.collect_step_observables(st, spec)
        @test obs.nhc_propagator == :gromacs
        @test isfinite(obs.extended_hamiltonian)
        @test isfinite(obs.thermostat_kinetic)
        @test isfinite(obs.thermostat_potential)

        for _ in 1:800
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        dof = 3 * N
        Tinst = Float64(2.0 * CUDA.sum(st.Ekin) / dof)
        @test isfinite(Tinst)
        @test abs(Tinst - 1.0) < 0.35
    end

    @testset "LAMMPS propagator smoke and control" begin
        seed_all!(0xE3005)
        dt = 1f-3
        N = 64
        st = build_tiny3d(
            N=N, T=Float64, box=(40.0, 40.0, 40.0), cutoff=2.5, skin=0.3, cap=Int32(64),
            neigh_interval=8, epsilon=0.0, sigma=1.0, gamma=nothing, temperature=0.0, nonbonded=:lj
        )

        vx0 = randn(Float64, N) .* 3.0
        vy0 = randn(Float64, N) .* 3.0
        vz0 = randn(Float64, N) .* 3.0
        set_velocities_3d!(st, vx0, vy0, vz0)

        spec = Simulation.nosehooverchain(
            st; temperature=1.0, tau=0.2, chain_length=5, substeps=2, propagator=:lammps
        )
        Simulation.step!(st, spec, dt; compute_energy=true)
        @test state_allfinite(st)

        obs = Simulation.collect_step_observables(st, spec)
        @test obs.nhc_propagator == :lammps
        @test isfinite(obs.extended_hamiltonian)
        @test isfinite(obs.thermostat_kinetic)
        @test isfinite(obs.thermostat_potential)

        for _ in 1:800
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        dof = 3 * N
        Tinst = Float64(2.0 * CUDA.sum(st.Ekin) / dof)
        @test isfinite(Tinst)
        @test abs(Tinst - 1.0) < 0.5
    end
end
