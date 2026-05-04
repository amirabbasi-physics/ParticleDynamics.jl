@testset "API and Filters" begin
    seed_all!(0xA1001)

    @test isdefined(ParticleDynamics, :SimulationState)
    @test isdefined(ParticleDynamics, :build_simulation)
    @test isdefined(ParticleDynamics, :step!)
    @test isdefined(ParticleDynamics, :step_graph!)
    @test isdefined(ParticleDynamics, :velocityverlet)
    @test isdefined(ParticleDynamics, :reset_bath_exchange_accumulators!)
    @test isdefined(ParticleDynamics, :baoab)
    @test isdefined(ParticleDynamics, :baoa)
    @test isdefined(ParticleDynamics, :gsm)
    @test isdefined(ParticleDynamics, :eulermaruyama)
    @test isdefined(ParticleDynamics, :eulerheun)
    @test isdefined(ParticleDynamics, :nosehooverchain)
    @test isdefined(ParticleDynamics, :NHCSpec)
    @test isdefined(ParticleDynamics, :csvr)
    @test isdefined(ParticleDynamics, :CSVRSpec)

    N = 16
    dt = 0.002f0
    st = build_tiny3d(N=N, T=Float32, gamma=1f0, temperature=1f0, dt=dt)

    type_host = vcat(fill(Int32(1), N ÷ 2), fill(Int32(2), N - N ÷ 2))
    st.typeid .= CuArray(type_host)

    cold_filter = Filters.TypeIDs(1)
    hot_filter = Filters.TypeIDs(2)

    cold_sel = Filters.selection(st, cold_filter)
    hot_sel = Filters.selection(st, hot_filter)
    vv = Simulation.velocityverlet(st; gamma=1f0, temperature=1f0, dt=dt)

    @test Filters.count(st, cold_filter) == N ÷ 2
    @test Filters.count(st, hot_filter) == N - N ÷ 2
    @test Filters.resolve(cold_filter, st) == cold_sel.host

    Filters.set_temperature!(vv, st, dt, cold_filter => 0.5f0, hot_filter => 2.0f0)

    gamma_host = Array(vv.params.gamma)
    ns = Array(vv.params.noise_scale)
    scale_cold = sqrt.(2f0 .* gamma_host[cold_sel.host] .* 0.5f0 .* dt)
    scale_hot = sqrt.(2f0 .* gamma_host[hot_sel.host] .* 2.0f0 .* dt)

    @test all(abs.(ns[cold_sel.host] .- scale_cold) .< 1f-6)
    @test all(abs.(ns[hot_sel.host] .- scale_hot) .< 1f-6)

    Filters.set_friction!(vv, st, 2.0f0; filter=cold_filter)
    Filters.set_temperature!(vv, st, dt, cold_filter => 0.5f0)
    gamma_host = Array(vv.params.gamma)
    ns = Array(vv.params.noise_scale)
    expected_cold = sqrt.(2f0 .* gamma_host[cold_sel.host] .* 0.5f0 .* dt)
    @test all(abs.(ns[cold_sel.host] .- expected_cold) .< 1f-6)

    bp = BrownianIntegrators.BrownianParams(vv.params.gamma, vv.params.noise_scale, vv.params.corr_time)
    Filters.set_friction!(bp, st, cold_filter, 3.0f0)
    gamma_host = Array(bp.gamma)
    @test all(abs.(gamma_host[cold_sel.host] .- 3.0f0) .< 1f-6)

    Filters.set_noise_scale!(bp, st, cold_filter, 0.25f0)
    ns_bp = Array(bp.noise_scale)
    @test all(abs.(ns_bp[cold_sel.host] .- 0.25f0) .< 1f-6)

    Filters.set_temperature!(bp, st, dt, 0.75f0; filter=hot_filter)
    ns_bp = Array(bp.noise_scale)
    gamma_host = Array(bp.gamma)
    expected_hot_bp = sqrt.(2f0 .* gamma_host[hot_sel.host] .* 0.75f0 .* dt)
    @test all(abs.(ns_bp[hot_sel.host] .- expected_hot_bp) .< 1f-6)

    Filters.assign_scalar!(st.dq, st; filter=cold_filter, value=1.0f0)
    Filters.assign_scalar!(st.dq, st; filter=hot_filter, value=2.0f0)
    CUDA.synchronize()

    dq_host = Array(st.dq)
    @test all(dq_host[cold_sel.host] .== 1.0f0)
    @test all(dq_host[hot_sel.host] .== 2.0f0)

    values = Float32.(1:Filters.count(cold_sel))
    Filters.assign_values!(st.dq, st; filter=cold_filter, values=values)
    CUDA.synchronize()

    dq_host = Array(st.dq)
    @test dq_host[cold_sel.host] == values
    @test Filters.sum(st.dq, st, cold_filter) ≈ Base.sum(values) atol=1f-5
    @test Filters.sum(dq_host, hot_sel.host) ≈ Base.sum(fill(2.0f0, Filters.count(hot_sel))) atol=1f-5

    gathered = Filters.gather(st.dq, st, cold_filter)
    @test gathered == values

    idx_sel = Filters.selection(st, Filters.Indices(collect(1:3:10)))
    Filters.assign_scalar!(st.dq, idx_sel, 5.0f0)
    CUDA.synchronize()
    dq_host = Array(st.dq)
    @test all(dq_host[idx_sel.host] .== 5.0f0)

    @testset "Integrator-spec controls and observables" begin
        spec = Simulation.velocityverlet(st; gamma=1f0, temperature=1f0, dt=dt)
        @test ParticleDynamics.IntegratorInterfaces.stage_sequence(spec) == (:kick1, :drift, :force, :kick2)

        Filters.set_friction!(spec, st, 1.5f0; filter=cold_filter)
        g_spec = Array(spec.params.gamma)
        @test all(abs.(g_spec[cold_sel.host] .- 1.5f0) .< 1f-6)

        Filters.set_temperature!(spec, st, dt, 0.8f0; filter=hot_filter)
        n_spec = Array(spec.params.noise_scale)
        expected_hot = sqrt.(2f0 .* g_spec[hot_sel.host] .* 0.8f0 .* dt)
        @test all(abs.(n_spec[hot_sel.host] .- expected_hot) .< 1f-6)

        Filters.set_corr_time!(spec, 0.2f0)
        @test spec.params.corr_time !== nothing
        @test all(abs.(Array(spec.params.corr_time) .- 0.2f0) .< 1f-6)

        Filters.set_ou_spectrum!(spec, st, 0.15f0, 0.4f0; filter=cold_filter, dt=dt)
        @test spec.params.ou !== nothing
        @test size(spec.params.ou.coeff_a, 1) == 1
        corr_host = Array(spec.params.corr_time)
        @test all(abs.(corr_host[cold_sel.host] .- 0.15f0) .< 1f-6)
        @test all(abs.(corr_host[hot_sel.host]) .< 1f-6)

        Filters.set_ou_spectrum!(spec, st, Float32[0.05f0, 0.3f0], Float32[0.2f0, 0.1f0]; filter=cold_filter, dt=dt)
        @test spec.params.ou !== nothing
        @test size(spec.params.ou.coeff_a, 1) == 2
        @test spec.params.corr_time === nothing

        Simulation.step!(st, spec, dt; compute_energy=true)
        obs = Simulation.collect_step_observables(st, spec)
        @test obs.integrator == :velocity_verlet
        @test obs.step == st.step
        @test isfinite(obs.Etot)
        @test isfinite(obs.Qtot)
        @test obs.thermostatted_dof == 3 * length(st.rx)
    end

    @testset "NHC controls" begin
        nhc = Simulation.nosehooverchain(st; temperature=1.0f0, tau=0.5f0, chain_length=4, substeps=3)
        @test ParticleDynamics.IntegratorInterfaces.stage_sequence(nhc) ==
              (:thermostat_pre, :kick1, :drift, :force, :kick2, :thermostat_post)
        @test nhc.params.propagator == Simulation.NHC_PROPAGATOR_GROMACS

        old_q = copy(nhc.params.chain_masses)
        Filters.set_thermostat_temperature!(nhc, 1.25f0)
        @test all(abs.(nhc.params.target_temperature .- 1.25f0) .< 1f-6)

        Filters.set_thermostat_timescale!(nhc, 0.25f0)
        @test all(abs.(nhc.params.tau .- 0.25f0) .< 1f-6)
        @test any(abs.(nhc.params.chain_masses .- old_q) .> 0f0)

        Filters.set_temperature!(nhc, dt, 0.9f0)
        @test all(abs.(nhc.params.target_temperature .- 0.9f0) .< 1f-6)

        Filters.set_temperature!(nhc, st, dt, cold_filter => 0.7f0, hot_filter => 1.3f0)
        @test length(nhc.params.target_temperature) == 2
        @test nhc.params.target_temperature[1] ≈ 0.7f0 atol=1f-6
        @test nhc.params.target_temperature[2] ≈ 1.3f0 atol=1f-6

        @test_throws ArgumentError Filters.set_friction!(nhc, 1.0f0)
        @test_throws ArgumentError Filters.set_noise_scale!(nhc, 1.0f0)
        @test_throws ArgumentError Filters.set_corr_time!(nhc, 0.1f0)

        nhc_legacy = Simulation.nosehooverchain(
            st; temperature=1.0f0, tau=0.5f0, chain_length=4, substeps=5, propagator=:legacy
        )
        @test nhc_legacy.params.propagator != nhc.params.propagator
        nhc_lammps = Simulation.nosehooverchain(
            st; temperature=1.0f0, tau=0.5f0, chain_length=4, substeps=1, propagator=:lammps
        )
        @test nhc_lammps.params.propagator == Simulation.NHC_PROPAGATOR_LAMMPS
        @test_throws ArgumentError Simulation.nosehooverchain(
            st; temperature=1.0f0, tau=0.5f0, chain_length=4, substeps=5, propagator=:invalid
        )
    end

    @testset "CSVR controls" begin
        csvr_spec = Simulation.csvr(st; temperature=1.0f0, tau=0.5f0)
        @test ParticleDynamics.IntegratorInterfaces.stage_sequence(csvr_spec) ==
              (:kick1, :drift, :force, :kick2, :thermostat)

        Filters.set_thermostat_temperature!(csvr_spec, 1.1f0)
        @test all(abs.(csvr_spec.params.target_temperature .- 1.1f0) .< 1f-6)

        Filters.set_thermostat_timescale!(csvr_spec, 0.25f0)
        @test all(abs.(csvr_spec.params.tau .- 0.25f0) .< 1f-6)

        Filters.set_temperature!(csvr_spec, dt, 0.9f0)
        @test all(abs.(csvr_spec.params.target_temperature .- 0.9f0) .< 1f-6)

        Filters.set_temperature!(csvr_spec, st, dt, cold_filter => 0.7f0, hot_filter => 1.3f0)
        @test length(csvr_spec.params.target_temperature) == 2
        @test csvr_spec.params.target_temperature[1] ≈ 0.7f0 atol=1f-6
        @test csvr_spec.params.target_temperature[2] ≈ 1.3f0 atol=1f-6

        @test_throws ArgumentError Filters.set_friction!(csvr_spec, 1.0f0)
        @test_throws ArgumentError Filters.set_noise_scale!(csvr_spec, 1.0f0)
        @test_throws ArgumentError Filters.set_corr_time!(csvr_spec, 0.1f0)
        @test_throws ArgumentError Simulation.csvr(st; temperature=1.0f0, tau=0.0f0)
    end
end
