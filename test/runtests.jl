using NonEqSimGPU
using NonEqSimGPU: Simulation, BrownianIntegrators, LangevinIntegrators
using NonEqSimGPU.Filters
using CUDA
using Test

CUDA.allowscalar(false)

@testset "NonEqSimGPU.jl" begin
    @testset "Filters" begin
        N = 16
        dt = 0.002f0
        st = Simulation.build_simulation(N=N,
                                         box=(20f0, 20f0, 20f0),
                                         cutoff=2.5f0,
                                         skin=0.3f0,
                                         cap=Int32(32),
                                         neigh_interval=5,
                                         epsilon=1f0,
                                         sigma=1f0,
                                         gamma=1f0,
                                         init_temperature=1f0)

        type_host = vcat(fill(Int32(1), N ÷ 2), fill(Int32(2), N - N ÷ 2))
        st.typeid .= CuArray(type_host)

        cold_filter = Filters.TypeIDs(1)
        hot_filter  = Filters.TypeIDs(2)

        cold_sel = Filters.selection(st, cold_filter)
        hot_sel  = Filters.selection(st, hot_filter)

        @test Filters.count(st, cold_filter) == N ÷ 2
        @test Filters.count(st, hot_filter) == N - N ÷ 2
        @test Filters.resolve(cold_filter, st) == cold_sel.host

        Filters.set_temperature!(st, dt,
            cold_filter => 0.5f0,
            hot_filter  => 2.0f0)

        gamma_host = Array(st.vv.gamma)
        ns = Array(st.vv.noise_scale)
        scale_cold = sqrt.(2f0 .* gamma_host[cold_sel.host] .* 0.5f0 .* dt)
        scale_hot  = sqrt.(2f0 .* gamma_host[hot_sel.host]  .* 2.0f0 .* dt)

        @test all(abs.(ns[cold_sel.host] .- scale_cold) .< 1f-6)
        @test all(abs.(ns[hot_sel.host]  .- scale_hot)  .< 1f-6)

        # Modify friction via filters and ensure noise scales track it
        Filters.set_friction!(st, 2.0f0; filter=cold_filter)
        Filters.set_temperature!(st, dt,
            cold_filter => 0.5f0)
        gamma_host = Array(st.vv.gamma)
        ns = Array(st.vv.noise_scale)
        expected_cold = sqrt.(2f0 .* gamma_host[cold_sel.host] .* 0.5f0 .* dt)
        @test all(abs.(ns[cold_sel.host] .- expected_cold) .< 1f-6)

        bp = BrownianIntegrators.BrownianParams(st)
        Filters.set_friction!(bp, st, cold_filter, 3.0f0)
        gamma_host = Array(bp.gamma)
        @test all(abs.(gamma_host[cold_sel.host] .- 3.0f0) .< 1f-6)
        @test all(abs.(Array(st.vv.gamma)[cold_sel.host] .- 3.0f0) .< 1f-6)

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
        @test all(dq_host[hot_sel.host]  .== 2.0f0)

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
    end

    @testset "Correlated noise" begin
        N = 8
        dt = 0.001f0
        st = Simulation.build_simulation(N=N,
                                         box=(10f0, 10f0),
                                         cutoff=2.5f0,
                                         skin=0.3f0,
                                         cap=Int32(16),
                                         neigh_interval=5,
                                         gamma=1f0,
                                         init_temperature=1f0,
                                         noise_corr_time=0.05f0)
        @test st.vv.corr_time !== nothing
        @test st.ou_x !== nothing && st.ou_y !== nothing

        LangevinIntegrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale;
                                              beta_z=nothing,
                                              corr_time=st.vv.corr_time,
                                              state_x=st.ou_x, state_y=st.ou_y, state_z=nothing,
                                              dt=dt)
        CUDA.synchronize()
        @test all(isfinite.(Array(st.rf_x)))

        Filters.set_corr_time!(st, 0.1f0; filter=Filters.All())
        CUDA.synchronize()
        @test all(abs.(Array(st.vv.corr_time) .- 0.1f0) .< 1f-6)

        bp = BrownianIntegrators.BrownianParams(st)
        @test bp.corr_time !== nothing
        bp = Filters.set_corr_time!(bp, 0.2f0)
        @test bp.corr_time !== nothing
        @test all(abs.(Array(bp.corr_time) .- 0.2f0) .< 1f-6)

        BrownianIntegrators.bd_prepare_noise_2d!(st.rf_x, st.rf_y;
                                                 noise_scale=bp.noise_scale,
                                                 corr_time=bp.corr_time,
                                                 state_x=st.ou_x, state_y=st.ou_y,
                                                 dt=dt)
        CUDA.synchronize()
        @test all(isfinite.(Array(st.rf_x)))
    end

end
