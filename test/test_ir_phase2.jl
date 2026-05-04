@testset "Phase 2 IR Reproductions" begin
    @testset "IR-001 test_step_graph_3d_smoke_no_freeze" begin
        seed_all!(0xD00101)
        st = build_tiny3d(
            N=8, T=Float32, box=(20f0, 20f0, 20f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(32), neigh_interval=1, use_neighborlist=true,
            nonbonded=:wca, gamma=1f0, temperature=1f0, dt=1f-3
        )
        err = nothing
        try
            vv = Simulation.velocityverlet(st; gamma=1f0, temperature=1f0, dt=1f-3)
            Simulation.step_graph!(st, vv, 1f-3; compute_energy=false)
            Simulation.step_graph!(st, vv.params, 1f-3; compute_energy=false)
        catch e
            err = e
        end
        @test err === nothing
    end

    @testset "IR-002 test_build_accepts_softrep_params_float32_float64" begin
        seed_all!(0xD00202)

        sp32 = SoftRepulsiveParams{Float32}(2.0f0, 1.25f0)
        st32 = nothing
        err32 = nothing
        try
            st32 = Simulation.build_simulation(
                N=6, box=(18f0, 18f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
                neigh_interval=2, use_neighborlist=true, nonbonded=:softrep,
                softrep_params=sp32, gamma=1f0, temperature=1f0, dt=1f-3, precision=:f32
            )
        catch e
            err32 = e
        end
        @test err32 === nothing
        if err32 === nothing
            @test st32.softrep !== nothing
            @test st32.softrep isa SoftRepulsiveParams{Float32}
            @test st32.softrep.ϵ == sp32.ϵ
            @test st32.softrep.σ == sp32.σ
        end

        sp64 = SoftRepulsiveParams{Float64}(2.0, 1.25)
        st64 = nothing
        err64 = nothing
        try
            st64 = Simulation.build_simulation(
                N=6, box=(18.0, 18.0), cutoff=2.5, skin=0.3, cap=Int32(32),
                neigh_interval=2, use_neighborlist=true, nonbonded=:softrep,
                softrep_params=sp64, gamma=1.0, temperature=1.0, dt=1e-3, precision=:f64
            )
        catch e
            err64 = e
        end
        @test err64 === nothing
        if err64 === nothing
            @test st64.softrep !== nothing
            @test st64.softrep isa SoftRepulsiveParams{Float64}
            @test st64.softrep.ϵ == sp64.ϵ
            @test st64.softrep.σ == sp64.σ
        end
    end

    @testset "IR-003 test_api_exports_presence" begin
        function missing_exports(mod::Module)
            syms = names(mod; all=false, imported=false)
            return [s for s in syms if !isdefined(mod, s)]
        end

        missing_root = missing_exports(ParticleDynamics)
        missing_sim = missing_exports(Simulation)
        missing_writers = missing_exports(ParticleDynamics.Writers)

        @test isempty(missing_root)
        @test isempty(missing_sim)
        @test isempty(missing_writers)

        # eulerheun policy in Phase 3: keep bound but unexported at top-level.
        @test isdefined(ParticleDynamics, :eulerheun)
        @test !(:eulerheun in names(ParticleDynamics; all=false, imported=false))
    end

    @testset "IR-004 test_em_rebuild_collision_state_consistency" begin
        seed_all!(0xD00404)
        st = build_tiny2d(
            N=12, T=Float32, box=(30f0, 30f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(32), neigh_interval=1, use_neighborlist=true,
            nonbonded=:wca, gamma=1f0, temperature=0.5f0, dt=1f-3
        )
        st.typeid .= CuArray(vcat(fill(Int32(1), 6), fill(Int32(2), 6)))
        ParticleDynamics.enable_collision_counting!(st; ntypes=2, bins=:all_pairs)

        @test st.coll_prev !== nothing
        @test st.coll_counts !== nothing
        @test st.coll_bins !== nothing

        # Force a rebuild in the next EM step.
        st.nbh.target_interval = 1
        st.nbh.last_build_step = -1
        fill!(st.coll_prev, UInt8(0x07))
        prev_len = length(st.coll_prev)

        em = Simulation.eulermaruyama(st; gamma=1f0, temperature=0.5f0, dt=1f-3)
        err = nothing
        try
            Simulation.step!(st, em, 1f-3; compute_energy=false)
        catch e
            err = e
        end
        @test err === nothing

        @test st.nbh.last_build_step == 0
        @test st.coll_prev !== nothing
        @test st.coll_counts !== nothing
        @test length(st.coll_prev) == prev_len
        @test length(st.coll_counts) == 3

        counts = ParticleDynamics.collisions_read_counts!(st)
        @test all(>=(0), counts)

        sentinel = Int(CUDA.sum(ifelse.(st.coll_prev .== UInt8(0x07), Int32(1), Int32(0))))
        @test sentinel == 0
    end

    @testset "IR-005 BAOAB gamma=0 handling" begin
        seed_all!(0xD00505)
        st = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(32), neigh_interval=1, use_neighborlist=true,
            nonbonded=:wca, gamma=0f0, temperature=1f0, dt=1f-3
        )
        spec = Simulation.baoab(st; gamma=0f0, temperature=1f0, dt=1f-3)
        err = nothing
        try
            Simulation.step!(st, spec, 1f-3; compute_energy=false)
        catch e
            err = e
        end
        @test err isa ArgumentError
        msg = lowercase(sprint(showerror, err))
        @test occursin("baoab", msg)
        @test occursin("gamma > 0", msg)
    end

    @testset "IR-006 Brownian/EM gamma=0 handling" begin
        seed_all!(0xD00606)

        st_mid = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(32), neigh_interval=1, use_neighborlist=true,
            nonbonded=:wca, gamma=0f0, temperature=1f0, dt=1f-3
        )
        mid = Simulation.eulerheun(st_mid; gamma=0f0, temperature=1f0, dt=1f-3)
        err_mid = nothing
        try
            Simulation.step!(st_mid, mid, 1f-3; compute_energy=false)
        catch e
            err_mid = e
        end
        @test err_mid isa ArgumentError
        msg_mid = lowercase(sprint(showerror, err_mid))
        @test occursin("brownian", msg_mid)
        @test occursin("gamma > 0", msg_mid)

        st_em = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(32), neigh_interval=1, use_neighborlist=true,
            nonbonded=:wca, gamma=0f0, temperature=1f0, dt=1f-3
        )
        em = Simulation.eulermaruyama(st_em; gamma=0f0, temperature=1f0, dt=1f-3)
        err_em = nothing
        try
            Simulation.step!(st_em, em, 1f-3; compute_energy=false)
        catch e
            err_em = e
        end
        @test err_em isa ArgumentError
        msg_em = lowercase(sprint(showerror, err_em))
        @test occursin("euler-maruyama", msg_em)
        @test occursin("gamma > 0", msg_em)
    end
end
