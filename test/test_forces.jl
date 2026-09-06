@testset "Force Kernels Smoke and Invariants" begin
    seed_all!(0xC1001)

    st_lj = build_tiny2d(
        N=2, T=Float32, box=(12f0, 12f0), use_neighborlist=false,
        nonbonded=:lj, gamma=0f0, temperature=0f0, epsilon=1f0, sigma=1f0
    )
    set_positions_2d!(st_lj, Float32[-0.75, 0.75], Float32[0.0, 0.0])
    set_velocities_2d!(st_lj, zeros(Float32, 2), zeros(Float32, 2))
    SimulationCore.step!(st_lj, SimulationCore.velocityverlet(st_lj; gamma=0f0, temperature=0f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_lj)
    @test abs(Float64(CUDA.sum(st_lj.fx))) <= 1e-5
    @test abs(Float64(CUDA.sum(st_lj.fy))) <= 1e-5

    st_wca = build_tiny2d(
        N=4, T=Float32, box=(16f0, 16f0), use_neighborlist=true,
        nonbonded=:wca, gamma=1f0, temperature=0.2f0, epsilon=2f0, sigma=1f0
    )
    SimulationCore.step!(st_wca, SimulationCore.velocityverlet(st_wca; gamma=1f0, temperature=0.2f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_wca)

    st_soft = build_tiny2d(
        N=4, T=Float32, box=(16f0, 16f0), use_neighborlist=true,
        nonbonded=:soft_repulsive, gamma=1f0, temperature=0.2f0, epsilon=10f0, sigma=1f0
    )
    SimulationCore.step!(st_soft, SimulationCore.velocityverlet(st_soft; gamma=1f0, temperature=0.2f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_soft)

    @testset "Stepped force matches direct evaluation with freeze spring" begin
        function build_force_eval_state()
            T = Float64
            st = SimulationCore.build_simulation(
                N=2,
                box=(T(12), T(12)),
                cutoff=T(2.5),
                skin=T(0.4),
                cap=Int32(4),
                neigh_interval=1,
                use_neighborlist=true,
                epsilon=T(0.8),
                sigma=T(1.0),
                gamma=T(0),
                temperature=T(0),
                bonds=[(Int32(1), Int32(2))],
                bonding=ParticleDynamics.harmonic_bond(k=T(5.0), r0=T(1.1)),
                nonbonded=:lj,
                precision=:f64,
            )
            set_positions_2d!(st, T[0.0, 1.35], T[0.0, 0.25])
            set_velocities_2d!(st, T[0.0, 0.0], T[0.0, 0.0])
            ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
            Filters.freeze_particles!(st; filter=Filters.Indices([1]), mode=:spring, k=T(7.5), include_energy=true)
            SimulationCore.zero_forces!(st)
            return st
        end

        st_direct = build_force_eval_state()
        st_step = build_force_eval_state()

        SimulationCore.step!(st_step, SimulationCore.velocityverlet(st_step; gamma=0.0, temperature=0.0, dt=0.001), 0.001; compute_energy=true)
        copyto!(st_direct.rx, st_step.rx)
        copyto!(st_direct.ry, st_step.ry)
        SimulationCore.evaluate_forces_into_f!(st_direct, true; freeze_spring=true)
        CUDA.synchronize()

        @test isapprox(Array(st_direct.fx), Array(st_step.fx); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.fy), Array(st_step.fy); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.Epot), Array(st_step.Epot); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.virial), Array(st_step.virial); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.virial_nonbonded), Array(st_step.virial_nonbonded); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.virial_bonded), Array(st_step.virial_bonded); atol=1e-12, rtol=1e-12)
        @test isapprox(Array(st_direct.virial_tensor), Array(st_step.virial_tensor); atol=1e-12, rtol=1e-12)
    end
end
