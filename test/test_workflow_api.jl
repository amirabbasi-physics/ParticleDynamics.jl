function _workflow_api_groups()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

function _workflow_api_system(; N=8, T=Float64, velocities=nothing, metadata=Dict{Symbol,Any}())
    cfg = hex_random_2d(N, 1.0, 0.30; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(
        cfg;
        types=[:C, :H],
        typeids=typeids,
        masses=Dict(:C => 1.0, :H => 1.0),
        velocities=velocities,
        metadata=metadata,
    )
end

@testset "Workflow Simulation auto-builds low-level state" begin
    mktempdir() do tmp
        T = Float64
        N = 8
        velocities = [(T(0.1 * i), T(-0.05 * i)) for i in 1:N]
        system = _workflow_api_system(N=N, T=T; velocities=velocities, metadata=Dict(:step => 7))
        cold, hot, all_particles, groups = _workflow_api_groups()

        sim = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=1e-3,
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=32, rebuild_interval=4))],
                methods=[
                    Langevin(cold; gamma=2.0, kT=0.5),
                    Langevin(hot; gamma=4.0, kT=1.5),
                ],
            ),
            observables=Observable[CollisionObservable()],
            writers=Writer[GSDWriter(joinpath(tmp, "traj.gsd"); write_start=false, write_unwrapped=true)],
            precision=Float64,
            seed=0xC9A319,
        )

        prepare!(sim)
        st = state(sim)
        @test sim.prepared
        @test st !== nothing
        @test st isa SimulationCore.SimulationState{Float64}
        @test st.step == 7
        @test st.rx_unwrap !== nothing
        @test st.ry_unwrap !== nothing
        @test sim.lowlevel_integrator isa SimulationCore.VVSpec{Float64}
        @test Array(st.typeid) == system.typeids
        @test Array(st.rx) ≈ [p[1] for p in system.positions]
        @test Array(st.ry) ≈ [p[2] for p in system.positions]
        @test Array(st.vx) ≈ [v[1] for v in velocities]
        @test Array(st.vy) ≈ [v[2] for v in velocities]
        @test st.coll_enabled
    end
end

@testset "Workflow Simulation prepares pair tables and free-particle workflows" begin
    T = Float64
    cfg = hex_random_2d(8, 1.0, 0.25; T=T)
    typeids = Int32[1, 2, 1, 2, 1, 2, 1, 2]
    system = ParticleSystem(cfg; types=[:small, :large], typeids=typeids)
    _, _, all_particles, groups = _workflow_api_groups()

    pair_table = PairTable(
        sigma=[1.0 1.5; 1.5 2.0],
        epsilon=[1.0 0.5; 0.5 2.0],
        cutoff=[1.122462048309373 1.6836930724640595; 1.6836930724640595 2.244924096618746],
        type_names=[:small, :large],
    )

    sim_pair = Simulation(
        system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=1e-3,
            forces=[WCA(pair_table=pair_table, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=24, rebuild_interval=3))],
            methods=[Brownian(all_particles; gamma=3.0, kT=0.25)],
        ),
        precision=Float64,
    )

    prepare!(sim_pair)
    st_pair = state(sim_pair)
    @test st_pair !== nothing
    @test st_pair.sigma_pair !== nothing
    @test st_pair.epsilon_pair !== nothing
    @test st_pair.rcut_pair !== nothing
    @test Array(st_pair.typeid) == typeids

    free_system = _workflow_api_system(N=6, T=T)
    free_sim = Simulation(
        free_system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=5e-4,
            methods=[ActiveOrnsteinUhlenbeck(all_particles; gamma=1.0, tau=0.05, noise_scale=0.2)],
        ),
        observables=Observable[MSDObservable(all_particles)],
        precision=Float64,
    )

    prepare!(free_sim)
    st_free = state(free_sim)
    @test st_free !== nothing
    @test !haskey(free_sim.metadata, :compiled_forces)
    @test st_free.rx_unwrap !== nothing
    @test free_sim.lowlevel_integrator isa SimulationCore.EMSpec{Float64}
end

@testset "Workflow bonded unwrapped initialization respects periodic bonds" begin
    system = ParticleSystem(
        Float64[-4.8 0.0 0.0; 4.8 0.0 0.0];
        box=PeriodicBox((10.0, 10.0, 10.0)),
        topology=Topology(bonds=[(1, 2)]),
        masses=Dict(:A => 0.0),
    )
    all_particles = Group(:all, AllSelection())
    sim = Simulation(
        system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=1e-3,
            methods=[Brownian(all_particles; gamma=1.0, kT=0.0)],
        ),
        writers=Writer[GSDWriter(joinpath(mktempdir(), "traj.gsd"); write_start=false, write_unwrapped=true)],
        precision=Float64,
    )

    prepare!(sim)
    st = state(sim)
    rxu = Array(st.rx_unwrap)
    dx = rxu[2] - rxu[1]
    @test isapprox(abs(dx), 0.4; atol=1e-12, rtol=1e-12)
end
