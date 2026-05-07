function _workflow_runloop_system(; N=8, T=Float64)
    cfg = hex_random_2d(N, 1.0, 0.30; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(cfg; types=[:C, :H], typeids=typeids, masses=Dict(:C => 1.0, :H => 1.0))
end

function _workflow_runloop_groups()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

function _workflow_runloop_sim(system, groups, st; writers=Writer[], observables=Observable[])
    cold, hot, _ = groups[1], groups[2], groups[3]
    return Simulation(
        system;
        groups=groups,
        observables=observables,
        writers=writers,
        integrator=Integrator(
            dt=1e-3,
            forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist)],
            methods=[
                Langevin(cold; gamma=2.0, kT=0.5),
                Langevin(hot; gamma=4.0, kT=1.5),
            ],
        ),
        state=st,
        precision=Float64,
    )
end

@testset "Workflow run! stage overrides and writers" begin
    mktempdir() do tmp
        system = _workflow_runloop_system(N=8, T=Float64)
        _, _, all_particles, groups = _workflow_runloop_groups()
        st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
        copyto!(st.typeid, system.typeids)

        thermo = ThermodynamicObservable(all_particles; name=:all)
        writers = Writer[
            TableWriter(joinpath(tmp, "stage_obs.csv"); every=2, observables=[thermo => [:potential_energy_accumulated]]),
            GSDWriter(joinpath(tmp, "stage_traj.gsd"); every=2, write_start=true),
        ]
        sim = _workflow_runloop_sim(system, groups, st; writers=writers, observables=Observable[thermo])

        @test !sim.prepared
        original_interval = st.neigh_interval
        run!(sim, Stage(:warmup, steps=2; dt=5e-4, neighbor_rebuild_interval=1, progress=false))

        @test sim.prepared
        @test st.step == 2
        @test st.neigh_interval == original_interval
        @test sim.lowlevel_integrator isa SimulationCore.VVSpec{Float64}
        @test isapprox(sim.lowlevel_integrator.params.dt, 1e-3; atol=1e-12)
        @test isapprox(sim.metadata[:workflow_time], 1e-3; atol=1e-12)

        lines = readlines(joinpath(tmp, "stage_obs.csv"))
        @test length(lines) == 2
        frame0 = ParticleDynamics.read_gsd_frame!(joinpath(tmp, "stage_traj.gsd"); step=0)
        frame_last = ParticleDynamics.read_gsd_frame!(joinpath(tmp, "stage_traj.gsd"))
        @test frame0.step == 0
        @test frame_last.step == 2
    end
end

@testset "Workflow run! auto compute_energy and reset controls" begin
    system = _workflow_runloop_system(N=8, T=Float64)
    _, _, all_particles, groups = _workflow_runloop_groups()

    st_no_energy = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
    copyto!(st_no_energy.typeid, system.typeids)
    sim_no_energy = _workflow_runloop_sim(system, groups, st_no_energy)
    run!(sim_no_energy, 2)
    @test all(Array(st_no_energy.Epot_accum) .== 0)
    @test all(Array(st_no_energy.virial_accum) .== 0)

    st_reset = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
    copyto!(st_reset.typeid, system.typeids)
    thermo = ThermodynamicObservable(all_particles; name=:all_reset)
    sim_reset = _workflow_runloop_sim(system, groups, st_reset; observables=Observable[thermo])
    fill!(st_reset.Epot_accum, 1.0)
    fill!(st_reset.dq, 2.0)
    run!(sim_reset, Stage(:production, steps=10_000; reset_observables=true, reset_step=0, max_seconds=0.0, progress=false))
    @test 1 <= st_reset.step < 10_000
    @test sim_reset.metadata[:workflow_time] > 0
end
