function _workflow_writer_system(; N=8, T=Float64, metadata=Dict{Symbol,Any}())
    cfg = hex_random_2d(N, 1.0, 0.30; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(cfg; types=[:C, :H], typeids=typeids, masses=Dict(:C => 1.0, :H => 1.0), metadata=metadata)
end

function _workflow_writer_groups()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

function _workflow_writer_sim(system, groups, st; writers, observables=Observable[])
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

@testset "Workflow TableWriter" begin
    mktempdir() do tmp
        system = _workflow_writer_system(N=8, T=Float64)
        _, _, all_particles, groups = _workflow_writer_groups()
        st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
        copyto!(st.typeid, system.typeids)
        fill!(st.dq, 2.0)

        thermo = ThermodynamicObservable(all_particles; name=:all)
        bath = BathExchangeObservable(name=:bath)
        writer = TableWriter(
            joinpath(tmp, "workflow_obs.csv");
            every=1,
            observables=[
                thermo => [:temperature, :kinetic_energy],
                bath => [:heat],
            ],
        )

        sim = _workflow_writer_sim(system, groups, st; writers=[writer], observables=Observable[thermo, bath])
        prepare!(sim)
        consumed = ParticleDynamics.Workflow.write_scheduled_outputs!(sim, st.step)
        ParticleDynamics.Workflow.close_writers!(sim)

        @test consumed
        lines = readlines(joinpath(tmp, "workflow_obs.csv"))
        @test length(lines) == 2
        @test occursin("step", lines[1])
        @test occursin("all.temperature", lines[1])
        @test occursin("all.kinetic_energy", lines[1])
        @test occursin("bath.heat", lines[1])
    end
end

@testset "Workflow GSDWriter write_start and append" begin
    mktempdir() do tmp
        diameters = Float64[1.0 + 0.1 * i for i in 1:8]
        system = _workflow_writer_system(N=8, T=Float64, metadata=Dict(:diameters => diameters))
        _, _, _, groups = _workflow_writer_groups()
        st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
        copyto!(st.typeid, system.typeids)

        path = joinpath(tmp, "workflow_traj.gsd")
        writer = GSDWriter(path; every=5, write_start=true, diameter=:automatic, types=:automatic)
        sim = _workflow_writer_sim(system, groups, st; writers=[writer])
        prepare!(sim)
        ParticleDynamics.Workflow.write_initial_frames!(sim)
        ParticleDynamics.Workflow.close_writers!(sim)

        frame0 = ParticleDynamics.read_gsd_frame!(path)
        @test frame0.step == st.step
        @test frame0.types == ["C", "H"] || isempty(frame0.types)
        @test frame0.particle_properties[:diameter] ≈ Float32.(diameters)

        st.step = 5
        writer_append = GSDWriter(path; every=5, write_start=true, append=true, diameter=:automatic, types=:automatic)
        sim_append = _workflow_writer_sim(system, groups, st; writers=[writer_append])
        prepare!(sim_append)
        ParticleDynamics.Workflow.write_initial_frames!(sim_append)
        ParticleDynamics.Workflow.close_writers!(sim_append)

        frame_last = ParticleDynamics.read_gsd_frame!(path)
        frame_first = ParticleDynamics.read_gsd_frame!(path; step=0)
        @test frame_first.step == 0
        @test frame_last.step == 5
    end
end
