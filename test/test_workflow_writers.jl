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

        @test consumed.wrote_any
        @test consumed.interval_consumed
        lines = readlines(joinpath(tmp, "workflow_obs.csv"))
        @test length(lines) == 2
        @test occursin("step", lines[1])
        @test occursin("all.temperature", lines[1])
        @test occursin("all.kinetic_energy", lines[1])
        @test occursin("bath.heat", lines[1])
    end
end

@testset "Workflow TableWriter log formatting" begin
    mktempdir() do tmp
        system = _workflow_writer_system(N=8, T=Float64)
        _, _, all_particles, groups = _workflow_writer_groups()
        st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
        copyto!(st.typeid, system.typeids)
        fill!(st.dq, 2.0)

        thermo = ThermodynamicObservable(all_particles; name=:all)
        bath = BathExchangeObservable(name=:bath)
        collisions = CollisionObservable(name=:collisions)
        writer = TableWriter(
            joinpath(tmp, "workflow_obs.log");
            every=1,
            observables=[
                thermo => [:temperature, :potential_energy, :virial],
                bath => [:heat, :entropy_production_rate],
                collisions => [:counts],
            ],
        )

        sim = _workflow_writer_sim(system, groups, st; writers=[writer], observables=Observable[thermo, bath, collisions])
        prepare!(sim)
        consumed = ParticleDynamics.Workflow.write_scheduled_outputs!(sim, st.step)
        ParticleDynamics.Workflow.close_writers!(sim)

        @test consumed.wrote_any
        @test consumed.interval_consumed
        lines = readlines(joinpath(tmp, "workflow_obs.log"))
        @test length(lines) == 2
        @test occursin("Time", lines[1])
        @test occursin("Temperature", lines[1])
        @test occursin("E_pot", lines[1])
        @test occursin("virial", lines[1])
        @test occursin("Bath Energy", lines[1])
        @test occursin("Entropy Production Rate", lines[1])
        @test occursin("collision rate", lines[1])
        @test occursin("|", lines[1])
        @test !occursin("bath.heat", lines[1])
        @test !occursin(",", lines[1])
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

@testset "Workflow GSDWriter virial writes require energy" begin
    mktempdir() do tmp
        system = _workflow_writer_system(N=8, T=Float64)
        _, _, all_particles, groups = _workflow_writer_groups()
        st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
        copyto!(st.typeid, system.typeids)

        path = joinpath(tmp, "workflow_virial.gsd")
        writer = GSDWriter(path; every=1, write_start=false, write_virial=true, diameter=:automatic, group=all_particles)
        sim = _workflow_writer_sim(system, groups, st; writers=[writer])
        prepare!(sim)

        @test ParticleDynamics.Workflow.active_writer_requires_energy(sim, st.step + 1)

        run!(sim, Stage(:virial, steps=1; progress=false))
        frame = ParticleDynamics.read_gsd_frame!(path)
        @test frame.step == 1
        @test haskey(frame.particle_properties, :virial)
        @test size(frame.particle_properties[:virial], 1) == length(system)
    end
end

@testset "Workflow GSDWriter emits particle images for bonded PBC systems" begin
    mktempdir() do tmp
        system = ParticleSystem(
            Float64[-4.8 0.0 0.0; 4.8 0.0 0.0];
            box=PeriodicBox((10.0, 10.0, 10.0)),
            topology=Topology(bonds=[(1, 2)]),
            masses=Dict(:A => 0.0),
        )
        all_particles = Group(:all, AllSelection())
        writer = GSDWriter(joinpath(tmp, "bonded_images.gsd"); every=1, write_start=true, write_unwrapped=true)
        sim = Simulation(
            system;
            groups=Groups(all_particles),
            writers=[writer],
            integrator=Integrator(
                dt=1e-3,
                methods=[Brownian(all_particles; gamma=1.0, kT=0.0)],
            ),
            precision=Float64,
        )

        prepare!(sim)
        ParticleDynamics.Workflow.write_initial_frames!(sim)
        ParticleDynamics.Workflow.close_writers!(sim)

        frame = ParticleDynamics.read_gsd_frame!(joinpath(tmp, "bonded_images.gsd"))
        @test haskey(frame.particle_properties, :image)
        image = frame.particle_properties[:image]
        @test size(image) == (2, 3)
        @test image[1, 2] == 0 && image[1, 3] == 0
        @test image[2, 2] == 0 && image[2, 3] == 0
        @test abs(image[2, 1] - image[1, 1]) == 1
    end
end

@testset "Writer session survives stage close and repeated prepare" begin
    mktempdir() do tmp
        system = ParticleSystem([(0.0, 0.0), (2.0, 0.0)]; box=(20.0, 20.0), velocities=[(0.0, 0.0), (0.0, 0.0)])
        group = Group(:all, AllSelection())
        thermo = ThermodynamicObservable(group)
        table = joinpath(tmp, "stages.csv")
        trajectory = joinpath(tmp, "stages.gsd")
        write(table, "previous run must be replaced\n")
        writers = [TableWriter(table; every=1, observables=[thermo => [:kinetic_energy]]),
                   GSDWriter(trajectory; every=1, write_start=true)]
        make_sim() = Simulation(system; groups=Groups(group), writers=writers,
            integrator=Integrator(dt=0.001, forces=[LennardJones(epsilon=0.0, sigma=1.0, cutoff=2.5)],
                                  methods=[ConstantVolume(group)]))
        sim = make_sim()
        run!(sim, 2)
        run!(sim, 2)
        lines = readlines(table)
        @test length(lines) == 5
        @test [parse(Int, first(split(line, ','))) for line in lines[2:end]] == collect(1:4)
        @test read_gsd_frame!(trajectory; step=0).step == 0
        @test read_gsd_frame!(trajectory; step=2).step == 2
        @test read_gsd_frame!(trajectory).step == 4
        @test all(ctx -> ctx.handle === nothing, ParticleDynamics.Workflow._writer_contexts(sim))
        # prepare! on the same writer objects retains the session, even if the
        # previous handles have already been closed at a stage boundary.
        prepare!(sim)
        run!(sim, 1)
        @test length(readlines(table)) == 6
        @test read_gsd_frame!(trajectory; step=0).step == 0
        @test read_gsd_frame!(trajectory).step == 5
        # A different Simulation starts a new replace-mode writer session.
        fresh = make_sim()
        run!(fresh, 1)
        @test length(readlines(table)) == 2
        @test read_gsd_frame!(trajectory).step == 1
    end
end

@testset "Initial writer failure restores stage overrides and closes handles" begin
    mktempdir() do tmp
        system = ParticleSystem([(0.0, 0.0), (2.0, 0.0)]; box=(20.0, 20.0))
        group = Group(:all, AllSelection())
        blocker = joinpath(tmp, "not_a_directory")
        write(blocker, "block output path")
        sim = Simulation(system; groups=Groups(group),
            writers=[GSDWriter(joinpath(tmp, "opened_first.gsd"); write_start=true),
                     GSDWriter(joinpath(blocker, "cannot_open.gsd"); write_start=true)],
            integrator=Integrator(dt=0.001, methods=[ConstantVolume(group)]))
        prepare!(sim)
        previous_spec = sim.lowlevel_integrator
        previous_interval = state(sim).neigh_interval
        @test_throws Base.IOError run!(sim, Stage(:failing, steps=1; dt=0.002, neighbor_rebuild_interval=1, progress=false))
        @test state(sim).neigh_interval == previous_interval
        @test sim.lowlevel_integrator === previous_spec
        @test state(sim).step == 0
        @test all(ctx -> ctx.handle === nothing, ParticleDynamics.Workflow._writer_contexts(sim))
    end
end
