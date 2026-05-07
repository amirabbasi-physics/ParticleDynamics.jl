function _workflow_smoke_groups_two_type()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

function _workflow_smoke_groups_single()
    all_particles = Group(:all, AllSelection())
    return all_particles, Groups(all_particles)
end

function _workflow_smoke_system_2d_two_type(; N=12, T=Float64)
    cfg = hex_random_2d(N, 1.0, 0.28; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(
        cfg;
        types=[:C, :H],
        typeids=typeids,
        masses=Dict(:C => 1.0, :H => 1.0),
    )
end

function _workflow_smoke_system_2d_single(; N=12, T=Float64)
    cfg = hex_random_2d(N, 1.0, 0.28; T=T)
    return ParticleSystem(
        cfg;
        types=[:A],
        typeids=fill(Int32(1), N),
        masses=Dict(:A => 1.0),
    )
end

function _workflow_smoke_system_3d_single(; N=16, T=Float64)
    cfg = fcc_random_3d(N, 1.0, 0.10; T=T)
    return ParticleSystem(
        cfg;
        types=[:A],
        typeids=fill(Int32(1), N),
        masses=Dict(:A => 1.0),
    )
end

function _workflow_smoke_pair_table()
    return PairTable(
        sigma=[1.0 1.4; 1.4 1.8],
        epsilon=[1.0 0.8; 0.8 1.2],
        cutoff=[1.122462048309373 1.571446867633122; 1.571446867633122 2.020431686956871],
        type_names=[:C, :H],
    )
end

function _workflow_polymer_system(; N=6, T=Float64)
    positions = [(T(0.9 * (i - 1)), zero(T)) for i in 1:N]
    topo = Topology(bonds=[(Int32(i), Int32(i + 1)) for i in 1:(N - 1)])
    return ParticleSystem(
        positions;
        box=PeriodicBox((T(20), T(20))),
        types=[:A],
        typeids=fill(Int32(1), N),
        masses=Dict(:A => 1.0),
        topology=topo,
    )
end

function _run_workflow_smoke(sim; steps=3)
    run!(sim, Stage(:smoke, steps=steps; progress=false))
    st = state(sim)
    @test st !== nothing
    @test st.step == steps
    @test state_allfinite(st)
    return st
end

@testset "Workflow example-style smoke runs" begin
    T = Float64

    @testset "2D WCA Langevin VelocityVerlet" begin
        system = _workflow_smoke_system_2d_single(N=12, T=T)
        all_particles, groups = _workflow_smoke_groups_single()
        sim = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=1e-3,
                scheme=VelocityVerlet(),
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=48, rebuild_interval=4))],
                methods=[Langevin(all_particles; gamma=2.0, kT=0.5)],
            ),
            precision=Float64,
            seed=0xA1,
        )
        st = _run_workflow_smoke(sim)
        @test sim.lowlevel_integrator isa SimulationCore.VVSpec{Float64}
        @test st.rz === nothing
    end

    @testset "3D WCA Brownian EulerMaruyama" begin
        system = _workflow_smoke_system_3d_single(N=16, T=T)
        all_particles, groups = _workflow_smoke_groups_single()
        sim = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                scheme=EulerMaruyama(),
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=64, rebuild_interval=3))],
                methods=[Brownian(all_particles; gamma=3.0, kT=0.25)],
            ),
            precision=Float64,
            seed=0xA2,
        )
        st = _run_workflow_smoke(sim)
        @test sim.lowlevel_integrator isa SimulationCore.EMSpec{Float64}
        @test st.rz !== nothing
    end

    @testset "Mixed Active OU and Brownian" begin
        system = _workflow_smoke_system_2d_two_type(N=12, T=T)
        cold, hot, _, groups = _workflow_smoke_groups_two_type()
        sim = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                scheme=EulerMaruyama(),
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=48, rebuild_interval=4))],
                methods=[
                    ActiveOrnsteinUhlenbeck(cold; gamma=2.0, kT=0.0, tau=0.05, noise_scale=0.2),
                    Brownian(hot; gamma=3.0, kT=0.5),
                ],
            ),
            precision=Float64,
            seed=0xA3,
        )
        _run_workflow_smoke(sim)
        @test sim.lowlevel_integrator isa SimulationCore.EMSpec{Float64}
        @test sim.lowlevel_integrator.params.ou !== nothing
    end

    @testset "Pair-table two-size run" begin
        system = _workflow_smoke_system_2d_two_type(N=12, T=T)
        _, _, all_particles, groups = _workflow_smoke_groups_two_type()
        sim = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                forces=[WCA(pair_table=_workflow_smoke_pair_table(), pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=48, rebuild_interval=4))],
                methods=[Brownian(all_particles; gamma=2.5, kT=0.35)],
            ),
            precision=Float64,
            seed=0xA4,
        )
        st = _run_workflow_smoke(sim)
        @test st.sigma_pair !== nothing
        @test st.epsilon_pair !== nothing
        @test st.rcut_pair !== nothing
    end

    @testset "Bonded polymer harmonic and FENE" begin
        all_particles, groups = _workflow_smoke_groups_single()

        harmonic_system = _workflow_polymer_system(N=6, T=T)
        sim_harm = Simulation(
            harmonic_system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                forces=[
                    WCA(epsilon=1.0, sigma=1.0, pairs=:all),
                    HarmonicBondForce(k=50.0, r0=0.9),
                ],
                methods=[Brownian(all_particles; gamma=4.0, kT=0.05)],
            ),
            precision=Float64,
            seed=0xA5,
        )
        st_harm = _run_workflow_smoke(sim_harm; steps=2)
        @test st_harm.bonds !== nothing
        @test st_harm.bonding isa ParticleDynamics.HarmonicBond{Float64}

        fene_system = _workflow_polymer_system(N=6, T=T)
        sim_fene = Simulation(
            fene_system;
            groups=groups,
            integrator=Integrator(
                dt=1e-4,
                forces=[
                    WCA(epsilon=1.0, sigma=1.0, pairs=:all),
                    FENEBondForce(k=10.0, R0=1.5),
                ],
                methods=[Brownian(all_particles; gamma=4.0, kT=0.01)],
            ),
            precision=Float64,
            seed=0xA6,
        )
        st_fene = _run_workflow_smoke(sim_fene; steps=2)
        @test st_fene.bonds !== nothing
        @test st_fene.bonding isa ParticleDynamics.FENEBond{Float64}
    end

    @testset "Two-temperature CSVR and NHC" begin
        system = _workflow_smoke_system_2d_two_type(N=12, T=T)
        cold, hot, _, groups = _workflow_smoke_groups_two_type()

        sim_csvr = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=48, rebuild_interval=4))],
                methods=[
                    ConstantVolume(cold; thermostat=CSVR(kT=0.5, tau=0.01)),
                    ConstantVolume(hot; thermostat=CSVR(kT=1.5, tau=0.02)),
                ],
            ),
            precision=Float64,
            seed=0xA7,
        )
        _run_workflow_smoke(sim_csvr)
        @test sim_csvr.lowlevel_integrator isa SimulationCore.CSVRSpec{Float64}

        sim_nhc = Simulation(
            system;
            groups=groups,
            integrator=Integrator(
                dt=5e-4,
                forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=48, rebuild_interval=4))],
                methods=[
                    ConstantVolume(cold; thermostat=NoseHooverChain(kT=0.6, tau=0.02, chain_length=4, substeps=3)),
                    ConstantVolume(hot; thermostat=NoseHooverChain(kT=1.4, tau=0.03, chain_length=4, substeps=3)),
                ],
            ),
            precision=Float64,
            seed=0xA8,
        )
        _run_workflow_smoke(sim_nhc)
        @test sim_nhc.lowlevel_integrator isa SimulationCore.NHCSpec{Float64}
    end
end
