function _workflow_groups_for_two_types()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

function _workflow_system_two_types(; N=8, T=Float64, masses=Dict(:C => 1.0, :H => 1.0))
    cfg = hex_random_2d(N, 1.0, 0.30; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(cfg; types=[:C, :H], typeids=typeids, masses=masses)
end

function _materialized_group_map(system, groups)
    out = Dict{Symbol,Any}()
    for group in groups
        out[group.name] = ParticleDynamics.Workflow.materialize_group(system, group)
    end
    return out
end

@testset "Workflow Langevin and Brownian Integrator Mapping" begin
    system = _workflow_system_two_types(N=8, T=Float64)
    cold, hot, _, groups = _workflow_groups_for_two_types()
    materialized = _materialized_group_map(system, groups)

    st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.0, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    dt = 1e-3
    langevin = Integrator(
        dt=dt,
        scheme=BAOAB(),
        methods=[
            Langevin(cold; gamma=2.0, kT=0.5),
            Langevin(hot; gamma=4.0, kT=1.5),
        ],
    )
    compiled_langevin = ParticleDynamics.Workflow.compile_integrator(system, langevin; precision=:f64)
    spec_langevin = ParticleDynamics.Workflow.build_lowlevel_integrator(compiled_langevin, st; system=system, materialized_groups=materialized)
    @test spec_langevin isa SimulationCore.BAOABSpec{Float64}
    gamma_host = Array(spec_langevin.params.gamma)
    noise_host = Array(spec_langevin.params.noise_scale)
    for i in eachindex(gamma_host)
        if isodd(i)
            @test gamma_host[i] == 2.0
            @test isapprox(noise_host[i], sqrt(2 * 2.0 * 0.5 * dt); atol=1e-12)
        else
            @test gamma_host[i] == 4.0
            @test isapprox(noise_host[i], sqrt(2 * 4.0 * 1.5 * dt); atol=1e-12)
        end
    end

    active_ou_vv = Integrator(
        dt=dt,
        scheme=VelocityVerlet(),
        methods=[
            ActiveOrnsteinUhlenbeck(cold; gamma=3.0, kT=0.0, tau=0.25, noise_scale=0.4),
            Langevin(hot; gamma=5.0, kT=1.25),
        ],
    )
    compiled_ou_vv = ParticleDynamics.Workflow.compile_integrator(system, active_ou_vv; precision=:f64)
    spec_ou_vv = ParticleDynamics.Workflow.build_lowlevel_integrator(compiled_ou_vv, st; system=system, materialized_groups=materialized)
    @test spec_ou_vv isa SimulationCore.VVSpec{Float64}
    @test spec_ou_vv.params.ou !== nothing
    gamma_ou_vv = Array(spec_ou_vv.params.gamma)
    noise_ou_vv = Array(spec_ou_vv.params.noise_scale)
    corr_ou_vv = Array(spec_ou_vv.params.corr_time)
    for i in eachindex(gamma_ou_vv)
        if isodd(i)
            @test gamma_ou_vv[i] == 3.0
            @test noise_ou_vv[i] == 0.4
            @test corr_ou_vv[i] == 0.25
        else
            @test gamma_ou_vv[i] == 5.0
            @test isapprox(noise_ou_vv[i], sqrt(2 * 5.0 * 1.25 * dt); atol=1e-12)
            @test corr_ou_vv[i] == 0.0
        end
    end

    active_ou = Integrator(
        dt=dt,
        scheme=EulerMaruyama(),
        methods=[
            ActiveOrnsteinUhlenbeck(cold; gamma=3.0, kT=0.0, spectrum=OUSpectrum([0.05, 0.2], [0.4, 0.1])),
            Brownian(hot; gamma=5.0, kT=1.25),
        ],
    )
    compiled_ou = ParticleDynamics.Workflow.compile_integrator(system, active_ou; precision=:f64)
    spec_ou = ParticleDynamics.Workflow.build_lowlevel_integrator(compiled_ou, st; system=system, materialized_groups=materialized)
    @test spec_ou isa SimulationCore.EMSpec{Float64}
    @test ParticleDynamics.IntegratorInterfaces.stage_sequence(spec_ou) == (:em_position, :force)
    @test spec_ou.params.ou !== nothing
    @test size(spec_ou.params.ou.tau, 1) == 2
    gamma_ou = Array(spec_ou.params.gamma)
    for i in eachindex(gamma_ou)
        @test gamma_ou[i] == (isodd(i) ? 3.0 : 5.0)
    end
end

@testset "Workflow Deterministic Thermostat Mapping" begin
    system = _workflow_system_two_types(N=8, T=Float64)
    cold, hot, _, groups = _workflow_groups_for_two_types()
    materialized = _materialized_group_map(system, groups)

    st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.0, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    csvr_integrator = Integrator(
        dt=1e-3,
        methods=[
            ConstantVolume(cold; thermostat=CSVR(kT=0.75, tau=0.10)),
            ConstantVolume(hot; thermostat=CSVR(kT=1.50, tau=0.20)),
        ],
    )
    compiled_csvr = ParticleDynamics.Workflow.compile_integrator(system, csvr_integrator; precision=:f64)
    spec_csvr = ParticleDynamics.Workflow.build_lowlevel_integrator(compiled_csvr, st; system=system, materialized_groups=materialized)
    @test spec_csvr isa SimulationCore.CSVRSpec{Float64}
    @test spec_csvr.params.target_temperature == [0.75, 1.50]
    @test spec_csvr.params.tau == [0.10, 0.20]
    @test Array(spec_csvr.workspace.particle_bath_id) == Int32[1, 2, 1, 2, 1, 2, 1, 2]

    nhc_integrator = Integrator(
        dt=1e-3,
        methods=[
            ConstantVolume(cold; thermostat=NoseHooverChain(kT=0.80, tau=0.15, chain_length=4, substeps=3)),
            ConstantVolume(hot; thermostat=NoseHooverChain(kT=1.20, tau=0.25, chain_length=4, substeps=3)),
        ],
    )
    compiled_nhc = ParticleDynamics.Workflow.compile_integrator(system, nhc_integrator; precision=:f64)
    spec_nhc = ParticleDynamics.Workflow.build_lowlevel_integrator(compiled_nhc, st; system=system, materialized_groups=materialized)
    @test spec_nhc isa SimulationCore.NHCSpec{Float64}
    @test spec_nhc.params.chain_length == 4
    @test spec_nhc.params.substeps == 3
    @test spec_nhc.params.target_temperature == [0.80, 1.20]
    @test spec_nhc.params.tau == [0.15, 0.25]
    @test Array(spec_nhc.workspace.particle_bath_id) == Int32[1, 2, 1, 2, 1, 2, 1, 2]
end

@testset "Workflow Deterministic Thermostats Support Heterogeneous Masses" begin
    seed_all!(0xA2105)
    system = _workflow_system_two_types(N=8, T=Float64, masses=Dict(:C => 1.0, :H => 2.0))
    cold, hot, _, groups = _workflow_groups_for_two_types()

    sim_csvr = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            methods=[
                ConstantVolume(cold; thermostat=CSVR(kT=0.75, tau=0.10)),
                ConstantVolume(hot; thermostat=CSVR(kT=1.50, tau=0.20)),
            ],
        ),
        precision=Float64,
    )
    prepare!(sim_csvr)
    @test sim_csvr.prepared
    @test sim_csvr.state !== nothing
    @test sim_csvr.lowlevel_integrator isa SimulationCore.CSVRSpec{Float64}
    @test sim_csvr.state.mass_particle isa CuArray{Float64,1}
    @test sim_csvr.state.inv_mass_particle isa CuArray{Float64,1}
    @test Array(sim_csvr.state.mass_particle) == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    @test Array(sim_csvr.state.inv_mass_particle) == [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
    SimulationCore.step!(sim_csvr.state, sim_csvr.lowlevel_integrator, 1e-3; compute_energy=false)
    @test state_allfinite(sim_csvr.state)

    sim_nhc = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            methods=[
                ConstantVolume(cold; thermostat=NoseHooverChain(kT=0.80, tau=0.15, chain_length=4, substeps=3)),
                ConstantVolume(hot; thermostat=NoseHooverChain(kT=1.20, tau=0.25, chain_length=4, substeps=3)),
            ],
        ),
        precision=Float64,
    )
    prepare!(sim_nhc)
    @test sim_nhc.prepared
    @test sim_nhc.state !== nothing
    @test sim_nhc.lowlevel_integrator isa SimulationCore.NHCSpec{Float64}
    SimulationCore.step!(sim_nhc.state, sim_nhc.lowlevel_integrator, 1e-3; compute_energy=false)
    @test state_allfinite(sim_nhc.state)
end

@testset "Workflow Brownian Mass Warning" begin
    seed_all!(0xA2106)
    system = _workflow_system_two_types(N=8, T=Float64, masses=Dict(:C => 1.0, :H => 2.0))
    cold, hot, _, groups = _workflow_groups_for_two_types()

    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            methods=[
                Brownian(cold; gamma=2.0, kT=0.5),
                Brownian(hot; gamma=4.0, kT=1.5),
            ],
        ),
        precision=Float64,
    )

    @test_logs (:warn, r"Brownian dynamics ignores particle masses") match_mode=:any prepare!(sim)
    @test sim.prepared
    @test sim.lowlevel_integrator isa SimulationCore.EMSpec{Float64}
end

@testset "Workflow Langevin Supports Heterogeneous Masses" begin
    system = _workflow_system_two_types(N=8, T=Float64, masses=Dict(:C => 1.0, :H => 2.0))
    cold, hot, _, groups = _workflow_groups_for_two_types()
    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            scheme=BAOAB(),
            methods=[
                Langevin(cold; gamma=2.0, kT=0.5),
                Langevin(hot; gamma=4.0, kT=1.5),
            ],
        ),
        precision=Float64,
    )
    prepare!(sim)
    @test sim.prepared
    @test sim.state !== nothing
    @test sim.lowlevel_integrator isa SimulationCore.BAOABSpec{Float64}
    @test sim.state.mass_particle isa CuArray{Float64,1}
    @test sim.state.inv_mass_particle isa CuArray{Float64,1}
    @test Array(sim.state.mass_particle) == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    SimulationCore.step!(sim.state, sim.lowlevel_integrator, 1e-3; compute_energy=false)
    @test state_allfinite(sim.state)
end

@testset "Workflow Langevin Rejects Zero Masses" begin
    system = _workflow_system_two_types(N=8, T=Float64, masses=Dict(:C => 0.0, :H => 1.0))
    cold, hot, _, groups = _workflow_groups_for_two_types()
    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            methods=[
                Langevin(cold; gamma=2.0, kT=0.5),
                Langevin(hot; gamma=4.0, kT=1.5),
            ],
        ),
        precision=Float64,
    )
    @test_throws ArgumentError prepare!(sim)
end

@testset "Workflow Deterministic MD Rejects Zero Masses" begin
    system = _workflow_system_two_types(N=8, T=Float64, masses=Dict(:C => 0.0, :H => 1.0))
    cold, hot, _, groups = _workflow_groups_for_two_types()
    sim = Simulation(
        system;
        groups=groups,
        integrator=Integrator(
            dt=1e-3,
            methods=[
                ConstantVolume(cold; thermostat=CSVR(kT=0.75, tau=0.10)),
                ConstantVolume(hot; thermostat=CSVR(kT=1.50, tau=0.20)),
            ],
        ),
        precision=Float64,
    )
    @test_throws ArgumentError prepare!(sim)
end

@testset "Workflow Integrator prepare!" begin
    system = _workflow_system_two_types(N=6, T=Float64)
    cold, hot, all_particles, groups = _workflow_groups_for_two_types()
    st = build_tiny2d(N=6, T=Float64, nonbonded=:wca, temperature=0.0, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    sim = Simulation(
        system;
        groups=groups,
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
    prepare!(sim)
    @test sim.prepared
    @test haskey(sim.metadata, :compiled_integrator)
    @test sim.lowlevel_integrator isa SimulationCore.VVSpec{Float64}
    @test Array(sim.lowlevel_integrator.params.gamma) == [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]
    @test haskey(sim.metadata, :compiled_forces)
    @test sim.state !== nothing
end
