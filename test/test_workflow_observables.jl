function _workflow_observable_system(; N=8, T=Float64)
    cfg = hex_random_2d(N, 1.0, 0.30; T=T)
    typeids = Int32[isodd(i) ? 1 : 2 for i in 1:N]
    return ParticleSystem(cfg; types=[:C, :H], typeids=typeids, masses=Dict(:C => 1.0, :H => 1.0))
end

function _workflow_observable_groups()
    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(:H))
    all_particles = Group(:all, AllSelection())
    return cold, hot, all_particles, Groups(cold, hot, all_particles)
end

@testset "Workflow Observables and Reset" begin
    system = _workflow_observable_system(N=8, T=Float64)
    cold, hot, all_particles, groups = _workflow_observable_groups()
    st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.2, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    thermo = ThermodynamicObservable(all_particles; name=:all_thermo)
    bath = BathExchangeObservable(name=:bath)
    collisions = CollisionObservable(name=:collisions)
    msd = MSDObservable(all_particles; name=:msd)
    vacf = VACFObservable(all_particles; name=:vacf)

    sim = Simulation(
        system;
        groups=groups,
        observables=[thermo, bath, collisions, msd, vacf],
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
    @test haskey(sim.metadata, :workflow_observables)
    @test st.coll_enabled

    thermodata = ParticleDynamics.Workflow.sample_observable(
        sim,
        thermo;
        fields=[:temperature, :kinetic_energy, :potential_energy, :total_energy, :virial],
    )
    @test isfinite(thermodata.temperature)
    @test isfinite(thermodata.kinetic_energy)
    @test isfinite(thermodata.potential_energy)
    @test isfinite(thermodata.total_energy)
    @test isfinite(thermodata.virial)

    fill!(st.Ekin_accum, 1.0)
    fill!(st.Epot_accum, 2.0)
    fill!(st.virial_accum, 3.0)
    fill!(st.dq, 4.0)
    fill!(st.dU, 5.0)

    thermo_acc = ParticleDynamics.Workflow.sample_observable(
        sim,
        thermo;
        fields=[:kinetic_energy_accumulated, :potential_energy_accumulated, :virial_accumulated],
    )
    @test thermo_acc.kinetic_energy_accumulated > 0
    @test thermo_acc.potential_energy_accumulated > 0
    @test thermo_acc.virial_accumulated > 0

    bath_before = ParticleDynamics.Workflow.sample_observable(sim, bath; fields=[:heat, :entropy])
    @test bath_before.heat != 0
    @test bath_before.entropy != 0

    coll_before = ParticleDynamics.Workflow.sample_observable(sim, collisions; fields=[:counts, :pair_counts])
    @test coll_before.counts >= 0
    @test length(coll_before.pair_counts) == 3

    st.rx .+= 0.25
    st.vx .*= -1
    CUDA.synchronize()
    msd_before = ParticleDynamics.Workflow.sample_observable(sim, msd; fields=[:msd]).msd
    vacf_before = ParticleDynamics.Workflow.sample_observable(sim, vacf; fields=[:vacf]).vacf
    @test msd_before > 0
    @test isfinite(vacf_before)

    reset_observables!(sim)

    thermo_after = ParticleDynamics.Workflow.sample_observable(
        sim,
        thermo;
        fields=[:kinetic_energy_accumulated, :potential_energy_accumulated, :virial_accumulated],
    )
    bath_after = ParticleDynamics.Workflow.sample_observable(sim, bath; fields=[:heat, :entropy])
    msd_after = ParticleDynamics.Workflow.sample_observable(sim, msd; fields=[:msd]).msd

    @test thermo_after.kinetic_energy_accumulated == 0
    @test thermo_after.potential_energy_accumulated == 0
    @test thermo_after.virial_accumulated == 0
    @test bath_after.heat == 0
    @test bath_after.entropy == 0
    @test isapprox(msd_after, 0.0; atol=1e-12)
end

@testset "Workflow Bath Observable for Deterministic Thermostats" begin
    system = _workflow_observable_system(N=8, T=Float64)
    cold, hot, _, groups = _workflow_observable_groups()
    st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.0, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    sim = Simulation(
        system;
        groups=groups,
        observables=[BathExchangeObservable(name=:bath_md)],
        integrator=Integrator(
            dt=1e-3,
            methods=[
                ConstantVolume(cold; thermostat=CSVR(kT=0.75, tau=0.10)),
                ConstantVolume(hot; thermostat=CSVR(kT=1.50, tau=0.20)),
            ],
        ),
        state=st,
        precision=Float64,
    )
    prepare!(sim)
    SimulationCore.step!(st, sim.lowlevel_integrator, 1e-3; compute_energy=true)
    sim.metadata[:workflow_time] = st.step * sim.integrator.dt

    bath_md = ParticleDynamics.Workflow.sample_observable(
        sim,
        BathExchangeObservable(name=:bath_md);
        fields=[:temperature_error, :extended_hamiltonian],
    )
    @test isfinite(bath_md.temperature_error)
    @test isfinite(bath_md.extended_hamiltonian)
end

@testset "Workflow Bath Observable for NVE" begin
    system = _workflow_observable_system(N=8, T=Float64)
    _, _, all_particles, groups = _workflow_observable_groups()
    st = build_tiny2d(N=8, T=Float64, nonbonded=:wca, temperature=0.0, dt=1e-3)
    copyto!(st.typeid, system.typeids)

    sim = Simulation(
        system;
        groups=groups,
        observables=[BathExchangeObservable(name=:bath_nve)],
        integrator=Integrator(
            dt=1e-3,
            methods=[ConstantVolume(all_particles)],
        ),
        state=st,
        precision=Float64,
    )
    prepare!(sim)
    SimulationCore.step!(st, sim.lowlevel_integrator, 1e-3; compute_energy=true)
    sim.metadata[:workflow_time] = st.step * sim.integrator.dt

    bath_nve = ParticleDynamics.Workflow.sample_observable(
        sim,
        BathExchangeObservable(name=:bath_nve);
        fields=[:heat, :entropy, :temperature_error, :extended_hamiltonian],
    )
    obs = SimulationCore.collect_step_observables(st, sim.lowlevel_integrator)
    @test bath_nve.heat == 0.0
    @test bath_nve.entropy == 0.0
    @test bath_nve.temperature_error == 0.0
    @test bath_nve.extended_hamiltonian ≈ obs.Etot atol=1e-12
end
