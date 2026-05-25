@testset "Workflow Force Mapping" begin
    cfg = hex_random_2d(12, 1.0, 0.30; T=Float64)
    system = ParticleSystem(
        cfg;
        types=[:A, :B],
        typeids=Int32[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        masses=Dict(:A => 1.0, :B => 1.5),
    )

    ff = ForceField()
    ParticleDynamics.Workflow.add!(ff, LennardJones(epsilon=2.0, sigma=1.1, cutoff=2.7, pairs=:all))
    @test length(ff) == 1
    @test first(collect(ff)) isa LennardJones

    compiled_lj = ParticleDynamics.Workflow.compile_forces(system, ff; precision=:f64)
    @test compiled_lj.build_kwargs[:nonbonded] == :lj
    @test compiled_lj.build_kwargs[:use_neighborlist] == false
    @test compiled_lj.build_kwargs[:epsilon] == 2.0
    @test compiled_lj.build_kwargs[:sigma] == 1.1
    @test compiled_lj.build_kwargs[:cutoff] == 2.7
    @test compiled_lj.metadata[:pair_style] == :uniform

    compiled_soft = ParticleDynamics.Workflow.compile_forces(
        system,
        [SoftRepulsive(epsilon=3.0, sigma=1.2, cutoff=1.2, pairs=:neighborlist, neighborlist=CellList(buffer=0.2, capacity=48, rebuild_interval=7))];
        precision=:f32,
    )
    @test compiled_soft.build_kwargs[:nonbonded] == :soft_repulsive
    @test compiled_soft.build_kwargs[:use_neighborlist] == true
    @test compiled_soft.build_kwargs[:cap] == Int32(48)
    @test compiled_soft.build_kwargs[:neigh_interval] == 7
    @test compiled_soft.build_kwargs[:softrep_params] isa ParticleDynamics.SoftRepulsiveParams{Float32}

    @test_throws ArgumentError ParticleDynamics.Workflow.compile_forces(
        system,
        [WCA(epsilon=1.0, sigma=1.0, pairs=:all), LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5, pairs=:all)];
        precision=:f64,
    )
end

@testset "Workflow PairTable and Bond Force Mapping" begin
    cfg = hex_random_2d(8, 1.0, 0.25; T=Float64)
    typeids = Int32[1, 2, 1, 2, 1, 2, 1, 2]
    topo = Topology(bonds=[(1, 2), (2, 3), (3, 4)])
    system = ParticleSystem(cfg; types=[:small, :large], typeids=typeids, topology=topo)

    pair_table = PairTable(
        sigma=[1.0 1.5; 1.5 2.0],
        epsilon=[1.0 0.5; 0.5 2.0],
        cutoff=[1.122462048309373 1.6836930724640595; 1.6836930724640595 2.244924096618746],
        type_names=[:small, :large],
    )
    wca = WCA(pair_table=pair_table, pairs=:neighborlist, neighborlist=CellList(buffer=0.25, capacity=32, rebuild_interval=5))
    compiled_pair = ParticleDynamics.Workflow.compile_forces(system, [wca]; precision=:f64)

    @test compiled_pair.build_kwargs[:nonbonded] == :wca
    @test compiled_pair.build_kwargs[:use_neighborlist] == true
    @test compiled_pair.build_kwargs[:cap] == Int32(32)
    @test compiled_pair.build_kwargs[:neigh_interval] == 5
    @test compiled_pair.metadata[:pair_style] == :pair_table

    kwargs = merge(
        Dict{Symbol,Any}(
            :N => length(system),
            :box => Tuple(system.box),
            :gamma => 1.0,
            :temperature => 0.0,
            :dt => 1e-3,
            :precision => :f64,
            :bonds => topo.bonds,
        ),
        compiled_pair.build_kwargs,
    )
    st = SimulationCore.build_simulation(; kwargs...)
    copyto!(st.rx, Float64[p[1] for p in system.positions])
    copyto!(st.ry, Float64[p[2] for p in system.positions])
    ParticleDynamics.Workflow.post_build!(compiled_pair, st)

    @test Array(st.typeid) == typeids
    @test st.sigma_pair !== nothing
    @test st.epsilon_pair !== nothing
    @test st.rcut_pair !== nothing
    @test size(st.sigma_pair) == (2, 2)
    @test SimulationCore._nonbonded_interaction(st).coefficients isa ParticleDynamics.NonBondedInteractions.PairMatrixCoefficients{Float64}
    @test SimulationCore._nonbonded_interaction(st).exclusions isa ParticleDynamics.NonBondedInteractions.BondExclusions

    harmonic = HarmonicBondForce(k=300.0, r0=1.1)
    compiled_bond = ParticleDynamics.Workflow.compile_forces(system, [harmonic]; precision=:f64)
    @test compiled_bond.build_kwargs[:bonds] == topo.bonds
    @test compiled_bond.build_kwargs[:bonding] isa ParticleDynamics.HarmonicBond{Float64}

    fene = FENEBondForce(k=30.0, R0=1.5)
    compiled_fene = ParticleDynamics.Workflow.compile_forces(system, [fene]; precision=:f64)
    @test compiled_fene.build_kwargs[:bonds] == topo.bonds
    @test compiled_fene.build_kwargs[:bonding] isa ParticleDynamics.FENEBond{Float64}

    kwargs_bond = merge(
        Dict{Symbol,Any}(
            :N => length(system),
            :box => Tuple(system.box),
            :gamma => 1.0,
            :temperature => 0.0,
            :dt => 1e-3,
            :precision => :f64,
            :cutoff => 2.5,
            :epsilon => 1.0,
            :sigma => 1.0,
            :nonbonded => :wca,
        ),
        compiled_bond.build_kwargs,
    )
    st_bond = SimulationCore.build_simulation(; kwargs_bond...)
    @test st_bond.bonds !== nothing
    @test st_bond.bonding isa ParticleDynamics.HarmonicBond{Float64}

    kwargs_fene = merge(
        Dict{Symbol,Any}(
            :N => length(system),
            :box => Tuple(system.box),
            :gamma => 1.0,
            :temperature => 0.0,
            :dt => 1e-3,
            :precision => :f64,
            :cutoff => 2.5,
            :epsilon => 1.0,
            :sigma => 1.0,
            :nonbonded => :wca,
        ),
        compiled_fene.build_kwargs,
    )
    st_fene = SimulationCore.build_simulation(; kwargs_fene...)
    @test st_fene.bonds !== nothing
    @test st_fene.bonding isa ParticleDynamics.FENEBond{Float64}
end

@testset "Workflow PairTable pairs=:all runs with bonded exclusions" begin
    system = ParticleSystem(
        [
            (-2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.5, 0.0, 0.0),
        ];
        box=PeriodicBox((12.0, 12.0, 12.0)),
        types=[:A],
        typeids=Int32[1, 1, 1],
        topology=Topology(bonds=[(1, 2)]),
    )
    all_particles = Group(:all, AllSelection())
    pair_table = PairTable(
        sigma=reshape([1.0], 1, 1),
        epsilon=reshape([1.0], 1, 1),
        cutoff=reshape([2^(1 / 6)], 1, 1),
        type_names=[:A],
    )

    sim = Simulation(
        system;
        groups=Groups(all_particles),
        integrator=Integrator(
            dt=1.0e-4,
            scheme=EulerMaruyama(),
            forces=[WCA(pair_table=pair_table, pairs=:all)],
            methods=[Brownian(all_particles; gamma=1.0, kT=0.0)],
        ),
        precision=Float64,
        seed=1,
    )

    @test_nowarn run!(sim, Stage(:production, steps=1; compute_energy=false, progress=false))
end

@testset "Workflow PairTable ignores zero-strength cutoff branches" begin
    cfg = hex_random_2d(4, 1.0, 0.20; T=Float64)
    system = ParticleSystem(cfg; types=[:A, :B], typeids=Int32[1, 1, 2, 2])

    pair_table = PairTable(
        sigma=[1.0 5.0; 5.0 2.0],
        epsilon=[1.0 0.0; 0.0 2.0],
        cutoff=[1.2 30.0; 30.0 2.4],
        type_names=[:A, :B],
    )
    compiled = ParticleDynamics.Workflow.compile_forces(
        system,
        [WCA(pair_table=pair_table, pairs=:neighborlist, neighborlist=CellList(buffer=0.2, capacity=16, rebuild_interval=4))];
        precision=:f64,
    )

    cutoff_eff = compiled.metadata[:cutoff_pair]
    @test cutoff_eff == [1.2 0.0; 0.0 2.4]
    @test compiled.build_kwargs[:cutoff] == 2.4

    kwargs = merge(
        Dict{Symbol,Any}(
            :N => length(system),
            :box => Tuple(system.box),
            :gamma => 1.0,
            :temperature => 0.0,
            :dt => 1e-3,
            :precision => :f64,
        ),
        compiled.build_kwargs,
    )
    st = SimulationCore.build_simulation(; kwargs...)
    copyto!(st.rx, Float64[p[1] for p in system.positions])
    copyto!(st.ry, Float64[p[2] for p in system.positions])
    ParticleDynamics.Workflow.post_build!(compiled, st)
    @test Array(st.rcut_pair) == cutoff_eff
end

@testset "Workflow warns on saturated initial neighbor capacity" begin
    system = ParticleSystem(
        [
            [-0.20, 0.0],
            [0.20, 0.0],
            [0.0, 0.20],
            [0.0, -0.20],
        ];
        box=PeriodicBox((4.0, 4.0)),
    )
    sim = Simulation(
        system;
        integrator=Integrator(
            dt=1e-3,
            forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist,
                        neighborlist=CellList(buffer=0.2, capacity=1, rebuild_interval=4))],
        ),
        precision=Float64,
    )
    @test_logs (:warn, r"Neighbor list reached the configured per-particle capacity") match_mode=:any prepare!(sim)
end

@testset "Workflow Force Compilation in prepare!" begin
    cfg = hex_random_2d(6, 1.0, 0.30; T=Float64)
    system = ParticleSystem(cfg; typeids=fill(Int32(1), 6))
    force = WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.3, capacity=24, rebuild_interval=4))
    integ = Integrator(dt=1e-3, forces=[force])
    sim = Simulation(system; integrator=integ)
    prepare!(sim)
    @test sim.prepared
    @test haskey(sim.metadata, :compiled_forces)
    @test sim.metadata[:compiled_forces] isa ParticleDynamics.Workflow.CompiledForces
end
