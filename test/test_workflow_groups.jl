@testset "Workflow Groups" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    system = ParticleSystem(
        positions;
        box=PeriodicBox((10.0, 10.0)),
        types=[:C, :H],
        typeids=Int32[1, 2, 1, 2],
        masses=Dict(:C => 1.0, :H => 1.0),
    )

    cold = Group(:cold, TypeSelection(:C))
    hot = Group(:hot, TypeSelection(2))
    all_particles = Group(:all, AllSelection())
    subset = Group(:subset, IndexSelection([1, 3]))

    groups = Groups(cold, hot, all_particles, subset)
    @test groups[:cold].name == :cold
    @test groups[:hot].selection isa TypeSelection
    @test length(groups) == 4

    @test_throws ArgumentError Groups(cold, Group(:cold, AllSelection()))

    cold_filter = ParticleDynamics.Workflow.materialize_group(system, cold)
    hot_filter = ParticleDynamics.Workflow.materialize_group(system, hot)
    all_filter = ParticleDynamics.Workflow.materialize_group(system, all_particles)
    subset_filter = ParticleDynamics.Workflow.materialize_group(system, subset)

    @test cold_filter isa Filters.TypeIDs
    @test cold_filter.ids == [1]
    @test hot_filter isa Filters.TypeIDs
    @test hot_filter.ids == [2]
    @test all_filter isa Filters.All
    @test subset_filter isa Filters.Indices
    @test subset_filter.idx == [1, 3]

    @test_throws ArgumentError ParticleDynamics.Workflow.materialize_group(system, Group(:ghost, TypeSelection(:Z)))
    @test_throws ArgumentError ParticleDynamics.Workflow.materialize_group(system, Group(:empty, IndexSelection(Int[])))

    sim = Simulation(system; groups=groups)
    prepare!(sim)
    @test sim.prepared
    @test haskey(sim.metadata, :materialized_groups)
    @test sim.metadata[:materialized_groups][:cold] isa Filters.TypeIDs
    @test sim.metadata[:materialized_groups][:all] isa Filters.All
end
