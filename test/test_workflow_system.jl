using StaticArrays: SVector

@testset "Workflow ParticleSystem and Topology" begin
    cfg = hex_random_2d(8, 1.0, 0.35; T=Float64)
    typeids = Int32[1, 2, 1, 2, 1, 2, 1, 2]
    topo = Topology(
        bonds=[(1, 2), (2, 3)],
        bond_types=[:backbone, :backbone],
        exclusions=[(1, 2)],
        molecules=[[1, 2, 3]],
        metadata=Dict(:label => "chain"),
    )

    system = ParticleSystem(
        cfg;
        types=[:C, :H],
        typeids=typeids,
        masses=Dict(:C => 1.0, :H => 2.0),
        topology=topo,
    )

    @test length(system) == 8
    @test system.box isa PeriodicBox{Float64,2}
    @test system.positions[1] isa SVector{2,Float64}
    @test system.typeids == typeids
    @test system.types == [:C, :H]
    @test system.masses[:C] == 1.0
    @test system.topology.bonds == Tuple{Int32,Int32}[(1, 2), (2, 3)]
    @test system.topology.bond_types == [:backbone, :backbone]
    @test system.topology.exclusions == Tuple{Int32,Int32}[(1, 2)]
    @test system.topology.metadata[:label] == "chain"
    @test haskey(system.metadata, :indices)
    @test haskey(system.metadata, :sites)

    matrix_positions = [0.0 0.0 0.0; 1.0 0.0 0.5; -1.0 1.5 -0.5]
    matrix_velocities = [0.1 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.3]
    system3 = ParticleSystem(
        matrix_positions;
        box=(12.0, 10.0, 8.0),
        velocities=matrix_velocities,
        typeids=[1, 1, 1],
    )
    @test system3.box isa PeriodicBox{Float64,3}
    @test system3.positions[2] == SVector(1.0, 0.0, 0.5)
    @test system3.velocities[3] == SVector(0.0, 0.0, 0.3)
    @test system3.types == [:T1]

    @test_throws ArgumentError ParticleSystem(cfg; types=[:C, :H])
    @test_throws ArgumentError ParticleSystem(cfg; types=[:C, :H], typeids=Int32[1, 2, 1, 2, 1, 2, 1, 3])
    @test_throws ArgumentError ParticleSystem(cfg; typeids=Int32[1, 1, 1, 1, 1, 1, 1, 1], masses=Dict(:B => 1.0))
    @test_throws ArgumentError Topology(bonds=[(1, 2)], bond_types=[:a, :b])
end

@testset "Workflow ParticleSystem from GSD" begin
    seed_all!(0xA45E)

    mktempdir() do tmp
        path = joinpath(tmp, "workflow_system.gsd")

        st = build_tiny2d(
            N=6, T=Float32, box=(18f0, 18f0), cutoff=2.5f0, skin=0.3f0,
            cap=Int32(24), neigh_interval=5, use_neighborlist=true,
            nonbonded=:wca, gamma=1f0, temperature=0.2f0,
        )
        st.typeid .= CuArray(Int32[1, 2, 1, 2, 1, 2])

        h = ParticleDynamics.gsd_open(path)
        try
            ParticleDynamics.write_gsd_frame!(h, st; step=11, types_names=["cold", "hot"])
        finally
            ParticleDynamics.gsd_close(h)
        end

        system = ParticleSystem.from_gsd(path)
        @test length(system) == 6
        @test system.box isa PeriodicBox{Float32,2}
        @test system.types == [:cold, :hot] || system.types == [:T1, :T2]
        @test system.typeids == Int32[1, 2, 1, 2, 1, 2]
        @test system.positions[1] isa SVector{2,Float32}
        @test system.velocities !== nothing
        @test system.metadata[:source_path] == path
        @test system.metadata[:step] == 11
        @test haskey(system.metadata, :configuration)
    end
end
