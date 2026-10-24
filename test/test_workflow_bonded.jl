using LinearAlgebra: dot, cross, norm

@testset "Workflow typed bonded forces" begin
    mic(d, L) = d .- L .* round.(d ./ L)

    # Independent host formulas (not the package geometry code).
    angle_of(a, b) = acos(clamp(dot(a, b) / (norm(a) * norm(b)), -1, 1))
    dihedral_of(b1, b2, b3) = atan(norm(b2) * dot(b1, cross(b2, b3)), dot(cross(b1, b2), cross(b2, b3)))
    wrap(x) = atan(sin(x), cos(x))

    function flexible_system(; T=Float64, L=7.0)
        rng = Random.MersenneTwister(42)
        N = 12
        x = zeros(3, N)
        for i in 2:N
            v = randn(rng, 3); x[:, i] = x[:, i - 1] .+ 1.0 .* v ./ norm(v)
        end
        x .+= 5.0                                    # chain crosses the box boundary
        wrapped = mod.(x, L)
        bonds = [(i, i + 1) for i in 1:N-1]
        angles = [(i, i + 1, i + 2) for i in 1:N-2]
        dihedrals = [(i, i + 1, i + 2, i + 3) for i in 1:N-3]
        impropers = [(1, 2, 3, 4), (5, 6, 7, 8)]
        topo = Topology(bonds=bonds, bond_types=[isodd(i) ? :a : :b for i in 1:N-1],
                        angles=angles, angle_types=[i <= 4 ? :stiff : :soft for i in 1:N-2],
                        dihedrals=dihedrals, impropers=impropers)
        sys = ParticleSystem([Tuple(T.(wrapped[:, i])) for i in 1:N]; box=PeriodicBox((T(L), T(L), T(L))),
                             types=[:A], typeids=fill(Int32(1), N), masses=Dict(:A => 1.0), topology=topo)
        return sys, x, L
    end

    forces = Force[
        HarmonicBondForce(k=50.0, r0=1.0, type=:a),
        FENEBondForce(k=30.0, R0=1.6, type=:b),
        HarmonicAngleForce(k=20.0, theta0=1.9, type=:stiff),
        CosineSquaredAngleForce(k=8.0, theta0=2.0, type=:soft),
        PeriodicDihedralForce(k=[1.2, -0.3], n=[1, 3], phase=[0.2, 0.0]),
        HarmonicImproperForce(k=5.0, chi0=0.3),
    ]

    function reference_energy(x, L, sys)
        topo = sys.topology
        U = 0.0
        for (t, (i, j)) in enumerate(topo.bonds)
            r = norm(mic(x[:, j] - x[:, i], L))
            U += topo.bond_types[t] === :a ? 0.5 * 50.0 * (r - 1.0)^2 : -0.5 * 30.0 * 1.6^2 * log(1 - r^2 / 1.6^2)
        end
        for (t, (i, j, k)) in enumerate(topo.angles)
            θ = angle_of(mic(x[:, i] - x[:, j], L), mic(x[:, k] - x[:, j], L))
            U += topo.angle_types[t] === :stiff ? 0.5 * 20.0 * (θ - 1.9)^2 : 0.5 * 8.0 * (cos(θ) - cos(2.0))^2
        end
        for (i, j, k, l) in topo.dihedrals
            φ = dihedral_of(mic(x[:, j] - x[:, i], L), mic(x[:, k] - x[:, j], L), mic(x[:, l] - x[:, k], L))
            U += 1.2 * (1 + cos(φ - 0.2)) - 0.3 * (1 + cos(3φ))
        end
        for (i, j, k, l) in topo.impropers
            χ = dihedral_of(mic(x[:, j] - x[:, i], L), mic(x[:, k] - x[:, j], L), mic(x[:, l] - x[:, k], L))
            U += 0.5 * 5.0 * wrap(χ - 0.3)^2
        end
        return U
    end

    function prepared(sys, forces; T=Float64, methods=nothing)
        allp = Group(:all, AllSelection())
        m = methods === nothing ? [Langevin(allp; gamma=1.0, kT=0.5)] : methods
        sim = Simulation(sys; groups=Groups(allp), integrator=Integrator(dt=1e-3, forces=forces, methods=m),
                         precision=T, seed=7)
        ParticleDynamics.Workflow.prepare!(sim)
        return sim, state(sim)
    end

    for T in (Float64, Float32)
        @testset "Mixed families energy $T" begin
            sys, x, L = flexible_system(; T=T)
            sim, st = prepared(sys, forces; T=T)
            @test st.bonding === nothing
            @test st.bonded isa ParticleDynamics.BondedForces.BondedTerms
            @test st.nb_kind == SimulationCore.NB_KIND_NONE
            E = total_potential_energy(st)
            @test isapprox(E, reference_energy(x, L, sys); rtol=(T === Float64 ? 1e-10 : 2e-5))
            # isolated internal terms: zero net force
            f = hcat(Float64.(Array(st.fx)), Float64.(Array(st.fy)), Float64.(Array(st.fz)))
            @test maximum(abs, sum(f; dims=1)) <= (T === Float64 ? 1e-10 : 1e-3) * maximum(abs, f)
            run!(sim, Stage(:md, steps=20; progress=false))
            @test state_allfinite(st)
        end
    end

    @testset "Assignment and dimension errors" begin
        sys, _, _ = flexible_system()
        @test_throws ArgumentError prepared(sys, Force[HarmonicBondForce(k=1.0, r0=1.0, type=:a)])   # :b unassigned
        @test_throws ArgumentError prepared(sys, Force[HarmonicBondForce(k=1.0, r0=1.0, type=:a),
                                                       FENEBondForce(k=1.0, R0=2.0, type=:a),
                                                       FENEBondForce(k=1.0, R0=2.0, type=:b)])        # :a twice
        @test_throws ArgumentError prepared(sys, Force[PeriodicDihedralForce(k=1.0, n=0, phase=0.0)])  # n must be ≥ 1
        sys2d = ParticleSystem([(0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (2.5, 1.0)]; box=PeriodicBox((10.0, 10.0)),
                               topology=Topology(dihedrals=[(1, 2, 3, 4)]))
        @test_throws ArgumentError prepared(sys2d, Force[PeriodicDihedralForce(k=1.0, n=1, phase=0.0)])
        sysnoang = ParticleSystem([(0.0, 0.0), (1.0, 0.0)]; box=PeriodicBox((10.0, 10.0)),
                                  topology=Topology(bonds=[(1, 2)]))
        @test_throws ArgumentError prepared(sysnoang, Force[HarmonicAngleForce(k=1.0, theta0=2.0)])
    end

    @testset "Legacy uniform bond path and topology warnings" begin
        sys, _, _ = flexible_system()
        sys1 = ParticleSystem(sys.positions; box=sys.box, topology=Topology(bonds=sys.topology.bonds,
                                                                          angles=sys.topology.angles))
        local st
        @test_logs (:warn, r"angles but no angles force") match_mode=:any begin
            _, st = prepared(sys1, Force[HarmonicBondForce(k=50.0, r0=1.0)])
        end
        @test st.bonding isa ParticleDynamics.HarmonicBond{Float64}
        @test st.bonded === nothing
    end

    @testset "Type labels: canonical constructor and metadata migration" begin
        meta = Dict{Symbol,Any}(:angle_types => [:x, :y], :label => "m")
        topo = Topology([(Int32(1), Int32(2))], Symbol[], [(Int32(1), Int32(2), Int32(3)), (Int32(2), Int32(3), Int32(4))],
                        NTuple{4,Int32}[], NTuple{4,Int32}[], Tuple{Int32,Int32}[], Vector{Int32}[], meta)
        @test topo.angle_types == [:x, :y]
        @test !haskey(topo.metadata, :angle_types) && topo.metadata[:label] == "m"
        @test haskey(meta, :angle_types)                       # caller's dict untouched
        @test_throws ArgumentError Topology(angles=[(1, 2, 3)], angle_types=[:a, :b])
        @test_throws ArgumentError Topology(angles=[(1, 2, 1)])
        @test Topology(dihedrals=[(1, 2, 3, 4)]).dihedral_types == Symbol[]
    end

    @testset "OPLS and RB compile to the documented functions" begin
        sys, x, L = flexible_system()
        topo = Topology(dihedrals=sys.topology.dihedrals)
        s = ParticleSystem(sys.positions; box=sys.box, topology=topo)
        K = (1.4, -0.6, 0.9, 0.2)
        _, st = prepared(s, Force[OPLSDihedralForce(K1=K[1], K2=K[2], K3=K[3], K4=K[4])])
        C = (9.28, 12.16, -13.12, -3.06, 26.24, -31.5)
        _, st_rb = prepared(s, Force[RyckaertBellemansDihedralForce(C=C)])
        Uo = 0.0; Ur = 0.0
        for (i, j, k, l) in topo.dihedrals
            φ = dihedral_of(mic(x[:, j] - x[:, i], L), mic(x[:, k] - x[:, j], L), mic(x[:, l] - x[:, k], L))
            Uo += 0.5 * (K[1] * (1 + cos(φ)) + K[2] * (1 - cos(2φ)) + K[3] * (1 + cos(3φ)) + K[4] * (1 - cos(4φ)))
            Ur += sum(C[n + 1] * cos(φ - π)^n for n in 0:5)
        end
        @test total_potential_energy(st) ≈ Uo rtol=1e-10
        @test total_potential_energy(st_rb) ≈ Ur rtol=1e-10
    end
end
