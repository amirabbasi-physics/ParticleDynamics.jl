using LinearAlgebra: dot, cross, norm

@testset "Energy minimization" begin
    methods = (("SD", SteepestDescent(initial_step=0.05)),
               ("FIRE", FIRE(dt_initial=0.005, dt_max=0.05, fictitious_mass=1.0)))

    function wf_state(positions, box, topo, forces; T=Float64, unwrapped=false, kwargs...)
        sys = ParticleSystem(positions; box=PeriodicBox(box), topology=topo)
        allp = Group(:all, AllSelection())
        obs = unwrapped ? [MSDObservable(allp)] : Observable[]
        sim = Simulation(sys; groups=Groups(allp), observables=obs,
                         integrator=Integrator(dt=1e-3, forces=forces, methods=[Langevin(allp; gamma=1.0, kT=0.1)]),
                         precision=T, seed=3)
        ParticleDynamics.Workflow.prepare!(sim)
        return state(sim)
    end
    hostpos(st) = hcat(Float64.(Array(st.rx)), Float64.(Array(st.ry)), (st.rz === nothing ? zeros(length(st.rx)) : Float64.(Array(st.rz))))
    mic(d, L) = d .- L .* round.(d ./ L)

    @testset "$name: stretched dimer, bent trimer, torsion" for (name, method) in methods
        # dimer: harmonic bond minimum at r0
        st = wf_state([(0.0, 0.0, 0.0), (1.6, 0.2, 0.0)], (10.0, 10.0, 10.0),
                      Topology(bonds=[(1, 2)]), Force[HarmonicBondForce(k=100.0, r0=1.0)])
        res = minimize!(st; method=method, force_tolerance=1e-8, max_displacement=0.1)
        @test res.converged && res.reason === :force_tolerance
        x = hostpos(st)
        @test abs(norm(x[2, :] - x[1, :]) - 1.0) < 1e-9
        @test res.final_energy < res.initial_energy && res.final_energy < 1e-15
        @test st.force_valid && st.energy_valid
        @test all(iszero, Array(st.vx))

        # bent trimer: two bonds + harmonic angle
        st = wf_state([(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (1.5, 1.4, 0.3)], (10.0, 10.0, 10.0),
                      Topology(bonds=[(1, 2), (2, 3)], angles=[(1, 2, 3)]),
                      Force[HarmonicBondForce(k=200.0, r0=1.0), HarmonicAngleForce(k=50.0, theta0=1.8)])
        res = minimize!(st; method=method, force_tolerance=1e-7, max_displacement=0.1)
        @test res.converged
        x = hostpos(st); a = x[1, :] - x[2, :]; b = x[3, :] - x[2, :]
        @test acos(dot(a, b) / (norm(a) * norm(b))) ≈ 1.8 atol=1e-8
        @test norm(a) ≈ 1.0 atol=1e-8

        # four-atom torsion: k(1+cos(φ-δ)) has its minimum at φ = δ + π
        δ = 0.7
        st = wf_state([(0.0, 1.0, 0.1), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.7, 0.8)], (10.0, 10.0, 10.0),
                      Topology(bonds=[(1, 2), (2, 3), (3, 4)], angles=[(1, 2, 3), (2, 3, 4)], dihedrals=[(1, 2, 3, 4)]),
                      Force[HarmonicBondForce(k=300.0, r0=1.0), HarmonicAngleForce(k=100.0, theta0=π / 2),
                            PeriodicDihedralForce(k=2.0, n=1, phase=δ)])
        # SD's Armijo test needs resolvable energy decreases; near this minimum
        # 1 + cos(φ-δ) cancels to ~2e-16·k, limiting SD to |F| ≈ 5e-7 in Float64.
        ftol = method isa SteepestDescent ? 1e-6 : 1e-7
        res = minimize!(st; method=method, force_tolerance=ftol, max_displacement=0.05, max_iterations=50_000,
                        max_evaluations=100_000)
        @test res.converged
        x = hostpos(st)
        b1, b2, b3 = x[2, :] - x[1, :], x[3, :] - x[2, :], x[4, :] - x[3, :]
        φ = atan(norm(b2) * dot(b1, cross(b2, b3)), dot(cross(b1, b2), cross(b2, b3)))
        @test abs(atan(sin(φ - δ - π), cos(φ - δ - π))) < 1e-5
    end

    @testset "$name: WCA packing with pinned subset ($T)" for (name, method) in methods, T in (Float64, Float32)
        rng = Random.MersenneTwister(11)
        N = 150; L = 14.0
        pos = [(L * rand(rng) - L / 2, L * rand(rng) - L / 2) for _ in 1:N]
        sys = ParticleSystem(pos; box=PeriodicBox((L, L)))
        allp = Group(:all, AllSelection())
        sim = Simulation(sys; groups=Groups(allp), observables=[MSDObservable(allp)],
                         integrator=Integrator(dt=1e-3,
                             forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist,
                                         neighborlist=CellList(buffer=0.3, capacity=48, rebuild_interval=1))],
                             methods=[Langevin(allp; gamma=1.0, kT=0.1)]),
                         precision=T, seed=5)
        ParticleDynamics.Workflow.prepare!(sim)
        st = state(sim)
        @test st.rx_unwrap !== nothing
        pinned = 1:10
        x0 = hostpos(st)
        # Attainable tolerances measured on this fixture: SD's energy-based line
        # search stalls once the total energy is within WCA's per-pair rounding
        # (Float32 SD: ~1e-2); FIRE only uses forces.
        tol = T === Float64 ? 1e-6 : (method isa SteepestDescent ? 1e-2 : 1e-4)
        res = minimize!(st; method=method, force_tolerance=tol, max_displacement=0.05,
                        mobile=setdiff(1:N, pinned), max_iterations=20_000, max_evaluations=60_000)
        @test res.converged
        @test res.final_energy < res.initial_energy
        @test res.max_force <= tol
        x = hostpos(st)
        @test x[pinned, :] == x0[pinned, :]                 # pinned coordinates never move
        # wrapped and unwrapped coordinates stay consistent
        xu = hcat(Float64.(Array(st.rx_unwrap)), Float64.(Array(st.ry_unwrap)))
        @test maximum(abs, mic(xu .- x[:, 1:2], L)) < (T === Float64 ? 1e-9 : 1e-3)
        @test res.evaluations >= res.iterations
    end

    @testset "Failure reporting" begin
        # exact overlap of two particles with a pair potential
        st = wf_state([(0.0, 0.0), (0.0, 0.0), (2.0, 0.0)], (10.0, 10.0), Topology(),
                      Force[WCA(epsilon=1.0, sigma=1.0)])
        for (_, method) in methods
            res = minimize!(st; method=method, force_tolerance=1e-6, max_displacement=0.1)
            @test !res.converged && res.reason === :invalid_initial_configuration
            @test occursin("overlap", res.message)
        end
        # FENE bond beyond its domain at the start
        st = wf_state([(0.0, 0.0), (2.0, 0.0)], (10.0, 10.0), Topology(bonds=[(1, 2)]),
                      Force[FENEBondForce(k=30.0, R0=1.5)])
        res = minimize!(st; method=SteepestDescent(initial_step=0.05), force_tolerance=1e-6, max_displacement=0.1)
        @test res.reason === :invalid_initial_configuration
        # all coordinates pinned
        st = wf_state([(0.0, 0.0), (1.4, 0.0)], (10.0, 10.0), Topology(bonds=[(1, 2)]),
                      Force[HarmonicBondForce(k=10.0, r0=1.0)])
        res = minimize!(st; method=FIRE(dt_initial=0.01, dt_max=0.1), force_tolerance=1e-6,
                        max_displacement=0.1, mobile=Int[])
        @test !res.converged && res.reason === :no_mobile_coordinates
        # exhausted budget is not convergence; state stays coherent
        for (_, method) in methods
            st = wf_state([(0.0, 0.0), (1.9, 0.0)], (10.0, 10.0), Topology(bonds=[(1, 2)]),
                          Force[HarmonicBondForce(k=10.0, r0=1.0)])
            res = minimize!(st; method=method, force_tolerance=1e-12, max_displacement=0.01, max_evaluations=5)
            @test !res.converged && res.reason in (:max_evaluations, :max_iterations)
            @test st.force_valid && st.energy_valid
            @test total_potential_energy(st) ≈ res.final_energy
        end
        # FENE trial steps beyond the domain are rejected, not fatal
        st = wf_state([(0.0, 0.0), (1.45, 0.0)], (10.0, 10.0), Topology(bonds=[(1, 2)]),
                      Force[FENEBondForce(k=30.0, R0=1.5), HarmonicBondForce(k=0.0, r0=0.0, type=:unused)])
        res = minimize!(st; method=SteepestDescent(initial_step=0.5), force_tolerance=1e-8, max_displacement=0.5)
        @test res.converged
    end

    @testset "SD and FIRE agree on a flexible chain" begin
        rng = Random.MersenneTwister(9)
        N = 16
        x = zeros(N, 3)
        for i in 2:N
            v = randn(rng, 3); x[i, :] = x[i - 1, :] .+ 1.1 .* v ./ norm(v)
        end
        pos = [Tuple(x[i, :]) for i in 1:N]
        topo = Topology(bonds=[(i, i + 1) for i in 1:N-1], bond_types=[isodd(i) ? :a : :b for i in 1:N-1],
                        angles=[(i, i + 1, i + 2) for i in 1:N-2], dihedrals=[(i, i + 1, i + 2, i + 3) for i in 1:N-3])
        # (FENE alone has its minimum at r = 0 once bonded pairs are excluded
        # from WCA, so this fixture uses two harmonic bond types.)
        forces = Force[HarmonicBondForce(k=100.0, r0=1.0, type=:a), HarmonicBondForce(k=60.0, r0=1.2, type=:b),
                       HarmonicAngleForce(k=20.0, theta0=2.0), PeriodicDihedralForce(k=[1.0, 0.5], n=[1, 3], phase=[0.0, 0.0]),
                       WCA(epsilon=1.0, sigma=1.0)]
        results = map(methods) do (_, method)
            st = wf_state(pos, (30.0, 30.0, 30.0), topo, forces)
            minimize!(st; method=method, force_tolerance=1e-6, max_displacement=0.05,
                      max_iterations=100_000, max_evaluations=200_000)
        end
        @test all(r -> r.converged, results)
        @test all(r -> r.max_force <= 1e-6, results)
        @info "flexible chain" SD=(results[1].iterations, results[1].evaluations, results[1].final_energy) FIRE=(results[2].iterations, results[2].evaluations, results[2].final_energy)
    end
end

@testset "Workflow minimize! without a thermostat" begin
    rng = Random.MersenneTwister(2)
    N = 60; L = 9.0
    pos = [(L * rand(rng) - L / 2, L * rand(rng) - L / 2) for _ in 1:N]
    sys = ParticleSystem(pos; box=PeriodicBox((L, L)), types=[:A, :B], typeids=Int32[i <= 5 ? 2 : 1 for i in 1:N])
    allp = Group(:all, AllSelection())
    mob = Group(:mobile, TypeSelection(:A))
    sim = Simulation(sys; groups=Groups(allp, mob), observables=[MSDObservable(allp)],
                     integrator=Integrator(dt=1e-3, forces=[WCA(epsilon=1.0, sigma=1.0)]), precision=Float64)
    res = minimize!(sim; method=FIRE(dt_initial=0.005, dt_max=0.05), force_tolerance=1e-6,
                    max_displacement=0.05, mobile=mob)
    @test res.converged
    st = state(sim)
    @test sim.metadata[:last_minimization] === res
    @test Float64.(Array(st.rx))[1:5] == [p[1] for p in pos[1:5]]       # type :B pinned
    @test st.step == 0
end
