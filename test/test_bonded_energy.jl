@testset "Bonded dimer energy, force and virial" begin
    # Isolated stretched dimer with no nonbonded term: the summed per-particle
    # energy must contain each bond once.
    harmonic_U(r, k, r0) = k * (r - r0)^2 / 2
    harmonic_dUdr(r, k, r0) = k * (r - r0)
    fene_U(r, k, R0) = -k * R0^2 * log(1 - r^2 / R0^2) / 2
    fene_dUdr(r, k, R0) = k * r / (1 - r^2 / R0^2)

    cases = (
        (:harmonic, ParticleDynamics.harmonic_bond(k=7.0, r0=1.1), (r -> harmonic_U(r, 7.0, 1.1)), (r -> harmonic_dUdr(r, 7.0, 1.1))),
        (:fene, ParticleDynamics.fene_bond(k=30.0, r0=1.5), (r -> fene_U(r, 30.0, 1.5)), (r -> fene_dUdr(r, 30.0, 1.5))),
    )
    for (name, bonding, U, dUdr) in cases, D in (2, 3), T in (Float32, Float64)
        @testset "$name D=$D $T" begin
            L = 12.0
            box = ntuple(_ -> T(L), D)
            st = SimulationCore.build_simulation(N=2, box=box, cutoff=T(2.5), cap=Int32(4), neigh_interval=1,
                gamma=T(0), temperature=T(0), bonds=[(Int32(1), Int32(2))], bonding=bonding,
                nonbonded=:none, precision=(T === Float64 ? :f64 : :f32), spatial_reorder=false)
            # The bond crosses the periodic x boundary.
            pos = [[5.6, 0.0, 0.0], [-5.7, 0.4, 0.2]]
            copyto!(st.rx, T[pos[1][1], pos[2][1]])
            copyto!(st.ry, T[pos[1][2], pos[2][2]])
            D == 3 && copyto!(st.rz, T[pos[1][3], pos[2][3]])
            d = [mod(pos[1][a] - pos[2][a] + L / 2, L) - L / 2 for a in 1:D]
            r = sqrt(sum(abs2, d))
            F1 = -dUdr(r) .* d ./ r

            SimulationCore.evaluate_forces_into_f!(st, true)
            rtol = T === Float64 ? 1e-12 : 2e-5
            @test isapprox(sum(Float64.(Array(st.Epot))), U(r); rtol=rtol)
            f = [Float64.(Array(c)) for c in (st.fx, st.fy, st.fz)[1:D]]
            @test isapprox([f[a][1] for a in 1:D], F1; rtol=rtol, atol=rtol * maximum(abs, F1))
            @test isapprox([f[a][2] for a in 1:D], -F1; rtol=rtol, atol=rtol * maximum(abs, F1))
            # Configurational virial trace: d . F1 for the single bond.
            W = Float64.(Array(st.virial_bonded))
            @test isapprox(sum(W[:, 1:D]), sum(d .* F1); rtol=rtol, atol=rtol * abs(sum(d .* F1)))

            # Force-only evaluation reproduces the same forces.
            fe = [copy(x) for x in f]
            SimulationCore.evaluate_forces_into_f!(st, false)
            f2 = [Float64.(Array(c)) for c in (st.fx, st.fy, st.fz)[1:D]]
            @test all(isapprox(f2[a], fe[a]; rtol=rtol, atol=rtol * maximum(abs, F1)) for a in 1:D)
        end
    end

    @testset "Workflow without a pair force has no hidden pair term" begin
        # 1-2 bonded at rest length; 3 is unbonded and 0.9 from 2.
        sys = ParticleSystem([(0.0, 0.0), (1.1, 0.0), (2.0, 0.0)];
                             box=PeriodicBox((20.0, 20.0)), types=[:A], typeids=fill(Int32(1), 3),
                             masses=Dict(:A => 1.0), topology=Topology(bonds=[(Int32(1), Int32(2))]))
        for forces in (Force[HarmonicBondForce(k=50.0, r0=1.1)], Force[])
            all_particles = Group(:all, AllSelection())
            sim = Simulation(sys; groups=Groups(all_particles),
                             integrator=Integrator(dt=1e-3, forces=forces,
                                                   methods=[Brownian(all_particles; gamma=1.0, kT=0.0)]),
                             precision=Float64, seed=1)
            ParticleDynamics.Workflow.prepare!(sim)
            st = state(sim)
            @test st.nb_kind == SimulationCore.NB_KIND_NONE
            SimulationCore.evaluate_forces_into_f!(st, true)
            @test all(iszero, Array(st.fx))
            @test sum(Array(st.Epot)) == 0
        end
    end

    @testset "Low-level harmonic energy-only overload" begin
        T = Float64
        bl = ParticleDynamics.BondedForces.build_bondlist(2, [(1, 2)])
        rx = CuArray(T[0.0, 1.3]); ry = CuArray(T[0.0, 0.4]); rz = CuArray(T[0.0, 0.2])
        fx = CUDA.zeros(T, 2); fy = CUDA.zeros(T, 2); fz = CUDA.zeros(T, 2); E = CUDA.zeros(T, 2)
        p = ParticleDynamics.Definitions.HarmonicBondParams{T}(T(7), T(1.1))
        ParticleDynamics.BondedForces.harmonic_forces_soa!(rx, ry, rz, fx, fy, fz, E, bl, (T(12), T(12), T(12)), p)
        @test isapprox(sum(Array(E)), harmonic_U(sqrt(1.3^2 + 0.4^2 + 0.2^2), 7.0, 1.1); rtol=1e-12)
        fill!(E, zero(T))
        ParticleDynamics.BondedForces.harmonic_forces_soa!(rx, ry, fx, fy, E, bl, (T(12), T(12)), p)
        @test isapprox(sum(Array(E)), harmonic_U(hypot(1.3, 0.4), 7.0, 1.1); rtol=1e-12)
    end
end
