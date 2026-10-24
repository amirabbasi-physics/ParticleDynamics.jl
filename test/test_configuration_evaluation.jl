@testset "Conservative configuration evaluation" begin
    function build_eval_state(T)
        st = SimulationCore.build_simulation(N=3, box=(T(10), T(10), T(10)), cutoff=T(2.5), cap=Int32(8),
            neigh_interval=1, epsilon=T(1), sigma=T(1), gamma=T(1), temperature=T(0.5),
            bonds=[(Int32(1), Int32(2))], bonding=ParticleDynamics.harmonic_bond(k=20.0, r0=1.0),
            nonbonded=:wca, precision=(T === Float64 ? :f64 : :f32), spatial_reorder=false)
        copyto!(st.rx, T[0.0, 1.2, 2.1]); copyto!(st.ry, T[0.0, 0.1, 0.3]); copyto!(st.rz, T[0.0, -0.1, 0.2])
        SimulationCore.invalidate_forces!(st)
        return st
    end

    for T in (Float32, Float64)
        @testset "$T" begin
            st = build_eval_state(T)
            snapshot(st) = (st.step, Array(st.dq), Array(st.dU), Array(st.Epot_accum), Array(st.virial_accum))
            before = snapshot(st)

            evaluate_configuration!(st, EvaluationRequest())
            @test st.force_valid
            @test !st.energy_valid            # force-only never marks energies current
            f_only = Array(st.fx)

            E = total_potential_energy(st)
            @test E isa Float64
            @test st.energy_valid && st.force_valid
            @test Array(st.fx) ≈ f_only
            @test E ≈ sum(Float64.(Array(st.Epot)))

            # Reuse: a second request does not recompute (buffers untouched).
            fill!(st.Epot_accum, one(T))   # sentinel elsewhere; Epot itself must stay
            Epot_before = Array(st.Epot)
            evaluate_configuration!(st, EvaluationRequest(energy=true))
            @test Array(st.Epot) == Epot_before
            fill!(st.Epot_accum, zero(T))

            # Virial totals agree with the diagnostic host helper.
            W = total_configurational_virial(st)
            ref = SimulationCore.virial_tensor(st)
            @test all(isapprox(W[c], Float64(ref[c]); rtol=(T === Float64 ? 1e-12 : 1e-5), atol=1e-6) for c in keys(W))
            @test total_virial_trace(st) ≈ W.xx + W.yy + W.zz rtol=(T === Float64 ? 1e-12 : 1e-5)

            # No MD side effects.
            after = snapshot(st)
            @test after[1] == before[1]
            @test all(after[k] == before[k] for k in 2:5)

            # Coordinate change + invalidation → fresh values.
            copyto!(st.rx, T[0.0, 1.3, 2.1])
            SimulationCore.invalidate_forces!(st)
            @test !st.force_valid && !st.energy_valid
            E2 = total_potential_energy(st)
            st_ref = build_eval_state(T)
            copyto!(st_ref.rx, T[0.0, 1.3, 2.1])
            SimulationCore.evaluate_forces_into_f!(st_ref, true)
            @test E2 ≈ sum(Float64.(Array(st_ref.Epot)))
            @test E2 != E
        end
    end
end
