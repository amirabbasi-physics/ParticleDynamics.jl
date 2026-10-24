@testset "Dynamic cell updates" begin
    SC = SimulationCore

    function lj_state(T, pos, L; neighborlist=true, nonbonded=:wca, cutoff=2.5, unwrapped=false, bonds=nothing)
        N = size(pos, 1)
        st = SC.build_simulation(N=N, box=(T(L), T(L), T(L)), cutoff=T(cutoff), skin=T(0.3), cap=Int32(128),
            neigh_interval=1, use_neighborlist=neighborlist, epsilon=T(1), sigma=T(1), gamma=T(1), temperature=T(1),
            nonbonded=nonbonded, precision=(T === Float64 ? :f64 : :f32), spatial_reorder=false,
            unwrapped_positions=unwrapped,
            bonds=bonds === nothing ? nothing : [(Int32(i), Int32(j)) for (i, j) in bonds],
            bonding=bonds === nothing ? nothing : ParticleDynamics.harmonic_bond(k=50.0, r0=1.0))
        copyto!(st.rx, T.(pos[:, 1])); copyto!(st.ry, T.(pos[:, 2])); copyto!(st.rz, T.(pos[:, 3]))
        unwrapped && SC.sync_unwrapped!(st)
        SC.invalidate_forces!(st)
        return st
    end
    function lattice(n, a)
        p = [((i - 0.5) * a, (j - 0.5) * a, (k - 0.5) * a) for i in 1:n, j in 1:n, k in 1:n][:]
        L = n * a
        return reduce(vcat, [collect(x)' .- L / 2 for x in p]), L
    end

    rng = Random.MersenneTwister(4)
    pos, L = lattice(8, 1.15)
    pos .+= 0.05 .* randn(rng, size(pos))
    pos .= mod.(pos .+ L / 2, L) .- L / 2

    @testset "Scaled cell equals a fresh build ($T, $nb, factor $f)" for T in (Float64, Float32), nb in (:wca, :lj),
                                                                      f in (0.93, 0.98, 1.04, 1.12)
        st = lj_state(T, pos, L; nonbonded=nb, unwrapped=true)
        E0 = total_potential_energy(st)
        nx0 = st.nbh.nx
        set_cell_and_positions!(st, (f * L, f * L, f * L))
        @test !st.force_valid && !st.energy_valid
        @test cell_volume(st) ≈ (f * L)^3 rtol=1e-6
        E = total_potential_energy(st)
        # independent references at the scaled coordinates: fresh dense list and all pairs
        scaled = Float64.(hcat(Array(st.rx), Array(st.ry), Array(st.rz)))
        @test maximum(abs, scaled .- mod.(f .* pos .+ f * L / 2, f * L) .+ f * L / 2) <= (T === Float64 ? 1e-12 : 1e-5) * L
        ref = lj_state(T, scaled, f * L; nonbonded=nb)
        ref_ap = lj_state(T, scaled, f * L; nonbonded=nb, neighborlist=false)
        tol = T === Float64 ? 1e-11 : 1e-4
        @test isapprox(E, total_potential_energy(ref); rtol=tol, atol=tol)
        @test isapprox(E, total_potential_energy(ref_ap); rtol=tol, atol=tol)
        @test isapprox(Array(st.fx), Array(ref.fx); rtol=tol, atol=tol * maximum(abs, Array(ref.fx)))
        # unwrapped coordinates follow the same scaling
        xu = Float64.(Array(st.rx_unwrap))
        @test maximum(abs, xu .- f .* pos[:, 1]) <= (T === Float64 ? 1e-12 : 1e-5) * L
        f < 1 && @test E != E0
    end

    @testset "Unsupported cells are rejected without side effects" begin
        st = lj_state(Float64, pos, L)
        x0 = Array(st.rx)
        E0 = total_potential_energy(st)
        small = (2.0, 2.0, 2.0)          # fewer than 3 cells of cutoff+skin
        @test cell_supported(st, small) isa String
        @test_throws DomainError set_cell_and_positions!(st, small)
        @test Array(st.rx) == x0 && st.box3 == (L, L, L)
        @test st.energy_valid && total_potential_energy(st) == E0
        @test_throws DomainError set_cell_and_positions!(st, (L, -L, L))
        st_ap = lj_state(Float64, pos, L; neighborlist=false, nonbonded=:lj, cutoff=2.5)
        @test cell_supported(st_ap, (4.9, 4.9, 4.9)) isa String     # rc > L/2 for all-pairs
        @test cell_supported(st_ap, (5.1, 5.1, 5.1)) === nothing
    end

    @testset "Trial save/restore is exact ($T)" for T in (Float64, Float32)
        st = lj_state(T, pos, L; unwrapped=true, bonds=[(1, 2), (3, 4)])
        total_potential_energy(st)
        snap = (Array(st.rx), Array(st.rx_unwrap), Array(st.fx), Array(st.Epot), Array(st.virial_tensor), st.box3)
        ws = SC.TrialWorkspace(st)
        SC.save_trial_state!(ws, st)
        set_cell_and_positions!(st, (0.95L, 0.95L, 0.95L))
        total_potential_energy(st)
        SC.restore_trial_state!(ws, st)
        @test (Array(st.rx), Array(st.rx_unwrap), Array(st.fx), Array(st.Epot), Array(st.virial_tensor), st.box3) == snap
        @test st.force_valid && st.energy_valid
        # the restored neighbor geometry gives the same forces on re-evaluation
        SC.invalidate_forces!(st)
        total_potential_energy(st)
        @test Array(st.fx) ≈ snap[3]
    end

    @testset "Virial is the strain derivative of the energy" begin
        # bonded chain pieces + WCA (continuous energy), Float64
        st = lj_state(Float64, pos, L; bonds=[(i, i + 1) for i in 1:2:40])
        W = total_virial_trace(st)
        ε = 1e-6
        Ep = (set_cell_and_positions!(st, ((1 + ε) * L, (1 + ε) * L, (1 + ε) * L)); total_potential_energy(st))
        Em = (set_cell_and_positions!(st, ((1 - ε) * L, (1 - ε) * L, (1 - ε) * L)); total_potential_energy(st))
        # U(s) with all coordinates scaled by s: dU/ds|₁ = -Σ r·F = -tr W
        @test isapprox((Ep - Em) / (2ε), -W; rtol=1e-6, atol=1e-8)
    end
end
