@testset "ELL neighbor completeness and capacity safety" begin
    NL = ParticleDynamics.NeighborLists
    NF = ParticleDynamics.NonBondedForces
    rng = Random.MersenneTwister(0xE110)

    function check_rows(nb, coords, box, radii)
        expected = reference_neighbor_rows(coords, box, radii)
        counts = Array(nb.counts)
        stored = reshape(Array(nb.neighbors_flat), length(counts), Int(nb.cap))
        @test nb.valid
        @test nb.required_capacity == maximum(length, expected)
        for i in eachindex(counts)
            row = stored[i, 1:Int(counts[i])]
            @test length(row) == length(unique(row))
            @test sort(row) == expected[i]
        end
    end

    for T in (Float32, Float64), D in (2, 3), N in (1, 2, 33)
        box = ntuple(_ -> T(20), D)
        coords = ntuple(_ -> T.(20 .* rand(rng, N) .- 10), D)
        device = map(CuArray, coords)
        nb = NL.build_neighbors_dense!(device...; box, cutoff=T(3), skin=T(0.4), cap=Int32(max(N - 1, 1)))
        check_rows(nb, coords, box, fill(T(3) + T(0.4), N))
        radii = T.(1 .+ 3 .* rand(rng, N))
        stencil = NL.build_neighbors_stencil!(device...; box, rcut_particle=radii, skin=T(0.4), cap=Int32(max(N - 1, 1)))
        check_rows(stencil, coords, box, radii .+ T(0.4))
    end

    @testset "Unique wrapped stencil cells" begin
        for T in (Float32, Float64), D in (2, 3)
            box = ntuple(_ -> T(6), D)
            coords = (T[-1.51, 0.01, -0.5], ntuple(_ -> zeros(T, 3), D - 1)...)
            for radii in (T[2.9, 1.5, 1.5], T[30, 1.5, 1.5])
                nb = NL.build_neighbors_stencil!(map(CuArray, coords)...; box, rcut_particle=radii, skin=zero(T), cap=Int32(3))
                check_rows(nb, coords, box, radii)
            end
        end
    end

    @testset "Overflow rejects rows and reports required capacity" begin
        for T in (Float32, Float64), D in (2, 3), stencil in (false, true)
            box = ntuple(_ -> T(20), D)
            coords = ntuple(_ -> CUDA.zeros(T, 4), D)
            build(cap) = stencil ?
                NL.build_neighbors_stencil!(coords...; box, rcut_particle=fill(T(2), 4), skin=T(0.3), cap=Int32(cap)) :
                NL.build_neighbors_dense!(coords...; box, cutoff=T(2), skin=T(0.3), cap=Int32(cap))
            @test_throws NL.NeighborCapacityError build(2)
            exact = build(3)
            @test exact.required_capacity == 3
            @test all(==(3), Array(exact.counts))
            # Start with a valid list whose capacity is sufficient for isolated particles.
            copyto!(coords[1], T[-6, -2, 2, 6])
            nb = build(2)
            refs = Array(nb.rref_x)
            fill!(coords[1], zero(T))
            err = try
                NL.update_neighbors_inplace!(nb, coords...; box, step=7)
                nothing
            catch e
                e
            end
            @test err isa NL.NeighborCapacityError
            @test err.capacity == 2 && err.required == 3
            @test !nb.valid
            @test nb.required_capacity == 3
            @test Array(nb.rref_x) == refs
            force = ntuple(_ -> CUDA.zeros(T, 4), D)
            @test_throws ArgumentError NF.lj_forces_soa_noE!(coords..., force..., nb, box, ParticleDynamics.Definitions.LJParams{T}(one(T), one(T), T(2)))
            @test_throws ArgumentError ParticleDynamics.Collisions._collision_neighbor_matrix(nb)
            # Recover with the same capacity once the configuration is valid again.
            copyto!(coords[1], T[-6, -2, 2, 6])
            NL.update_neighbors_inplace!(nb, coords...; box, step=8)
            @test nb.valid && nb.last_build_step == 8
        end
    end

    @testset "Deferred construction and invalid inputs" begin
        st = build_simulation(N=128, box=(20.0, 20.0), cutoff=2.5, skin=0.4,
                              cap=Int32(16), temperature=0.0, precision=:f64, spatial_reorder=false)
        @test !st.nbh.valid
        @test st.nbh.required_capacity == 0
        x = [1.1 * mod(i - 1, 16) - 9 for i in 1:128]
        y = [2.0 * ((i - 1) ÷ 16) - 8 for i in 1:128]
        copyto!(st.rx, x); copyto!(st.ry, y)
        SimulationCore.evaluate_forces_into_f!(st, false)
        @test st.nbh.valid
        @test all(isfinite, Array(st.fx))
        @test_throws ArgumentError NL.build_neighbors_dense!(CUDA.zeros(Float64, 0), CUDA.zeros(Float64, 0);
            box=(20.0, 20.0), cutoff=2.0, skin=0.3, cap=Int32(4))
        @test_throws ArgumentError NL.build_neighbors_dense!(CUDA.zeros(Float64, 2), CUDA.zeros(Float64, 2);
            box=(20.0, 20.0), cutoff=2.0, skin=-0.3, cap=Int32(4))
        @test_throws DimensionMismatch NL.update_neighbors_inplace!(st.nbh, CUDA.zeros(Float64, 1), st.ry; box=st.box2)
    end
end

@testset "ELL collision consumers: dimension, cutoffs and exclusions" begin
    NL = ParticleDynamics.NeighborLists
    for T in (Float32, Float64), D in (2, 3), paircut in (false, true), excluded in (false, true)
        N = 40
        box = ntuple(_ -> T(100), D)
        bonds = excluded ? [(Int32(1), Int32(2))] : nothing
        st = build_simulation(; N, box, cutoff=one(T), skin=T(0.8), cap=Int32(8), temperature=zero(T),
            bonds, bonding=excluded ? harmonic_bond(k=zero(T), r0=one(T)) : nothing,
            precision=T === Float32 ? :f32 : :f64, spatial_reorder=false)
        x = T[10 * mod((i - 1) ÷ 2, 5) - 25 + (iseven(i) ? 1.2 : 0) for i in 1:N]
        y = T[10 * ((i - 1) ÷ 10) - 20 for i in 1:N]
        copyto!(st.rx, x); copyto!(st.ry, y)
        st.rz === nothing || fill!(st.rz, zero(T))
        copyto!(st.typeid, Int32[isodd(i) ? 1 : 2 for i in 1:N])
        coords = D == 2 ? (st.rx, st.ry) : (st.rx, st.ry, st.rz)
        NL.update_neighbors_inplace!(st.nbh, coords...; box)
        paircut && ParticleDynamics.set_collision_pair_cutoffs!(st, ones(T, 2, 2))
        ParticleDynamics.enable_collision_counting!(st; ntypes=2)
        @test ParticleDynamics.collisions_read_counts!(st) == zeros(Int64, 3)
        # All twenty isolated pairs enter; a bonded pair is excluded when requested.
        x[2:2:end] .-= T(0.4)
        copyto!(st.rx, x)
        ParticleDynamics.Collisions._collisions_update_after_positions!(st)
        expected = Int64[0, 20 - Int(excluded), 0]
        @test ParticleDynamics.collisions_read_counts!(st) == expected
        ParticleDynamics.Collisions._collisions_update_after_positions!(st)
        @test ParticleDynamics.collisions_read_counts!(st) == expected
        # Reinitializing a rebuilt list must not count existing contacts again.
        NL.update_neighbors_inplace!(st.nbh, coords...; box, step=1)
        ParticleDynamics.Collisions._collisions_reinit_on_rebuild!(st)
        ParticleDynamics.Collisions._collisions_update_after_positions!(st)
        @test ParticleDynamics.collisions_read_counts!(st) == expected
    end
end
