using StaticArrays
using LinearAlgebra: dot, cross, norm

@testset "Typed bonded families" begin
    BF = ParticleDynamics.BondedForces

    # Host reference: energy and forces of one interaction from local positions.
    function local_eval(kind, c, pos)
        K = length(pos)
        if K == 2
            F1, F2, U, ok = BF.pair_forces(kind, c, pos[2] - pos[1]); return (F1, F2), U, ok
        elseif K == 3
            Fi, Fj, Fk, U, ok = BF.angle_forces(kind, c, pos[1] - pos[2], pos[3] - pos[2]); return (Fi, Fj, Fk), U, ok
        else
            b1, b2, b3 = pos[2] - pos[1], pos[3] - pos[2], pos[4] - pos[3]
            if kind isa BF.FourierTorsionKind
                kf, nf, δf = c
                F1, F2, F3, F4, U, ok = BF.fourier_torsion_forces(kf, nf, δf, 1, length(kf), b1, b2, b3)
            else
                F1, F2, F3, F4, U, ok = BF.torsion_forces(kind, c, b1, b2, b3)
            end
            return (F1, F2, F3, F4), U, ok
        end
    end
    energy(kind, c, pos) = local_eval(kind, c, pos)[2]

    function fd_forces(kind, c, pos; h=1e-6)
        K = length(pos); D = length(pos[1])
        return ntuple(K) do a
            SVector{D}(ntuple(D) do d
                e = SVector{D}(ntuple(j -> j == d ? h : 0.0, D))
                p⁺ = ntuple(b -> b == a ? pos[b] + e : pos[b], K)
                p⁻ = ntuple(b -> b == a ? pos[b] - e : pos[b], K)
                -(energy(kind, c, p⁺) - energy(kind, c, p⁻)) / (2h)
            end)
        end
    end

    rng = Random.MersenneTwister(0xB0DED)
    randvec(D) = SVector{D}(ntuple(_ -> randn(rng), D))
    normalize_rand(rng, D) = (v = randn(rng, D); v ./ norm(v))

    torsion_terms = ([1.3, -0.4, 0.25], [1.0, 2.0, 3.0], [0.2, 1.1, -0.7])
    cases = Any[
        (BF.HarmonicBondKind(), (7.0, 1.1), 2, (2, 3)),
        (BF.FENEBondKind(), (30.0, 1.5), 2, (2, 3)),
        (BF.HarmonicAngleKind(), (40.0, 1.9), 3, (2, 3)),
        (BF.CosineSquaredAngleKind(), (25.0, cos(2.1)), 3, (2, 3)),
        (BF.UreyBradleyKind(), (40.0, 1.9, 15.0, 1.6), 3, (2, 3)),
        (BF.HarmonicImproperKind(), (12.0, 0.4), 4, (3,)),
        (BF.FourierTorsionKind(), torsion_terms, 4, (3,)),
        (BF.RyckaertBellemansKind(), (9.28, 12.16, -13.12, -3.06, 26.24, -31.5), 4, (3,)),
    ]

    @testset "Float64 energy-force consistency: $(nameof(typeof(kind))) $(D)D" for (kind, c, K, dims) in cases, D in dims
        for trial in 1:50
            pos = ntuple(_ -> 0.8 * randvec(D), K)
            if kind isa BF.FENEBondKind
                d = pos[2] - pos[1]; pos = (pos[1], pos[1] + 1.2 * rand(rng) * d / norm(d))
            end
            F, U, ok = local_eval(kind, c, pos)
            @test ok
            Ffd = fd_forces(kind, c, pos)
            scale = max(1.0, maximum(norm, F))
            @test all(norm(F[a] - Ffd[a]) <= 1e-6 * scale for a in 1:K)
            @test norm(sum(F)) <= 1e-10 * scale                 # no net force
            # Strain derivative: -dU/dε (all coordinates scaled) = Σ r·F.
            ε = 1e-6
            dU = (energy(kind, c, map(p -> (1 + ε) * p, pos)) - energy(kind, c, map(p -> (1 - ε) * p, pos))) / (2ε)
            @test isapprox(-dU, sum(dot(pos[a], F[a]) for a in 1:K); atol=1e-6 * scale * max(1.0, maximum(norm, pos)))
        end
    end

    @testset "Torsion sign convention (IUPAC)" begin
        # φ = +90°: looking from atom 2 to atom 3, atom 4 is rotated clockwise from atom 1.
        φ, = BF.torsion_angle_gradient(SVector(0.0, -1.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0))
        @test φ ≈ π / 2
        for _ in 1:100
            b1, b2, b3 = randvec(3), randvec(3), randvec(3)
            φ, = BF.torsion_angle_gradient(b1, b2, b3)
            ref = atan(norm(b2) * dot(b1, cross(b2, b3)), dot(cross(b1, b2), cross(b2, b3)))
            @test φ ≈ ref atol=1e-12
        end
        # trans = π, cis = 0
        @test abs(BF.torsion_angle_gradient(SVector(0.0, 1.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))[1]) ≈ π
        @test BF.torsion_angle_gradient(SVector(0.0, 1.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0))[1] ≈ 0 atol=1e-14
        # RB at trans (φ = π, ψ = 0) gives ΣCn.
        Crb = (9.28, 12.16, -13.12, -3.06, 26.24, -31.5)
        trans = (SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, -1.0, 0.0))
        @test energy(BF.RyckaertBellemansKind(), Crb, trans) ≈ sum(Crb) atol=1e-12
        # cis (φ = 0, ψ = π): Σ Cn (-1)^n
        cis = (SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0))
        @test energy(BF.RyckaertBellemansKind(), Crb, cis) ≈ sum(Crb[n + 1] * (-1)^n for n in 0:5)
        # Wrapped harmonic improper is 2π periodic around χ0.
        @test energy(BF.HarmonicImproperKind(), (3.0, π - 0.1), trans) ≈ 0.5 * 3.0 * 0.1^2
    end

    @testset "Degenerate geometry is reported, linear equilibrium is not" begin
        line = (SVector(-1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), SVector(1.3, 0.0, 0.0))
        F, U, ok = local_eval(BF.HarmonicAngleKind(), (10.0, 2.0), line)
        @test !ok
        F, U, ok = local_eval(BF.HarmonicAngleKind(), (10.0, Float64(π)), line)
        @test ok && U == 0 && all(f -> all(iszero, f), F)
        # Slightly bent around θ0 = π: finite, consistent with finite differences.
        bent = (SVector(-1.0, 1e-5, 0.0), SVector(0.0, 0.0, 0.0), SVector(1.3, 0.0, 0.0))
        F, U, ok = local_eval(BF.HarmonicAngleKind(), (10.0, Float64(π)), bent)
        @test ok && all(isfinite, reduce(vcat, F))
        @test !local_eval(BF.FENEBondKind(), (30.0, 1.5), (SVector(0.0, 0.0), SVector(1.6, 0.0)))[3]
        collinear = (SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(2.0, 0.0, 0.0), SVector(2.5, 1.0, 0.0))
        @test !local_eval(BF.FourierTorsionKind(), torsion_terms, collinear)[3]
    end

    @testset "GPU families vs host reference $(T) $(D)D" for T in (Float64, Float32), D in (2, 3)
        N = 40
        L = 6.0
        box = ntuple(_ -> T(L), D)
        # A wrapped random chain; consecutive atoms stay close, crossing the box.
        x = zeros(D, N)
        for i in 2:N
            x[:, i] = x[:, i - 1] .+ 0.9 .* normalize_rand(rng, D)
        end
        wrapped = mod.(x, L)
        rx = CuArray(T.(wrapped[1, :])); ry = CuArray(T.(wrapped[2, :]))
        rz = D == 3 ? CuArray(T.(wrapped[3, :])) : rx

        fams = Any[]
        refs = Any[]
        function add_family!(kind, K, coeff_host, ptype_of, sel)
            tuples = [ntuple(a -> Int32(s + a - 1), K) for s in sel]
            ptype = [ptype_of(s) for s in sel]
            tbl = BF.build_interaction_table(N, tuples, ptype)
            if kind isa BF.FourierTorsionKind
                offs, cnts, kf, nf, δf = coeff_host
                coeffs = (CuArray(Int32.(offs)), CuArray(Int32.(cnts)), CuArray(T.(kf)), CuArray(T.(nf)), CuArray(T.(δf)))
            else
                coeffs = map(v -> CuArray(T.(v)), coeff_host)
            end
            push!(fams, BF.BondedFamily(kind, tbl, coeffs, T, D; name=Symbol(nameof(typeof(kind)))))
            push!(refs, (kind, coeff_host, tuples, ptype))
        end
        add_family!(BF.HarmonicBondKind(), 2, ([7.0, 20.0], [1.0, 0.8]), s -> isodd(s) ? 1 : 2, 1:2:N-1)
        add_family!(BF.FENEBondKind(), 2, ([30.0], [1.5]), s -> 1, 2:2:N-1)
        add_family!(BF.HarmonicAngleKind(), 3, ([40.0, 15.0], [1.9, Float64(π)]), s -> isodd(s) ? 1 : 2, 1:N-2)
        add_family!(BF.CosineSquaredAngleKind(), 3, ([25.0], [cos(2.1)]), s -> 1, 1:3:N-2)
        if D == 3
            # two parameter sets: 2 and 1 Fourier terms
            add_family!(BF.FourierTorsionKind(), 4, ([0, 2], [2, 1], [1.3, -0.4, 0.7], [1.0, 3.0, 2.0], [0.25, 1.0, -0.3]),
                        s -> isodd(s) ? 1 : 2, 1:N-3)
            add_family!(BF.HarmonicImproperKind(), 4, ([12.0], [0.4]), s -> 1, 2:5:N-3)
        end
        terms = BF.BondedTerms(Tuple(fams), D)

        fx = CUDA.zeros(T, N); fy = CUDA.zeros(T, N); fz = D == 3 ? CUDA.zeros(T, N) : fx
        E = CUDA.zeros(T, N); V = CUDA.zeros(T, N, D == 3 ? 6 : 3)
        BF.evaluate_bonded!(terms, rx, ry, rz, box, fx, fy, fz, E, V, true)

        # Host reference in Float64 from the unwrapped chain.
        Fref = zeros(D, N); Uref = 0.0; Wtr = 0.0
        for (kind, ch, tuples, ptype) in refs
            for (tup, p) in zip(tuples, ptype)
                pos = ntuple(a -> SVector{D}(x[:, tup[a]] .- x[:, tup[1]]), length(tup))
                c = if kind isa BF.FourierTorsionKind
                    offs, cnts, kf, nf, δf = ch
                    r = (offs[p] + 1):(offs[p] + cnts[p]); (kf[r], nf[r], δf[r])
                else
                    map(v -> v[p], ch)
                end
                F, U, ok = local_eval(kind, c, pos)
                @test ok
                for a in eachindex(tup)
                    Fref[:, tup[a]] .+= F[a]
                end
                Uref += U
                Wtr += sum(dot(pos[a], F[a]) for a in eachindex(tup))
            end
        end
        tol = T === Float64 ? 1e-10 : 2e-4
        Fgpu = vcat(Array(fx)', Array(fy)', (D == 3 ? [Array(fz)'] : [])...)
        @test maximum(abs, Fgpu .- Fref) <= tol * max(1.0, maximum(abs, Fref))
        @test isapprox(sum(Float64.(Array(E))), Uref; rtol=tol)
        Vh = Float64.(Array(V))
        @test isapprox(sum(Vh[:, 1:D]), Wtr; rtol=tol, atol=tol * abs(Uref))
        @test BF.bonded_status(terms) === nothing

        # Force-only pass reproduces the forces and leaves E untouched.
        fx2 = CUDA.zeros(T, N); fy2 = CUDA.zeros(T, N); fz2 = D == 3 ? CUDA.zeros(T, N) : fx2
        E2 = CUDA.zeros(T, N)
        BF.evaluate_bonded!(terms, rx, ry, rz, box, fx2, fy2, fz2, E2, V, false)
        @test Array(fx2) ≈ Array(fx) && Array(fy2) ≈ Array(fy)
        @test all(iszero, Array(E2))
    end

    @testset "Invalid geometry raises DomainError with the topology index" begin
        T = Float64
        rx = CuArray(T[0.0, 1.0, 2.0]); ry = CuArray(T[0.0, 0.0, 0.0])
        tbl = BF.build_interaction_table(3, [(Int32(1), Int32(2), Int32(3))], [1])
        fam = BF.BondedFamily(BF.HarmonicAngleKind(), tbl, (CuArray(T[10.0]), CuArray(T[2.0])), T, 2;
                              name=:harmonic_angle, source_index=[7])
        terms = BF.BondedTerms((fam,), 2)
        fx = CUDA.zeros(T, 3); fy = CUDA.zeros(T, 3); E = CUDA.zeros(T, 3); V = CUDA.zeros(T, 3, 3)
        err = try
            BF.evaluate_bonded!(terms, rx, ry, rx, (T(10), T(10)), fx, fy, fx, E, V, true); nothing
        catch e
            e
        end
        @test err isa DomainError && err.val == 7
        @test BF.bonded_status(terms) == (:harmonic_angle, 7)
        @test_throws ArgumentError BF.build_interaction_table(3, [(Int32(1), Int32(1), Int32(3))], [1])
        @test_throws ArgumentError BF.build_interaction_table(3, [(Int32(1), Int32(4))], [1])
        @test_throws ArgumentError BF.BondedFamily(BF.HarmonicImproperKind(),
            BF.build_interaction_table(4, [(Int32(1), Int32(2), Int32(3), Int32(4))], [1]),
            (CuArray(T[1.0]), CuArray(T[0.0])), T, 2; name=:improper)
    end
end
