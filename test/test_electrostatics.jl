using LinearAlgebra
using Random

if !isdefined(@__MODULE__, :EwaldReference)
    include("ewald_reference.jl")
end
using .EwaldReference

const ES = ParticleDynamics.Electrostatics

struct _ESDummyPotential <: SimulationCore.AbstractExternalPotential end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function _es_cloud(N, L; seed=1, neutral=true, rng=MersenneTwister(seed))
    R = rand(rng, 3, N) .* collect(L) .- collect(L) ./ 2
    q = randn(rng, N)
    neutral && (q .-= sum(q) / N)
    return R, q
end

function _es_state(R, q, L, spec; T=Float64, nonbonded=:none, pairs=true, cap=Int32(160),
                   exceptions=(), spatial_reorder=false, kwargs...)
    N = size(R, 2)
    st = build_simulation(; N=N, box=L, nonbonded=nonbonded, cutoff=spec.real_cutoff, skin=0.075 * spec.real_cutoff,
                          cap=cap, precision=T === Float64 ? :f64 : :f32, charges=q,
                          spatial_reorder=spatial_reorder, use_neighborlist=pairs, kwargs...)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    attach_electrostatics!(st, spec; exceptions=exceptions)
    return st
end

function _es_eval(st)
    SimulationCore.evaluate_forces_into_f!(st, true)
    F = vcat(Array(st.fx)', Array(st.fy)', Array(st.fz)')
    E = sum(Float64, Array(st.Epot))
    W = virial_tensor(st)
    return E, F, [W.xx W.xy W.xz; W.xy W.yy W.yz; W.xz W.yz W.zz]
end

_relrms(F, Fref) = sqrt(sum(abs2, F .- Fref) / sum(abs2, Fref))

_spme(C, rc, α, mesh, P) = ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc,
    split=ES.GaussianSplit(α), window=ES.BSplineWindow(P), mesh=mesh, influence=:spme)
_p3m(C, rc, α, mesh, P, kind) = ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc,
    split=ES.GaussianSplit(α), window=ES.BSplineWindow(P), mesh=mesh, influence=kind)
_esp(C, rc, eps, mesh, P) = ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc,
    split=ES.ProlateSplit(rc, ES.prolate_shape_for_tolerance(eps)),
    window=ES.ProlateWindow(P, ES.prolate_shape_for_tolerance(eps / 2)), mesh=mesh, influence=:esp)

const _ES_L = (14.0, 15.0, 16.0)

@testset "Electrostatics: prolate functions and windows" begin
    # Table 2 of Liang et al. (2026)
    for (eps, c) in ((1e-3, 9.5392), (5e-4, 10.290), (1e-4, 12.024), (5e-5, 12.762), (1e-5, 14.471))
        @test isapprox(ES.prolate_shape_for_tolerance(eps), c; rtol=5e-4)
    end
    pf = ES.ProlateFunction(10.0)
    x, w = ES.gauss_legendre(160)
    ratio(t) = sum(w .* pf.(x) .* cos.(10.0 .* t .* x)) / pf(t)
    @test isapprox(ratio(0.2), ratio(0.9); rtol=1e-10)                    # finite Fourier eigenfunction
    @test isapprox(ES.prolate_derivative(pf, 0.3), (pf(0.3 + 1e-6) - pf(0.3 - 1e-6)) / 2e-6; rtol=1e-7)

    rng = MersenneTwister(5)
    for P in 3:8
        st = ES.device_stencil(ES.BSplineWindow(P), Float64)
        for _ in 1:20
            u = 40 * rand(rng)
            k0, wt, dw = ES.window_stencil(st, u)
            @test isapprox(sum(wt), 1.0; atol=1e-14)                       # partition of unity
            @test isapprox(sum(dw), 0.0; atol=1e-13)
            @test k0 == floor(Int, u) - P + 1
            # derivative of the weights
            _, wp, _ = ES.window_stencil(st, u + 1e-7)
            _, wm, _ = ES.window_stencil(st, u - 1e-7)
            floor(u + 1e-7) == floor(u - 1e-7) && @test maximum(abs.((wp .- wm) ./ 2e-7 .- dw)) < 1e-6
        end
        # continuity when crossing a grid line: node weights agree on both sides
        kL, wL, _ = ES.window_stencil(st, 7.0 - 1e-12)
        kR, wR, _ = ES.window_stencil(st, 7.0)
        nodesL = Dict(kL + j - 1 => wL[j] for j in 1:P)
        nodesR = Dict(kR + j - 1 => wR[j] for j in 1:P)
        for k in union(keys(nodesL), keys(nodesR))
            @test isapprox(get(nodesL, k, 0.0), get(nodesR, k, 0.0); atol=1e-10)
        end
    end
    pw = ES.ProlateWindow(6, 12.0)
    pst = ES.device_stencil(pw, Float64)
    k0, wt, dw = ES.window_stencil(pst, 10.5)
    @test k0 == 8 && isapprox(wt[3], wt[4]; rtol=1e-12)                  # symmetric about the particle
    @test isapprox(wt[3], pw.pf(1 / 6); rtol=1e-12)
    D = ES.window_deconvolution(ES.BSplineWindow(4), 16)
    @test isapprox(D[1], 1.0; rtol=1e-14) && all(isfinite, D)
    @test all(isfinite, ES.window_deconvolution(ES.BSplineWindow(5), 16))  # odd-order Nyquist handled
end

@testset "Electrostatics: cuFFT wrapper normalisation" begin
    ws = ES.MeshWorkspace(Float64, (8, 6, 5), false)
    g = zeros(8, 6, 5); g[1, 1, 1] = 1.0
    copyto!(ws.grid, g)
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    @test size(ws.spec) == (5, 6, 5)
    @test all(isapprox.(Array(ws.spec), 1.0; atol=1e-14))                 # impulse → flat spectrum
    fill!(ws.grid, 2.0)
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    S = Array(ws.spec)
    @test isapprox(S[1, 1, 1], 2.0 * 240; rtol=1e-14) && maximum(abs.(S[2:end])) < 1e-10
    m = (1, 2, 1)
    g = [cos(2pi * (m[1] * (i - 1) / 8 + m[2] * (j - 1) / 6 + m[3] * (k - 1) / 5)) for i in 1:8, j in 1:6, k in 1:5]
    copyto!(ws.grid, g)
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    S = Array(ws.spec)
    @test isapprox(S[2, 3, 2], 120.0; rtol=1e-12)                          # single mode: M/2
    mul!(ws.grid, ws.plan_bwd, ws.spec)
    @test maximum(abs.(Array(ws.grid) ./ 240 .- g)) < 1e-13               # backward is unnormalised
end

@testset "Electrostatics: real-space pair kernels" begin
    for kernel in (ES.device_pair_kernel(ES.GaussianSplit(1.1), Float64),
                   ES.device_pair_kernel(ES.ProlateSplit(3.0, 11.0), Float64))
        r = CuArray(collect(range(0.3, 2.9; length=40)))
        e(rr) = ES.pair_kernel(kernel, rr, 1 / rr)[1]
        f(rr) = ES.pair_kernel(kernel, rr, 1 / rr)[2]
        h = 1e-6
        dE = (Array(e.(r .+ h)) .- Array(e.(r .- h))) ./ (2h)
        @test maximum(abs.(-dE .- Array(f.(r)) .* Array(r))) < 1e-7          # F = −dE/dr, f = F/r
    end
    sp = ES.ProlateSplit(3.0, 11.0)
    k = ES.device_pair_kernel(sp, Float64)
    @test Array(map(rr -> ES.pair_kernel(k, rr, 1 / rr)[1], CuArray([3.0, 3.5]))) == [0.0, 0.0]
    # E_s(t) = 1 − χ(0) t + O(t³) near the origin
    @test isapprox(Array(map(rr -> ES.pair_kernel(k, rr, 1 / rr)[1], CuArray([1e-3])))[1] * 1e-3,
                   1 - sp.chi0 * 1e-3 / 3.0; atol=1e-9)
end

@testset "Electrostatics: GPU solvers against the Ewald oracle" begin
    C = 1.7; rc = 4.0
    R, q = _es_cloud(120, _ES_L; seed=3)
    H = Matrix(Diagonal(collect(_ES_L)))
    Eref, Fref, Wref = ewald_reference(R, q, H; C=C)
    cases = [
        ("SPME", _spme(C, rc, 0.9, (48, 48, 54), 6), 3e-5, 1e-4),
        ("SPME refined", _spme(C, rc, 0.9, (64, 64, 72), 8), 1e-6, 1e-6),
        ("P3M-AD", _p3m(C, rc, 0.9, (48, 48, 54), 6, :p3m_ad), 3e-5, 3e-6),
        ("P3M-ik", _p3m(C, rc, 0.9, (48, 48, 54), 6, :p3m_ik), 3e-6, 3e-6),
        ("ESP", _esp(C, rc, 1e-6, (32, 32, 36), 8), 1e-5, 1e-6),
    ]
    for (name, spec, ftol, etol) in cases
        @testset "$(name)" begin
            st = _es_state(R, q, _ES_L, spec)
            E, F, W = _es_eval(st)
            @test _relrms(F, Fref) < ftol
            @test abs(E - Eref) / abs(Eref) < etol
            @test maximum(abs.(W .- Wref)) / abs(tr(Wref)) < 10 * max(ftol, etol)
            # component breakdown sums to the per-particle total
            comp = electrostatic_energy(st)
            @test isapprox(comp.total, E; rtol=1e-11)
            @test isapprox(comp.self, -C * (spec.split isa ES.GaussianSplit ? 2 * 0.9 / sqrt(pi) : ES.self_coefficient(spec.split)) / 2 * sum(abs2, q); rtol=1e-12)
        end
    end
    # all-pairs traversal equals the neighbor-list traversal
    spec = _spme(C, rc, 0.9, (48, 48, 54), 6)
    E1, F1, _ = _es_eval(_es_state(R, q, _ES_L, spec))
    E2, F2, _ = _es_eval(_es_state(R, q, _ES_L, spec; pairs=false))
    @test isapprox(E1, E2; rtol=1e-12) && maximum(abs.(F1 .- F2)) < 1e-11
    # noncubic, strongly anisotropic box and a cubic one
    for L in ((13.0, 13.0, 26.0), (13.5, 13.5, 13.5))
        R2, q2 = _es_cloud(90, L; seed=7)
        Er, Fr, Wr = ewald_reference(R2, q2, Matrix(Diagonal(collect(L))); C=1.0)
        mesh = ntuple(d -> ES.fft_friendly(ceil(Int, 4 * L[d])), 3)
        for spec in (_spme(1.0, 3.8, 0.95, mesh, 6), _esp(1.0, 3.8, 1e-6, ntuple(d -> ES.fft_friendly(ceil(Int, 2.5 * L[d])), 3), 8))
            E, F, W = _es_eval(_es_state(R2, q2, L, spec))
            @test _relrms(F, Fr) < 3e-5
            @test abs(E - Er) / abs(Er) < 3e-5
        end
    end
end

@testset "Electrostatics: discrete derivatives and strain" begin
    C = 1.0; rc = 3.8
    L = (13.0, 13.5, 14.0)
    R, q = _es_cloud(80, L; seed=9)
    for spec in (_spme(C, rc, 0.9, (24, 25, 27), 5), _esp(C, rc, 1e-4, (18, 18, 20), 6),
                 _p3m(C, rc, 0.9, (24, 25, 27), 5, :p3m_ad))
        st = _es_state(R, q, L, spec)
        E0, F0, W0 = _es_eval(st)
        # coordinate derivative of the same discrete energy (AD forces are exact gradients)
        h = 1e-5
        for (i, d) in ((3, 1), (40, 2), (77, 3))
            buf = d == 1 ? st.rx : (d == 2 ? st.ry : st.rz)
            x = Array(buf); x[i] += h; copyto!(buf, x); Ep, _, _ = _es_eval(st)
            x[i] -= 2h; copyto!(buf, x); Em, _, _ = _es_eval(st)
            x[i] += h; copyto!(buf, x)
            @test abs(-(Ep - Em) / (2h) - F0[d, i]) < 1e-6 * maximum(abs.(F0))
        end
        # diagonal strain at fixed fractional coordinates: W_aa = −dU/dε_aa
        for a in 1:3
            eps = 1e-5
            Es = map((eps, -eps)) do e
                scale = ntuple(d -> d == a ? 1 + e : 1.0, 3)
                Ls = ntuple(d -> L[d] * scale[d], 3)
                Rs = R .* collect(scale)
                E, _, _ = _es_eval(_es_state(Rs, q, Ls, spec))
                E
            end
            @test isapprox(-(Es[1] - Es[2]) / (2eps), W0[a, a]; rtol=1e-6, atol=1e-6 * abs(tr(W0)))
        end
    end
    # shear components: converged solver against the oracle's analytic tensor
    Er, Fr, Wr = ewald_reference(R, q, Matrix(Diagonal(collect(L))); C=C)
    _, _, W = _es_eval(_es_state(R, q, L, _spme(C, rc, 0.9, (54, 54, 60), 8)))
    for (a, b) in ((1, 2), (1, 3), (2, 3))
        @test abs(W[a, b] - Wr[a, b]) < 1e-5 * abs(tr(Wr))
    end
end

@testset "Electrostatics: exceptions, background, zero charges" begin
    C = 1.3; rc = 4.0
    R, q = _es_cloud(100, _ES_L; seed=13)
    # one exception pair straddles the periodic boundary
    R[:, 1] .= (6.8, 0.0, 0.0); R[:, 2] .= (-6.6, 0.3, 0.1)
    exc = [(1, 2, 0.0), (5, 9, 0.5), (10, 11, 0.0), (10, 12, 0.8333)]
    Er, Fr, Wr = ewald_reference(R, q, Matrix(Diagonal(collect(_ES_L))); C=C, exceptions=exc)
    for spec in (_spme(C, rc, 0.9, (48, 48, 54), 6), _esp(C, rc, 1e-6, (32, 32, 36), 8))
        st = _es_state(R, q, _ES_L, spec; exceptions=exc)
        E, F, W = _es_eval(st)
        @test _relrms(F, Fr) < 3e-5
        @test abs(E - Er) / abs(Er) < 3e-5
        @test maximum(abs.(W .- Wr)) / abs(tr(Wr)) < 3e-4
    end
    @test_throws ArgumentError ES.normalize_exceptions([(1, 2, 0.0), (2, 1, 0.5)], 10)
    @test_throws ArgumentError ES.normalize_exceptions([(1, 1, 0.0)], 10)
    @test isempty(ES.normalize_exceptions([(1, 2, 1.0)], 10))

    # net charge: rejected by default, explicit uniform background otherwise
    qn = q .+ 0.05
    spec_err = _spme(C, rc, 0.9, (48, 48, 54), 6)
    @test_throws ArgumentError _es_state(R, qn, _ES_L, spec_err)
    spec_bg = ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc, split=ES.GaussianSplit(0.9),
                                    window=ES.BSplineWindow(6), mesh=(48, 48, 54), influence=:spme, net_charge=:background)
    Eb, Fb, Wb = ewald_reference(R, qn, Matrix(Diagonal(collect(_ES_L))); C=C, background=true)
    st = _es_state(R, qn, _ES_L, spec_bg)
    E, F, W = _es_eval(st)
    @test abs(E - Eb) / abs(Eb) < 3e-5 && _relrms(F, Fb) < 3e-5
    @test maximum(abs.(W .- Wb)) / abs(tr(Wb)) < 3e-4
    @test isapprox(electrostatic_energy(st).background, -C * pi * sum(qn)^2 / (2 * 0.81 * prod(_ES_L)); rtol=1e-12)

    # zero charges: no force, no energy
    st = _es_state(R, zeros(100), _ES_L, spec_err)
    E, F, W = _es_eval(st)
    @test E == 0 && maximum(abs.(F)) == 0 && maximum(abs.(W)) == 0
end

@testset "Electrostatics: unit conventions and symmetries" begin
    R, q = _es_cloud(90, _ES_L; seed=17)
    rc = 4.0; α = 0.9; mesh = (40, 40, 45)
    E1, F1, W1 = _es_eval(_es_state(R, q, _ES_L, _spme(1.0, rc, α, mesh, 6)))
    # same physical system in another length/energy/charge convention:
    # x' = λx, q' = κq, E' = μE ⇒ prefactor' = μ λ / κ², α' = α/λ, F' = μ/λ F
    for (λ, κ, μ) in ((10.0, 1.0, 96.485), (0.1, 3.0, 2.5))
        Ls = _ES_L .* λ
        spec = _spme(μ * λ / κ^2, rc * λ, α / λ, mesh, 6)
        E2, F2, W2 = _es_eval(_es_state(R .* λ, q .* κ, Ls, spec))
        @test isapprox(E2, μ * E1; rtol=1e-10)
        @test maximum(abs.(F2 .- (μ / λ) .* F1)) < 1e-10 * (μ / λ) * maximum(abs.(F1))
        @test maximum(abs.(W2 .- μ .* W1)) < 1e-10 * μ * abs(tr(W1))
    end
    # dielectric scaling and charge doubling
    specε = ES.ElectrostaticsSpec(coulomb_prefactor=1.0, relative_permittivity=4.0, real_cutoff=rc,
                                  split=ES.GaussianSplit(α), window=ES.BSplineWindow(6), mesh=mesh, influence=:spme)
    E3, F3, _ = _es_eval(_es_state(R, q, _ES_L, specε))
    @test isapprox(E3, E1 / 4; rtol=1e-12)
    E4, F4, _ = _es_eval(_es_state(R, 2 .* q, _ES_L, _spme(1.0, rc, α, mesh, 6)))
    @test isapprox(E4, 4 * E1; rtol=1e-12) && maximum(abs.(F4 .- 4 .* F1)) < 1e-11 * maximum(abs.(F1))
    # permutation
    p = randperm(MersenneTwister(1), 90)
    E5, F5, _ = _es_eval(_es_state(R[:, p], q[p], _ES_L, _spme(1.0, rc, α, mesh, 6)))
    @test isapprox(E5, E1; rtol=1e-11) && maximum(abs.(F5 .- F1[:, p])) < 1e-10
    # integer box translation is exact; arbitrary translation within the mesh error
    Rt = R .+ collect(_ES_L) .* [1.0, -2.0, 1.0]
    E6, F6, _ = _es_eval(_es_state(Rt, q, _ES_L, _spme(1.0, rc, α, mesh, 6)))
    @test isapprox(E6, E1; rtol=1e-11)
    E7, F7, _ = _es_eval(_es_state(R .+ [0.37, -0.11, 0.53], q, _ES_L, _spme(1.0, rc, α, mesh, 6)))
    @test abs(E7 - E1) / abs(E1) < 1e-4 && _relrms(F7, F1) < 1e-3
    # net force is a mesh-convergence quantity, not forced to zero
    @test norm(sum(F1; dims=2)) < 1e-3 * norm(F1)
end

@testset "Electrostatics: composition and lifecycle" begin
    rng = MersenneTwister(21)
    n = 6; a = 1.6; L = (n * a, n * a, n * a) .+ 4.0
    R = hcat([[a * i, a * j, a * k] .- L[1] / 2 .+ 0.2 .* randn(rng, 3) for i in 1:n for j in 1:n for k in 1:n]...)
    N = size(R, 2)
    q = [isodd(i) ? 0.8 : -0.8 for i in 1:N]
    spec = _spme(1.0, 3.5, 1.0, (32, 32, 32), 5)
    # LJ + PME equals LJ alone plus PME alone
    st_lj = build_simulation(N=N, box=L, nonbonded=:lj, cutoff=2.5, epsilon=1.0, sigma=1.0, skin=0.3,
                             cap=Int32(160), precision=:f64, spatial_reorder=false)
    copyto!(st_lj.rx, R[1, :]); copyto!(st_lj.ry, R[2, :]); copyto!(st_lj.rz, R[3, :])
    Elj, Flj, Wlj = _es_eval(st_lj)
    Ees, Fes, Wes = _es_eval(_es_state(R, q, L, spec))
    st_both = build_simulation(N=N, box=L, nonbonded=:lj, cutoff=2.5, epsilon=1.0, sigma=1.0, skin=0.3,
                               cap=Int32(160), precision=:f64, spatial_reorder=false, charges=q)
    copyto!(st_both.rx, R[1, :]); copyto!(st_both.ry, R[2, :]); copyto!(st_both.rz, R[3, :])
    attach_electrostatics!(st_both, spec)                       # charged → cluster backend (user rule, 13 Oct)
    @test nonbonded_backend(st_both) === :cluster
    E, F, W = _es_eval(st_both)
    @test isapprox(E, Elj + Ees; rtol=1e-11)
    @test maximum(abs.(F .- (Flj .+ Fes))) < 1e-10 * maximum(abs.(F))
    @test maximum(abs.(W .- (Wlj .+ Wes))) < 1e-10 * abs(tr(W))

    # repeated evaluations do not accumulate; f0 slot and energy-off evaluations agree
    E2, F2, _ = _es_eval(st_both)
    # atomic spreading is not bitwise reproducible; stale accumulation would be O(1)
    @test isapprox(E2, E; rtol=1e-12) && maximum(abs.(F2 .- F)) < 1e-11 * maximum(abs.(F))
    SimulationCore.evaluate_forces_into_f!(st_both, false)
    @test vcat(Array(st_both.fx)', Array(st_both.fy)', Array(st_both.fz)') ≈ F

    # set_charges! and detach
    set_charges!(st_both, 2 .* q)
    E3, _, _ = _es_eval(st_both)
    @test isapprox(E3 - Elj, 4 * Ees; rtol=1e-10)
    @test charges(st_both) ≈ 2 .* q
    detach_electrostatics!(st_both)
    @test nonbonded_backend(st_both) === :ell                   # uncharged again: back to ELL
    E4, _, W4 = _es_eval(st_both)
    @test isapprox(E4, Elj; rtol=1e-12) && W4 ≈ Wlj

    # guards
    @test_throws ArgumentError attach_electrostatics!(build_simulation(N=4, box=(10.0, 10.0), precision=:f64,
                                                                       charges=[1.0, -1.0, 1.0, -1.0]), spec)
    big = _spme(1.0, 7.0, 1.0, (32, 32, 32), 5)
    @test_throws ArgumentError _es_state(R, q, L, big; pairs=false)          # r_c ≥ L/2
    stx = _es_state(R, q, L, spec)
    @test_throws ErrorException attach_external_potential!(stx, _ESDummyPotential())
end

@testset "Electrostatics: reorder, integrators, residency" begin
    rng = MersenneTwister(31)
    n = 10; a = 1.35; Lb = n * a; L = (Lb, Lb, Lb)
    R = hcat([[a * (i - 0.5), a * (j - 0.5), a * (k - 0.5)] .- Lb / 2 .+ 0.03 .* randn(rng, 3)
              for i in 1:n for j in 1:n for k in 1:n]...)
    N = size(R, 2)
    q = [isodd(i + j + k) ? 1.0 : -1.0 for i in 1:n for j in 1:n for k in 1:n]
    spec = _spme(2.0, 3.4, 1.0, (30, 30, 30), 5)
    V0 = 0.5 .* randn(rng, 3, N)
    V0 .-= sum(V0; dims=2) ./ N
    function run_nve(reorder)
        st = build_simulation(N=N, box=L, nonbonded=:wca, epsilon=1.0, sigma=1.0, skin=0.3, cap=Int32(160),
                              precision=:f64, charges=q, spatial_reorder=reorder, reorder_interval=10,
                              neigh_interval=5, temperature=0.3, neighbor_cutoff=3.4)
        copyto!(st.rx, R[1, :]); copyto!(st.ry, R[2, :]); copyto!(st.rz, R[3, :])
        copyto!(st.vx, V0[1, :]); copyto!(st.vy, V0[2, :]); copyto!(st.vz, V0[3, :])
        attach_electrostatics!(st, spec)
        integ = nve(st; dt=0.002)
        for _ in 1:200
            step!(st, integ, 0.002; compute_energy=false)
        end
        SimulationCore.evaluate_forces_into_f!(st, true)
        return st, sum(Float64, Array(st.Epot))
    end
    st_r, E_r = run_nve(true)
    st_n, E_n = run_nve(false)
    @test st_r.last_reorder_step < 0                                   # never reordered: the cluster backend sorts its own copy
    @test isapprox(E_r, E_n; rtol=1e-8)
    @test charges(st_r) == q                                           # charges follow their particles

    # Brownian midpoint and Langevin integrators run through the shared force path
    st = _es_state(R, q, L, spec; nonbonded=:wca, epsilon=1.0, sigma=1.0, neighbor_cutoff=3.4)
    eh = eulerheun(st; gamma=1.0, temperature=0.3, dt=1e-4)
    for _ in 1:20
        step!(st, eh, 1e-4)
    end
    @test all(isfinite, Array(st.rx))
    st = _es_state(R, q, L, spec; nonbonded=:wca, epsilon=1.0, sigma=1.0, neighbor_cutoff=3.4)
    vv = velocityverlet(st; gamma=1.0, temperature=0.3, dt=1e-3)
    for _ in 1:20
        step!(st, vv, 1e-3)
    end
    @test all(isfinite, Array(st.vx))

    # warmed electrostatics path: no device allocations, no host transfers needed
    solver = st.electrostatics
    args = (st.rx, st.ry, st.rz, st.charge, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded, st.virial_global, st.nbh)
    ES.compute!(solver, args..., true); ES.compute!(solver, args..., false)
    @test CUDA.@allocated(ES.compute!(solver, args..., false)) == 0
    @test CUDA.@allocated(ES.compute!(solver, args..., true)) == 0
end

@testset "Electrostatics: Float32 and workflow integration" begin
    C = 1.7; rc = 4.0
    R, q = _es_cloud(120, _ES_L; seed=3)
    Eref, Fref, _ = ewald_reference(R, q, Matrix(Diagonal(collect(_ES_L))); C=C)
    for spec in (_spme(C, rc, 0.9, (48, 48, 54), 6), _esp(C, rc, 1e-5, (28, 28, 32), 7))
        E32, F32, _ = _es_eval(_es_state(R, q, _ES_L, spec; T=Float32))
        E64, F64, _ = _es_eval(_es_state(R, q, _ES_L, spec; T=Float64))
        @test _relrms(F32, F64) < 1e-5                    # Float32 storage, mesh and FFT roundoff
        @test abs(E32 - E64) / abs(E64) < 1e-5
        @test _relrms(F32, Fref) < 1e-4
        @test abs(E32 - Eref) / abs(Eref) < 1e-4
    end

    # workflow: descriptors, exceptions from topology, observables, restart
    mktempdir() do tmp
        rng = MersenneTwister(41)
        n = 10; a = 1.35; Lb = n * a
        pos = [(a * (i - 0.5) - Lb / 2, a * (j - 0.5) - Lb / 2, a * (k - 0.5) - Lb / 2) .+ 0.03 .* (randn(rng), randn(rng), randn(rng))
               for i in 1:n for j in 1:n for k in 1:n]
        typeids = Int32[isodd(i + j + k) ? 1 : 2 for i in 1:n for j in 1:n for k in 1:n]
        topo = Topology(exclusions=[(1, 2), (3, 4)])
        system = ParticleSystem(pos; box=(Lb, Lb, Lb), types=[:Na, :Cl], typeids=typeids,
                                masses=Dict(:Na => 1.0, :Cl => 1.0), charges=Dict(:Na => 1.0, :Cl => -1.0),
                                topology=topo)
        allg = Group(:all, AllSelection())
        na = Group(:na, TypeSelection(:Na))
        thermo = ThermodynamicObservable(allg; name=:all)
        thermo_na = ThermodynamicObservable(na; name=:na)
        for elec in (PME(coulomb_prefactor=2.0, real_cutoff=3.4, alpha=1.0, mesh=(30, 30, 30), order=5,
                         scaled_pairs=[(5, 6, 0.5)]),
                     PPPM(coulomb_prefactor=2.0, real_cutoff=3.4, real_space_tolerance=1e-5,
                          mesh_spacing=0.45, differentiation=:ik),
                     ESP(coulomb_prefactor=2.0, real_cutoff=3.4, tolerance=1e-5))
            path = joinpath(tmp, "es_$(nameof(typeof(elec))).gsd")
            sim = Simulation(system; groups=Groups(allg, na), observables=[thermo, thermo_na],
                writers=[GSDWriter(path; every=10, write_start=false)],
                integrator=Integrator(dt=0.002,
                    forces=[WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(capacity=200)), elec],
                    methods=[ConstantVolume(allg)]),
                precision=Float64, seed=3)
            run!(sim, Stage(:nve, steps=20; progress=false))
            st = state(sim)
            @test st.electrostatics isa ES.ElectrostaticsSolver
            SimulationCore.evaluate_forces_into_f!(st, true)
            full = ParticleDynamics.Workflow.sample_observable(sim, thermo; fields=[:potential_energy, :virial])
            @test isfinite(full.virial)
            W = virial_tensor(st)
            @test isapprox(full.virial, W.xx + W.yy + W.zz; rtol=1e-10)
            @test_throws ArgumentError ParticleDynamics.Workflow.sample_observable(sim, thermo_na; fields=[:virial])
            sub = ParticleDynamics.Workflow.sample_observable(sim, thermo_na; fields=[:potential_energy])
            @test isfinite(sub.potential_energy)
            # restart: descriptor, mesh and exceptions are reproduced exactly
            rest = electrostatics_from_gsd(path)
            @test typeof(rest) == typeof(elec)
            spec_old = st.electrostatics.spec
            spec_new = electrostatics_spec(rest)
            @test spec_new.influence == spec_old.influence
            @test spec_new.real_cutoff == spec_old.real_cutoff
            @test ES.resolve_mesh(spec_new, (Lb, Lb, Lb)) == ES.solver_mesh(st.electrostatics)
            @test sort(ES.normalize_exceptions(rest.scaled_pairs, length(pos))) == sort(st.electrostatics.exceptions)
            sys2 = ParticleSystem.from_gsd(path)
            @test sys2.charges == system.charges
        end
        # pure Coulomb + bonds (no vdW force) compiles to the :none kind
        sysb = ParticleSystem(pos; box=(Lb, Lb, Lb), types=[:Na, :Cl], typeids=typeids,
                              masses=Dict(:Na => 1.0, :Cl => 1.0), charges=Dict(:Na => 0.2, :Cl => -0.2),
                              topology=Topology(bonds=[(1, 2), (3, 4)]))
        sim = Simulation(sysb; groups=Groups(allg),
            integrator=Integrator(dt=0.001, forces=[HarmonicBondForce(k=50.0, r0=1.35),
                                                    PME(coulomb_prefactor=1.0, real_cutoff=3.4, alpha=1.0, mesh=(30, 30, 30),
                                                        exclude_bonded=true)],
                                  methods=[ConstantVolume(allg)]), precision=Float64)
        prepare!(sim)
        @test state(sim).nb_kind == SimulationCore.NB_KIND_NONE
        @test length(state(sim).electrostatics.exceptions) == 2
        run!(sim, Stage(:nve, steps=5; progress=false))
        @test all(isfinite, Array(state(sim).rx))
    end
end

@testset "Electrostatics: measured accuracy targeting" begin
    C = 1.7; rc = 4.0
    R, q = _es_cloud(120, _ES_L; seed=3)
    Eref, Fref, _ = ewald_reference(R, q, Matrix(Diagonal(collect(_ES_L))); C=C)
    st = _es_state(R, q, _ES_L, _spme(C, rc, 0.9, (16, 16, 16), 4))
    for (force, target) in ((PME(coulomb_prefactor=C, real_cutoff=rc, alpha=0.9, order=5), 1e-4),
                            (PPPM(coulomb_prefactor=C, real_cutoff=rc, alpha=0.9, order=5, differentiation=:ik), 1e-4),
                            (ESP(coulomb_prefactor=C, real_cutoff=rc, tolerance=1e-4), 1e-4))
        res = tune_electrostatics(st, force; target=target)
        @test res.error <= target
        @test res.reference_error_bound < target / 10
        @test res.force.mesh isa NTuple{3,Int} && res.time_ms > 0
        # the tuned explicit descriptor meets the target against the independent oracle
        _, F, _ = _es_eval(_es_state(R, q, _ES_L, electrostatics_spec(res.force)))
        @test _relrms(F, Fref) < 1.5 * target
    end
end

@testset "Electrostatics: potential shift, validity, box setter, mesh identities" begin
    C = 1.3; rc = 4.0; α = 0.9
    R, q = _es_cloud(100, _ES_L; seed=51)
    exc = [(1, 2, 0.0), (3, 4, 0.5)]
    R[:, 2] .= R[:, 1] .+ [0.6, 0.0, 0.0]; R[:, 4] .= R[:, 3] .+ [0.0, 0.7, 0.0]
    shifted = ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc, split=ES.GaussianSplit(α),
                                    window=ES.BSplineWindow(6), mesh=(48, 48, 54), influence=:spme,
                                    real_space_modifier=:potential_shift)
    E0, F0, W0 = _es_eval(_es_state(R, q, _ES_L, _spme(C, rc, α, (48, 48, 54), 6); exceptions=exc))
    E1, F1, W1 = _es_eval(_es_state(R, q, _ES_L, shifted; exceptions=exc))
    # expected offset: −C erfc(α r_c)/r_c Σ q_i q_j over non-excluded pairs inside r_c
    pair_sum = 0.0
    for i in 1:100, j in (i + 1):100
        (i, j) in ((1, 2), (3, 4)) && continue
        d = R[:, i] .- R[:, j]
        d .-= collect(_ES_L) .* round.(d ./ collect(_ES_L))
        sum(abs2, d) < rc^2 && (pair_sum += q[i] * q[j])
    end
    # (a small difference of two large energies: compare absolutely)
    @test abs((E1 - E0) - (-C * ES.cutoff_energy(ES.GaussianSplit(α), rc) * pair_sum)) < 1e-10 * abs(E0)
    @test F1 ≈ F0 && W1 ≈ W0                       # energy-only modifier
    @test ES.cutoff_energy(ES.ProlateSplit(rc, 11.0), rc) == 0.0
    @test_throws ArgumentError ES.ElectrostaticsSpec(coulomb_prefactor=C, real_cutoff=rc, split=ES.GaussianSplit(α),
        window=ES.BSplineWindow(6), mesh=(48, 48, 54), influence=:spme, real_space_modifier=:force_switch)

    # energy validity: force-only evaluations mark energies stale
    st = _es_state(R, q, _ES_L, _spme(C, rc, α, (48, 48, 54), 6); exceptions=exc)
    SimulationCore.evaluate_forces_into_f!(st, false)
    @test !st.energy_valid
    SimulationCore.ensure_energies!(st)
    @test st.energy_valid && isapprox(sum(Float64, Array(st.Epot)), E0; rtol=1e-10)
    copyto!(st.rx, R[1, :] .+ 0.01)
    SimulationCore.invalidate_forces!(st)
    @test !st.energy_valid

    # set_box!: identical to a fresh build in the new box
    L2 = (15.0, 15.5, 16.5)
    st = _es_state(R, q, _ES_L, _spme(C, rc, α, (48, 48, 54), 6))
    set_box!(st, L2)
    Ea, Fa, Wa = _es_eval(st)
    Eb, Fb, Wb = _es_eval(_es_state(R, q, L2, _spme(C, rc, α, (48, 48, 54), 6)))
    @test isapprox(Ea, Eb; rtol=1e-11) && maximum(abs.(Fa .- Fb)) < 1e-10 * maximum(abs.(Fb)) && Wa ≈ Wb
    @test_throws ArgumentError set_box!(st, (7.0, 15.0, 15.0))      # r_c ≥ L/2

    # mesh identities: B-spline spreading conserves charge; packed R2C = full complex FFT
    st = _es_state(R, q .+ 0.0, _ES_L, _spme(C, rc, α, (20, 22, 24), 5))
    s = st.electrostatics; ws = s.ws
    fill!(ws.grid, 0.0)
    K = Int32.(ws.mesh)
    @cuda threads=128 blocks=cld(100, 128) ES._spread_kernel!(ws.grid_flat, st.rx, st.ry, st.rz, st.charge,
        s.identity_order, _ES_L, K..., s.stencil)
    G = Array(ws.grid)
    @test isapprox(sum(G), sum(q); atol=1e-12)
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    full = Array(CUDA.CUFFT.fft(CuArray(complex.(G))))
    @test maximum(abs.(Array(ws.spec) .- full[1:11, :, :])) < 1e-10 * maximum(abs.(full))
end

@testset "Electrostatics: dynamics across unit conventions and thermostat" begin
    rng = MersenneTwister(61)
    n = 8; a = 1.7; Lb = n * a
    R = hcat([[a * (i - 0.5), a * (j - 0.5), a * (k - 0.5)] .- Lb / 2 .+ 0.05 .* randn(rng, 3)
              for i in 1:n for j in 1:n for k in 1:n]...)
    N = size(R, 2)
    q = [isodd(i + j + k) ? 0.7 : -0.7 for i in 1:n for j in 1:n for k in 1:n]
    V0 = 0.3 .* randn(rng, 3, N); V0 .-= sum(V0; dims=2) ./ N
    # length λ, energy μ, mass m: time τ = λ sqrt(m/μ), charge κ ⇒ prefactor ∝ μ λ / κ²
    function trajectory(λ, μ, m, κ)
        τ = λ * sqrt(m / μ)
        spec = _spme(μ * λ / κ^2, 3.4 * λ, 1.0 / λ, (24, 24, 24), 5)
        st = build_simulation(N=N, box=(Lb, Lb, Lb) .* λ, nonbonded=:wca, epsilon=μ, sigma=λ, skin=0.3λ,
                              cap=Int32(160), precision=:f64, charges=q .* κ, mass=m, spatial_reorder=false,
                              neighbor_cutoff=3.4λ)
        copyto!(st.rx, R[1, :] .* λ); copyto!(st.ry, R[2, :] .* λ); copyto!(st.rz, R[3, :] .* λ)
        copyto!(st.vx, V0[1, :] .* λ ./ τ); copyto!(st.vy, V0[2, :] .* λ ./ τ); copyto!(st.vz, V0[3, :] .* λ ./ τ)
        attach_electrostatics!(st, spec)
        integ = nve(st; dt=0.002τ)
        for _ in 1:100
            step!(st, integ, 0.002τ; compute_energy=false)
        end
        return vcat(Array(st.rx)', Array(st.ry)', Array(st.rz)') ./ λ
    end
    X1 = trajectory(1.0, 1.0, 1.0, 1.0)
    X2 = trajectory(10.0, 96.485, 22.99, 2.0)
    @test maximum(abs.(X1 .- X2)) < 1e-8

    # Langevin thermostat with electrostatics reaches the target temperature
    st = build_simulation(N=N, box=(Lb, Lb, Lb), nonbonded=:wca, epsilon=1.0, sigma=1.0, skin=0.3,
                          cap=Int32(160), precision=:f64, charges=q, spatial_reorder=false,
                          neighbor_cutoff=3.4, temperature=0.4)
    copyto!(st.rx, R[1, :]); copyto!(st.ry, R[2, :]); copyto!(st.rz, R[3, :])
    attach_electrostatics!(st, _spme(1.0, 3.4, 1.0, (24, 24, 24), 5))
    integ = baoab(st; gamma=2.0, temperature=0.4, dt=0.004)
    temps = Float64[]
    for it in 1:3000
        step!(st, integ, 0.004; compute_energy=false)
        if it > 1000 && it % 10 == 0
            v2 = sum(abs2, Array(st.vx)) + sum(abs2, Array(st.vy)) + sum(abs2, Array(st.vz))
            push!(temps, v2 / (3N))
        end
    end
    Tm = sum(temps) / length(temps)
    @test abs(Tm - 0.4) < 0.02
end

@testset "Electrostatics: prolate window width bound" begin
    # P ≤ 2c₁/π − ½: Table 2 widths pass; a window at the band limit is rejected
    for eps in (1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-8, 1e-10)
        P = ParticleDynamics.Workflow.esp_order_for_tolerance(eps)
        @test P <= ES.max_prolate_order(ES.prolate_shape_for_tolerance(eps / 2))
    end
    c1 = ES.prolate_shape_for_tolerance(5e-7)                    # 2c₁/π ≈ 11.2
    @test_throws ArgumentError ES.ElectrostaticsSpec(coulomb_prefactor=1.0, real_cutoff=3.0,
        split=ES.ProlateSplit(3.0, ES.prolate_shape_for_tolerance(1e-6)), window=ES.ProlateWindow(12, c1),
        mesh=(24, 24, 24), influence=:esp)
end
