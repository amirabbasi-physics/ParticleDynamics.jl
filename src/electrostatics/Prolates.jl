# =========================
#   Prolate spheroidal wave function of order zero, ψ₀ᶜ on [-1, 1]
# =========================
# ψ₀ᶜ is the eigenfunction with the smallest eigenvalue of the prolate operator
#   L ψ = -((1 - x²) ψ')' + c² x² ψ,
# which commutes with the finite Fourier transform on [-1, 1]. In the
# normalised Legendre basis P̄_n = √(n + ½) P_n the operator is symmetric
# tridiagonal and couples only n, n ± 2, so ψ₀ᶜ (even) is a combination of
# even-degree Legendre polynomials (Bouwkamp's method). Everything here is
# host-side Float64 setup code.

"""
    ProlateFunction(c)

Order-zero prolate spheroidal wave function `ψ₀ᶜ` with bandlimit `c`,
normalised so that `ψ(0) = 1`. Stored as coefficients of the even Legendre
polynomials `P_0, P_2, P_4, …`.
"""
struct ProlateFunction
    c::Float64
    coeffs::Vector{Float64}     # coeffs[k] multiplies P_{2(k-1)}(x)
end

function ProlateFunction(c::Real; nterms::Integer=ceil(Int, c) + 48)
    c > 0 || throw(ArgumentError("Prolate bandlimit c must be positive."))
    cc = Float64(c)^2
    diag = Vector{Float64}(undef, nterms)
    off = Vector{Float64}(undef, nterms - 1)
    for k in 1:nterms
        n = 2 * (k - 1)
        diag[k] = n * (n + 1) + cc * (2.0 * n * (n + 1) - 1.0) / ((2n + 3) * (2n - 1))
        if k < nterms
            off[k] = cc * (n + 2) * (n + 1) / ((2n + 3) * sqrt((2.0 * n + 1) * (2.0 * n + 5)))
        end
    end
    F = eigen(SymTridiagonal(diag, off), 1:1)
    v = F.vectors[:, 1]
    beta = [v[k] * sqrt(2 * (k - 1) + 0.5) for k in 1:nterms]
    # P_{2m}(0) = (-1)^m (2m-1)!! / (2m)!!
    p0 = 1.0
    psi0 = beta[1]
    for k in 2:nterms
        m = k - 1
        p0 *= -(2m - 1) / (2m)
        psi0 += beta[k] * p0
    end
    return ProlateFunction(Float64(c), beta ./ psi0)
end

function _prolate_legendre(pf::ProlateFunction, x::Float64)
    nmax = 2 * length(pf.coeffs) - 1
    return _legendre_all(x, nmax), nmax
end

"""
    (pf::ProlateFunction)(x) -> ψ(x)
"""
function (pf::ProlateFunction)(x::Real)
    P, _ = _prolate_legendre(pf, Float64(x))
    acc = 0.0
    for k in eachindex(pf.coeffs)
        acc += pf.coeffs[k] * P[2 * (k - 1) + 1]
    end
    return acc
end

"""
    prolate_derivative(pf, x) -> ψ'(x)
"""
function prolate_derivative(pf::ProlateFunction, x::Real)
    P, nmax = _prolate_legendre(pf, Float64(x))
    dP = _legendre_derivs(P, nmax)
    acc = 0.0
    for k in eachindex(pf.coeffs)
        acc += pf.coeffs[k] * dP[2 * (k - 1) + 1]
    end
    return acc
end

"""
    prolate_integral(pf, x) -> ∫₀ˣ ψ(t) dt
"""
function prolate_integral(pf::ProlateFunction, x::Real)
    xf = Float64(x)
    P, _ = _prolate_legendre(pf, xf)
    acc = pf.coeffs[1] * xf
    for k in 2:length(pf.coeffs)
        n = 2 * (k - 1)
        # ∫₀ˣ P_n = (P_{n+1}(x) - P_{n-1}(x)) / (2n+1); the odd values vanish at 0.
        acc += pf.coeffs[k] * (P[n + 2] - P[n]) / (2n + 1)
    end
    return acc
end

"""
    prolate_l2_norm(pf) -> ‖ψ‖ on [-1, 1]
"""
function prolate_l2_norm(pf::ProlateFunction)
    # ∫ P_m P_n = 2/(2n+1) δ_mn
    acc = 0.0
    for k in eachindex(pf.coeffs)
        n = 2 * (k - 1)
        acc += pf.coeffs[k]^2 * 2.0 / (2n + 1)
    end
    return sqrt(acc)
end

"""
    prolate_shape_for_tolerance(ε) -> c

Bandlimit `c` with `ψ₀ᶜ(1) = ε` for the L²-normalised (`‖ψ₀ᶜ‖ = 1` on
`[-1, 1]`) prolate function: the selection rule of Liang et al. (2026),
eq. (15), used for both the splitting kernel and the assignment window. It
reproduces the `c` values of the paper's Table 2 (e.g. `ε = 1e-4 → c ≈ 12.024`).
"""
function prolate_shape_for_tolerance(eps::Real)
    0 < eps < 1 || throw(ArgumentError("Prolate tolerance must lie in (0, 1)."))
    target = log(Float64(eps))
    f(c) = (pf = ProlateFunction(c); log(abs(pf(1.0)) / prolate_l2_norm(pf)) - target)
    lo, hi = 0.5, 80.0
    f(lo) > 0 || throw(ArgumentError("Tolerance $(eps) is too loose for a prolate split."))
    f(hi) < 0 || throw(ArgumentError("Tolerance $(eps) is too tight for a prolate split."))
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-12 * hi && break
    end
    return 0.5 * (lo + hi)
end

# ==============================================================================
#   Generalized (Jacobi-weighted) Slepian Prolate Functions Φ_α^c(x) on [-1, 1]
# ==============================================================================
# Ground-state eigenfunction of Slepian's generalized prolate operator:
#   L_α φ = -(1 - x²)^{-α} d/dx [ (1 - x²)^{α+1} φ' ] + c² x² φ
# represented as Φ_α^c(x) = (1 - x²)^α Σ_{k≥0} β_k P_{2k}^{(α,α)}(x), normalised
# to Φ_α^c(0) = 1. For α = 2, Φ(±1) = Φ'(±1) = 0 (exact C² radial split);
# for α = 3, Φ(±1) = Φ'(±1) = Φ''(±1) = 0 (exact C² grid assignment window).

"""
    GeneralizedProlateFunction(c, alpha)

Jacobi-weighted generalized prolate spheroidal wave function `Φ_α^c(x)` on
`[-1, 1]` with boundary vanishing order `alpha ≥ 0`, normalised so `Φ_α^c(0) = 1`.
"""
struct GeneralizedProlateFunction
    c::Float64
    alpha::Int
    coeffs::Vector{Float64}     # coeffs[k] multiplies P_{2(k-1)}^{(α,α)}(x)
end

# Overflow-free squared norm ∫_{-1}^1 (1-x²)^a [P_n^{(a,a)}(x)]² dx:
#   2^{2a+1} / (2n + 2a + 1) · Π_{j=1}^a (n + j)/(n + a + j).
function _jacobi_sym_norm_sq(n::Int, a::Int)
    ratio = 1.0
    for j in 1:a
        ratio *= (n + j) / (n + a + j)
    end
    return ldexp(ratio, 2 * a + 1) / (2 * n + 2 * a + 1)
end

function GeneralizedProlateFunction(c::Real, alpha::Integer; nterms::Integer=ceil(Int, c) + 48)
    c > 0 || throw(ArgumentError("Generalized prolate parameter c must be positive."))
    alpha >= 0 || throw(ArgumentError("Generalized prolate order alpha must be nonnegative."))
    a = Int(alpha)
    cc = Float64(c)^2
    bn(m::Int) = m >= 0 ? sqrt((m + 1.0) * (m + 2a + 1.0) / ((2m + 2a + 1.0) * (2m + 2a + 3.0))) : 0.0
    diag = Vector{Float64}(undef, nterms)
    off = Vector{Float64}(undef, nterms - 1)
    for k in 1:nterms
        n = 2 * (k - 1)
        x2d = bn(n)^2 + (n >= 1 ? bn(n - 1)^2 : 0.0)
        diag[k] = n * (n + 2a + 1.0) + cc * x2d
        if k < nterms
            off[k] = cc * bn(n) * bn(n + 1)
        end
    end
    F = eigen(SymTridiagonal(diag, off), 1:1)
    v = F.vectors[:, 1]
    beta = [v[k] / sqrt(_jacobi_sym_norm_sq(2 * (k - 1), a)) for k in 1:nterms]
    P0 = _jacobi_sym_all(0.0, 2 * (nterms - 1), a)
    phi0 = sum(beta[k] * P0[2 * (k - 1) + 1] for k in 1:nterms)
    return GeneralizedProlateFunction(Float64(c), a, beta ./ phi0)
end

function (pf::GeneralizedProlateFunction)(x::Real)
    xf = Float64(x)
    a = pf.alpha
    (a > 0 && abs(xf) >= 1.0) && return 0.0
    nmax = 2 * (length(pf.coeffs) - 1)
    P = _jacobi_sym_all(xf, nmax, a)
    poly = 0.0
    @inbounds for k in eachindex(pf.coeffs)
        poly += pf.coeffs[k] * P[2 * (k - 1) + 1]
    end
    a == 0 && return poly
    return (1.0 - xf * xf)^a * poly
end

function prolate_derivative(pf::GeneralizedProlateFunction, x::Real)
    xf = Float64(x)
    a = pf.alpha
    (a >= 2 && abs(xf) >= 1.0) && return 0.0
    nmax = 2 * (length(pf.coeffs) - 1)
    P = _jacobi_sym_all(xf, nmax, a)
    dP = _jacobi_sym_derivs(xf, nmax, a)
    poly = 0.0
    dpoly = 0.0
    @inbounds for k in eachindex(pf.coeffs)
        idx = 2 * (k - 1) + 1
        c = pf.coeffs[k]
        poly += c * P[idx]
        dpoly += c * dP[idx]
    end
    a == 0 && return dpoly
    omx2 = max(1.0 - xf * xf, 0.0)
    return omx2^(a - 1) * (-2.0 * a * xf * poly + omx2 * dpoly)
end

"""
    prolate_integral(pf::GeneralizedProlateFunction, x) -> ∫₀ˣ Φ_α^c(t) dt

Exact closed-form antiderivative via DLMF 18.9.16:
`∫₀ˣ (1 − t²)^α P_{2k}^{(α,α)}(t) dt = −(1/4k)(1 − x²)^{α+1} P_{2k−1}^{(α+1,α+1)}(x)` for `k ≥ 1`.
"""
function prolate_integral(pf::GeneralizedProlateFunction, x::Real)
    xf = clamp(Float64(x), -1.0, 1.0)
    a = pf.alpha
    # k = 0 term: β₀ ∫₀ˣ (1 − t²)^a dt = β₀ Σ_{m=0}^a (-1)^m binomial(a, m) x^{2m+1}/(2m+1)
    k0 = 0.0
    xpow = xf
    x2 = xf * xf
    for m in 0:a
        sgn = isodd(m) ? -1.0 : 1.0
        k0 += sgn * binomial(a, m) * xpow / (2m + 1)
        xpow *= x2
    end
    acc = pf.coeffs[1] * k0
    length(pf.coeffs) == 1 && return acc
    omx2 = max(1.0 - x2, 0.0)
    omx2 == 0.0 && return acc
    nmax = 2 * (length(pf.coeffs) - 1) - 1
    P1 = _jacobi_sym_all(xf, nmax, a + 1)
    w = omx2^(a + 1)
    @inbounds for k in 2:length(pf.coeffs)
        m = k - 1
        acc -= pf.coeffs[k] * w * P1[2m] / (4.0 * m)
    end
    return acc
end

function prolate_l2_norm(pf::GeneralizedProlateFunction; nquad::Integer=192)
    x, w = gauss_legendre(nquad)
    return sqrt(sum(w .* (pf.(x) .^ 2)))
end

"""
    generalized_prolate_sidelobe(pf::GeneralizedProlateFunction) -> (max_sidelobe, bandlimit_u0)

Maximum relative spectral sidelobe `max_{u ≥ u₀} |Φ̂(u)| / Φ̂(0)` beyond the first
zero crossing `u₀` of `Φ̂(u) = ∫_{-1}^1 Φ(x) cos(u x) dx`.
"""
function generalized_prolate_sidelobe(pf::GeneralizedProlateFunction; nquad::Integer=192)
    x, w = gauss_legendre(nquad)
    vals = w .* pf.(x)
    f0 = sum(vals)
    umax = max(50.0, 2.5 * pf.c + 12.0)
    du = 0.04
    nu = ceil(Int, umax / du) + 1
    found_zero = false
    u0 = umax
    sl = 0.0
    for j in 1:nu
        u = (j - 1) * du
        hat = sum(vals[q] * cos(u * x[q]) for q in eachindex(x)) / f0
        if !found_zero && hat <= 0.0
            found_zero = true
            u0 = u
        end
        if found_zero
            sl = max(sl, abs(hat))
        end
    end
    return sl, u0
end

generalized_prolate_bandlimit(pf::GeneralizedProlateFunction) = generalized_prolate_sidelobe(pf)[2]

"""
    generalized_prolate_shape_for_tolerance(ε, alpha) -> c

Shape `c` of `GeneralizedProlateFunction(c, alpha)` whose spectral sidelobe bound
matches the `ESP` calibration (`max_sidelobe = 0.30 ε`, reproducing Table 2 of
Liang et al. 2026 within 0.5% at `alpha = 0` while extending cleanly to `alpha = 2, 3`).
"""
function generalized_prolate_shape_for_tolerance(eps::Real, alpha::Integer)
    0 < eps < 1 || throw(ArgumentError("Prolate tolerance must lie in (0, 1)."))
    target = log(0.30 * Float64(eps))
    f(c) = log(max(generalized_prolate_sidelobe(GeneralizedProlateFunction(c, alpha))[1], 1e-300)) - target
    lo, hi = 1.5, 75.0
    f(lo) > 0 || throw(ArgumentError("Tolerance $(eps) is too loose for a generalized prolate (alpha=$(alpha))."))
    f(hi) < 0 || throw(ArgumentError("Tolerance $(eps) is too tight for a generalized prolate (alpha=$(alpha))."))
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        f(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-10 * hi && break
    end
    return 0.5 * (lo + hi)
end
