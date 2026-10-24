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
