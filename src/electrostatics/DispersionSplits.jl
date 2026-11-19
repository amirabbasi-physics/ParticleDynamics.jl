# ==============================================================================
#   Radial short/long-range splits for 1/r⁶ lattice dispersion
# ==============================================================================
# Every dispersion split decomposes 1/r⁶ = V_{6,s}(r) + Φ₆(r), where Φ₆(r) is a
# smooth even function of r (analytic in r² at r = 0, with no 1/r⁶ singularity):
#   - Self removal per particle:  +½ Φ₆(0) s6ᵢ²  (stored with C = −1 via selfc = Φ₆(0))
#   - Mandatory k = 0 term:       −½ (Φ̂₆(0)/V) (Σ s6ᵢ)²  (energy and diagonal virial)
#   - Reciprocal Green's factor:  g₆(k) = ∫_{ℝ³} Φ₆(r) e^{−i k·r} d³r  (with C = −1)

abstract type AbstractDispersionSplit end

"""
    GaussianDispersionSplit(beta)

Essmann / Wennberg Gaussian `1/r⁶` split (`beta` in inverse length units):
`V_{6,s}(r) = e^{−β²r²}(1 + β²r² + ½ β⁴r⁴) / r⁶`, `Φ₆(0) = β⁶ / 6`.
"""
struct GaussianDispersionSplit <: AbstractDispersionSplit
    beta::Float64
    function GaussianDispersionSplit(beta::Real)
        isfinite(beta) && beta > 0 || throw(ArgumentError("Dispersion beta must be finite and positive."))
        return new(Float64(beta))
    end
end

"""
    C2ProlateDispersionSplit(cutoff, shape)

C²-continuous compactly supported `1/r⁶` split built from `Φ₂ᶜ(x)` on `[0, 1]`:
`A₆(t) = ∫₀ᵗ x⁵ Φ₂ᶜ(x) dx / m₅`, `V_{6,s}(r) = (1 − A₆(r/r_c)) / r⁶`.
Satisfies `V_{6,s}(r_c) = V_{6,s}'(r_c) = V_{6,s}''(r_c) = 0` identically,
while `Φ₆(r) = A₆(r/r_c) / r⁶` is even analytic in `r²` at `r = 0`.
"""
struct C2ProlateDispersionSplit <: AbstractDispersionSplit
    cutoff::Float64
    shape::Float64
    pf::GeneralizedProlateFunction
    m5::Float64
    phi0::Float64             # Φ₆(0) = 1 / (6 m₅ r_c⁶)
    phi6_hat0::Float64        # Φ̂₆(0)
    beta_ref::Float64         # 7 / r_c reference Gaussian for singularity-free spectral transform
    rq::Vector{Float64}       # quadrature nodes on [0, r_c]
    w_dphi::Vector{Float64}   # 4π w_q r_q² [Φ₆(r_q) − Φ₆^Gauss(r_q; β_ref)]
    u::Vector{Float64}        # quadrature nodes on [0, 1] for Φ₆(r) evaluation
    wu::Vector{Float64}
end

@inline function _g_ess_host(x2::Float64)
    return exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2)
end

function _phi6_gauss_host(s::Float64, beta::Float64)
    z = beta * beta * s
    if z < 1e-3
        acc = 0.0
        term = 1.0
        for m in 0:28
            acc += term / (m + 3.0)
            term *= -z / (m + 1.0)
        end
        return 0.5 * (beta^6) * acc
    end
    return (1.0 - _g_ess_host(z)) / (s * s * s)
end

function _pf6_gauss_host(s::Float64, beta::Float64)
    z = beta * beta * s
    if z < 1e-3
        acc = 0.0
        term = 1.0
        for n in 0:28
            acc += term / (n + 4.0)
            term *= -z / (n + 1.0)
        end
        return (beta^8) * acc
    end
    return (6.0 * (1.0 - _g_ess_host(z)) - (z^3) * exp(-z)) / (s^4)
end

function _phi6_hat_gauss_host(k::Float64, beta::Float64)
    u = k / (2.0 * beta)
    u2 = u * u
    eu = exp(-u2)
    erfc_u = _host_erfc(u)
    pref = (pi^1.5) * (beta^3) / 3.0
    g = pref * ((1.0 - 2.0 * u2) * eu + 2.0 * (u^3) * sqrt(pi) * erfc_u)
    gpk = 0.5 * (pi^1.5) * beta * (u * sqrt(pi) * erfc_u - eu)
    return g, gpk
end

function C2ProlateDispersionSplit(cutoff::Real, shape::Real; nquad::Integer=128)
    isfinite(cutoff) && cutoff > 0 || throw(ArgumentError("Dispersion cutoff must be finite and positive."))
    isfinite(shape) && shape > 0 || throw(ArgumentError("Dispersion split shape must be finite and positive."))
    rc = Float64(cutoff)
    pf = GeneralizedProlateFunction(shape, 2)
    u, wu = gauss_legendre(nquad; a=0.0, b=1.0)
    m5 = sum(wu .* (u .^ 5) .* pf.(u))
    inv_m5 = 1.0 / m5
    rc6 = rc^6
    phi0 = inv_m5 / (6.0 * rc6)
    beta_ref = 7.0 / rc
    rq, wrq = gauss_legendre(nquad; a=0.0, b=rc)
    w_dphi = Vector{Float64}(undef, nquad)
    for q in 1:nquad
        r = rq[q]
        t = r / rc
        phi_r = (inv_m5 / rc6) * sum(wu[j] * (u[j]^5) * pf(t * u[j]) for j in 1:nquad)
        dphi = phi_r - _phi6_gauss_host(r * r, beta_ref)
        w_dphi[q] = 4.0 * pi * wrq[q] * (r * r) * dphi
    end
    phi6_hat0 = _phi6_hat_gauss_host(0.0, beta_ref)[1] + sum(w_dphi)
    return C2ProlateDispersionSplit(rc, Float64(shape), pf, m5, phi0, phi6_hat0, beta_ref, rq, w_dphi, u, wu)
end

dispersion_self_coefficient(sp::GaussianDispersionSplit) = (sp.beta^6) / 6.0
dispersion_self_coefficient(sp::C2ProlateDispersionSplit) = sp.phi0

dispersion_zero_mode_coefficient(sp::GaussianDispersionSplit) = 0.5 * (pi^1.5) * (sp.beta^3) / 3.0
dispersion_zero_mode_coefficient(sp::C2ProlateDispersionSplit) = 0.5 * sp.phi6_hat0

function dispersion_cutoff_energy(sp::GaussianDispersionSplit, rc::Real)
    rc2 = Float64(rc)^2
    return _g_ess_host((sp.beta^2) * rc2) / (rc2^3)
end
dispersion_cutoff_energy(::C2ProlateDispersionSplit, ::Real) = 0.0

"""
    dispersion_long_range_correction(split, s) -> (phi6, pf6)

Host (Float64) smooth long-range part `phi6 = Φ₆(√s)` and `pf6 = −(1/r) dΦ₆/dr`
at `s = r² ∈ [0, r_c²]`, fitted by `ClusterPairs.dispersion_polynomial`.
"""
function dispersion_long_range_correction(sp::GaussianDispersionSplit, s::Float64)
    return _phi6_gauss_host(s, sp.beta), _pf6_gauss_host(s, sp.beta)
end

function dispersion_long_range_correction(sp::C2ProlateDispersionSplit, s::Float64)
    rc = sp.cutoff
    r = sqrt(max(s, 0.0))
    t = min(r / rc, 1.0)
    inv_m5 = 1.0 / sp.m5
    rc6 = rc^6
    phi = (inv_m5 / rc6) * sum(sp.wu[j] * (sp.u[j]^5) * sp.pf(t * sp.u[j]) for j in eachindex(sp.u))
    if t < 1e-4
        h = 1e-4
        d2 = (sp.pf(h) - 2.0 * sp.pf(0.0) + sp.pf(-h)) / (h * h)
        return phi, -(inv_m5 / (rc6 * rc * rc)) * (d2 / 8.0)
    end
    dp = sum(sp.wu[j] * (sp.u[j]^6) * prolate_derivative(sp.pf, t * sp.u[j]) for j in eachindex(sp.u))
    return phi, -(inv_m5 / (rc6 * rc * r)) * dp
end

"""
    dispersion_beta_from_tolerance(cutoff, tol) -> β

Gaussian dispersion parameter solving `e^{−β²r_c²}(1 + β²r_c² + ½ β⁴r_c⁴) = tol`.
"""
function dispersion_beta_from_tolerance(cutoff::Real, tol::Real)
    0 < tol < 1 || throw(ArgumentError("Dispersion real_space_tolerance must lie in (0, 1)."))
    f(x) = _g_ess_host(x * x) - Float64(tol)
    lo, hi = 0.0, 12.0
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi) / Float64(cutoff)
end

# ------------------------------------------------------------------------------
# Device-callable spectral transforms (compatible with Influence.jl kernels)
# ------------------------------------------------------------------------------
struct GaussianDispersionSpectral
    beta::Float64
end

struct ProlateDispersionSpectral{V}
    beta_ref::Float64
    rq::V
    w_dphi::V
end

struct TabulatedDispersionSpectral{V}
    rc::Float64
    du::Float64
    g::V
    dg::V
    exact::ProlateDispersionSpectral{V}
end

@inline function spectral_g(s::GaussianDispersionSpectral, k::Float64)
    u = k / (2.0 * s.beta)
    u2 = u * u
    eu = exp(-u2)
    erfc_u = _erfc(u)
    pref = (pi * sqrt(pi)) * (s.beta * s.beta * s.beta) / 3.0
    g = pref * ((1.0 - 2.0 * u2) * eu + 2.0 * (u2 * u) * sqrt(pi) * erfc_u)
    gpk = 0.5 * (pi * sqrt(pi)) * s.beta * (u * sqrt(pi) * erfc_u - eu)
    return g, gpk
end
@inline spectral_g(s::GaussianDispersionSpectral, k::Float32) = map(Float32, spectral_g(s, Float64(k)))

@inline function spectral_g(s::ProlateDispersionSpectral, k::Float64)
    g0, gpk0 = spectral_g(GaussianDispersionSpectral(s.beta_ref), k)
    dg = 0.0
    dgpk = 0.0
    @inbounds for q in 1:length(s.rq)
        r = s.rq[q]
        w = s.w_dphi[q]
        kr = k * r
        if kr < 1e-3
            kr2 = kr * kr
            j0 = 1.0 - kr2 / 6.0 + (kr2 * kr2) / 120.0
            dj0k = -(r * r) * (1.0 / 3.0 - kr2 / 30.0)
        else
            sn, cs = sincos(kr)
            j0 = sn / kr
            dj0k = (r * r) * (kr * cs - sn) / (kr * kr * kr)
        end
        dg += w * j0
        dgpk += w * dj0k
    end
    return g0 + dg, gpk0 + dgpk
end
@inline spectral_g(s::ProlateDispersionSpectral, k::Float32) = map(Float32, spectral_g(s, Float64(k)))

@inline function spectral_g(s::TabulatedDispersionSpectral, k::Float64)
    u = k * s.rc
    t = u / s.du
    j = unsafe_trunc(Int, t)
    j >= length(s.g) - 1 && return spectral_g(s.exact, k)
    θ = t - j; θ2 = θ * θ; θ3 = θ2 * θ
    @inbounds begin
        g0 = s.g[j + 1]; g1 = s.g[j + 2]; d0 = s.dg[j + 1] * s.du; d1 = s.dg[j + 2] * s.du
    end
    g = (2θ3 - 3θ2 + 1) * g0 + (θ3 - 2θ2 + θ) * d0 + (-2θ3 + 3θ2) * g1 + (θ3 - θ2) * d1
    dg_du = ((6θ2 - 6θ) * g0 + (3θ2 - 4θ + 1) * d0 + (-6θ2 + 6θ) * g1 + (3θ2 - 2θ) * d1) / s.du
    gpk = k > 1e-12 ? (dg_du * s.rc / k) : spectral_g(s.exact, k)[2]
    return g, gpk
end
@inline spectral_g(s::TabulatedDispersionSpectral, k::Float32) = map(Float32, spectral_g(s, Float64(k)))

function _dispersion_table_kernel!(gt, dgt, exact::ProlateDispersionSpectral, rc::Float64, du::Float64)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > length(gt) && return
    u = (j - 1) * du
    k = u / rc
    g, gpk = spectral_g(exact, max(k, 1e-12))
    @inbounds gt[j] = g
    @inbounds dgt[j] = (gpk * k) / rc
    return
end
