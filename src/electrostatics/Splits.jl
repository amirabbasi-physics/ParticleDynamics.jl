# =========================
#   Radial short/long-range splits
# =========================
# Every split is described by a dimensionless kernel χ (∫χ = 1) and its
# cumulative a(t) = ∫₀ᵗ χ, t = r/r_c (see the implementation log, Step 1):
#
#   1/r = E_s(t)/r + a(t)/r,                E_s = 1 − a,
#   short-range force factor  f(r) = [E_s/r − E_s'(t)/r_c] / r²  (F = C qᵢqⱼ f d),
#   long-range transform       g(k) = 4π ĉ(k r_c)/k²,  ĉ(u) = ∫ χ(x) cos(ux) dx,
#   self energy per particle  −(C/2) χ(0)/r_c qᵢ²,
#   neutralising background   −C π r_c² m₂ Q²/V,  m₂ = ∫ x² χ(x) dx.

abstract type AbstractSplit end

"""
    GaussianSplit(alpha)

Conventional Ewald split `erfc(αr)/r + erf(αr)/r` with `alpha` in inverse
length units.
"""
struct GaussianSplit <: AbstractSplit
    alpha::Float64
    function GaussianSplit(alpha::Real)
        isfinite(alpha) && alpha > 0 || throw(ArgumentError("Ewald alpha must be finite and positive."))
        return new(Float64(alpha))
    end
end

"""
    ProlateSplit(cutoff, shape)

Ewald split with prolates (ESP): `χ(x) ∝ ψ₀ᶜ(x)` on `[0, 1]` with bandlimit
`c = shape`, zero beyond `x = 1`, so the short-range part vanishes exactly at
`r = cutoff`. Device evaluation uses a Chebyshev interpolant of `E_s(t)` and
the exact derivative of that same polynomial for forces.
"""
struct ProlateSplit <: AbstractSplit
    cutoff::Float64
    shape::Float64
    es::Vector{Float64}       # Chebyshev coefficients of E_s(t), s = 2t − 1
    des::Vector{Float64}      # coefficients of dE_s/dt (same polynomial, differentiated)
    chi0::Float64             # χ(0)
    m2::Float64               # ∫₀¹ x² χ(x) dx
    xq::Vector{Float64}       # quadrature nodes on [0, 1]
    wchi::Vector{Float64}     # quadrature weights times χ(x_q)
end

function ProlateSplit(cutoff::Real, shape::Real; nquad::Integer=128)
    isfinite(cutoff) && cutoff > 0 || throw(ArgumentError("ESP cutoff must be finite and positive."))
    isfinite(shape) && shape > 0 || throw(ArgumentError("ESP split shape must be finite and positive."))
    pf = ProlateFunction(shape)
    norm = prolate_integral(pf, 1.0)
    a(t) = prolate_integral(pf, t) / norm
    es = chebyshev_fit_adaptive(s -> 1.0 - a(0.5 * (s + 1.0)))
    des = 2.0 .* chebyshev_derivative(es)          # d/dt = 2 d/ds
    xq, wq = gauss_legendre(nquad; a=0.0, b=1.0)
    chi = pf.(xq) ./ norm
    m2 = sum(wq .* xq .^ 2 .* chi)
    return ProlateSplit(Float64(cutoff), Float64(shape), es, des, 1.0 / norm, m2, xq, wq .* chi)
end

"""
    self_coefficient(split) -> χ(0)/r_c

The self energy of particle `i` is `−(C/2) self_coefficient · qᵢ²`.
"""
self_coefficient(sp::GaussianSplit) = 2.0 * sp.alpha / sqrt(pi)
self_coefficient(sp::ProlateSplit) = sp.chi0 / sp.cutoff

"""
    cutoff_energy(split, rc) -> E_s(r_c)/r_c

Short-range pair energy per unit `C qᵢqⱼ` at the cutoff, the constant removed
by the potential-shift modifier (`erfc(α r_c)/r_c` for the Gaussian split,
exactly zero for the prolate split).
"""
cutoff_energy(sp::GaussianSplit, rc::Real) = _host_erfc(sp.alpha * rc) / rc
cutoff_energy(sp::ProlateSplit, rc::Real) = 0.0

"""
    background_coefficient(split) -> π r_c² m₂

The uniform neutralising background energy is `−C background_coefficient Q²/V`.
"""
background_coefficient(sp::GaussianSplit) = pi / (2.0 * sp.alpha^2)
background_coefficient(sp::ProlateSplit) = pi * sp.cutoff^2 * sp.m2

# -------------------------
# Device representations
# -------------------------
struct GaussianPairKernel{T}
    alpha::T
    two_alpha_sqrtpi::T
end

struct ProlatePairKernel{T,N,M}
    inv_rc::T
    es::NTuple{N,T}
    des::NTuple{M,T}
end

device_pair_kernel(sp::GaussianSplit, ::Type{T}) where {T} =
    GaussianPairKernel{T}(T(sp.alpha), T(2.0 * sp.alpha / sqrt(pi)))
device_pair_kernel(sp::ProlateSplit, ::Type{T}) where {T} =
    ProlatePairKernel{T,length(sp.es),length(sp.des)}(T(1.0 / sp.cutoff), Tuple(T.(sp.es)), Tuple(T.(sp.des)))

@inline _erfc(x::Float64) = ccall("extern __nv_erfc", llvmcall, Float64, (Float64,), x)
@inline _erfc(x::Float32) = ccall("extern __nv_erfcf", llvmcall, Float32, (Float32,), x)

"""
    pair_kernel(kernel, r, invr) -> (e, f)

Short-range energy `e = E_s/r` and force factor `f` (force `= C qᵢqⱼ f d`)
per unit `C qᵢ qⱼ`.
"""
@inline function pair_kernel(k::GaussianPairKernel{T}, r::T, invr::T) where {T}
    ar = k.alpha * r
    e = _erfc(ar) * invr
    f = (e + k.two_alpha_sqrtpi * exp(-ar * ar)) * invr * invr
    return e, f
end

@inline function pair_kernel(k::ProlatePairKernel{T}, r::T, invr::T) where {T}
    t = r * k.inv_rc
    t >= one(T) && return zero(T), zero(T)
    s = t + t - one(T)
    Es = clenshaw(k.es, s)
    dEs = clenshaw(k.des, s)
    e = Es * invr
    f = (e - dEs * k.inv_rc) * invr * invr
    return e, f
end

"""
    long_range_correction(split, s) -> (corr, fcorr)

Host (Float64) evaluation of the smooth long-range part of the split at
`s = r²`: `corr = 1/r − E_s(r)/r` and `fcorr = −(1/r) d corr/dr`, both even
analytic functions of `r` inside the cutoff. The cluster-pair kernels fit them
by polynomials in `r²` (`ClusterPairs.coulomb_polynomial`).
"""
function long_range_correction(sp::GaussianSplit, s::Float64)
    setprecision(BigFloat, 256) do
        a = BigFloat(sp.alpha)
        z = a * a * BigFloat(s)
        ce = one(BigFloat)                 # Σ (−z)ⁿ / (n! (2n+1))
        cf = zero(BigFloat)                # −2 d/ds of the series
        prev = one(BigFloat)               # z^(n−1)/(n−1)!
        tiny = BigFloat(10)^-45
        for n in 1:1000
            term = prev * z / n
            sgn = isodd(n) ? -1 : 1
            ce += sgn * term / (2n + 1)
            cf -= 2 * sgn * a * a * prev / (2n + 1)
            prev = term
            n > 10 && abs(term) < tiny && break
        end
        pre = 2a / sqrt(BigFloat(pi))
        return Float64(pre * ce), Float64(pre * cf)
    end
end

# Prolate split: with a(t) = ∫₀ᵗ χ, corr = a/r = (1/r_c) ∫₀¹ χ(t u) du and
# fcorr = (1/(r_c r²)) ∫₀¹ (χ(t u) − χ(t)) du, evaluated by Gauss–Legendre
# quadrature without the cancellation of 1 − E_s near r = 0.
struct ProlateCorrection
    rc::Float64
    pf::ProlateFunction
    inv_norm::Float64
    u::Vector{Float64}
    w::Vector{Float64}
end

function ProlateCorrection(sp::ProlateSplit; nquad::Integer=96)
    pf = ProlateFunction(sp.shape)
    u, w = gauss_legendre(nquad; a=0.0, b=1.0)
    return ProlateCorrection(sp.cutoff, pf, 1.0 / prolate_integral(pf, 1.0), u, w)
end

function long_range_correction(pc::ProlateCorrection, s::Float64)
    rc = pc.rc
    r = sqrt(s)
    t = min(r / rc, 1.0)
    chit = pc.pf(t) * pc.inv_norm
    ic = 0.0
    id = 0.0
    @inbounds for k in eachindex(pc.u)
        chiu = pc.pf(t * pc.u[k]) * pc.inv_norm
        ic += pc.w[k] * chiu
        id += pc.w[k] * (chiu - chit)
    end
    corr = ic / rc
    if t < 1e-4
        # limit: ∫₀¹ (χ(tu) − χ(t)) du / r² → χ''(0) (1/3 − 1)/2 / r_c²
        h = 1e-3
        d2 = (pc.pf(h) - 2 * pc.pf(0.0) + pc.pf(-h)) * pc.inv_norm / h^2
        return corr, d2 * (1 / 3 - 1) / 2 / rc^3
    end
    return corr, id / (rc * r^2)
end

long_range_correction(sp::ProlateSplit, s::Float64) = long_range_correction(ProlateCorrection(sp), s)

# Spectral transforms used by the setup kernels (always Float64).
struct GaussianSpectral
    alpha::Float64
end

struct ProlateSpectral{V}
    rc::Float64
    xq::V
    wchi::V
end

"""
    spectral_g(spec, k) -> (g, g′/k)

`g(k) = 4π ĉ(k r_c)/k²` and its radial derivative divided by `k`.
"""
@inline function spectral_g(s::GaussianSpectral, k::Float64)
    k2 = k * k
    g = 4.0 * pi * exp(-k2 / (4.0 * s.alpha^2)) / k2
    return g, g * (-2.0 / k2 - 1.0 / (2.0 * s.alpha^2))
end

@inline function spectral_g(s::ProlateSpectral, k::Float64)
    u = k * s.rc
    c = 0.0
    dc = 0.0
    @inbounds for q in 1:length(s.xq)
        x = s.xq[q]
        w = s.wchi[q]
        sn, cs = sincos(u * x)
        c += w * cs
        dc -= w * x * sn
    end
    k2 = k * k
    g = 4.0 * pi * c / k2
    gp = 4.0 * pi * (s.rc * dc / k2 - 2.0 * c / (k2 * k))
    return g, gp / k
end

"""
    TabulatedSpectral

The prolate split's ĉ(u) and dĉ/du (u = k r_c) on uniform nodes, interpolated by cubic
Hermite (error ≈ 1e-8 of ĉ at the default spacing). Device refreshes of the influence
function (device-resident barostat) use it instead of the 128-point quadrature per
wavevector; beyond the table the exact quadrature is used.
"""
struct TabulatedSpectral{V}
    rc::Float64
    du::Float64
    c::V
    dc::V
    exact::ProlateSpectral{V}
end

# Float32 forms (influence sums of Float32 solvers): the same tables, Float32 arithmetic.
@inline spectral_g(s::GaussianSpectral, k::Float32) = map(Float32, spectral_g(s, Float64(k)))
@inline spectral_g(s::ProlateSpectral, k::Float32) = map(Float32, spectral_g(s, Float64(k)))
@inline function spectral_g(s::TabulatedSpectral, k::Float32)
    rc = Float32(s.rc); du = Float32(s.du)
    t = k * rc / du
    j = unsafe_trunc(Int, t)
    j >= length(s.c) - 1 && return map(Float32, spectral_g(s.exact, Float64(k)))
    θ = t - Float32(j); θ2 = θ * θ; θ3 = θ2 * θ
    @inbounds begin
        c0 = Float32(s.c[j + 1]); c1 = Float32(s.c[j + 2]); d0 = Float32(s.dc[j + 1]) * du; d1 = Float32(s.dc[j + 2]) * du
    end
    c = (2θ3 - 3θ2 + 1) * c0 + (θ3 - 2θ2 + θ) * d0 + (-2θ3 + 3θ2) * c1 + (θ3 - θ2) * d1
    dc = ((6θ2 - 6θ) * c0 + (3θ2 - 4θ + 1) * d0 + (-6θ2 + 6θ) * c1 + (3θ2 - 2θ) * d1) / du
    k2 = k * k
    g = 4.0f0 * Float32(pi) * c / k2
    gp = 4.0f0 * Float32(pi) * (rc * dc / k2 - 2.0f0 * c / (k2 * k))
    return g, gp / k
end
@inline function spectral_g(s::TabulatedSpectral, k::Float64)
    u = k * s.rc
    t = u / s.du
    j = unsafe_trunc(Int, t)
    j >= length(s.c) - 1 && return spectral_g(s.exact, k)
    θ = t - j; θ2 = θ * θ; θ3 = θ2 * θ
    @inbounds begin
        c0 = s.c[j + 1]; c1 = s.c[j + 2]; d0 = s.dc[j + 1] * s.du; d1 = s.dc[j + 2] * s.du
    end
    c = (2θ3 - 3θ2 + 1) * c0 + (θ3 - 2θ2 + θ) * d0 + (-2θ3 + 3θ2) * c1 + (θ3 - θ2) * d1
    dc = ((6θ2 - 6θ) * c0 + (3θ2 - 4θ + 1) * d0 + (-6θ2 + 6θ) * c1 + (3θ2 - 2θ) * d1) / s.du
    k2 = k * k
    g = 4.0 * pi * c / k2
    gp = 4.0 * pi * (s.rc * dc / k2 - 2.0 * c / (k2 * k))
    return g, gp / k
end

# ĉ(u) and dĉ/du at u = (j − 1)·du, by the exact quadrature (one thread per node).
function _prolate_table_kernel!(c, dc, xq, wchi, du::Float64)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > length(c) && return
    u = (j - 1) * du
    a = 0.0; b = 0.0
    @inbounds for q in 1:length(xq)
        x = xq[q]; w = wchi[q]
        sn, cs = sincos(u * x)
        a += w * cs
        b -= w * x * sn
    end
    @inbounds c[j] = a; @inbounds dc[j] = b
    return
end
