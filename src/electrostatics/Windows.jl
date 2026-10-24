# =========================
#   Particle–grid assignment windows
# =========================
# A window maps a particle at fractional grid coordinate u ∈ [0, K) onto P
# consecutive grid nodes k0, …, k0 + P − 1 (taken modulo K) with weights
# W(u − k) and derivatives dW/du. The same window is used for spreading and
# for interpolation/gathering.

abstract type AbstractWindow end

"""
    BSplineWindow(order)

Cardinal B-spline `M_P` (Essmann et al. 1995) of order `P` = stencil width;
order 4 is cubic interpolation. Node `k` receives `M_P(u − k)`, so the support
is `floor(u) − P + 1 … floor(u)`.
"""
struct BSplineWindow <: AbstractWindow
    order::Int
    function BSplineWindow(order::Integer)
        3 <= order <= 12 || throw(ArgumentError("B-spline assignment order must be between 3 and 12; got $(order)."))
        return new(Int(order))
    end
end

"""
    ProlateWindow(order, shape)

Prolate assignment window of ESP: `W(x) = ψ₀ᶜ(2x/P)` for `|x| ≤ P/2` grid
spacings (`c = shape`), centred on the particle. Device evaluation uses a
Chebyshev interpolant of `ψ` and its exact derivative.
"""
struct ProlateWindow <: AbstractWindow
    order::Int
    shape::Float64
    coeffs::Vector{Float64}
    dcoeffs::Vector{Float64}
    pf::ProlateFunction
end

function ProlateWindow(order::Integer, shape::Real)
    2 <= order <= 16 || throw(ArgumentError("Prolate window width must be between 2 and 16 grid points; got $(order)."))
    isfinite(shape) && shape > 0 || throw(ArgumentError("Prolate window shape must be finite and positive."))
    pf = ProlateFunction(shape)
    c = chebyshev_fit_adaptive(y -> pf(y))
    return ProlateWindow(Int(order), Float64(shape), c, chebyshev_derivative(c), pf)
end

window_order(w::AbstractWindow) = w.order

# -------------------------
# Device stencils
# -------------------------
struct BSplineStencil{P} end

struct ProlateStencil{P,T,N,M}
    c::NTuple{N,T}
    dc::NTuple{M,T}
end

device_stencil(w::BSplineWindow, ::Type{T}) where {T} = BSplineStencil{w.order}()
device_stencil(w::ProlateWindow, ::Type{T}) where {T} =
    ProlateStencil{w.order,T,length(w.coeffs),length(w.dcoeffs)}(Tuple(T.(w.coeffs)), Tuple(T.(w.dcoeffs)))

# One B-spline order step: from a[j+1] = M_n(t + j), j = 0..n−1, to M_{n+1}.
@inline function _bspline_step(a::NTuple{N,T}, t::T) where {N,T}
    inv = one(T) / T(N)
    return ntuple(Val(N + 1)) do jj
        j = T(jj - 1)
        left = jj <= N ? a[jj] : zero(T)
        right = jj >= 2 ? a[jj - 1] : zero(T)
        ((t + j) * left + (T(N + 1) - t - j) * right) * inv
    end
end

@inline _bspline_upto(a::NTuple{N,T}, t::T, ::Val{N}) where {N,T} = a
@inline _bspline_upto(a::NTuple{N,T}, t::T, v::Val{Q}) where {N,Q,T} =
    _bspline_upto(_bspline_step(a, t), t, v)

"""
    window_stencil(stencil, u) -> (k0, w, dw)

First node `k0` (may be negative; callers wrap modulo `K`) and the `P` weights
`w[j] = W(u − (k0 + j − 1))` and derivatives `dw[j] = dW/du` in ascending node
order.
"""
@inline function window_stencil(::BSplineStencil{P}, u::T) where {P,T<:AbstractFloat}
    fl = floor(u)
    t = u - fl
    a = _bspline_upto((t, one(T) - t), t, Val(P - 1))          # M_{P−1}(t + j)
    full = _bspline_step(a, t)                                  # M_P(t + j)
    # node floor(u) − j carries M_P(t + j); reverse into ascending node order
    w = ntuple(i -> full[P + 1 - i], Val(P))
    dw = ntuple(Val(P)) do i
        j = P - i                                               # 0-based j for this node
        left = j + 1 <= P - 1 ? a[j + 1] : zero(T)
        right = j >= 1 ? a[j] : zero(T)
        left - right
    end
    return unsafe_trunc(Int32, fl) - Int32(P - 1), w, dw
end

@inline function window_stencil(s::ProlateStencil{P,T}, u::T) where {P,T<:AbstractFloat}
    k0 = iseven(P) ? unsafe_trunc(Int32, floor(u)) - Int32(P ÷ 2 - 1) :
                     unsafe_trunc(Int32, floor(u + T(0.5))) - Int32((P - 1) ÷ 2)
    scale = T(2) / T(P)
    w = ntuple(Val(P)) do i
        y = (T(k0 + Int32(i - 1)) - u) * scale
        abs(y) <= one(T) ? clenshaw(s.c, y) : zero(T)
    end
    dw = ntuple(Val(P)) do i
        y = (T(k0 + Int32(i - 1)) - u) * scale
        abs(y) <= one(T) ? -scale * clenshaw(s.dc, y) : zero(T)
    end
    return k0, w, dw
end

# -------------------------
# Host-side window transforms
# -------------------------

# M_P at the integers 0..P−1 (Float64).
function _bspline_integer_values(P::Int)
    a = (0.0, 1.0)
    while length(a) < P
        a = _bspline_step(a, 0.0)
    end
    return collect(a)
end

@inline _signed_mode(m::Integer, K::Integer) = m <= K ÷ 2 ? m : m - K

"""
    window_deconvolution(window, K) -> Vector{Float64}

Per-dimension correction `D(m)`, `m = 0..K−1`, multiplying the influence
function: SPME's exact discrete B-spline factor `1/|Σ_j M_P(j) e^{2πimj/K}|²`,
or `1/φ̂(2πm/K)²` with the continuum transform of the prolate window in grid
units for ESP.
"""
function window_deconvolution(w::BSplineWindow, K::Integer)
    P = w.order
    P <= K || throw(ArgumentError("Mesh dimension $(K) is smaller than the assignment order $(P)."))
    M = _bspline_integer_values(P)
    D = Vector{Float64}(undef, K)
    for m in 0:(K - 1)
        re = 0.0
        im = 0.0
        for j in 0:(P - 1)
            s, c = sincos(2pi * m * j / K)
            re += M[j + 1] * c
            im += M[j + 1] * s
        end
        den = re^2 + im^2
        D[m + 1] = den > 1e-12 ? 1.0 / den : NaN
    end
    # odd orders vanish at the Nyquist mode: use the neighbour average (Essmann)
    for m in 0:(K - 1)
        if isnan(D[m + 1])
            l = D[mod(m - 1, K) + 1]
            r = D[mod(m + 1, K) + 1]
            D[m + 1] = 0.5 * (l + r)
        end
    end
    return D
end

"""
    prolate_window_transform(window, θ) -> φ̂(θ)

Continuum Fourier transform of the prolate window in grid units,
`φ̂(θ) = (P/2) ∫_{-1}^{1} ψ(t) cos(θ P t / 2) dt`.
"""
function prolate_window_transform(w::ProlateWindow, theta::Real; nquad::Integer=192)
    x, wq = gauss_legendre(nquad)
    vals = w.pf.(x)
    half = 0.5 * w.order
    return half * sum(wq .* vals .* cos.(theta * half .* x))
end

function window_deconvolution(w::ProlateWindow, K::Integer)
    w.order <= K || throw(ArgumentError("Mesh dimension $(K) is smaller than the window width $(w.order)."))
    x, wq = gauss_legendre(192)
    vals = w.pf.(x)
    half = 0.5 * w.order
    D = Vector{Float64}(undef, K)
    for m in 0:(K - 1)
        theta = 2pi * _signed_mode(m, K) / K
        phat = half * sum(wq .* vals .* cos.(theta * half .* x))
        phat > 0 || throw(ArgumentError("Prolate window transform vanishes inside the mesh band; increase the window shape or width."))
        D[m + 1] = 1.0 / phat^2
    end
    return D
end

"""
    window_alias_table(window, K, nmax) -> Matrix{Float64}

Window transform at the alias frequencies used by optimal influence
functions: entry `[n + nmax + 1, m + 1]` is `U(θ_m + 2πn)` with
`θ_m = 2π m′/K` (signed mode `m′`). B-spline: `(sin(θ/2)/(θ/2))^P`
(transform of `M_P` up to a phase). Prolate: the continuum transform
`φ̂(θ)` of the truncated prolate window in grid units.
"""
function window_alias_table(w::BSplineWindow, K::Integer, nmax::Integer)
    U = Matrix{Float64}(undef, 2nmax + 1, K)
    for m in 0:(K - 1), n in -nmax:nmax
        x = pi * (_signed_mode(m, K) + K * n) / K
        s = abs(x) < 1e-12 ? 1.0 : sin(x) / x
        U[n + nmax + 1, m + 1] = s^w.order
    end
    return U
end

function window_alias_table(w::ProlateWindow, K::Integer, nmax::Integer)
    x, wq = gauss_legendre(256)
    vals = w.pf.(x)
    half = 0.5 * w.order
    U = Matrix{Float64}(undef, 2nmax + 1, K)
    for m in 0:(K - 1), n in -nmax:nmax
        theta = 2pi * (_signed_mode(m, K) + K * n) / K
        U[n + nmax + 1, m + 1] = half * sum(wq .* vals .* cos.(theta * half .* x))
    end
    return U
end
