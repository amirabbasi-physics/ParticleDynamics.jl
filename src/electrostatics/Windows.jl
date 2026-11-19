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

Prolate assignment window of ESP: `W(x) = ψ₀ᶜ(2x/P) − ψ₀ᶜ(1)` for `|x| ≤ P/2`
grid spacings (`c = shape`), centred on the particle. The constant `edge =
ψ₀ᶜ(1)` makes the weights vanish at the edge of the support, so the mesh
energy is continuous when a particle moves from one cell to the next; the
window transform, the influence function and the deconvolution all use the
same shifted window. Device evaluation uses one polynomial per stencil node
and its exact derivative (`ProlatePolyStencil`); the Chebyshev interpolant of
the shifted `ψ` stored here is the reference evaluation.
"""
struct ProlateWindow <: AbstractWindow
    order::Int
    shape::Float64
    coeffs::Vector{Float64}
    dcoeffs::Vector{Float64}
    pf::ProlateFunction
    edge::Float64
end

function ProlateWindow(order::Integer, shape::Real)
    2 <= order <= 16 || throw(ArgumentError("Prolate window width must be between 2 and 16 grid points; got $(order)."))
    isfinite(shape) && shape > 0 || throw(ArgumentError("Prolate window shape must be finite and positive."))
    pf = ProlateFunction(shape)
    edge = pf(1.0)
    c = chebyshev_fit_adaptive(y -> pf(y) - edge)
    return ProlateWindow(Int(order), Float64(shape), c, chebyshev_derivative(c), pf, edge)
end

"""
    C2ProlateWindow(order, shape)

C²-continuous assignment window built from Slepian's generalized prolate
`Φ₃ᶜ(y) = (1 − y³)… = (1 − y²)³ φ(y)` (`W(x) = Φ₃ᶜ(2x/P)` on `|x| ≤ P/2`).
Because `Φ₃ᶜ(±1) = (Φ₃ᶜ)'(±1) = (Φ₃ᶜ)''(±1) = 0` identically (`edge = 0`),
mesh energies, analytic-differentiation mesh forces (`dW/du`), and force
derivatives (`d²W/du²`) are all continuous across every grid plane.
"""
struct C2ProlateWindow <: AbstractWindow
    order::Int
    shape::Float64
    coeffs::Vector{Float64}
    dcoeffs::Vector{Float64}
    pf::GeneralizedProlateFunction
    edge::Float64
end

function C2ProlateWindow(order::Integer, shape::Real)
    2 <= order <= 16 || throw(ArgumentError("C2 prolate window width must be between 2 and 16 grid points; got $(order)."))
    isfinite(shape) && shape > 0 || throw(ArgumentError("C2 prolate window shape must be finite and positive."))
    pf = GeneralizedProlateFunction(shape, 3)
    c = chebyshev_fit_adaptive(y -> pf(y))
    return C2ProlateWindow(Int(order), Float64(shape), c, chebyshev_derivative(c), pf, 0.0)
end

const AnyProlateWindow = Union{ProlateWindow,C2ProlateWindow}

window_order(w::AbstractWindow) = w.order

# -------------------------
# Device stencils
# -------------------------
struct BSplineStencil{P} end

struct ProlateStencil{P,T,N,M}
    c::NTuple{N,T}
    dc::NTuple{M,T}
end

# Prolate window as one polynomial per stencil node. With τ the offset of the particle from the
# centre of its cell (τ ∈ [−½, ½)), node j (0-based) carries W_j(τ) = ψ((j − (P−1)/2 − τ)·2/P),
# which spans only 1/P of the support of ψ and needs a much lower degree than the series of ψ
# on [−1, 1]. ψ is even, W_{P−1−j}(τ) = W_j(−τ), so the first H = ⌈P/2⌉ nodes are stored.
# `c[k][h]` multiplies τ^(k−1) for node h − 1. The derivative weight is the derivative of the
# same polynomial (one Horner pass gives both).
struct ProlatePolyStencil{P,T,D,H}
    c::NTuple{D,NTuple{H,T}}
end

# Monomial coefficients in s ∈ [−1, 1] of the degree-d interpolant of f at the Chebyshev–Lobatto
# points. They include s = ±1, so neighbouring node polynomials agree where a particle crosses
# a cell boundary and the window stays continuous there.
function _lobatto_monomials(f, d::Int)
    v = [f(cospi(i / d)) for i in 0:d]
    a = [(2 / d) * sum((i == 0 || i == d ? 0.5 : 1.0) * v[i + 1] * cospi(i * k / d) for i in 0:d) for k in 0:d]
    a[1] *= 0.5
    a[end] *= 0.5
    m = zeros(d + 1)
    tp = zeros(d + 1); tp[1] = 1.0                                   # T_0
    tc = zeros(d + 1); tc[2] = 1.0                                   # T_1
    m .+= a[1] .* tp
    for k in 1:d
        m .+= a[k + 1] .* tc
        tn = zeros(d + 1)
        tn[2:end] .= 2 .* tc[1:end - 1]
        tn .-= tp
        tp, tc = tc, tn
    end
    return m
end

"""
    prolate_node_polynomials(w::ProlateWindow, T) -> Matrix{Float64} or nothing

Coefficients `C[h, k]` of `τ^(k−1)` of the node polynomials of `w` (nodes `0 … ⌈P/2⌉ − 1`,
`τ ∈ [−½, ½]`), at the lowest even degree whose error is below the round-off of `T`:
weights within `eps(T)/64` of `ψ` (not below `4e-15`) and derivatives within `eps(T)/4` of the
largest derivative (not below `1e-12`). `nothing` if no degree up to 24 reaches that.
"""
function prolate_node_polynomials(w::AnyProlateWindow, ::Type{T}) where {T<:AbstractFloat}
    P = w.order
    H = (P + 1) ÷ 2
    y(h, tau) = (h - (P - 1) / 2 - tau) * 2 / P
    check = range(-0.5, 0.5; length=97)
    wref = [w.pf(y(h, tau)) - w.edge for h in 0:H - 1, tau in check]
    dref = [-(2 / P) * prolate_derivative(w.pf, y(h, tau)) for h in 0:H - 1, tau in check]
    tolw = max(eps(T) / 64, 4e-15)
    told = max(eps(T) / 4, 1e-12) * maximum(abs, dref)
    for d in 6:2:24
        C = Matrix{Float64}(undef, H, d + 1)
        for h in 0:H - 1
            C[h + 1, :] .= _lobatto_monomials(s -> w.pf(y(h, s / 2)) - w.edge, d) .* 2.0 .^ (0:d)
        end
        # odd P: the centre node is its own mirror image, ψ(−2τ/P), an even function of τ
        isodd(P) && (C[H, 2:2:end] .= 0.0)
        ew = 0.0
        ed = 0.0
        for h in 1:H, (n, tau) in enumerate(check)
            p = C[h, d + 1]
            dp = 0.0
            for k in d:-1:1
                dp = muladd(dp, tau, p)
                p = muladd(p, tau, C[h, k])
            end
            ew = max(ew, abs(p - wref[h, n]))
            ed = max(ed, abs(dp - dref[h, n]))
        end
        ew <= tolw && ed <= told && return C
    end
    return nothing
end

device_stencil(w::BSplineWindow, ::Type{T}) where {T} = BSplineStencil{w.order}()
# Prolate window: per-node polynomials (`:polynomial`, the solver's evaluation) or one series of
# `ψ` on `[-1, 1]` (`:chebyshev`, the reference, also used where no polynomial meets the bounds).
function device_stencil(w::AnyProlateWindow, ::Type{T}; evaluator::Symbol=:polynomial) where {T}
    evaluator in (:polynomial, :chebyshev) ||
        throw(ArgumentError("Prolate window evaluator must be :polynomial or :chebyshev; got $(evaluator)."))
    if evaluator === :polynomial
        C = prolate_node_polynomials(w, T)
        if C !== nothing
            H, D = size(C)
            return ProlatePolyStencil{w.order,T,D,H}(ntuple(k -> ntuple(h -> T(C[h, k]), H), D))
        end
    end
    return ProlateStencil{w.order,T,length(w.coeffs),length(w.dcoeffs)}(Tuple(T.(w.coeffs)), Tuple(T.(w.dcoeffs)))
end

# First node of the prolate stencil at u and the offset τ of u from the centre of its cell.
@inline function _prolate_cell(::Val{P}, u::T) where {P,T<:AbstractFloat}
    if iseven(P)
        fl = floor(u)
        return unsafe_trunc(Int32, fl) - Int32(P ÷ 2 - 1), (u - fl) - T(0.5)
    else
        fl = floor(u + T(0.5))
        return unsafe_trunc(Int32, fl) - Int32((P - 1) ÷ 2), u - fl
    end
end

# p(σ) and p'(σ) of node polynomial h by one Horner pass. `Val(h)`: 1-based and static;
# integer h: 0-based and thread-dependent, taken by unrolled selects (`_tsel`).
@generated function _node_poly(c::NTuple{D,NTuple{H,T}}, sigma::T, ::Val{h}) where {D,H,T,h}
    ex = Expr(:block, Expr(:meta, :inline), :(p = c[$D][$h]), :(dp = zero(T)))
    for k in D - 1:-1:1
        push!(ex.args, :(dp = muladd(dp, sigma, p)), :(p = muladd(p, sigma, c[$k][$h])))
    end
    push!(ex.args, :(return p, dp))
    return ex
end
@generated function _node_poly(c::NTuple{D,NTuple{H,T}}, sigma::T, h::Integer) where {D,H,T}
    ex = Expr(:block, Expr(:meta, :inline), :(p = _tsel(c[$D], h)), :(dp = zero(T)))
    for k in D - 1:-1:1
        push!(ex.args, :(dp = muladd(dp, sigma, p)), :(p = muladd(p, sigma, _tsel(c[$k], h))))
    end
    push!(ex.args, :(return p, dp))
    return ex
end

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

@generated function window_stencil(s::ProlatePolyStencil{P,T,D,H}, u::T) where {P,T<:AbstractFloat,D,H}
    ex = Expr(:block, Expr(:meta, :inline), :((k0, tau) = _prolate_cell(Val($P), u)), :(mtau = -tau))
    w = Expr(:tuple)
    dw = Expr(:tuple)
    for j in 0:P - 1
        p = Symbol(:p, j)
        d = Symbol(:d, j)
        if j < H
            push!(ex.args, :(($p, $d) = _node_poly(s.c, tau, Val($(j + 1)))))
            push!(dw.args, d)
        else                                                        # mirror node P − 1 − j at −τ
            push!(ex.args, :(($p, $d) = _node_poly(s.c, mtau, Val($(P - j)))))
            push!(dw.args, :(-$d))
        end
        push!(w.args, p)
    end
    push!(ex.args, :(return k0, $w, $dw))
    return ex
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
`φ̂(θ) = (P/2) ∫_{-1}^{1} (ψ(t) − ψ(1)) cos(θ P t / 2) dt`.
"""
function prolate_window_transform(w::AnyProlateWindow, theta::Real; nquad::Integer=192)
    x, wq = gauss_legendre(nquad)
    vals = w.pf.(x) .- w.edge
    half = 0.5 * w.order
    return half * sum(wq .* vals .* cos.(theta * half .* x))
end

"""
    window_nyquist_attenuation(order, shape) -> Float64
    window_nyquist_attenuation(window::Union{ProlateWindow,C2ProlateWindow}) -> Float64

`(φ̂(π)/φ̂(0))²` of a prolate window (shifted to zero at its edge, as
`ProlateWindow` is): its squared transform at the mesh Nyquist
frequency relative to zero frequency. The influence function divides by the
squared window transform of each mesh direction, so the modes at the corners
of the mesh band are scaled by the inverse cube of this number.
"""
function window_nyquist_attenuation(order::Integer, pf::Union{ProlateFunction,GeneralizedProlateFunction}, edge::Real=pf(1.0))
    x, wq = gauss_legendre(192)
    vals = pf.(x) .- edge
    half = 0.5 * order
    return (sum(wq .* vals .* cos.(pi * half .* x)) / sum(wq .* vals))^2
end
window_nyquist_attenuation(order::Integer, shape::Real) = window_nyquist_attenuation(order, ProlateFunction(shape))
window_nyquist_attenuation(w::AnyProlateWindow) = window_nyquist_attenuation(w.order, w.pf, w.edge)

"""
    min_nyquist_attenuation(T) -> Float64

Smallest `window_nyquist_attenuation` that a mesh solve in precision `T`
resolves: `0.02 eps(T)^(2/3)` (4.8e-7 in Float32, 7.3e-13 in Float64). For a
smaller attenuation `a` the round-off of the charge mesh and of the assignment
weights, amplified by the deconvolution, enters the forces. Measured in both
precisions, that noise is the split tolerance times a factor that grows as
`eps(T)²/a³`; at the bound the factor is below 0.1.
"""
min_nyquist_attenuation(::Type{T}) where {T<:AbstractFloat} = 0.02 * Float64(eps(T))^(2 / 3)

"""
    window_resolved(window, T) -> Bool
    window_resolved(order, shape, T) -> Bool

Whether a mesh solve in precision `T` resolves the window (always true for
B-splines, whose transform decays only algebraically).
"""
window_resolved(w::AnyProlateWindow, ::Type{T}) where {T<:AbstractFloat} =
    window_nyquist_attenuation(w) >= min_nyquist_attenuation(T)
window_resolved(::AbstractWindow, ::Type{T}) where {T<:AbstractFloat} = true
window_resolved(order::Integer, shape::Real, ::Type{T}) where {T<:AbstractFloat} =
    window_nyquist_attenuation(order, shape) >= min_nyquist_attenuation(T)
c2_window_resolved(order::Integer, shape::Real, ::Type{T}) where {T<:AbstractFloat} =
    window_nyquist_attenuation(order, GeneralizedProlateFunction(shape, 3), 0.0) >= min_nyquist_attenuation(T)

function window_deconvolution(w::AnyProlateWindow, K::Integer)
    w.order <= K || throw(ArgumentError("Mesh dimension $(K) is smaller than the window width $(w.order)."))
    x, wq = gauss_legendre(192)
    vals = w.pf.(x) .- w.edge
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

function window_alias_table(w::AnyProlateWindow, K::Integer, nmax::Integer)
    x, wq = gauss_legendre(256)
    vals = w.pf.(x) .- w.edge
    half = 0.5 * w.order
    U = Matrix{Float64}(undef, 2nmax + 1, K)
    for m in 0:(K - 1), n in -nmax:nmax
        theta = 2pi * (_signed_mode(m, K) + K * n) / K
        U[n + nmax + 1, m + 1] = half * sum(wq .* vals .* cos.(theta * half .* x))
    end
    return U
end
