# =========================
#   Host-side quadrature, Legendre and Chebyshev helpers
# =========================
# These run once at solver setup (Float64, host). Nothing here is called from
# a GPU kernel except `clenshaw`, which evaluates a fixed-length coefficient
# tuple with Clenshaw's recurrence.

"""
    gauss_legendre(n; a=-1.0, b=1.0) -> (nodes, weights)

`n`-point Gauss–Legendre rule on `[a, b]` (Golub–Welsch eigenvalue method).
"""
function gauss_legendre(n::Integer; a::Real=-1.0, b::Real=1.0)
    n >= 1 || throw(ArgumentError("Gauss–Legendre rule needs at least one node."))
    if n == 1
        return [0.5 * (a + b)], [Float64(b - a)]
    end
    off = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    F = eigen(SymTridiagonal(zeros(n), off))
    x = F.values
    w = 2.0 .* F.vectors[1, :] .^ 2
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    return mid .+ half .* x, half .* w
end

# Legendre values P_0..P_nmax at x (unnormalised, P_n(1) = 1).
function _legendre_all(x::Float64, nmax::Int)
    P = Vector{Float64}(undef, nmax + 2)
    P[1] = 1.0
    nmax + 1 >= 2 && (P[2] = x)
    for n in 1:nmax
        P[n + 2] = ((2n + 1) * x * P[n + 1] - n * P[n]) / (n + 1)
    end
    return P   # P[n+1] = P_n(x), valid up to n = nmax + 1
end

# Derivatives P_0'..P_nmax' via P'_{n+1} = P'_{n-1} + (2n+1) P_n.
function _legendre_derivs(P::Vector{Float64}, nmax::Int)
    dP = zeros(Float64, nmax + 1)
    nmax >= 1 && (dP[2] = 1.0)
    for n in 1:(nmax - 1)
        dP[n + 2] = dP[n] + (2n + 1) * P[n + 1]
    end
    return dP
end

# -------------------------
# Chebyshev series on [-1, 1]
# -------------------------

"""
    chebyshev_fit(f, degree) -> Vector{Float64}

Coefficients `c[1:degree+1]` of the interpolant `Σ c[j+1] T_j(s)` of `f` at the
Chebyshev points of the first kind on `[-1, 1]`.
"""
function chebyshev_fit(f, degree::Integer)
    n = degree + 1
    fs = [f(cos(pi * (k + 0.5) / n)) for k in 0:(n - 1)]
    c = Vector{Float64}(undef, n)
    for j in 0:(n - 1)
        acc = 0.0
        for k in 0:(n - 1)
            acc += fs[k + 1] * cos(pi * j * (k + 0.5) / n)
        end
        c[j + 1] = 2.0 * acc / n
    end
    c[1] *= 0.5
    return c
end

"""
    chebyshev_derivative(c) -> Vector{Float64}

Coefficients of `d/ds Σ c[j+1] T_j(s)`; exact differentiation of the series.
"""
function chebyshev_derivative(c::AbstractVector{<:Real})
    n = length(c)
    n <= 1 && return [0.0]
    d = zeros(Float64, n)          # d[j+1] multiplies T_j; top entry stays 0
    for k in (n - 1):-1:1          # k is the polynomial degree index
        d[k] = (k + 1 < n ? d[k + 2] : 0.0) + 2.0 * k * c[k + 1]
    end
    d[1] *= 0.5
    return d[1:(n - 1)]
end

function chebyshev_eval(c::AbstractVector{<:Real}, s::Real)
    b1 = 0.0
    b2 = 0.0
    for j in length(c):-1:2
        b0 = 2.0 * s * b1 - b2 + c[j]
        b2 = b1
        b1 = b0
    end
    return s * b1 - b2 + c[1]
end

"""
    chebyshev_fit_adaptive(f; tol=2e-14, max_degree=96) -> Vector{Float64}

Smallest Chebyshev interpolant whose deviation from `f` on a dense check grid
is below `tol` times the maximum of `|f|`. Trailing negligible coefficients
are trimmed.
"""
function chebyshev_fit_adaptive(f; tol::Real=2e-14, max_degree::Integer=96)
    check = [cos(pi * (k + 0.25) / 1024) for k in 0:1023]
    fcheck = f.(check)
    scale = max(maximum(abs, fcheck), floatmin(Float64))
    best = Float64[]
    for degree in 8:2:max_degree
        c = chebyshev_fit(f, degree)
        err = maximum(abs(chebyshev_eval(c, s) - fs) for (s, fs) in zip(check, fcheck))
        best = c
        err <= tol * scale && break
    end
    while length(best) > 2 && abs(best[end]) <= 0.01 * tol * scale
        pop!(best)
    end
    return best
end

"""
    clenshaw(c::NTuple, s)

Device-safe evaluation of `Σ c[j] T_{j-1}(s)` for a coefficient tuple.
"""
@inline function clenshaw(c::NTuple{N,T}, s::T) where {N,T<:AbstractFloat}
    b1 = zero(T)
    b2 = zero(T)
    s2 = s + s
    @inbounds for j in N:-1:2
        b0 = muladd(s2, b1, c[j] - b2)
        b2 = b1
        b1 = b0
    end
    return muladd(s, b1, c[1] - b2)
end
