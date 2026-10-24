# ------------------------------------------------------------------
# Bonded potential kinds and allocation-free local geometry
# ------------------------------------------------------------------
#
# Every function here takes locally unwrapped coordinates (SVector) and is
# callable on the host and inside CUDA kernels. Each returns the forces on
# the tuple's atoms, the interaction energy, and a validity flag; invalid
# geometry is reported, never clamped.

"""
Concrete bonded potential kinds. Coefficient tables are plain device vectors
indexed by a per-interaction parameter id.
"""
abstract type BondedPotential end

struct HarmonicBondKind <: BondedPotential end        # (k, r0)
struct FENEBondKind <: BondedPotential end            # (k, R0)
struct HarmonicAngleKind <: BondedPotential end       # (k, θ0)
struct CosineSquaredAngleKind <: BondedPotential end  # (k, cosθ0)
struct UreyBradleyKind <: BondedPotential end         # (kθ, θ0, kr, r13_0)
struct FourierTorsionKind <: BondedPotential end      # (offset, count) into (k_m, n_m, δ_m)
struct HarmonicImproperKind <: BondedPotential end    # (k, χ0)
struct RyckaertBellemansKind <: BondedPotential end   # (C0..C5), ψ = φ - π

arity(::HarmonicBondKind) = 2
arity(::FENEBondKind) = 2
arity(::HarmonicAngleKind) = 3
arity(::CosineSquaredAngleKind) = 3
arity(::UreyBradleyKind) = 3
arity(::FourierTorsionKind) = 4
arity(::HarmonicImproperKind) = 4
arity(::RyckaertBellemansKind) = 4

# Relative degeneracy threshold for sines of bond/plane angles.
@inline _degenerate_tol(::Type{T}) where {T<:AbstractFloat} = T(16) * eps(T)

@inline _cross_norm(u::SVector{2,T}, v::SVector{2,T}) where {T} = abs(u[1] * v[2] - u[2] * v[1])
@inline _cross_norm(u::SVector{3,T}, v::SVector{3,T}) where {T} = norm(cross(u, v))

# ---------------------------- pairs ----------------------------------

"""
    pair_forces(kind, c, d) -> (F1, F2, U, ok)

`d = r2 - r1` in the local image; `c` holds this interaction's coefficients.
"""
@inline function pair_forces(::HarmonicBondKind, c::NTuple{2,T}, d::SVector{D,T}) where {D,T}
    k, r0 = c
    r = norm(d)
    if !(r > zero(T))
        # Zero length: the direction is undefined unless r0 == 0.
        ok = r0 == zero(T)
        z = zero(d)
        return z, z, (ok ? zero(T) : T(NaN)), ok
    end
    diff = r - r0
    F2 = (-k * diff / r) * d
    return -F2, F2, T(0.5) * k * diff * diff, isfinite(r)
end

@inline function pair_forces(::FENEBondKind, c::NTuple{2,T}, d::SVector{D,T}) where {D,T}
    k, R0 = c
    R02 = R0 * R0
    r2 = dot(d, d)
    denom = (R02 - r2) / R02
    if !(r2 < R02) || !(denom > zero(T))
        z = zero(d)
        return z, z, T(NaN), false
    end
    lg = r2 < R02 / 2 ? log1p(-r2 / R02) : log(denom)
    F2 = (-k / denom) * d
    return -F2, F2, -T(0.5) * k * R02 * lg, true
end

# ---------------------------- angles ---------------------------------

# Shared cosine geometry: a = ri - rj, b = rk - rj.
@inline function _angle_frame(a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    A = norm(a); B = norm(b)
    ok = A > zero(T) && B > zero(T) && isfinite(A) && isfinite(B)
    Ainv = ok ? inv(A) : zero(T)
    Binv = ok ? inv(B) : zero(T)
    u = a * Ainv; v = b * Binv
    c = clamp(dot(u, v), -one(T), one(T))
    s = _cross_norm(u, v)
    return u, v, Ainv, Binv, c, s, ok
end

# Forces from dU/dθ through (dU/dθ)/sinθ = g.
@inline function _angle_forces_from_g(g::T, u, v, Ainv::T, Binv::T, c::T) where {T}
    Fi = (g * Ainv) * (v - c * u)
    Fk = (g * Binv) * (u - c * v)
    return Fi, -Fi - Fk, Fk
end

# Forces from dU/dc (no division by sinθ).
@inline function _angle_forces_from_dUdc(dUdc::T, u, v, Ainv::T, Binv::T, c::T) where {T}
    Fi = (-dUdc * Ainv) * (v - c * u)
    Fk = (-dUdc * Binv) * (u - c * v)
    return Fi, -Fi - Fk, Fk
end

# x / sin(x) for small |x|, used for the linear-equilibrium limit.
@inline _x_over_sin(x::T) where {T} = abs(x) < T(1e-4) ? one(T) + x * x / T(6) : x / sin(x)

# (dU/dθ)/sinθ for k(θ-θ0)²/2. θ0 exactly π (linear chains) or 0 has a finite
# limit at the collinear geometry; any other θ0 is singular there.
@inline function _harmonic_angle_g(k::T, θ0::T, θ::T, s::T) where {T}
    if θ0 == T(π)
        return -k * _x_over_sin(T(π) - θ), true
    elseif θ0 == zero(T)
        return k * _x_over_sin(θ), true
    end
    ok = s > _degenerate_tol(T)
    return (ok ? k * (θ - θ0) / s : T(NaN)), ok
end

"""
    angle_forces(kind, c, a, b) -> (Fi, Fj, Fk, U, ok)

`a = ri - rj`, `b = rk - rj` (vertex j) in the local image.
"""
@inline function angle_forces(::HarmonicAngleKind, c::NTuple{2,T}, a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    k, θ0 = c
    u, v, Ainv, Binv, cθ, s, ok = _angle_frame(a, b)
    θ = atan(s, cθ)
    g, okg = _harmonic_angle_g(k, θ0, θ, s)
    Fi, Fj, Fk = _angle_forces_from_g(g, u, v, Ainv, Binv, cθ)
    dθ = θ - θ0
    return Fi, Fj, Fk, T(0.5) * k * dθ * dθ, ok & okg
end

@inline function angle_forces(::CosineSquaredAngleKind, c::NTuple{2,T}, a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    k, c0 = c
    u, v, Ainv, Binv, cθ, s, ok = _angle_frame(a, b)
    dc = cθ - c0
    Fi, Fj, Fk = _angle_forces_from_dUdc(k * dc, u, v, Ainv, Binv, cθ)
    return Fi, Fj, Fk, T(0.5) * k * dc * dc, ok
end

@inline function angle_forces(::UreyBradleyKind, c::NTuple{4,T}, a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    kθ, θ0, kr, r130 = c
    Fi, Fj, Fk, U, ok = angle_forces(HarmonicAngleKind(), (kθ, θ0), a, b)
    # 1–3 distance term on i and k: d = rk - ri = b - a.
    Fa, Fb, Ur, okr = pair_forces(HarmonicBondKind(), (kr, r130), b - a)
    return Fi + Fa, Fj, Fk + Fb, U + Ur, ok & okr
end

# ---------------------------- torsions --------------------------------

"""
    torsion_angle_gradient(b1, b2, b3) -> (φ, G1, G2, G3, G4, ok)

Signed dihedral `φ = atan2(h·(n1×n2), n1·n2)` with `n1 = b1×b2`,
`n2 = b2×b3`, `h = b2/|b2|`, and its gradients `∂φ/∂r_a`.
"""
@inline function torsion_angle_gradient(b1::SVector{3,T}, b2::SVector{3,T}, b3::SVector{3,T}) where {T}
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    b2sq = dot(b2, b2)
    n1sq = dot(n1, n1)
    n2sq = dot(n2, n2)
    tol = _degenerate_tol(T)^2
    ok = b2sq > zero(T) &&
         n1sq > tol * dot(b1, b1) * b2sq &&
         n2sq > tol * dot(b3, b3) * b2sq &&
         isfinite(n1sq) && isfinite(n2sq)
    b2n = sqrt(b2sq)
    h = ok ? b2 / b2n : zero(b2)
    x = dot(n1, n2)
    y = dot(h, cross(n1, n2))
    φ = atan(y, x)
    Dinv = ok ? inv(x * x + y * y) : zero(T)
    g1 = (x * cross(n2, h) - y * n2) * Dinv
    g2 = (x * cross(h, n1) - y * n1) * Dinv
    gh = (x * Dinv) * cross(n1, n2)
    gb1 = cross(b2, g1)
    gb2 = cross(g1, b1) + cross(b3, g2) + (ok ? (gh - h * dot(h, gh)) / b2n : zero(b2))
    gb3 = cross(g2, b2)
    return φ, -gb1, gb1 - gb2, gb2 - gb3, gb3, ok
end

# dU/dφ and U for each torsion-style kind.
@inline function _torsion_energy(::HarmonicImproperKind, c::NTuple{2,T}, φ::T) where {T}
    k, χ0 = c
    d = atan(sin(φ - χ0), cos(φ - χ0))
    return T(0.5) * k * d * d, k * d
end

@inline function _torsion_energy(::RyckaertBellemansKind, c::NTuple{6,T}, φ::T) where {T}
    # U = Σ Cn cos(ψ)^n with ψ = φ - π, so cosψ = -cosφ.
    cψ = -cos(φ)
    U = c[6]; dUdc = zero(T)
    for n in 5:-1:1
        dUdc = muladd(dUdc, cψ, U)
        U = muladd(U, cψ, c[n])
    end
    # dcosψ/dφ = sinφ
    return U, dUdc * sin(φ)
end

"""
    torsion_forces(kind, c, b1, b2, b3) -> (F1, F2, F3, F4, U, ok)
"""
@inline function torsion_forces(kind::BondedPotential, c, b1::SVector{3,T}, b2::SVector{3,T}, b3::SVector{3,T}) where {T}
    φ, G1, G2, G3, G4, ok = torsion_angle_gradient(b1, b2, b3)
    U, dUdφ = _torsion_energy(kind, c, φ)
    s = -dUdφ
    return s * G1, s * G2, s * G3, s * G4, U, ok
end

# Fourier sum over this interaction's terms; `kf, nf, δf` are indexable.
@inline function fourier_torsion_forces(kf, nf, δf, first::Int, count::Int,
                                        b1::SVector{3,T}, b2::SVector{3,T}, b3::SVector{3,T}) where {T}
    φ, G1, G2, G3, G4, ok = torsion_angle_gradient(b1, b2, b3)
    U = zero(T); dUdφ = zero(T)
    for m in first:(first + count - 1)
        km = kf[m]; nm = nf[m]; δm = δf[m]
        arg = nm * φ - δm
        U += km * (one(T) + cos(arg))
        dUdφ -= km * nm * sin(arg)
    end
    s = -dUdφ
    return s * G1, s * G2, s * G3, s * G4, U, ok
end
