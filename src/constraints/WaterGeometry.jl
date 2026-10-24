# Allocation-free local algebra shared by the CPU oracle tests and GPU
# kernels. Atom order is (O, H1, H2); edge order is (O-H1, O-H2, H1-H2).
# SHAKE uses the OLD gradients, not a nearest rigid-body fit, which would
# change the constrained velocity-Verlet discretization.

@inline _edges(x::NTuple{3,SVector{3,T}}) where {T} =
    (x[1] - x[2], x[1] - x[3], x[2] - x[3])

@inline _dot3(a::SVector{3,T}, b::SVector{3,T}) where {T} =
    a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

# StaticArrays.cross includes a dynamic size-error path that GPUCompiler
# cannot lower in a device kernel. These operands are always SVector{3}.
@inline function _cross3(a::SVector{3,T}, b::SVector{3,T}) where {T}
    return SVector{3,T}(a[2]*b[3] - a[3]*b[2],
                        a[3]*b[1] - a[1]*b[3],
                        a[1]*b[2] - a[2]*b[1])
end

@inline function _solve3(a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T},
                         rhs::SVector{3,T}) where {T}
    bc = _cross3(b, c)
    det = _dot3(a, bc)
    scale = sqrt(_dot3(a, a) * _dot3(b, b) * _dot3(c, c))
    ok = isfinite(det) && isfinite(scale) && abs(det) > 16eps(T) * scale
    ok || return SVector{3,T}(zero(T), zero(T), zero(T)), false
    v = (rhs[1] * bc + rhs[2] * _cross3(c, a) + rhs[3] * _cross3(a, b)) / det
    return v, isfinite(v[1]) && isfinite(v[2]) && isfinite(v[3])
end

@inline function _mass_rows(d::NTuple{3,SVector{3,T}}, b::NTuple{3,SVector{3,T}},
                            w::NTuple{3,T}) where {T}
    a = SVector{3,T}((w[1] + w[2]) * _dot3(d[1], b[1]), w[1] * _dot3(d[1], b[2]), -w[2] * _dot3(d[1], b[3]))
    c = SVector{3,T}(w[1] * _dot3(d[2], b[1]), (w[1] + w[3]) * _dot3(d[2], b[2]), w[3] * _dot3(d[2], b[3]))
    e = SVector{3,T}(-w[2] * _dot3(d[3], b[1]), w[3] * _dot3(d[3], b[2]), (w[2] + w[3]) * _dot3(d[3], b[3]))
    return a, c, e
end

@inline function _increments(b::NTuple{3,SVector{3,T}}, w::NTuple{3,T},
                             λ::SVector{3,T}) where {T}
    return (w[1] * (λ[1] * b[1] + λ[2] * b[2]),
            w[2] * (-λ[1] * b[1] + λ[3] * b[3]),
            w[3] * (-λ[2] * b[2] - λ[3] * b[3]))
end

@inline function distance_error(x::NTuple{3,SVector{3,T}}, lengths2::NTuple{3,T}) where {T}
    d = _edges(x)
    e1 = abs(_dot3(d[1], d[1]) / lengths2[1] - one(T))
    e2 = abs(_dot3(d[2], d[2]) / lengths2[2] - one(T))
    e3 = abs(_dot3(d[3], d[3]) / lengths2[3] - one(T))
    return max(max(e1, e2), e3)
end

"""
    shake_water(old, trial, inverse_masses, squared_lengths, tolerance, maxiter)

Return `(positions, converged, iterations)` for one whole, locally unwrapped
molecule. Tolerance is the maximum relative **squared-distance** residual.
No coordinates are published by the GPU caller unless the solve converges.
"""
@inline function shake_water(old::NTuple{3,SVector{3,T}}, trial::NTuple{3,SVector{3,T}},
                             w::NTuple{3,T}, lengths2::NTuple{3,T}, tolerance::T,
                             maxiter::Integer) where {T}
    b = _edges(old)
    x = trial
    for it in 0:maxiter
        err = distance_error(x, lengths2)
        isfinite(err) || return trial, false, it
        err <= tolerance && return x, true, it
        it == maxiter && break
        d = _edges(x)
        rhs = SVector{3,T}((lengths2[1] - _dot3(d[1], d[1])) / T(2),
                           (lengths2[2] - _dot3(d[2], d[2])) / T(2),
                           (lengths2[3] - _dot3(d[3], d[3])) / T(2))
        a, c, e = _mass_rows(d, b, w)
        λ, ok = _solve3(a, c, e, rhs)
        ok || return trial, false, it
        dx = _increments(b, w, λ)
        x = (x[1] + dx[1], x[2] + dx[2], x[3] + dx[3])
    end
    return trial, false, maxiter
end

"""
    rattle_water(positions, velocities, inverse_masses)

Mass-metric velocity projection onto `r_ij ⋅ v_ij = 0`. Returns
`(velocities, success)`. Removes only the three constrained velocity modes.
"""
@inline function rattle_water(x::NTuple{3,SVector{3,T}}, v::NTuple{3,SVector{3,T}},
                              w::NTuple{3,T}) where {T}
    d, dv = _edges(x), _edges(v)
    rhs = SVector{3,T}(-_dot3(d[1], dv[1]), -_dot3(d[2], dv[2]), -_dot3(d[3], dv[3]))
    a, c, e = _mass_rows(d, d, w)
    λ, ok = _solve3(a, c, e, rhs)
    ok || return v, false
    correction = _increments(d, w, λ)
    return (v[1] + correction[1], v[2] + correction[2], v[3] + correction[3]), true
end

@inline _image(v::SVector{3,T}, box::NTuple{3,T}) where {T} =
    SVector{3,T}(v[1] - round(v[1] / box[1]) * box[1],
                 v[2] - round(v[2] / box[2]) * box[2],
                 v[3] - round(v[3] / box[3]) * box[3])

# Relative coordinates near the oxygen avoid subtracting two large absolute
# coordinates during iteration and handle molecules split across the PBC.
@inline function local_water(x::NTuple{3,SVector{3,T}}, origin::SVector{3,T},
                             box::NTuple{3,T}) where {T}
    return (_image(x[1] - origin, box), _image(x[2] - origin, box),
            _image(x[3] - origin, box))
end

"""
    constraint_force_water(x, v, force, inverse_masses)

Instantaneous continuous-time constraint forces. Differentiating each
`dot(rij, vij) = 0` gives `dot(rij, aij) + dot(vij, vij) = 0`.
Solve the mass-metric system for the three multipliers. These forces are
for the pressure estimator only: SHAKE/RATTLE already applies the discrete
constraint impulses in the integrator, so adding them to its kicks would
double count constraints.
"""
@inline function constraint_force_water(x::NTuple{3,SVector{3,T}},
                                        v::NTuple{3,SVector{3,T}},
                                        force::NTuple{3,SVector{3,T}},
                                        w::NTuple{3,T}) where {T}
    d, dv = _edges(x), _edges(v)
    da = _edges((w[1]*force[1], w[2]*force[2], w[3]*force[3]))
    rhs = SVector{3,T}(-_dot3(d[1], da[1]) - _dot3(dv[1], dv[1]),
                       -_dot3(d[2], da[2]) - _dot3(dv[2], dv[2]),
                       -_dot3(d[3], da[3]) - _dot3(dv[3], dv[3]))
    a, c, e = _mass_rows(d, d, w)
    λ, ok = _solve3(a, c, e, rhs)
    f = _increments(d, (one(T), one(T), one(T)), λ)
    return f, ok
end

@inline function constraint_tensor_water(x::NTuple{3,SVector{3,T}},
                                         f::NTuple{3,SVector{3,T}}) where {T}
    # Zero net constraint force makes the tensor origin independent.
    r2, r3 = x[2] - x[1], x[3] - x[1]
    return SVector{6,T}(r2[1]*f[2][1] + r3[1]*f[3][1],
                        r2[2]*f[2][2] + r3[2]*f[3][2],
                        r2[3]*f[2][3] + r3[3]*f[3][3],
                        r2[1]*f[2][2] + r3[1]*f[3][2],
                        r2[1]*f[2][3] + r3[1]*f[3][3],
                        r2[2]*f[2][3] + r3[2]*f[3][3])
end
