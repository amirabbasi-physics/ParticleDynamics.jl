# Force geometry must retain the low bits of a coordinate difference before
# subtracting a box length. In Float32, rounding (xi - xj) near L can otherwise
# lose many significant bits of the short periodic separation.
@inline function periodic_delta(xi::T, xj::T, halfL::T, L::T) where {T<:AbstractFloat}
    d = xi - xj
    if abs(d) > halfL
        # Error-free TwoDiff, followed by an exact (Sterbenz) subtraction of L
        # for coordinates in a single image. Do not reassociate with fastmath.
        bvirtual = xi - d
        residual = (xi - (d + bvirtual)) + (bvirtual - xj)
        return (d - copysign(L, d)) + residual
    end
    return d
end

@inline periodic_delta(xi::T, xj::T, L::T) where {T<:AbstractFloat} =
    periodic_delta(xi, xj, T(0.5) * L, L)

# Keep already centered coordinates unchanged. Translating negative values
# into [0,L) can round away bits even before the pair kernel sees them.
@inline function force_coordinate(x::T, L::T) where {T<:AbstractFloat}
    h = T(0.5) * L
    -h <= x <= h && return x
    return x - L * floor(x / L + T(0.5))
end
