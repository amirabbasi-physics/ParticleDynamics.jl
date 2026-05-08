# ───────────────────────────────────────────────────────────────────────────────
# Math helpers
# ───────────────────────────────────────────────────────────────────────────────

# MIC tuned for positions in [-L/2, L/2)
# (works even if values drift slightly outside; floor-wrap in neighbor build makes it robust)
@inline function mic_fast(dx::T, halfL::T, L::T) where {T<:AbstractFloat}
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# Row-wise neighbor kernels visit each physical pair twice, once from each
# endpoint. Store half the dyadic contribution per row so the particle-summed
# virial equals the exact pairwise configurational virial.
@inline function _half_virial2(dx::T, dy::T, fx::T, fy::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dx * fy
end

@inline function _half_virial3(dx::T, dy::T, dz::T, fx::T, fy::T, fz::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dz * fz,
           half * dx * fy, half * dx * fz, half * dy * fz
end

@inline function _launch_config_energy(N::Integer)
    threads = (N < 100_000) ? 128 : 256
    return threads, cld(N, threads)
end

@inline function _launch_config_force_only(N::Integer)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    return threads, cld(N, threads)
end

# Lennard-Jones, returns force components and pair energy
@inline function lj_pair_2d(dx::T, dy::T, r2::T, ϵ::T, σ::T) where {T<:AbstractFloat}
    invr2 = one(T) / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = T(24) * ϵ * (T(2) * s12 - s6) * invr2
    # force on i due to j (no extra minus sign; dx,dy are r_i - r_j)
    fx = f_over_r * dx
    fy = f_over_r * dy
    ep = T(4) * ϵ * (s12 - s6)
    return fx, fy, ep
end
@inline function lj_pair_3d(dx::T, dy::T, dz::T, r2::T, ϵ::T, σ::T) where {T<:AbstractFloat}
    invr2 = one(T) / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
    fx = f_over_r * dx
    fy = f_over_r * dy
    fz = f_over_r * dz
    ep = T(4)*ϵ*(s12 - s6)
    return fx, fy, fz, ep
end
