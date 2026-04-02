"""
Bonded interaction kernels (harmonic and FENE) used by the polymer examples.
"""
module BondedForces

using CUDA
using ..Definitions

export BondList, build_bondlist,
       harmonic_forces_soa!, harmonic_forces_soa_noE!,
       fene_forces_soa!, fene_forces_soa_noE!

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@inline function mic_fast(dx::T, halfL::T, L::T) where {T<:AbstractFloat}
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# Bond rows are symmetric in the CSR list, so each physical bond is visited
# twice. Store half the dyadic contribution per row so the particle-summed
# virial matches the single-count configurational virial.
@inline function _half_virial2(dx::T, dy::T, fx::T, fy::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dx * fy
end

@inline function _half_virial3(dx::T, dy::T, dz::T, fx::T, fy::T, fz::T) where {T<:AbstractFloat}
    half = T(0.5)
    return half * dx * fx, half * dy * fy, half * dz * fz,
           half * dx * fy, half * dx * fz, half * dy * fz
end

@inline _bond_threads(N::Int) = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)

"""
CSR-style adjacency list describing bead connectivity.
"""
struct BondList
    index::CuArray{Int32,1}
    flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}
end

"""
    build_bondlist(N, bonds) -> BondList

Construct a GPU-ready bond list from a collection of `(i, j)` tuples (1-based).
`examples/2D_polymer_bonded.jl` builds its chains via:

```julia
chain = collect(zip(1:(n-1), 2:n))
bond_list = build_bondlist(n, chain)
```
"""
function build_bondlist(N::Integer, bonds)
    N = Int(N)
    deg = zeros(Int32, N)
    for (i,j) in bonds
        @assert 1 <= i <= N && 1 <= j <= N "bond index out of range"
        deg[Int(i)] += 1
        deg[Int(j)] += 1
    end
    index = similar(deg)
    offs = Int32(0)
    for i in 1:N
        index[i] = offs
        offs += deg[i]
    end
    total = Int(offs)
    flat = Vector{Int32}(undef, total)
    counts = zeros(Int32, N)
    for (i,j) in bonds
        ii = Int(i); jj = Int(j)
        bi = index[ii] + counts[ii]
        flat[Int(bi)+1] = Int32(jj)
        counts[ii] += 1
        bj = index[jj] + counts[jj]
        flat[Int(bj)+1] = Int32(ii)
        counts[jj] += 1
    end
    return BondList(CUDA.CuArray(index), CUDA.CuArray(flat), CUDA.CuArray(counts))
end

# ------------------------------------------------------------------
# Harmonic kernels
# ------------------------------------------------------------------

function _harmonic2_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            r = sqrt(r2)
            diff = r - r0
            f_over_r = -k * diff / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * k * diff * diff
        end
    end
    fx[i] += accx; fy[i] += accy; E[i] += eacc
    return
end

function _harmonic2_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            r = sqrt(r2)
            diff = r - r0
            f_over_r = -k * diff / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            eacc += T(0.5) * k * diff * diff
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx
            vyy += dvyy
            vxy += dvxy
        end
    end
    fx[i] += accx; fy[i] += accy; E[i] += eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _harmonic3_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            r = sqrt(r2)
            diff = r - r0
            f_over_r = -k * diff / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * k * diff * diff
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz; E[i] += eacc
    return
end

function _harmonic3_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            r = sqrt(r2)
            diff = r - r0
            f_over_r = -k * diff / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            eacc += T(0.5) * k * diff * diff
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx
            vyy += dvyy
            vzz += dvzz
            vxy += dvxy
            vxz += dvxz
            vyz += dvyz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz; E[i] += eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _harmonic2_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] += accx; fy[i] += accy
    return
end

function _harmonic3_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, r0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base = index[i]; nb = counts[i]
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    return
end

# ------------------------------------------------------------------
# FENE kernels
# ------------------------------------------------------------------

function _fene2_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * (-T(0.5) * k * (R0*R0) * log(denom))
        end
    end
    fx[i] += accx; fy[i] += accy; E[i] += eacc
    return
end

function _fene2_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            eacc += T(0.5) * (-T(0.5) * k * (R0*R0) * log(denom))
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx
            vyy += dvyy
            vxy += dvxy
        end
    end
    fx[i] += accx; fy[i] += accy; E[i] += eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _fene3_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * (-T(0.5) * k * (R0*R0) * log(denom))
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz; E[i] += eacc
    return
end

function _fene3_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            eacc += T(0.5) * (-T(0.5) * k * (R0*R0) * log(denom))
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx
            vyy += dvyy
            vzz += dvzz
            vxy += dvxy
            vxz += dvxz
            vyz += dvyz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz; E[i] += eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _fene2_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] += accx; fy[i] += accy
    return
end

function _fene3_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base = index[i]; nb = counts[i]
    invR02 = one(T) / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > zero(T)
            denom = one(T) - r2*invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    return
end

# ------------------------------------------------------------------
# Public wrappers
# ------------------------------------------------------------------

"""
    harmonic_forces_soa!(rx, ry[, rz], fx, fy[, fz], E, bonds, box, params)

Evaluate harmonic bond forces and per-particle energies. The bead–spring chains
in `examples/2D_polymer_bonded.jl` use this helper after calling
`build_bondlist`.
"""
function harmonic_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmonic2_E!(rx, ry, fx, fy, E,
                                              bonds.index, bonds.flat, bonds.counts,
                                              Lx, Ly, halfLx, halfLy,
                                              params.k, params.r0)
    k(rx, ry, fx, fy, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.r0; threads, blocks)
    return nothing
end

function harmonic_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmonic2_EV!(rx, ry, fx, fy, E, V,
                                               bonds.index, bonds.flat, bonds.counts,
                                               Lx, Ly, halfLx, halfLy,
                                               params.k, params.r0)
    k(rx, ry, fx, fy, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.r0; threads, blocks)
    return nothing
end

"""
    harmonic_forces_soa_noE!(rx, ry[, rz], fx, fy[, fz], bonds, box, params)

Force-only harmonic bonds. Handy for warmup segments when energies are not
recorded.
"""
function harmonic_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmonic2_noE!(rx, ry, fx, fy,
                                                bonds.index, bonds.flat, bonds.counts,
                                                Lx, Ly, halfLx, halfLy,
                                                params.k, params.r0)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.r0; threads, blocks)
    return nothing
end

function harmonic_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmonic3_E!(rx, ry, rz, fx, fy, fz, E,
                                              bonds.index, bonds.flat, bonds.counts,
                                              Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                              params.k, params.r0)
    k(rx, ry, rz, fx, fy, fz, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.r0; threads, blocks)
    return nothing
end

function harmonic_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmonic3_EV!(rx, ry, rz, fx, fy, fz, E, V,
                                               bonds.index, bonds.flat, bonds.counts,
                                               Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                               params.k, params.r0)
    k(rx, ry, rz, fx, fy, fz, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.r0; threads, blocks)
    return nothing
end

function harmonic_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.HarmonicBondParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmonic3_noE!(rx, ry, rz, fx, fy, fz,
                                                bonds.index, bonds.flat, bonds.counts,
                                                Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                                params.k, params.r0)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.r0; threads, blocks)
    return nothing
end

"""
    fene_forces_soa!(rx, ry[, rz], fx, fy[, fz], E, bonds, box, params)

Finite extensible nonlinear elastic bonds. Matches the `fene_bond(k=300, r0=1.5)`
configuration used in `examples/2D_polymer_bonded_BP.jl`.
"""
function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _fene2_E!(rx, ry, fx, fy, E,
                                          bonds.index, bonds.flat, bonds.counts,
                                          Lx, Ly, halfLx, halfLy,
                                          params.k, params.R0)
    k(rx, ry, fx, fy, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _fene2_EV!(rx, ry, fx, fy, E, V,
                                           bonds.index, bonds.flat, bonds.counts,
                                           Lx, Ly, halfLx, halfLy,
                                           params.k, params.R0)
    k(rx, ry, fx, fy, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

"""
    fene_forces_soa_noE!(rx, ry[, rz], fx, fy[, fz], bonds, box, params)

FENE bonds without per-particle energy accumulation.
"""
function fene_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly = box; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _fene2_noE!(rx, ry, fx, fy,
                                            bonds.index, bonds.flat, bonds.counts,
                                            Lx, Ly, halfLx, halfLy,
                                            params.k, params.R0)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _fene3_E!(rx, ry, rz, fx, fy, fz, E,
                                          bonds.index, bonds.flat, bonds.counts,
                                          Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                          params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _fene3_EV!(rx, ry, rz, fx, fy, fz, E, V,
                                           bonds.index, bonds.flat, bonds.counts,
                                           Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                           params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(_bond_threads(N), N); blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _fene3_noE!(rx, ry, rz, fx, fy, fz,
                                            bonds.index, bonds.flat, bonds.counts,
                                            Lx, Ly, Lz, halfLx, halfLy, halfLz,
                                            params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end

end # module BondedForces
