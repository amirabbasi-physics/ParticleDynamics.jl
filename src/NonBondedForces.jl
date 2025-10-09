module NonBondedForces

using CUDA
using ..Definitions
using ..NeighborLists  # so we can dispatch on NeighborLists.NeighborMatrix
using ..BondedForces   # for BondList in exclusions

export lj_forces_soa!, lj_forces_soa_noE!,
       wca_forces_soa!, wca_forces_soa_noE!,
       harmonic_rep_forces_soa!, harmonic_rep_forces_soa_noE!

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

# ───────────────────────────────────────────────────────────────────────────────
# No-energy (no Epot) CSR variants
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end

    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.NeighborMatrix{T},
                            box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.NeighborMatrix{T},
                            box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# Overloads for stencil neighbor lists (reuse the same CSR kernels)
function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.StencilNeighborMatrix{T},
                            box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.StencilNeighborMatrix{T},
                            box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
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

# ───────────────────────────────────────────────────────────────────────────────
# New (fast) CSR kernels — for NeighborLists.NeighborMatrix (your “newer” NL)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep   # half to avoid double counting
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Mixed-σ LJ (per-particle size; Lorentz mixing, global ϵ)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_kernel_mixed!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T,
    σp::CuDeviceVector{T}, rcut_factor::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        σij = T(0.5) * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > zero(T)) & (r2 < rcut_ij*rcut_ij)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * (T(4)*ϵ*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_mixed!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T,
    σp::CuDeviceVector{T}, rcut_factor::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        σij = T(0.5) * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > zero(T)) & (r2 < rcut_ij*rcut_ij)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * (T(4)*ϵ*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_mixed!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T,
    σp::CuDeviceVector{T}, rcut_factor::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        σij = T(0.5) * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > zero(T)) & (r2 < rcut_ij*rcut_ij)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_mixed!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T,
    σp::CuDeviceVector{T}, rcut_factor::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        σij = T(0.5) * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > zero(T)) & (r2 < rcut_ij*rcut_ij)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_mixed!(rx::CuArray{T,1}, ry::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box2{T},
                              ϵ::T,
                              σp::CuArray{T,1}, rcut_factor::T) where {T<:AbstractFloat} 
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_kernel_mixed!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        ϵ, σp, rcut_factor)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      ϵ, σp, rcut_factor; threads, blocks)
    return nothing
end

function lj_forces_soa_mixed!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box3{T},
                              ϵ::T,
                              σp::CuArray{T,1}, rcut_factor::T) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_kernel_mixed!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        ϵ, σp, rcut_factor)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      ϵ, σp, rcut_factor; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_mixed!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box2{T},
                                  ϵ::T,
                                  σp::CuArray{T,1}, rcut_factor::T) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel_mixed!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        ϵ, σp, rcut_factor)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      ϵ, σp, rcut_factor; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_mixed!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box3{T},
                                  ϵ::T,
                                  σp::CuArray{T,1}, rcut_factor::T) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel_mixed!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        ϵ, σp, rcut_factor)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      ϵ, σp, rcut_factor; threads, blocks)
    return nothing
end

function _lj3_csr_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# CSR kernels with bonded exclusions
# ───────────────────────────────────────────────────────────────────────────────

@inline function _is_bonded(i::Int32, j::Int32,
                            bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32})
    base = bindex[i]
    nb = bcounts[i]
    @inbounds for t in 0:Int(nb-1)
        if bflat[base + t + 1] == j
            return true
        end
    end
    return false
end

function _lj2_csr_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        # skip bonded pairs
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, eps, sig)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, eps, sig)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (sig*sig) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*eps*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    eps::T, sig::T, cutoff2::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (sig*sig) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*eps*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Legacy “matrix neighbors” kernels — for older neighbor list (nbh.neighbors)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_mat_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)

    @inbounds for k in 1:cap
        j = nbr[i,k]; if j <= 0; break; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij; eacc += T(0.5)*ep
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_mat_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)

    @inbounds for k in 1:cap
        j = nbr[i,k]; if j <= 0; break; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5)*ep
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Public API (dispatch to the correct path automatically)
# ───────────────────────────────────────────────────────────────────────────────

# ---- 2D, CSR (NeighborLists.NeighborMatrix)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_csr_kernel!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# ---- 3D, CSR (NeighborLists.NeighborMatrix)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_csr_kernel!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# Overloads for stencil neighbor lists (CSR paths)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_csr_kernel!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_csr_kernel!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# ---- Variants with bonded exclusions ----
function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1},
                             Epot::CuArray{T,1},
                             nbh::NeighborLists.NeighborMatrix{T},
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_csr_kernel_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                             Epot::CuArray{T,1},
                             nbh::NeighborLists.NeighborMatrix{T},
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_csr_kernel_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# stencil variants
function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1},
                             Epot::CuArray{T,1},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_kernel_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                             Epot::CuArray{T,1},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_kernel_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# no-energy variants with exclusions
function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 nbh::NeighborLists.NeighborMatrix{T},
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 nbh::NeighborLists.NeighborMatrix{T},
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 nbh::NeighborLists.StencilNeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 nbh::NeighborLists.StencilNeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end


# ---- 2D, legacy matrix neighbor list (fallback)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh,  # duck-typed; must have .neighbors::CuArray{Int32,2} and .cap
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    @assert hasproperty(nbh, :neighbors) "nbh lacks 'neighbors' field"
    @assert hasproperty(nbh, :cap)       "nbh lacks 'cap' field"

    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    cap = Int32(nbh.cap)

    k = CUDA.@cuda launch=false _lj2_mat_kernel!(
        rx, ry, fx, fy, Epot, nbh.neighbors, cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot, nbh.neighbors, cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# ---- 3D, legacy matrix neighbor list (fallback)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh,
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    @assert hasproperty(nbh, :neighbors) "nbh lacks 'neighbors' field"
    @assert hasproperty(nbh, :cap)       "nbh lacks 'cap' field"

    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    cap = Int32(nbh.cap)

    k = CUDA.@cuda launch=false _lj3_mat_kernel!(
        rx, ry, rz, fx, fy, fz, Epot, nbh.neighbors, cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot, nbh.neighbors, cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end



# Pairwise-per-type LJ (sigma_ij and epsilon_ij per pair, with rcut_ij)
function _lj2_csr_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc*rc)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*εij*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * (T(4)*εij*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc*rc)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*εij*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * (T(4)*εij*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc*rc)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*εij*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc*rc)
            invr2 = one(T) / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*εij*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box2{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_kernel_pairs!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function lj_forces_oa_pairs_bugfix() end  

function lj_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box3{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              )  where {T<:AbstractFloat}

    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_kernel_pairs!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box2{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                  ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_csr_noE_kernel_pairs!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box3{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2} 
                                  ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_csr_noE_kernel_pairs!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end


# =============================
# WCA (truncated-shifted LJ)
# =============================

function _wca2_csr_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = T(4)*ϵ*(s12 - s6) + ϵ
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# Exclusion variants (skip bonded pairs)
function _wca2_csr_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = T(4)*ϵ*(s12 - s6) + ϵ
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _wca3_csr_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = T(4)*ϵ*(s12 - s6) + ϵ
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca2_csr_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_csr_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.NeighborMatrix{T},
                              bonds::BondedForces.BondList,
                              box::Definitions.Box2{T}, params::Definitions.LJParams{T} ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_csr_kernel_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.NeighborMatrix{T},
                              bonds::BondedForces.BondList,
                              box::Definitions.Box3{T}, params::Definitions.LJParams{T} ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_csr_kernel_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box2{T}, params::Definitions.LJParams{T} ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_csr_noE_kernel_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_csr_noE_kernel_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ; threads, blocks)
    return nothing
end

function _wca3_csr_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = T(4)*ϵ*(s12 - s6) + ϵ
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca2_csr_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_csr_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = T(1.2599211) * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < rc2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# Wrappers
function wca_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T} ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_csr_kernel!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}
                        ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_csr_kernel!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1},
                             nbh::NeighborLists.NeighborMatrix{T},
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}
                             ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_csr_noE_kernel!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                             nbh::NeighborLists.NeighborMatrix{T},
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}
                             ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_csr_noE_kernel!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

# =============================
# Soft repulsive harmonic (nonbonded)
# =============================

function _harmrep2_csr_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = T(0.5) * ϵ * (one(T) - r/σ)*(one(T) - r/σ)
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# Exclusion variants for soft-repulsive harmonic
function _harmrep2_csr_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * (T(0.5) * ϵ*(one(T) - r/σ)*(one(T) - r/σ))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _harmrep3_csr_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * (T(0.5) * ϵ*(one(T) - r/σ)*(one(T) - r/σ))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _harmrep2_csr_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _harmrep3_csr_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function harmonic_rep_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                       fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                                       nbh::NeighborLists.NeighborMatrix{T},
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                       ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_kernel_excl!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                       fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                                       nbh::NeighborLists.NeighborMatrix{T},
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                       ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_kernel_excl!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                           fx::CuArray{T,1}, fy::CuArray{T,1},
                                           nbh::NeighborLists.NeighborMatrix{T},
                                           bonds::BondedForces.BondList,
                                           box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                           ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_noE_kernel_excl!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                           fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                           nbh::NeighborLists.NeighborMatrix{T},
                                           bonds::BondedForces.BondList,
                                           box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                           ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_noE_kernel_excl!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function _harmrep3_csr_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = T(0.5) * ϵ * (one(T) - r/σ)*(one(T) - r/σ)
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _harmrep2_csr_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _harmrep3_csr_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# Wrappers for soft repulsive harmonic
function harmonic_rep_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                  ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_kernel!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                  ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_kernel!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                      fx::CuArray{T,1}, fy::CuArray{T,1},
                                      nbh::NeighborLists.NeighborMatrix{T},
                                      box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                      ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_noE_kernel!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                      fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                      nbh::NeighborLists.NeighborMatrix{T},
                                      box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T} ) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_noE_kernel!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end
end # module
