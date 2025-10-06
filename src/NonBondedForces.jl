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
@inline function mic_fast(dx::Float32, halfL::Float32, L::Float32)
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

# Lennard-Jones, returns force components and pair energy
@inline function lj_pair_2d(dx::Float32, dy::Float32, r2::Float32, ϵ::Float32, σ::Float32)
    invr2 = 1f0 / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
    # force on i due to j (no extra minus sign; dx,dy are r_i - r_j)
    fx = f_over_r * dx
    fy = f_over_r * dy
    ep = 4f0*ϵ*(s12 - s6)
    return fx, fy, ep
end

# ───────────────────────────────────────────────────────────────────────────────
# No-energy (no Epot) CSR variants
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_noE_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < cutoff2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end

    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < cutoff2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                            nbh::NeighborLists.NeighborMatrix,
                            box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly

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

function lj_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                            nbh::NeighborLists.NeighborMatrix,
                            box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz

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
function lj_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                            nbh::NeighborLists.StencilNeighborMatrix,
                            box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly

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

function lj_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                            nbh::NeighborLists.StencilNeighborMatrix,
                            box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz

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


 

@inline function lj_pair_3d(dx::Float32, dy::Float32, dz::Float32, r2::Float32, ϵ::Float32, σ::Float32)
    invr2 = 1f0 / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
    fx = f_over_r * dx
    fy = f_over_r * dy
    fz = f_over_r * dz
    ep = 4f0*ϵ*(s12 - s6)
    return fx, fy, fz, ep
end

# ───────────────────────────────────────────────────────────────────────────────
# New (fast) CSR kernels — for NeighborLists.NeighborMatrix (your “newer” NL)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += 0.5f0 * ep   # half to avoid double counting
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Mixed-σ LJ (per-particle size; Lorentz mixing, global ϵ)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_csr_kernel_mixed!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32,
    σp::CuDeviceVector{Float32}, rcut_factor::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        σij = 0.5f0 * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > 0f0) & (r2 < rcut_ij*rcut_ij)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += 0.5f0 * (4f0*ϵ*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_mixed!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32,
    σp::CuDeviceVector{Float32}, rcut_factor::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        σij = 0.5f0 * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > 0f0) & (r2 < rcut_ij*rcut_ij)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += 0.5f0 * (4f0*ϵ*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_mixed!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32,
    σp::CuDeviceVector{Float32}, rcut_factor::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        σij = 0.5f0 * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > 0f0) & (r2 < rcut_ij*rcut_ij)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_mixed!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32,
    σp::CuDeviceVector{Float32}, rcut_factor::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σi = σp[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        σij = 0.5f0 * (σi + σp[j])
        rcut_ij = rcut_factor * σij
        if (r2 > 0f0) & (r2 < rcut_ij*rcut_ij)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_mixed!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box2,
                              ϵ::Float32,
                              σp::CuArray{Float32,1}, rcut_factor::Float32)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_mixed!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box3,
                              ϵ::Float32,
                              σp::CuArray{Float32,1}, rcut_factor::Float32)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function lj_forces_soa_noE_mixed!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box2,
                                  ϵ::Float32,
                                  σp::CuArray{Float32,1}, rcut_factor::Float32)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_noE_mixed!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box3,
                                  ϵ::Float32,
                                  σp::CuArray{Float32,1}, rcut_factor::Float32)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += 0.5f0 * ep
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
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    eps::Float32, sig::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        # skip bonded pairs
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, eps, sig)
            accx += fxij; accy += fyij
            eacc += 0.5f0 * ep
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_excl!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    eps::Float32, sig::Float32, cutoff2::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0

    base  = neighbors_index[i]
    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, eps, sig)
            accx += fxij; accy += fyij; accz += fzij
            eacc += 0.5f0 * ep
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_excl!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    eps::Float32, sig::Float32, cutoff2::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < cutoff2)
            invr2 = 1f0 / r2
            s2    = (sig*sig) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*eps*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_excl!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    eps::Float32, sig::Float32, cutoff2::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < cutoff2)
            invr2 = 1f0 / r2
            s2    = (sig*sig) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*eps*(2f0*s12 - s6)*invr2
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
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0

    @inbounds for k in 1:cap
        j = nbr[i,k]; if j <= 0; break; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij; eacc += 0.5f0*ep
        end
    end

    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_mat_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0

    @inbounds for k in 1:cap
        j = nbr[i,k]; if j <= 0; break; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += 0.5f0*ep
        end
    end

    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

# ───────────────────────────────────────────────────────────────────────────────
# Public API (dispatch to the correct path automatically)
# ───────────────────────────────────────────────────────────────────────────────

# ---- 2D, CSR (NeighborLists.NeighborMatrix)
function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh::NeighborLists.NeighborMatrix,
                        box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly

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
function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh::NeighborLists.NeighborMatrix,
                        box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz

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
function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly

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

function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz

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
function lj_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                             Epot::CuArray{Float32,1},
                             nbh::NeighborLists.NeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly

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

function lj_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                             Epot::CuArray{Float32,1},
                             nbh::NeighborLists.NeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz

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
function lj_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                             Epot::CuArray{Float32,1},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                             Epot::CuArray{Float32,1},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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
function lj_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                 fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                 nbh::NeighborLists.NeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                 fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                 nbh::NeighborLists.NeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function lj_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                 fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                 nbh::NeighborLists.StencilNeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                 fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                 nbh::NeighborLists.StencilNeighborMatrix,
                                 bonds::BondedForces.BondList,
                                 box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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
function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh,  # duck-typed; must have .neighbors::CuArray{Int32,2} and .cap
                        box::Definitions.Box2, params::Definitions.LJParams{Float32})
    @assert hasproperty(nbh, :neighbors) "nbh lacks 'neighbors' field"
    @assert hasproperty(nbh, :cap)       "nbh lacks 'cap' field"

    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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
function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh,
                        box::Definitions.Box3, params::Definitions.LJParams{Float32})
    @assert hasproperty(nbh, :neighbors) "nbh lacks 'neighbors' field"
    @assert hasproperty(nbh, :cap)       "nbh lacks 'cap' field"

    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{Float32}, epsilon_pair::CuDeviceMatrix{Float32}, rcut_pair::CuDeviceMatrix{Float32}
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
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
        if (r2 > 0f0) & (r2 < rc*rc)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*εij*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += 0.5f0 * (4f0*εij*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_csr_kernel_pairs!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{Float32}, epsilon_pair::CuDeviceMatrix{Float32}, rcut_pair::CuDeviceMatrix{Float32}
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
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
        if (r2 > 0f0) & (r2 < rc*rc)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*εij*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += 0.5f0 * (4f0*εij*(s12 - s6))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj2_csr_noE_kernel_pairs!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{Float32}, epsilon_pair::CuDeviceMatrix{Float32}, rcut_pair::CuDeviceMatrix{Float32}
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
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
        if (r2 > 0f0) & (r2 < rc*rc)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*εij*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_csr_noE_kernel_pairs!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{Float32}, epsilon_pair::CuDeviceMatrix{Float32}, rcut_pair::CuDeviceMatrix{Float32}
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
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
        if (r2 > 0f0) & (r2 < rc*rc)
            invr2 = 1f0 / r2
            s2    = (σij*σij) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*εij*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function lj_forces_soa_pairs!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box2,
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{Float32,2}, epsilon_pair::CuArray{Float32,2}, rcut_pair::CuArray{Float32,2})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_oa_pairs_bugfix() end  # keep file parsable if above adds no trailing newline

function lj_forces_soa_pairs!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              box::Definitions.Box3,
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{Float32,2}, epsilon_pair::CuArray{Float32,2}, rcut_pair::CuArray{Float32,2})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function lj_forces_soa_noE_pairs!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box2,
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{Float32,2}, epsilon_pair::CuArray{Float32,2}, rcut_pair::CuArray{Float32,2})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function lj_forces_soa_noE_pairs!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  box::Definitions.Box3,
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{Float32,2}, epsilon_pair::CuArray{Float32,2}, rcut_pair::CuArray{Float32,2})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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
const _CBRT2_F32 = 1.2599211f0  # 2^(1/3)

function _wca2_csr_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = 4f0*ϵ*(s12 - s6) + ϵ
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# Exclusion variants (skip bonded pairs)
function _wca2_csr_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = 4f0*ϵ*(s12 - s6) + ϵ
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _wca3_csr_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = 4f0*ϵ*(s12 - s6) + ϵ
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca2_csr_noE_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_csr_noE_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function wca_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.NeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function wca_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              nbh::NeighborLists.NeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function wca_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                  nbh::NeighborLists.NeighborMatrix,
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function wca_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                  nbh::NeighborLists.NeighborMatrix,
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function _wca3_csr_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = 4f0*ϵ*(s12 - s6) + ϵ
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca2_csr_noE_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_csr_noE_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    rc2 = _CBRT2_F32 * (σ*σ)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < rc2)
            invr2 = 1f0 / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# Wrappers
function wca_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                        nbh::NeighborLists.NeighborMatrix,
                        box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _wca2_csr_kernel!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                        nbh::NeighborLists.NeighborMatrix,
                        box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _wca3_csr_kernel!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                             nbh::NeighborLists.NeighborMatrix,
                             box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _wca2_csr_noE_kernel!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function wca_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                             fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                             nbh::NeighborLists.NeighborMatrix,
                             box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function _harmrep2_csr_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = 0.5f0 * ϵ * (1f0 - r/σ)*(1f0 - r/σ)
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

# Exclusion variants for soft-repulsive harmonic
function _harmrep2_csr_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += 0.5f0 * (0.5f0 * ϵ*(1f0 - r/σ)*(1f0 - r/σ))
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _harmrep3_csr_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
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
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += 0.5f0 * (0.5f0 * ϵ*(1f0 - r/σ)*(1f0 - r/σ))
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _harmrep2_csr_noE_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _harmrep3_csr_noE_kernel_excl!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
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
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function harmonic_rep_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                       fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                                       nbh::NeighborLists.NeighborMatrix,
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box2, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function harmonic_rep_forces_soa_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                       fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                                       nbh::NeighborLists.NeighborMatrix,
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box3, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function harmonic_rep_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                           fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                           nbh::NeighborLists.NeighborMatrix,
                                           bonds::BondedForces.BondList,
                                           box::Definitions.Box2, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
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

function harmonic_rep_forces_soa_noE_excl!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                           fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                           nbh::NeighborLists.NeighborMatrix,
                                           bonds::BondedForces.BondList,
                                           box::Definitions.Box3, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
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

function _harmrep3_csr_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, Epot::CuDeviceVector{Float32},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = 0.5f0 * ϵ * (1f0 - r/σ)*(1f0 - r/σ)
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _harmrep2_csr_noE_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _harmrep3_csr_noE_kernel!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32, ϵ::Float32, σ::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base  = neighbors_index[i]
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > 0f0) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (1f0 - r/σ) / r
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# Wrappers for soft repulsive harmonic
function harmonic_rep_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                                  nbh::NeighborLists.NeighborMatrix,
                                  box::Definitions.Box2, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_kernel!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                                  nbh::NeighborLists.NeighborMatrix,
                                  box::Definitions.Box3, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_kernel!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                      fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                      nbh::NeighborLists.NeighborMatrix,
                                      box::Definitions.Box2, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _harmrep2_csr_noE_kernel!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                      fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                      nbh::NeighborLists.NeighborMatrix,
                                      box::Definitions.Box3, params::Definitions.SoftRepulsiveParams{Float32})
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _harmrep3_csr_noE_kernel!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end
end # module
