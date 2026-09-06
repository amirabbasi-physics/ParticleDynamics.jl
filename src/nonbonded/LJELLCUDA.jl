# No-energy (no Epot) ELL variants
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_ell_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj3_ell_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_ell_noE_kernel!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.NeighborMatrix{T},
                            box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_ell_noE_kernel!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# Overloads for stencil neighbor lists (reuse the same ELL kernels)
function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            nbh::NeighborLists.StencilNeighborMatrix{T},
                            box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_ell_noE_kernel!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            nbh::NeighborLists.StencilNeighborMatrix{T},
                            box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_ell_noE_kernel!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end


# ───────────────────────────────────────────────────────────────────────────────
# New (fast) ELL kernels — for NeighborLists.NeighborMatrix (your “newer” NL)
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_ell_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj2_ell_kernel_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _lj3_ell_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj3_ell_kernel_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx; vyy += dvyy; vzz += dvzz
            vxy += dvxy; vxz += dvxz; vyz += dvyz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _lj2_ell_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj2_ell_kernel_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, eps, sig)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _lj3_ell_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj3_ell_kernel_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded(Int32(i), j, bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, eps, sig)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx; vyy += dvyy; vzz += dvzz
            vxy += dvxy; vxz += dvxz; vyz += dvyz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _lj2_ell_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    eps::T, sig::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _lj3_ell_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    eps::T, sig::T, cutoff2::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    nlist = counts[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

# ---- 2D, ELL (NeighborLists.NeighborMatrix)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_ell_kernel!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1}, V::CuArray{T,2},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_kernel_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# ---- 3D, ELL (NeighborLists.NeighborMatrix)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_ell_kernel!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1}, V::CuArray{T,2},
                        nbh::NeighborLists.NeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_kernel_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

# Overloads for stencil neighbor lists (ELL paths)
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_ell_kernel!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1},
                        Epot::CuArray{T,1}, V::CuArray{T,2},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_kernel_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_ell_kernel!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                        Epot::CuArray{T,1}, V::CuArray{T,2},
                        nbh::NeighborLists.StencilNeighborMatrix,
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_kernel_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly

    k = CUDA.@cuda launch=false _lj2_ell_kernel_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1},
                             Epot::CuArray{T,1}, V::CuArray{T,2},
                             nbh::NeighborLists.NeighborMatrix{T},
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_kernel_excl_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)

    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz

    k = CUDA.@cuda launch=false _lj3_ell_kernel_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                             Epot::CuArray{T,1}, V::CuArray{T,2},
                             nbh::NeighborLists.NeighborMatrix{T},
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_kernel_excl_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_kernel_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1},
                             Epot::CuArray{T,1}, V::CuArray{T,2},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_kernel_excl_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_kernel_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                             Epot::CuArray{T,1}, V::CuArray{T,2},
                             nbh::NeighborLists.StencilNeighborMatrix,
                             bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_kernel_excl_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_noE_kernel_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_noE_kernel_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_ell_noE_kernel_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks  = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_ell_noE_kernel_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.ϵ, params.σ, cutoff2
    )
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
