# =============================
# Soft repulsive harmonic (nonbonded)
# =============================

function _harmrep2_ell_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _harmrep2_ell_kernel_virial!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            ep = T(0.5) * ϵ * (one(T) - r/σ)*(one(T) - r/σ)
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

# Exclusion variants for soft-repulsive harmonic
function _harmrep2_ell_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
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

function _harmrep2_ell_kernel_excl_virial!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            eacc += T(0.5) * (T(0.5) * ϵ*(one(T) - r/σ)*(one(T) - r/σ))
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _harmrep3_ell_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
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

function _harmrep3_ell_kernel_excl_virial!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            eacc += T(0.5) * (T(0.5) * ϵ*(one(T) - r/σ)*(one(T) - r/σ))
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

function _harmrep2_ell_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
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

function _harmrep3_ell_noE_kernel_excl!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    bbase, bnb, b1, b2 = _bond_cache(Int32(i), bindex, bflat, bcounts)
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        if _is_bonded_cached(j, bbase, bnb, b1, b2, bflat); continue; end
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

@inline _softrep_threads(N::Int) = (N < 50_000) ? 32 : ((N < 200_000) ? 64 : 128)

function harmonic_rep_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                       fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                                       nbh::NeighborLists.NeighborMatrix{T},
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                       ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_kernel_excl!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                       fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                                       nbh::NeighborLists.NeighborMatrix{T},
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                       ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_kernel_excl_virial!(rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_kernel_excl!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                       fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                                       nbh::NeighborLists.NeighborMatrix{T},
                                       bonds::BondedForces.BondList,
                                       box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                       ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_kernel_excl_virial!(rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_noE_kernel_excl!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_noE_kernel_excl!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function _harmrep3_ell_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _harmrep3_ell_kernel_virial!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < σ2)
            r = sqrt(r2)
            f_over_r = (ϵ/σ) * (one(T) - r/σ) / r
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            ep = T(0.5) * ϵ * (one(T) - r/σ)*(one(T) - r/σ)
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

function _harmrep2_ell_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _harmrep3_ell_noE_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T, ϵ::T, σ::T
    ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    nlist = counts[i]
    σ2 = σ*σ
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_kernel!(rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                  ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_kernel_virial!(rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                  ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_kernel!(rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                                  nbh::NeighborLists.NeighborMatrix{T},
                                  box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T}
                                  ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_kernel_virial!(rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                      fx::CuArray{T,1}, fy::CuArray{T,1},
                                      nbh::NeighborLists.NeighborMatrix{T},
                                      box::Definitions.Box2{T}, params::Definitions.SoftRepulsiveParams{T}
                                      ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _harmrep2_ell_noE_kernel!(rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy, params.ϵ, params.σ)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy, params.ϵ, params.σ; threads, blocks)
    return nothing
end

function harmonic_rep_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                      fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                      nbh::NeighborLists.NeighborMatrix{T},
                                      box::Definitions.Box3{T}, params::Definitions.SoftRepulsiveParams{T} ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx); threads = min(_softrep_threads(N), N); blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _harmrep3_ell_noE_kernel!(rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ; threads, blocks)
    return nothing
end
