function _wca2_ell_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
            ep = T(4)*εij*(s12 - s6) + εij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _wca2_ell_kernel_pairs_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            ep = T(4)*εij*(s12 - s6) + εij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _wca3_ell_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
            ep = T(4)*εij*(s12 - s6) + εij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca3_ell_kernel_pairs_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            ep = T(4)*εij*(s12 - s6) + εij
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

function _wca2_ell_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _wca3_ell_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    nlist = counts[i]
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[_ell_index(i, t, N)]
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

function _wca2_ell_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            ep = T(4) * εij * (s12 - s6) + εij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _wca2_ell_kernel_pairs_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            ep = T(4) * εij * (s12 - s6) + εij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _wca3_ell_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            ep = T(4) * εij * (s12 - s6) + εij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _wca3_ell_kernel_pairs_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            ep = T(4) * εij * (s12 - s6) + εij
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

function _wca2_ell_noE_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_ell_noE_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32}, neighbors_flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32}, cap::Int32,
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    nlist = counts[i]
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for t in 0:Int(nlist - 1)
        j = neighbors_flat[_ell_index(i, t, N)]
        _is_bonded_cached(j, bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function _wca2_allpairs_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    ti = typeid[i]
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_allpairs_noE_kernel_pairs!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    ti = typeid[i]
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function _wca2_allpairs_noE_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for j in 1:N
        _is_bonded_cached(Int32(j), bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _wca3_allpairs_noE_kernel_pairs_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    typeid::CuDeviceVector{Int32},
    sigma_pair::CuDeviceMatrix{T}, epsilon_pair::CuDeviceMatrix{T}, rcut_pair::CuDeviceMatrix{T}
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    ii = Int32(i)
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    bbase, bnb, b1, b2 = _bond_cache(ii, bindex, bflat, bcounts)
    ti = typeid[i]
    @inbounds for j in 1:N
        _is_bonded_cached(Int32(j), bbase, bnb, b1, b2, bflat) && continue
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        tj = typeid[j]
        σij = sigma_pair[ti, tj]
        εij = epsilon_pair[ti, tj]
        rc = rcut_pair[ti, tj]
        if (r2 > zero(T)) & (r2 < rc * rc)
            invr2 = one(T) / r2
            s2 = (σij * σij) * invr2
            s6 = s2 * s2 * s2
            s12 = s6 * s6
            f_over_r = T(24) * εij * (T(2) * s12 - s6) * invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function wca_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                               nbh::NeighborLists.AbstractNeighborMatrix,
                               box::Definitions.Box2{T},
                               typeid::CuArray{Int32,1},
                               sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                               ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_ell_kernel_pairs!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                               nbh::NeighborLists.AbstractNeighborMatrix,
                               box::Definitions.Box2{T},
                               typeid::CuArray{Int32,1},
                               sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                               ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_ell_kernel_pairs_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                               nbh::NeighborLists.AbstractNeighborMatrix,
                               box::Definitions.Box3{T},
                               typeid::CuArray{Int32,1},
                               sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                               ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_ell_kernel_pairs!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                               nbh::NeighborLists.AbstractNeighborMatrix,
                               box::Definitions.Box3{T},
                               typeid::CuArray{Int32,1},
                               sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                               ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_ell_kernel_pairs_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                   fx::CuArray{T,1}, fy::CuArray{T,1},
                                   nbh::NeighborLists.AbstractNeighborMatrix,
                                   box::Definitions.Box2{T},
                                   typeid::CuArray{Int32,1},
                                   sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                   ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_force_only(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _wca2_ell_noE_kernel_pairs!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                   fx::CuArray{T,1}, fy::CuArray{T,1},
                                   ::NeighborLists.AllPairsNeighborMatrix{T},
                                   box::Definitions.Box2{T},
                                   typeid::CuArray{Int32,1},
                                   sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                   ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _wca2_allpairs_noE_kernel_pairs!(
        rx, ry, fx, fy,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                   fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                   nbh::NeighborLists.AbstractNeighborMatrix,
                                   box::Definitions.Box3{T},
                                   typeid::CuArray{Int32,1},
                                   sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                   ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_force_only(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _wca3_ell_noE_kernel_pairs!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_pairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                   fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                   ::NeighborLists.AllPairsNeighborMatrix{T},
                                   box::Definitions.Box3{T},
                                   typeid::CuArray{Int32,1},
                                   sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                   ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _wca3_allpairs_noE_kernel_pairs!(
        rx, ry, rz, fx, fy, fz,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box2{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _wca2_ell_kernel_pairs_excl!(
        rx, ry, fx, fy, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box2{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _wca2_ell_kernel_pairs_excl_virial!(
        rx, ry, fx, fy, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box3{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _wca3_ell_kernel_pairs_excl!(
        rx, ry, rz, fx, fy, fz, Epot,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz, Epot,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                              fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                              nbh::NeighborLists.AbstractNeighborMatrix,
                              bonds::BondedForces.BondList,
                              box::Definitions.Box3{T},
                              typeid::CuArray{Int32,1},
                              sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                              ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_energy(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _wca3_ell_kernel_pairs_excl_virial!(
        rx, ry, rz, fx, fy, fz, Epot, V,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz, Epot, V,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box2{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                  ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_force_only(N)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _wca2_ell_noE_kernel_pairs_excl!(
        rx, ry, fx, fy,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1},
                                  ::NeighborLists.AllPairsNeighborMatrix{T},
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box2{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                  ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _wca2_allpairs_noE_kernel_pairs_excl!(
        rx, ry, fx, fy,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                  nbh::NeighborLists.AbstractNeighborMatrix,
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box3{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                  ) where {T<:AbstractFloat}
    NeighborLists.require_valid_neighbors(nbh)
    N = length(rx)
    threads, blocks = _launch_config_force_only(N)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _wca3_ell_noE_kernel_pairs_excl!(
        rx, ry, rz, fx, fy, fz,
        nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz,
      nbh.neighbors_index, nbh.neighbors_flat, nbh.counts, nbh.cap,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end

function wca_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                  fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                  ::NeighborLists.AllPairsNeighborMatrix{T},
                                  bonds::BondedForces.BondList,
                                  box::Definitions.Box3{T},
                                  typeid::CuArray{Int32,1},
                                  sigma_pair::CuArray{T,2}, epsilon_pair::CuArray{T,2}, rcut_pair::CuArray{T,2}
                                  ) where {T<:AbstractFloat}
    N = length(rx)
    threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256)
    blocks = cld(N, threads)
    Lx = T(box[1]); Ly = T(box[2]); Lz = T(box[3])
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _wca3_allpairs_noE_kernel_pairs_excl!(
        rx, ry, rz, fx, fy, fz,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        typeid, sigma_pair, epsilon_pair, rcut_pair)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      typeid, sigma_pair, epsilon_pair, rcut_pair; threads, blocks)
    return nothing
end
