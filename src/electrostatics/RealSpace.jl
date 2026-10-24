# =========================
#   Real-space Coulomb, exceptions and self energy (one row per particle)
# =========================
# Full neighbor rows (both i→j and j→i are stored), so each particle's group
# of threads owns particle i and adds its force; pair energies and virials
# carry the factor ½. Every real-space pair is included regardless of
# molecular exclusions; the exception rows then add (s − 1) C qᵢqⱼ/r for each
# scaled/excluded pair at its minimum-image distance, independent of the
# cutoff.
# With the potential-shift modifier every non-excluded pair inside the cutoff
# has its energy lowered by C qᵢqⱼ E_s(r_c)/r_c; exception pairs do not carry
# the shift, so their rows add it back.
#
# Thread layout: TPP consecutive lanes share one particle and stride over its
# neighbor row (slots lane, lane + TPP, …), then combine with warp shuffles.
# TPP = 1 for large systems (one thread per particle, as the vdW kernels);
# small systems use TPP up to 32 so that the GPU has enough resident threads
# (a single thread walking a ~500-entry row left most SMs idle). Threads keep
# the storage order, so slot-major ELL reads stay coalesced across particles.

@inline function _mic(d::T, half::T, L::T) where {T}
    d -= (d > half) * L
    d += (d < -half) * L
    return d
end

@inline function _group_sum(v, ::Val{TPP}) where {TPP}
    off = TPP ÷ 2
    while off >= 1
        v += shfl_down_sync(0xffffffff, v, off)
        off ÷= 2
    end
    return v
end

function _coulomb_real_kernel!(fx, fy, fz, Epot, V, rx, ry, rz, q,
                               nbr::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
                               exo::CuDeviceVector{Int32}, exj::CuDeviceVector{Int32},
                               exs::CuDeviceVector{T},
                               Lx::T, Ly::T, Lz::T, C::T, rc2::T, selfc::T, eshift::T,
                               kernel::K, ::Val{ALLPAIRS}, ::Val{ENERGY},
                               ::Val{TPP}) where {T,K,ALLPAIRS,ENERGY,TPP}
    gid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    i = Int(gid ÷ Int32(TPP)) + 1
    lane = Int(gid % Int32(TPP))
    N = length(rx)
    # Threads past the end stay alive until the shuffles; they contribute zeros.
    active = i <= N
    ax = zero(T); ay = zero(T); az = zero(T); e = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    qi = active ? (@inbounds q[i]) : zero(T)
    @inbounds if qi != zero(T)
        xi = rx[i]; yi = ry[i]; zi = rz[i]
        hx = T(0.5) * Lx; hy = T(0.5) * Ly; hz = T(0.5) * Lz
        nlist = ALLPAIRS ? N : Int(counts[i])
        t = lane
        while t < nlist
            j = ALLPAIRS ? t + 1 : Int(nbr[NeighborLists._ell_index(i, t, N)])
            t += TPP
            (ALLPAIRS && j == i) && continue
            qj = q[j]
            qj == zero(T) && continue
            dx = _mic(xi - rx[j], hx, Lx)
            dy = _mic(yi - ry[j], hy, Ly)
            dz = _mic(zi - rz[j], hz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
            if (r2 > zero(T)) & (r2 < rc2)
                invr = one(T) / sqrt(r2)
                ep, fp = pair_kernel(kernel, r2 * invr, invr)
                s = C * qi * qj
                f = s * fp
                ax += f * dx; ay += f * dy; az += f * dz
                if ENERGY
                    e += s * (ep - eshift)
                    vxx += f * dx * dx; vyy += f * dy * dy; vzz += f * dz * dz
                    vxy += f * dx * dy; vxz += f * dx * dz; vyz += f * dy * dz
                end
            end
        end
        p = exo[i] + Int32(lane)
        while p < exo[i + 1]
            j = exj[p]
            s = C * qi * q[j] * (exs[p] - one(T))
            dx = _mic(xi - rx[j], hx, Lx)
            dy = _mic(yi - ry[j], hy, Ly)
            dz = _mic(zi - rz[j], hz, Lz)
            r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
            invr = one(T) / sqrt(r2)
            f = s * invr * invr * invr
            ax += f * dx; ay += f * dy; az += f * dz
            if ENERGY
                # excluded/scaled pairs carry no potential shift (GROMACS convention)
                e += s * invr + ifelse(r2 < rc2, C * qi * q[j] * eshift, zero(T))
                vxx += f * dx * dx; vyy += f * dy * dy; vzz += f * dz * dz
                vxy += f * dx * dy; vxz += f * dx * dz; vyz += f * dy * dz
            end
            p += Int32(TPP)
        end
    end
    if TPP > 1
        ax = _group_sum(ax, Val(TPP)); ay = _group_sum(ay, Val(TPP)); az = _group_sum(az, Val(TPP))
        if ENERGY
            e = _group_sum(e, Val(TPP))
            vxx = _group_sum(vxx, Val(TPP)); vyy = _group_sum(vyy, Val(TPP)); vzz = _group_sum(vzz, Val(TPP))
            vxy = _group_sum(vxy, Val(TPP)); vxz = _group_sum(vxz, Val(TPP)); vyz = _group_sum(vyz, Val(TPP))
        end
    end
    @inbounds if (lane == 0) & (qi != zero(T))
        fx[i] += ax; fy[i] += ay; fz[i] += az
        if ENERGY
            Epot[i] += T(0.5) * e - T(0.5) * C * selfc * qi * qi
            h = T(0.5)
            V[i, 1] += h * vxx; V[i, 2] += h * vyy; V[i, 3] += h * vzz
            V[i, 4] += h * vxy; V[i, 5] += h * vxz; V[i, 6] += h * vyz
        end
    end
    return
end
