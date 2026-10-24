# =========================
#   Fused cluster-pair kernel, scatter, scaled pairs
# =========================
# Block = 64 threads = one super-cluster. Thread (ti, tj) = (tid & 7, tid >> 3):
# i-atom ti of each of the 8 i-clusters against j-atom tj of the current
# j-cluster; warp w holds tj ∈ {4w, …, 4w+3} (one j half) and skips the
# i-clusters cleared in its 8-bit mask. i-forces accumulate in registers and
# are reduced once per block; j-forces are reduced over the 8 ti lanes after
# every j-cluster and added with one atomic per component.

@inline function _sum8(v)
    v += shfl_down_sync(0xffffffff, v, 4)
    v += shfl_down_sync(0xffffffff, v, 2)
    v += shfl_down_sync(0xffffffff, v, 1)
    return v
end

# (x, y, z) summed over the 8 lanes of a ti group; lanes ti = 0, 1, 2 return
# the x, y, z totals.
@inline function _reduce3(fx::T, fy::T, fz::T, ti::Int32) where {T}
    fx += shfl_down_sync(0xffffffff, fx, 1)
    fy += shfl_up_sync(0xffffffff, fy, 1)
    fz += shfl_down_sync(0xffffffff, fz, 1)
    fx = ifelse((ti & Int32(1)) != Int32(0), fy, fx)
    fx += shfl_down_sync(0xffffffff, fx, 2)
    fz += shfl_up_sync(0xffffffff, fz, 2)
    fx = ifelse((ti & Int32(2)) != Int32(0), fz, fx)
    fx += shfl_down_sync(0xffffffff, fx, 4)
    return fx
end

@inline function _reduce_i(v)
    v += shfl_down_sync(0xffffffff, v, 16)
    v += shfl_down_sync(0xffffffff, v, 8)
    return v
end

function _cluster_pair_kernel!(fc, Ec, Vc, xs, ys, zs, qs, ts, s6, s12, sci, cjl, m8, xid, masks, counts, skip::Int32,
                               box::Union{NTuple{3,T},AbstractVector{T}}, ljp, ntypes::Int32, lrc2u::T, irc6::T, coul::CP,
                               nsplit::Int32, m8out, rp2::T, ::Val{SPLIT}, vlj::Val{LJK}, vljx::Val{LJEXCL},
                               vc::Val{COUL}, ve::Val{E}, ::Val{SHIFT}, ::Val{PRUNE}, vtr,
                               ::Val{VT}) where {T,CP,SPLIT,LJK,LJEXCL,COUL,E,SHIFT,PRUNE,VT}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    tid = threadIdx().x - Int32(1)
    ti = tid & Int32(7); tj = tid >> Int32(3); w = tid >> Int32(5)
    # `nsplit` blocks share one super-cluster, each taking a contiguous part
    # of its entries (small systems: enough blocks to fill the device)
    S = SPLIT ? (blockIdx().x - Int32(1)) ÷ nsplit : blockIdx().x - Int32(1)
    # launched at the super-cluster bound; blocks beyond the device count and
    # every block after an outer-list overflow do nothing (uniform per block)
    @inbounds (S >= counts[CL_NS] || counts[CL_ERR] != Int32(0)) && return
    # graph steps: the plain pass does nothing when the device chose the pruning pass
    @inbounds (skip > Int32(0) && counts[skip] != Int32(0)) && return
    sxq = CuStaticSharedArray(NTuple{4,T}, SUPER)
    sab = CuStaticSharedArray(NTuple{2,T}, (LJK == LJ_SEPARABLE || LJK == LJ_MIXED) ? SUPER : 1)
    stb = CuStaticSharedArray(Int32, SUPER)
    sljp = CuStaticSharedArray(NTuple{4,T}, LJK == LJ_TABLE_SHARED ? MAX_SHARED_TABLE : 1)
    xs = CUDA.Const(xs); ys = CUDA.Const(ys); zs = CUDA.Const(zs); qs = CUDA.Const(qs); ts = CUDA.Const(ts)
    s6 = CUDA.Const(s6); s12 = CUDA.Const(s12); ljp = CUDA.Const(ljp)
    cjl = CUDA.Const(cjl); m8 = CUDA.Const(m8); masks = CUDA.Const(masks); xid = CUDA.Const(xid)
    @inbounds begin
        a = S * Int32(SUPER) + tid + Int32(1)
        sxq[tid + 1] = (xs[a], ys[a], zs[a], qs[a])
        (LJK == LJ_SEPARABLE || LJK == LJ_MIXED) && (sab[tid + 1] = (s6[a], s12[a]))
        mix4e, _, mixf2, mixsh = LJK == LJ_MIXED ? ljp[1] : (zero(T), zero(T), zero(T), zero(T))
        (LJK == LJ_TABLE_SHARED || LJK == LJ_TABLE_GLOBAL) && (stb[tid + 1] = (ts[a] - Int32(1)) * ntypes)
        if LJK == LJ_TABLE_SHARED
            k = tid + Int32(1)
            while k <= ntypes * ntypes
                sljp[k] = ljp[k]
                k += Int32(SUPER)
            end
        end
        sync_threads()
        hx = T(0.5) * Lx; hy = T(0.5) * Ly; hz = T(0.5) * Lz
        Base.Cartesian.@nexprs 8 c -> (fxi_c = zero(T); fyi_c = zero(T); fzi_c = zero(T))
        wt = zero(T)   # VT: this thread's Σ f·r² (virial trace of its pairs)
        if E
            Base.Cartesian.@nexprs 8 c -> (ei_c = zero(T); w1i_c = zero(T); w2i_c = zero(T); w3i_c = zero(T);
                                           w4i_c = zero(T); w5i_c = zero(T); w6i_c = zero(T))
        end
        if SPLIT
            part = (blockIdx().x - Int32(1)) % nsplit
            e0 = sci[S + 1]; ne = sci[S + 2] - e0
            ea = e0 + (ne * part) ÷ nsplit; eb = e0 + (ne * (part + Int32(1))) ÷ nsplit
        else
            ea = sci[S + 1]; eb = sci[S + 2]
        end
        for e in (ea + Int32(1)):eb
            imask = UInt32(m8[Int32(2) * e - Int32(1) + w])
            if imask == UInt32(0)
                (PRUNE && (tid & Int32(31)) == Int32(0)) && (m8out[Int32(2) * e - Int32(1) + w] = UInt8(0))
                continue
            end
            pm = UInt32(0)
            ent = cjl[e]
            cj = Int32(ent >> 6)
            hasx = (ent & UInt32(1)) != UInt32(0)
            xb = hasx ? (xid[e] - Int32(1)) * Int32(8) : Int32(0)
            js = cj * Int32(CLUSTER) + tj + Int32(1)
            xj = xs[js]; yj = ys[js]; zj = zs[js]; qj = qs[js]
            tyj = (LJK == LJ_TABLE_SHARED || LJK == LJ_TABLE_GLOBAL) ? ts[js] : Int32(0)
            a6j = (LJK == LJ_SEPARABLE || LJK == LJ_MIXED) ? s6[js] : zero(T)
            a12j = LJK == LJ_SEPARABLE ? s12[js] : zero(T)
            fjx = zero(T); fjy = zero(T); fjz = zero(T)
            ej = zero(T); vj1 = zero(T); vj2 = zero(T); vj3 = zero(T); vj4 = zero(T); vj5 = zero(T); vj6 = zero(T)
            Base.Cartesian.@nexprs 8 c -> begin
                if (imask >> (c - 1)) & UInt32(1) != UInt32(0)
                    ii = (c - 1) * 8 + ti + 1
                    xi, yi, zi, qi = sxq[ii]
                    # Encoded shifts describe build-time bounding boxes.
                    # Applying them to Float32 coordinates before subtraction
                    # loses short-range precision. Use centered force geometry
                    # with a compensated periodic difference on either list.
                    # The compensated image path is needed only near a box face; take it only
                    # when a lane of the warp (one j-cluster) needs it. Identical deltas otherwise.
                    dx = xi - xj; dy = yi - yj; dz = zi - zj
                    if vote_any_sync(0xffffffff, (abs(dx) > hx) | (abs(dy) > hy) | (abs(dz) > hz))
                        dx = periodic_delta(xi, xj, hx, Lx)
                        dy = periodic_delta(yi, yj, hy, Ly)
                        dz = periodic_delta(zi, zj, hz, Lz)
                    end
                    r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
                    ok = ((cj != S * Int32(8) + Int32(c - 1)) | (tj > ti)) & (r2 < T(Inf))
                    excluded = hasx ? (((masks[xb + c] >> (ti * Int32(8) + tj)) & UInt64(1)) != UInt64(0)) : false
                    if PRUNE
                        # keep the tile if any pair is inside the prune radius or excluded
                        keep = vote_any_sync(0xffffffff, ok & ((r2 < rp2) | excluded))
                        pm |= ifelse(keep, UInt32(1) << (c - 1), UInt32(0))
                    end
                    rinv = CUDA.rsqrt(r2)
                    if LJK == LJ_TABLE_SHARED
                        c6, c12, lrc2, esh = sljp[stb[ii] + tyj]
                    elseif LJK == LJ_TABLE_GLOBAL
                        c6, c12, lrc2, esh = ljp[stb[ii] + tyj]
                    elseif LJK == LJ_SEPARABLE
                        a6i, a12i = sab[ii]
                        c6 = a6i * a6j; c12 = a12i * a12j; lrc2 = lrc2u
                        esh = E ? (c12 * irc6 - c6) * irc6 : zero(T)
                    elseif LJK == LJ_MIXED
                        sij = T(0.5) * (sab[ii][1] + a6j); s2 = sij * sij; s6v = s2 * s2 * s2
                        c6 = mix4e * s6v; c12 = c6 * s6v; lrc2 = mixf2 * s2; esh = mixsh
                    else
                        c6 = zero(T); c12 = zero(T); lrc2 = zero(T); esh = zero(T)
                    end
                    f, en = pair_force(r2, rinv, qi * qj, c6, c12, lrc2, esh, excluded, coul, vlj, vljx, vc, ve)
                    # padding (NaN coordinates) and the lower triangle of the
                    # diagonal tile contribute exact zeros
                    f = ifelse(ok, f, zero(T))
                    dx = ifelse(ok, dx, zero(T)); dy = ifelse(ok, dy, zero(T)); dz = ifelse(ok, dz, zero(T))
                    VT && (wt = muladd(f, ifelse(ok, r2, zero(T)), wt))
                    fxp = f * dx; fyp = f * dy; fzp = f * dz
                    fxi_c += fxp; fyi_c += fyp; fzi_c += fzp
                    fjx -= fxp; fjy -= fyp; fjz -= fzp
                    if E
                        en = ifelse(ok, T(0.5) * en, zero(T))
                        w1 = T(0.5) * fxp * dx; w2 = T(0.5) * fyp * dy; w3 = T(0.5) * fzp * dz
                        w4 = T(0.5) * fxp * dy; w5 = T(0.5) * fxp * dz; w6 = T(0.5) * fyp * dz
                        ei_c += en; w1i_c += w1; w2i_c += w2; w3i_c += w3; w4i_c += w4; w5i_c += w5; w6i_c += w6
                        ej += en; vj1 += w1; vj2 += w2; vj3 += w3; vj4 += w4; vj5 += w5; vj6 += w6
                    end
                end
            end
            (PRUNE && (tid & Int32(31)) == Int32(0)) && (m8out[Int32(2) * e - Int32(1) + w] = UInt8(pm))
            r = _reduce3(fjx, fjy, fjz, ti)
            ti < Int32(3) && (Definitions.@red fc[js, ti + Int32(1)] += r)
            if E
                rv1 = _reduce3(vj1, vj2, vj3, ti)
                rv2 = _reduce3(vj4, vj5, vj6, ti)
                ej = _sum8(ej)
                ti < Int32(3) && (Definitions.@red Vc[js, ti + Int32(1)] += rv1)
                ti < Int32(3) && (Definitions.@red Vc[js, ti + Int32(4)] += rv2)
                ti == Int32(0) && (Definitions.@red Ec[js] += ej)
            end
        end
        lane = tid & Int32(31)
        if VT   # one Float64 atomic per warp
            w64 = Float64(wt)
            for off in (16, 8, 4, 2, 1)
                w64 += shfl_down_sync(0xffffffff, w64, off)
            end
            lane == Int32(0) && w64 != 0.0 && Definitions.@red vtr[1] += w64
        end
        Base.Cartesian.@nexprs 8 c -> begin
            fxi_c = _reduce_i(fxi_c); fyi_c = _reduce_i(fyi_c); fzi_c = _reduce_i(fzi_c)
            if E
                ei_c = _reduce_i(ei_c); w1i_c = _reduce_i(w1i_c); w2i_c = _reduce_i(w2i_c); w3i_c = _reduce_i(w3i_c)
                w4i_c = _reduce_i(w4i_c); w5i_c = _reduce_i(w5i_c); w6i_c = _reduce_i(w6i_c)
            end
        end
        if lane < Int32(8)
            Base.Cartesian.@nexprs 8 c -> begin
                ai = S * Int32(SUPER) + Int32((c - 1) * 8) + lane + Int32(1)
                Definitions.@red fc[ai, 1] += fxi_c; Definitions.@red fc[ai, 2] += fyi_c; Definitions.@red fc[ai, 3] += fzi_c
                if E
                    Definitions.@red Ec[ai] += ei_c
                    Definitions.@red Vc[ai, 1] += w1i_c; Definitions.@red Vc[ai, 2] += w2i_c; Definitions.@red Vc[ai, 3] += w3i_c
                    Definitions.@red Vc[ai, 4] += w4i_c; Definitions.@red Vc[ai, 5] += w5i_c; Definitions.@red Vc[ai, 6] += w6i_c
                end
            end
        end
    end
    return
end

# Slot buffers → storage order (overwriting the force slot), self energy, and
# zeroing of the slot buffers for the next evaluation.
# ADD: the slots already hold the mesh force and energy (a gather in set mode ran first).
function _scatter_kernel!(fx, fy, fz, Epot, V, fc, Ec, Vc, perm, q, halfCself::T,
                          ::Val{E}, ::Val{CHARGED}, ::Val{ADD}=Val(false)) where {T,E,CHARGED,ADD}
    s = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s > length(perm) && return
    @inbounds begin
        i = perm[s]
        if i != Int32(0)
            if ADD
                fx[i] += fc[s, 1]; fy[i] += fc[s, 2]; fz[i] += fc[s, 3]
            else
                fx[i] = fc[s, 1]; fy[i] = fc[s, 2]; fz[i] = fc[s, 3]
            end
            if E
                ei = Ec[s]
                if CHARGED
                    qi = q[i]
                    ei -= halfCself * qi * qi
                end
                Epot[i] = ADD ? Epot[i] + ei : ei
                for k in 1:6
                    V[i, k] = Vc[s, k]
                end
            end
        end
        fc[s, 1] = zero(T); fc[s, 2] = zero(T); fc[s, 3] = zero(T)
        if E
            Ec[s] = zero(T)
            for k in 1:6
                Vc[s, k] = zero(T)
            end
        end
    end
    return
end

# Bond exclusions (see `bond_exclusion_rows`): the Lennard-Jones term of masked
# pairs that are not bonded, with the table's cutoff and shift.
function _lj_restore_kernel!(fx, fy, fz, Epot, V, li, lj, rx, ry, rz, typ, ljp, ntypes::Int32,
                             box::Union{NTuple{3,T},AbstractVector{T}}, counts, ::Val{E}, ::Val{MIXED}) where {T,E,MIXED}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds (k > length(li) || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        i = li[k]; j = lj[k]
        dx = periodic_delta(rx[i], rx[j], Lx)
        dy = periodic_delta(ry[i], ry[j], Ly)
        dz = periodic_delta(rz[i], rz[j], Lz)
        r2 = dx * dx + dy * dy + dz * dz
        if MIXED   # `typ` holds σ by particle; ljp[1] = (4ε, –, factor², shift)
            m4e, _, mf2, esh = ljp[1]
            sij = T(0.5) * (typ[i] + typ[j]); s2 = sij * sij; s6v = s2 * s2 * s2
            c6 = m4e * s6v; c12 = c6 * s6v; lrc2 = mf2 * s2
        else
            c6, c12, lrc2, esh = ljp[(typ[i] - Int32(1)) * ntypes + typ[j]]
        end
        r2 < lrc2 || return
        rinv2 = one(T) / r2
        ir6 = rinv2 * rinv2 * rinv2
        f = (T(12) * c12 * ir6 - T(6) * c6) * ir6 * rinv2
        Definitions.@red fx[i] += f * dx; Definitions.@red fy[i] += f * dy; Definitions.@red fz[i] += f * dz
        Definitions.@red fx[j] -= f * dx; Definitions.@red fy[j] -= f * dy; Definitions.@red fz[j] -= f * dz
        if E
            h = T(0.5) * ((c12 * ir6 - c6) * ir6 - esh)
            Definitions.@red Epot[i] += h
            Definitions.@red Epot[j] += h
            w1 = T(0.5) * f * dx * dx; w2 = T(0.5) * f * dy * dy; w3 = T(0.5) * f * dz * dz
            w4 = T(0.5) * f * dx * dy; w5 = T(0.5) * f * dx * dz; w6 = T(0.5) * f * dy * dz
            Definitions.@red V[i, 1] += w1; Definitions.@red V[j, 1] += w1
            Definitions.@red V[i, 2] += w2; Definitions.@red V[j, 2] += w2
            Definitions.@red V[i, 3] += w3; Definitions.@red V[j, 3] += w3
            Definitions.@red V[i, 4] += w4; Definitions.@red V[j, 4] += w4
            Definitions.@red V[i, 5] += w5; Definitions.@red V[j, 5] += w5
            Definitions.@red V[i, 6] += w6; Definitions.@red V[j, 6] += w6
        end
    end
    return
end

# Bonded pairs restored to full Coulomb with scale 1 (no shift, as exception pairs):
# the potential shift an ordinary pair inside the cutoff carries (energy only).
function _coulomb_shift_kernel!(Epot, ci, cj, rx, ry, rz, q, C::T, rc2::T, eshift::T,
                                box::Union{NTuple{3,T},AbstractVector{T}}, counts) where {T}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds (k > length(ci) || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        i = ci[k]; j = cj[k]
        dx = periodic_delta(rx[i], rx[j], Lx)
        dy = periodic_delta(ry[i], ry[j], Ly)
        dz = periodic_delta(rz[i], rz[j], Lz)
        r2 = dx * dx + dy * dy + dz * dz
        r2 < rc2 || return
        h = -T(0.5) * C * q[i] * q[j] * eshift
        Definitions.@red Epot[i] += h
        Definitions.@red Epot[j] += h
    end
    return
end

# coef · C qᵢqⱼ / r for scaled pairs inside the list and exception pairs outside it.
function _scaled_pairs_kernel!(fx, fy, fz, Epot, V, pi, pj, pc, rx, ry, rz, q, C::T,
                               box::Union{NTuple{3,T},AbstractVector{T}}, counts, ::Val{E}, vtr=nothing) where {T,E}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds (k > counts[CL_NP] || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        i = pi[k]; j = pj[k]
        dx = periodic_delta(rx[i], rx[j], Lx)
        dy = periodic_delta(ry[i], ry[j], Ly)
        dz = periodic_delta(rz[i], rz[j], Lz)
        r2 = dx * dx + dy * dy + dz * dz
        rinv = one(T) / sqrt(r2)
        s = pc[k] * C * q[i] * q[j]
        f = s * rinv * rinv * rinv
        Definitions.@red fx[i] += f * dx; Definitions.@red fy[i] += f * dy; Definitions.@red fz[i] += f * dz
        Definitions.@red fx[j] -= f * dx; Definitions.@red fy[j] -= f * dy; Definitions.@red fz[j] -= f * dz
        vtr === nothing || Definitions.@red vtr[1] += Float64(f * r2)
        if E
            h = T(0.5) * s * rinv
            Definitions.@red Epot[i] += h
            Definitions.@red Epot[j] += h
            w1 = T(0.5) * f * dx * dx; w2 = T(0.5) * f * dy * dy; w3 = T(0.5) * f * dz * dz
            w4 = T(0.5) * f * dx * dy; w5 = T(0.5) * f * dx * dz; w6 = T(0.5) * f * dy * dz
            Definitions.@red V[i, 1] += w1; Definitions.@red V[j, 1] += w1
            Definitions.@red V[i, 2] += w2; Definitions.@red V[j, 2] += w2
            Definitions.@red V[i, 3] += w3; Definitions.@red V[j, 3] += w3
            Definitions.@red V[i, 4] += w4; Definitions.@red V[j, 4] += w4
            Definitions.@red V[i, 5] += w5; Definitions.@red V[j, 5] += w5
            Definitions.@red V[i, 6] += w6; Definitions.@red V[j, 6] += w6
        end
    end
    return
end


# Distance-only pruning pass (15 Oct 2026): the outer list's tiles at the current coordinates,
# kept when any pair lies inside the prune radius or is excluded, without forces; the plain pass
# then runs on the compacted working list.
# Block (64 threads: 8 i-atoms × 8 j-atoms) per super-cluster, or `nsplit` blocks sharing one.
# The keep test needs no correctly rounded separation: a branch-free nearest image and a radius
# widened by its rounding bound keep every tile the exact criterion keeps (a superset; extra
# tiles lie beyond the cutoff and add exact zeros).
function _prune_only_kernel!(m8o, xs, ys, zs, scio, cjf, m8f, xid, masks, counts,
                             box::Union{NTuple{3,T},AbstractVector{T}}, rp2::T, nsplit::Int32) where {T}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    tid = threadIdx().x - Int32(1)
    ti = tid & Int32(7); tj = tid >> Int32(3); w = tid >> Int32(5)
    S = (blockIdx().x - Int32(1)) ÷ nsplit
    @inbounds (S >= counts[CL_NS] || counts[CL_ERR] != Int32(0)) && return
    sx = CuStaticSharedArray(NTuple{3,T}, SUPER)
    @inbounds begin
        a = S * Int32(SUPER) + tid + Int32(1)
        sx[tid + 1] = (xs[a], ys[a], zs[a])
        sync_threads()
        iLx = one(T) / Lx; iLy = one(T) / Ly; iLz = one(T) / Lz
        # |error| of each component ≤ eps·L (difference, then the fused shift), of r² ≤ eps·(6 r L + 3 r²)
        rp2 = rp2 + eps(T) * (T(16) * sqrt(rp2) * max(Lx, Ly, Lz) + T(16) * rp2)
        part = (blockIdx().x - Int32(1)) % nsplit
        e0 = scio[S + 1]; ne = scio[S + 2] - e0
        ea = e0 + (ne * part) ÷ nsplit; eb = e0 + (ne * (part + Int32(1))) ÷ nsplit
        for e in (ea + Int32(1)):eb
            imask = UInt32(m8f[Int32(2) * e - Int32(1) + w])
            pm = UInt32(0)
            if imask != UInt32(0)   # uniform per warp
                ent = cjf[e]
                cj = Int32(ent >> 6)
                hasx = (ent & UInt32(1)) != UInt32(0)
                xb = hasx ? (xid[e] - Int32(1)) * Int32(8) : Int32(0)
                js = cj * Int32(CLUSTER) + tj + Int32(1)
                xj = xs[js]; yj = ys[js]; zj = zs[js]
                Base.Cartesian.@nexprs 8 c -> begin
                    if (imask >> (c - 1)) & UInt32(1) != UInt32(0)
                        xi, yi, zi = sx[(c - 1) * 8 + ti + 1]
                        dx = xi - xj; dy = yi - yj; dz = zi - zj
                        dx = muladd(-Lx, round(dx * iLx), dx); dy = muladd(-Ly, round(dy * iLy), dy); dz = muladd(-Lz, round(dz * iLz), dz)
                        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
                        ok = ((cj != S * Int32(8) + Int32(c - 1)) | (tj > ti)) & (r2 < T(Inf))
                        excluded = hasx ? (((masks[xb + Int32(c)] >> (ti * Int32(8) + tj)) & UInt64(1)) != UInt64(0)) : false
                        keep = vote_any_sync(0xffffffff, ok & ((r2 < rp2) | excluded))
                        pm |= ifelse(keep, UInt32(1) << (c - 1), UInt32(0))
                    end
                end
            end
            (tid & Int32(31)) == Int32(0) && (m8o[Int32(2) * e - Int32(1) + w] = UInt8(pm))
        end
    end
    return
end
