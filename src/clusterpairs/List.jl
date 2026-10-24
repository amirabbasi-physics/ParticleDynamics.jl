# =========================
#   Cluster-pair list: clustering, search, exclusion masks, pruning
# =========================
# Slots: super-cluster S (0-based) owns slots 64S … 64S+63, cluster c = slot ÷ 8.
# Particles are binned into x–y columns of about one super-cluster edge,
# sorted by z within a column, cut into super-clusters of 64 and split 2×2×2
# into clusters of up to 8 (x halves, then y halves, then z halves). Unused
# slots are padding (`perm = 0`, NaN coordinates, zero charge).
#
# Outer list (built at r_list = r_max + skin): per super-cluster, the
# j-clusters with cluster index ≥ the i-cluster's and a bounding-box gap below
# r_list, as `(cj << 9) | (excl flag << 8) | imask`.
# Working list (rebuilt by pruning): entries with at least one pair inside
# the prune radius, as `(cj << 6) | (shift << 1) | excl flag` plus one 8-bit
# i-cluster mask per warp (j-cluster half), and the periodic shift of the
# j-cluster relative to the super-cluster (3³ codes).
#
# Device residency: every buffer is allocated once per box geometry
# (`setup!`) and every count lives on the device (`counts`). Build, prune and
# force launch at capacity and stop at the device counts, with no host reads
# and no allocations, so the sequence can be captured in a CUDA graph. The
# super-cluster count has the hard bound ⌈N/64⌉ + columns, the working list is
# a subset of the outer list, and exclusion entries and scaled pairs are
# bounded by the exclusion list; only the outer-entry count has a capacity
# (`necap`). An overflow latches `counts[CL_ERR]` with the required size in
# `counts[CL_REQ]`; later list and force kernels then do nothing, and the
# owner resizes ([`ensure_capacity!`](@ref)) and rebuilds at a boundary.

const CLUSTER = 8
const SUPER = 64

# device counters
const CL_NS = 1        # super-clusters in use
const CL_NE = 2        # outer-list entries
const CL_NP = 3        # scaled/out-of-list exception pairs
const CL_UNSORTED = 4  # a super-cluster segment longer than the bitonic sort
const CL_ERR = 5       # outer-list capacity exceeded
const CL_REQ = 6       # outer entries required when exceeded
# device list decision: this step's actions, a forced-build request
# (first build, box change), and the build/prune totals for diagnostics
const CL_BUILD = 7
const CL_PRUNE = 8
const CL_REQUEST = 9
const CL_NBUILD = 10
const CL_NPRUNE = 11
const CL_DSTEP = 12   # device decisions so far (the step of fixed list intervals)
const CL_LEN = 12
# CL_ERR codes
const CL_ERR_CAPACITY = Int32(1)
const CL_ERR_NONFINITE = Int32(2)

# Persistent hierarchical inclusive scan (Int32) of an input of fixed length.
struct DeviceScan
    values::Vector{CuVector{Int32}}
    totals::Vector{CuVector{Int32}}
end
function DeviceScan(n::Integer)
    values = CuVector{Int32}[]; totals = CuVector{Int32}[]
    n = max(Int(n), 1)
    while true
        push!(values, CUDA.zeros(Int32, n))
        blocks = cld(n, 256)
        push!(totals, CUDA.zeros(Int32, blocks))
        blocks == 1 && break
        n = blocks
    end
    return DeviceScan(values, totals)
end

function _scan_tiles_kernel!(values, totals, input)
    t = threadIdx().x
    i = (blockIdx().x - Int32(1)) * blockDim().x + t
    shared = CuStaticSharedArray(Int32, 256)
    @inbounds shared[t] = i <= length(input) ? input[i] : Int32(0)
    sync_threads()
    offset = 1
    while offset < 256
        @inbounds addend = t > offset ? shared[t - offset] : Int32(0)
        sync_threads()
        @inbounds shared[t] += addend
        sync_threads()
        offset *= 2
    end
    i <= length(values) && (@inbounds values[i] = shared[t])
    t == 256 && (@inbounds totals[blockIdx().x] = shared[t])
    return
end

function _scan_add_kernel!(values, scanned_totals)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(values) && blockIdx().x > 1
        @inbounds values[i] += scanned_totals[blockIdx().x - 1]
    end
    return
end

"""Inclusive scan of `input` (same length as the workspace) into `sc.values[1]`."""
function scan!(sc::DeviceScan, input)
    for level in eachindex(sc.values)
        x = level == 1 ? input : sc.totals[level - 1]
        @cuda threads=256 blocks=cld(length(sc.values[level]), 256) _scan_tiles_kernel!(sc.values[level], sc.totals[level], x)
    end
    for level in (length(sc.values) - 1):-1:1
        @cuda threads=256 blocks=cld(length(sc.values[level]), 256) _scan_add_kernel!(sc.values[level], sc.values[level + 1])
    end
    return sc.values[1]
end

"""Smallest box-length ratio (new/at setup) the cluster grid covers without a host setup."""
const GEOMETRY_SHRINK = 0.98

mutable struct ClusterPairList{T}
    N::Int
    box::NTuple{3,T}         # host copy of the box (`setup!`)
    geom::CuVector{T}        # device box lengths read by the kernels; an engine run shares its own
    rlist::T
    # geometry, fixed per box (`setup!`)
    ncx::Int32; ncy::Int32; nzb::Int32; mx::Int32; my::Int32
    ncol::Int; nbin::Int; NSmax::Int; nslot::Int
    necap::Int            # outer-entry capacity
    nexcap::Int           # exclusion-mask entries (bounded by the exclusion list)
    npmax::Int            # exception pairs (length of the exclusion list)
    counts::CuVector{Int32}
    # clustering scratch
    bin::CuVector{Int32}; col::CuVector{Int32}; bcnt::CuVector{Int32}; cursor::CuVector{Int32}
    border::CuVector{Int32}; cnt::CuVector{Int32}; nsc::CuVector{Int32}
    colstart::CuVector{Int32}; col_sc_off::CuVector{Int32}
    sc_first::CuVector{Int32}; sc_n::CuVector{Int32}; sc_col::CuVector{Int32}
    ones::CuVector{Int32}; dummy::CuVector{T}; occ::CuVector{Int32}
    perm::CuVector{Int32}           # slot → particle (0 = padding)
    slot_of::CuVector{Int32}        # particle → slot (0-based)
    order::CuVector{Int32}          # particles in slot order (mesh traversal order)
    # Cluster-ordered force coordinates refreshed in the centered image each
    # evaluation. Build-time [0,L) coordinates are only for list geometry;
    # force differences use compensated MIC rather than translated positions.
    xs::CuVector{T}; ys::CuVector{T}; zs::CuVector{T}; qs::CuVector{T}; ts::CuVector{Int32}
    xw0::CuVector{T}; yw0::CuVector{T}; zw0::CuVector{T}
    s6::CuVector{T}; s12::CuVector{T}
    # bounding boxes at build time
    bcx::CuVector{T}; bcy::CuVector{T}; bcz::CuVector{T}; bhx::CuVector{T}; bhy::CuVector{T}; bhz::CuVector{T}
    sbcx::CuVector{T}; sbcy::CuVector{T}; sbcz::CuVector{T}; sbhx::CuVector{T}; sbhy::CuVector{T}; sbhz::CuVector{T}
    # outer list
    cntS::CuVector{Int32}            # entries per (super-cluster, column slot), `ncs` slots each
    scio::CuVector{Int32}
    cjo::CuVector{UInt32}
    xido::CuVector{Int32}
    flag::CuVector{Int32}
    masks::CuVector{UInt64}
    m8o::CuVector{UInt8}             # per-warp masks of the outer list (pruning output)
    full::CuVector{UInt8}            # all-live masks of the outer list (build)
    # the whole outer list in working format (pruning pass input)
    cjf::CuVector{UInt32}
    m8f::CuVector{UInt8}
    scif::CuVector{Int32}; xidf::CuVector{Int32}     # its offsets/ids (equal to scio/xido)
    # working list
    cntc::CuVector{Int32}
    sciw::CuVector{Int32}
    cjw::CuVector{UInt32}
    m8w::CuVector{UInt8}
    xidw::CuVector{Int32}
    # scaled pairs inside the list (coef = s) and exception pairs outside it (coef = s − 1)
    pi::CuVector{Int32}; pj::CuVector{Int32}; pc::CuVector{T}
    # persistent scans
    scan_bin::DeviceScan; scan_col::DeviceScan; scan_sc::DeviceScan; scan_e::DeviceScan; scan_slot::DeviceScan
    # reference coordinates for the rebuild and prune triggers
    rx0::CuVector{T}; ry0::CuVector{T}; rz0::CuVector{T}
    px0::CuVector{T}; py0::CuVector{T}; pz0::CuVector{T}
    flags::CuVector{UInt32}
    flags_host::Vector{UInt32}      # pinned
    prune_pending::Bool             # next force pass runs on the full list and prunes it
    nbuild::Int
    nprune::Int
    # fixed list intervals (GROMACS-style, opt-in): rebuild every `nstlist`, prune every `nstprune`
    # device decisions instead of the displacement triggers; 0 = displacement triggers (exact)
    nstlist::Int32
    nstprune::Int32
    prune_only::Bool                # pruning by a distance-only pass (`PRUNE_ONLY`)
    ncs::Int32                      # searched column slots per super-cluster (≤ capacity of `cntS`)
    scan_scs::DeviceScan            # scan over `cntS`
end

# Host reads for diagnostics and tests (never in the step).
function Base.getproperty(L::ClusterPairList, s::Symbol)
    s === :NS && return Int(Array(getfield(L, :counts))[CL_NS])
    s === :npairs && return Int(Array(getfield(L, :counts))[CL_NP])
    return getfield(L, s)
end

@inline _launch1(n::Integer, threads::Integer=256) = (threads, cld(max(Int(n), 1), threads))

@inline _wrapc(x::T, L::T) where {T} = x - L * floor(x / L)

@inline function _mic(d::T, L::T) where {T}
    h = T(0.5) * L
    d -= (d > h) * L
    d += (d < -h) * L
    return d
end

@inline function _shift_of(d::T, h::T) where {T}
    return ifelse(d > h, Int32(1), ifelse(d < -h, Int32(-1), Int32(0)))
end

@inline function _bbgap2(c1x::T, c1y::T, c1z::T, h1x::T, h1y::T, h1z::T,
                         c2x::T, c2y::T, c2z::T, h2x::T, h2y::T, h2z::T, Lx::T, Ly::T, Lz::T) where {T}
    gx = max(abs(_mic(c1x - c2x, Lx)) - h1x - h2x, zero(T))
    gy = max(abs(_mic(c1y - c2y, Ly)) - h1y - h2y, zero(T))
    gz = max(abs(_mic(c1z - c2z, Lz)) - h1z - h2z, zero(T))
    return gx * gx + gy * gy + gz * gz
end

# ---------- clustering ----------
# Particles are ordered by (x–y column, z bin) with a counting sort; z bins
# hold ~4 particles, so cutting a column into super-clusters of 64 in this
# order gives z-slabs whose boundaries are fuzzy by at most one bin.
function _column_key_kernel!(bin, col, counts, rx, ry, rz, box::Union{NTuple{3,T},AbstractVector{T}}, ncx::Int32, ncy::Int32, nzb::Int32) where {T}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        x = _wrapc(rx[i], Lx); y = _wrapc(ry[i], Ly); z = _wrapc(rz[i], Lz)
        if !(isfinite(x) & isfinite(y) & isfinite(z))   # keep the bin in range; the list is invalid
            counts[CL_ERR] = CL_ERR_NONFINITE
            x = zero(T); y = zero(T); z = zero(T)
        end
        cx = min(unsafe_trunc(Int32, x / Lx * T(ncx)), ncx - Int32(1))
        cy = min(unsafe_trunc(Int32, y / Ly * T(ncy)), ncy - Int32(1))
        cz = min(unsafe_trunc(Int32, z / Lz * T(nzb)), nzb - Int32(1))
        c = cx + ncx * cy
        col[i] = c
        bin[i] = c * nzb + cz
    end
    return
end

function _bin_scatter_kernel!(order, cursor, bin)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(bin) && return
    @inbounds begin
        p = CUDA.atomic_add!(pointer(cursor, bin[i] + Int32(1)), Int32(1))
        order[p + Int32(1)] = Int32(i)
    end
    return
end

function _column_histogram_kernel!(cnt, col)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(col) && return
    @inbounds CUDA.@atomic cnt[col[i] + Int32(1)] += Int32(1)
    return
end

# exclusive = inclusive − own count
function _exclusive_kernel!(out, incl, cnt)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(out) && return
    @inbounds out[i] = incl[i] - cnt[i]
    return
end

function _super_count_kernel!(nsc, cnt)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    c > length(cnt) && return
    @inbounds nsc[c] = cld(cnt[c], Int32(SUPER))
    return
end

# col_sc_off = [0; inclusive scan of super-clusters per column]; NS = its last entry
function _super_offsets_kernel!(col_sc_off, incl, counts)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    c > length(incl) && return
    @inbounds begin
        c == 1 && (col_sc_off[1] = Int32(0))
        col_sc_off[c + 1] = incl[c]
        if c == length(incl)
            counts[CL_NS] = incl[c]
            counts[CL_NP] = Int32(0); counts[CL_UNSORTED] = Int32(0)
        end
    end
    return
end

function _super_meta_kernel!(sc_first, sc_n, sc_col, colstart, cnt, col_sc_off)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    c > length(cnt) && return
    @inbounds begin
        n = cnt[c]
        s0 = col_sc_off[c]
        k = Int32(0)
        while k * Int32(SUPER) < n
            S = s0 + k + Int32(1)
            sc_first[S] = colstart[c] + k * Int32(SUPER)
            sc_n[S] = min(Int32(SUPER), n - k * Int32(SUPER))
            sc_col[S] = c - Int32(1)
            k += Int32(1)
        end
    end
    return
end

# One block of 64 threads per super-cluster: x halves, y halves, z halves.
function _cluster_assign_kernel!(perm, slot_of, order, sc_first, sc_n, counts, rx, ry, rz, box::Union{NTuple{3,T},AbstractVector{T}}) where {T}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    S = blockIdx().x
    t = threadIdx().x - Int32(1)
    sx = CuStaticSharedArray(T, SUPER); sy = CuStaticSharedArray(T, SUPER); sz = CuStaticSharedArray(T, SUPER)
    shx = CuStaticSharedArray(Int32, SUPER); shy = CuStaticSharedArray(Int32, SUPER)
    @inbounds S > counts[CL_NS] && return
    @inbounds begin
        n = sc_n[S]
        i = Int32(0)
        if t < n
            i = Int32(order[sc_first[S] + t + Int32(1)])
            sx[t + 1] = _wrapc(rx[i], Lx); sy[t + 1] = _wrapc(ry[i], Ly); sz[t + 1] = _wrapc(rz[i], Lz)
        end
        sync_threads()
        hx = Int32(0); hy = Int32(0)
        if t < n
            xt = sx[t + 1]; r = Int32(0)
            for u in Int32(0):(n - Int32(1))
                xu = sx[u + 1]
                r += ((xu < xt) | ((xu == xt) & (u < t))) ? Int32(1) : Int32(0)
            end
            hx = r >= (n + Int32(1)) >> 1 ? Int32(1) : Int32(0)
            shx[t + 1] = hx
        end
        sync_threads()
        if t < n
            m = Int32(0); r = Int32(0); yt = sy[t + 1]
            for u in Int32(0):(n - Int32(1))
                if shx[u + 1] == hx
                    m += Int32(1)
                    yu = sy[u + 1]
                    r += ((yu < yt) | ((yu == yt) & (u < t))) ? Int32(1) : Int32(0)
                end
            end
            hy = r >= (m + Int32(1)) >> 1 ? Int32(1) : Int32(0)
            shy[t + 1] = hy
        end
        sync_threads()
        if t < n
            m = Int32(0); r = Int32(0); zt = sz[t + 1]
            for u in Int32(0):(n - Int32(1))
                if (shx[u + 1] == hx) & (shy[u + 1] == hy)
                    m += Int32(1)
                    zu = sz[u + 1]
                    r += ((zu < zt) | ((zu == zt) & (u < t))) ? Int32(1) : Int32(0)
                end
            end
            m0 = (m + Int32(1)) >> 1
            hz = r >= m0 ? Int32(1) : Int32(0)
            pos = r - hz * m0
            slot = (S - Int32(1)) * Int32(SUPER) + (hx * Int32(4) + hy * Int32(2) + hz) * Int32(CLUSTER) + pos
            perm[slot + 1] = i
            slot_of[i] = slot
        end
    end
    return
end

# particles in slot order (mesh traversal): order[rank of occupied slot s] = perm[s]
function _occupancy_kernel!(occ, perm)
    s = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s > length(perm) && return
    @inbounds occ[s] = perm[s] != Int32(0) ? Int32(1) : Int32(0)
    return
end
function _order_kernel!(order, perm, rank)
    s = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s > length(perm) && return
    @inbounds begin
        i = perm[s]
        i != Int32(0) && (order[rank[s]] = i)
    end
    return
end

function _pack_kernel!(xs, ys, zs, qs, ts, s6s, s12s, perm, rx, ry, rz, q, typ, s6t, s12t,
                       xw0, yw0, zw0, rx0, ry0, rz0,
                       box::Union{NTuple{3,T},AbstractVector{T}}, ::Val{CHARGED}, ::Val{SEP}, ::Val{BUILD},
                       ::Val{MIX}) where {T,CHARGED,SEP,BUILD,MIX}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    s = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    s > length(perm) && return
    @inbounds begin
        i = perm[s]
        if i == Int32(0)
            xs[s] = T(NaN); ys[s] = T(NaN); zs[s] = T(NaN); qs[s] = zero(T); ts[s] = Int32(1)
            if SEP || MIX
                s6s[s] = zero(T); s12s[s] = zero(T)
            end
        else
            if BUILD
                xs[s] = _wrapc(rx[i], Lx); ys[s] = _wrapc(ry[i], Ly); zs[s] = _wrapc(rz[i], Lz)
            else
                # Force coordinates use their centered image. Build/prune
                # membership is periodic; carrying the build-time [0,L) image
                # here needlessly rounds both the image and its displacement.
                xs[s] = force_coordinate(rx[i], Lx)
                ys[s] = force_coordinate(ry[i], Ly)
                zs[s] = force_coordinate(rz[i], Lz)
            end
            ty = typ[i]
            qs[s] = CHARGED ? q[i] : zero(T)
            ts[s] = ty
            if SEP
                s6s[s] = s6t[ty]; s12s[s] = s12t[ty]
            elseif MIX   # per-particle σ (s6t holds σ by particle)
                s6s[s] = s6t[i]
            end
        end
    end
    return
end

function _bbox_kernel!(bcx, bcy, bcz, bhx, bhy, bhz, xs, ys, zs)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    c > length(bcx) && return
    T = eltype(xs)
    @inbounds begin
        lx = T(Inf); ly = T(Inf); lz = T(Inf); ux = T(-Inf); uy = T(-Inf); uz = T(-Inf)
        for k in 1:CLUSTER
            s = (c - 1) * CLUSTER + k
            x = xs[s]
            if !isnan(x)
                y = ys[s]; z = zs[s]
                lx = min(lx, x); ux = max(ux, x); ly = min(ly, y); uy = max(uy, y); lz = min(lz, z); uz = max(uz, z)
            end
        end
        if lx == T(Inf)
            bcx[c] = T(NaN); bcy[c] = T(NaN); bcz[c] = T(NaN)
            bhx[c] = zero(T); bhy[c] = zero(T); bhz[c] = zero(T)
        else
            bcx[c] = T(0.5) * (lx + ux); bcy[c] = T(0.5) * (ly + uy); bcz[c] = T(0.5) * (lz + uz)
            bhx[c] = T(0.5) * (ux - lx); bhy[c] = T(0.5) * (uy - ly); bhz[c] = T(0.5) * (uz - lz)
        end
    end
    return
end

function _super_bbox_kernel!(sbcx, sbcy, sbcz, sbhx, sbhy, sbhz, bcx, bcy, bcz, bhx, bhy, bhz)
    S = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    S > length(sbcx) && return
    T = eltype(bcx)
    @inbounds begin
        lx = T(Inf); ly = T(Inf); lz = T(Inf); ux = T(-Inf); uy = T(-Inf); uz = T(-Inf)
        for k in 1:CLUSTER
            c = (S - 1) * CLUSTER + k
            x = bcx[c]
            if !isnan(x)
                lx = min(lx, x - bhx[c]); ux = max(ux, x + bhx[c])
                ly = min(ly, bcy[c] - bhy[c]); uy = max(uy, bcy[c] + bhy[c])
                lz = min(lz, bcz[c] - bhz[c]); uz = max(uz, bcz[c] + bhz[c])
            end
        end
        sbcx[S] = T(0.5) * (lx + ux); sbcy[S] = T(0.5) * (ly + uy); sbcz[S] = T(0.5) * (lz + uz)
        sbhx[S] = T(0.5) * (ux - lx); sbhy[S] = T(0.5) * (uy - ly); sbhz[S] = T(0.5) * (uz - lz)
    end
    return
end

# ---------- search ----------
# Warp-parallel search: one warp per (super-cluster, column slot), so that every
# super-cluster's neighbouring columns are searched concurrently. Lanes take
# candidate super-clusters of the column in parallel, each emits up to 8 entries,
# and a warp prefix sum places them. Slots follow the (dy, dx) column order and
# entries of a column come out in increasing cluster order, so the outer list is
# laid out as by a serial loop over the columns; `_sort_entries_kernel!` then
# orders each super-cluster's segment for the binary searches of the exclusion mapping.
@inline function _warp_exclusive_scan(v::Int32, lane::Int32)
    x = v
    for off in (1, 2, 4, 8, 16)
        y = shfl_up_sync(0xffffffff, x, off)
        lane >= Int32(off) && (x += y)
    end
    return x - v, shfl_sync(0xffffffff, x, 32)
end

function _search_warp_kernel!(cnt, incl, cjout, ncs::Int32, counts, sc_col, col_sc_off, ncx::Int32, ncy::Int32,
                              mx::Int32, my::Int32, bcx, bcy, bcz, bhx, bhy, bhz, sbcx, sbcy, sbcz, sbhx, sbhy, sbhz,
                              box::Union{NTuple{3,T},AbstractVector{T}}, rl2::T, ::Val{FILL}) where {T,FILL}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    gw = (blockIdx().x - Int32(1)) * (blockDim().x >> Int32(5)) + ((threadIdx().x - Int32(1)) >> Int32(5))
    S = gw ÷ ncs + Int32(1); k = gw % ncs
    @inbounds S > counts[CL_NS] && return
    @inbounds (FILL && counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        c = sc_col[S]
        cx = c % ncx; cy = c ÷ ncx
        nxr = min(Int32(2) * mx + Int32(1), ncx); nyr = min(Int32(2) * my + Int32(1), ncy)
        k >= nxr * nyr && return
        dy = k ÷ nxr; dx = k % nxr
        x0 = nxr == ncx ? Int32(0) : cx - mx
        y0 = nyr == ncy ? Int32(0) : cy - my
        slot = gw + Int32(1)
        out = FILL ? incl[slot] - cnt[slot] : Int32(0)   # exclusive offset of this slot
        n = Int32(0)
        s1x = sbcx[S]; s1y = sbcy[S]; s1z = sbcz[S]; h1x = sbhx[S]; h1y = sbhy[S]; h1z = sbhz[S]
        ci0 = (S - Int32(1)) * Int32(CLUSTER)
        cc = mod(x0 + dx, ncx) + ncx * mod(y0 + dy, ncy) + Int32(1)
        lo = col_sc_off[cc] + Int32(1); hi = col_sc_off[cc + Int32(1)]
        base = lo
        while base <= hi
            S2 = base + lane
            ok = (S2 <= hi) && (S2 >= S)
            if ok
                ok = _bbgap2(s1x, s1y, s1z, h1x, h1y, h1z, sbcx[S2], sbcy[S2], sbcz[S2],
                             sbhx[S2], sbhy[S2], sbhz[S2], Lx, Ly, Lz) < rl2
            end
            # masks of this lane's 8 j-clusters, packed 8 bits each
            mlo = UInt32(0); mhi = UInt32(0); k8 = Int32(0)
            if ok
                for j in Int32(1):Int32(CLUSTER)
                    cj = (S2 - Int32(1)) * Int32(CLUSTER) + j
                    jx = bcx[cj]
                    m = UInt32(0)
                    if !isnan(jx)
                        jy = bcy[cj]; jz = bcz[cj]; hjx = bhx[cj]; hjy = bhy[cj]; hjz = bhz[cj]
                        for kk in Int32(1):Int32(CLUSTER)
                            ci = ci0 + kk
                            cj >= ci || continue
                            ix = bcx[ci]
                            isnan(ix) && continue
                            if _bbgap2(ix, bcy[ci], bcz[ci], bhx[ci], bhy[ci], bhz[ci], jx, jy, jz, hjx, hjy, hjz,
                                       Lx, Ly, Lz) < rl2
                                m |= UInt32(1) << (kk - Int32(1))
                            end
                        end
                    end
                    if j <= Int32(4)
                        mlo |= m << (Int32(8) * (j - Int32(1)))
                    else
                        mhi |= m << (Int32(8) * (j - Int32(5)))
                    end
                    k8 += (m != UInt32(0)) ? Int32(1) : Int32(0)
                end
            end
            off, total = _warp_exclusive_scan(k8, lane)
            if FILL && k8 > Int32(0)
                w = out + n + off
                for j in Int32(1):Int32(CLUSTER)
                    m = j <= Int32(4) ? (mlo >> (Int32(8) * (j - Int32(1)))) & UInt32(0xff) :
                                        (mhi >> (Int32(8) * (j - Int32(5)))) & UInt32(0xff)
                    if m != UInt32(0)
                        w += Int32(1)
                        cj = (S2 - Int32(1)) * Int32(CLUSTER) + j
                        cjout[w] = (UInt32(cj - Int32(1)) << 9) | m
                    end
                end
            end
            n += total
            base += Int32(32)
        end
        (!FILL && lane == Int32(0)) && (cnt[slot] = n)
    end
    return
end

# scio = [0; entries up to and including each super-cluster] from the inclusive scan
# over (super-cluster, column slot) counts; NE = the last value, checked against the capacity
function _entry_offsets_kernel!(scio, incl, counts, necap::Int32, ncs::Int32)
    S = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    S > length(scio) - 1 && return
    @inbounds begin
        S == 1 && (scio[1] = Int32(0))
        scio[S + 1] = incl[S * ncs]
        if S == length(scio) - 1
            ne = incl[S * ncs]
            counts[CL_NE] = ne
            if ne > necap && counts[CL_ERR] == Int32(0)
                counts[CL_ERR] = CL_ERR_CAPACITY; counts[CL_REQ] = ne
            end
        end
    end
    return
end

# Bitonic sort of each super-cluster's entry segment (≤ 1024 entries) by
# cluster index; longer segments are left unsorted and flagged.
function _sort_entries_kernel!(cjo, scio, counts)
    S = blockIdx().x
    t = threadIdx().x
    sh = CuStaticSharedArray(UInt32, 1024)
    @inbounds (S > counts[CL_NS] || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        e0 = scio[S]; n = scio[S + 1] - e0
        if n > Int32(1024)
            t == Int32(1) && (counts[CL_UNSORTED] = Int32(1))
            return
        end
        n <= Int32(1) && return
        P = Int32(1)
        while P < n
            P <<= 1
        end
        k = t
        while k <= P
            sh[k] = k <= n ? cjo[e0 + k] : typemax(UInt32)
            k += blockDim().x
        end
        sync_threads()
        size = Int32(2)
        while size <= P
            stride = size >> 1
            while stride > Int32(0)
                k = t - Int32(1)
                while k < P
                    l = xor(k, stride)
                    if l > k
                        a = sh[k + 1]; b = sh[l + 1]
                        up = (k & size) == Int32(0)
                        ((a > b) == up) && (sh[k + 1] = b; sh[l + 1] = a)
                    end
                    k += blockDim().x
                end
                sync_threads()
                stride >>= 1
            end
            size <<= 1
        end
        k = t
        while k <= n
            cjo[e0 + k] = sh[k]
            k += blockDim().x
        end
    end
    return
end

# ---------- exclusions ----------
# Canonical orientation: the pair is stored under the lower cluster (lower
# slot inside one cluster). PASS 1 flags entries, PASS 2 sets mask bits and
# collects scaled in-list pairs (coef s) and out-of-list pairs (coef s − 1).
function _exclusion_kernel!(flag, masks, xid, counts, pi, pj, pc, exo, exj, exs, slot_of, scio, cjo,
                            ::Val{PASS}) where {PASS}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(slot_of) && return
    @inbounds counts[CL_ERR] != Int32(0) && return
    @inbounds begin
        sorted = counts[CL_UNSORTED] == Int32(0)
        si = slot_of[i]
        for p in exo[i]:(exo[i + 1] - Int32(1))
            j = exj[p]
            s = exs[p]
            sj = slot_of[j]
            ci = si >> 3; cj = sj >> 3
            ((ci < cj) | ((ci == cj) & ((si & Int32(7)) < (sj & Int32(7))))) || continue
            S = ci >> 3; k = ci & Int32(7)
            e = Int32(-1)
            if sorted
                lo = scio[S + 1] + Int32(1); hi = scio[S + 2]
                while lo <= hi
                    mid = (lo + hi) >> 1
                    v = cjo[mid] >> 9
                    if v < UInt32(cj)
                        lo = mid + Int32(1)
                    elseif v > UInt32(cj)
                        hi = mid - Int32(1)
                    else
                        ((cjo[mid] >> k) & UInt32(1)) != UInt32(0) && (e = mid)
                        break
                    end
                end
            else
                for q in (scio[S + 1] + Int32(1)):scio[S + 2]
                    ent = cjo[q]
                    if (ent >> 9) == UInt32(cj) && ((ent >> k) & UInt32(1)) != UInt32(0)
                        e = q
                        break
                    end
                end
            end
            if PASS == 1
                e > 0 && (flag[e] = Int32(1))
            elseif e > 0
                bit = UInt64(1) << ((si & Int32(7)) * Int32(8) + (sj & Int32(7)))
                CUDA.@atomic masks[(xid[e] - Int32(1)) * Int32(8) + k + Int32(1)] |= bit
                if s != zero(s)
                    n = CUDA.@atomic counts[CL_NP] += Int32(1)
                    pi[n + 1] = i; pj[n + 1] = j; pc[n + 1] = s
                end
            else
                n = CUDA.@atomic counts[CL_NP] += Int32(1)
                pi[n + 1] = i; pj[n + 1] = j; pc[n + 1] = s - one(s)
            end
        end
    end
    return
end

function _flag_entries_kernel!(cjo, flag, xid, counts)
    e = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds (e > counts[CL_NE] || counts[CL_ERR] != Int32(0)) && return
    @inbounds if flag[e] != Int32(0)
        cjo[e] |= UInt32(1) << 8
    else
        xid[e] = Int32(0)
    end
    return
end

# ---------- pruning ----------
# Pruning happens inside the force kernel (`Val(PRUNE)` on the full list): it
# writes per-warp masks of the tiles that hold a pair inside the prune radius
# or an excluded pair. `_compact_warp_kernel!` then turns the outer list plus
# such masks into a working list: one warp per super-cluster, ballots over
# chunks of 32 entries, and the periodic shift of each kept j-cluster.

function _duplicate_masks_kernel!(m8, cjo, counts)
    e = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds (e > counts[CL_NE] || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        m = UInt8(cjo[e] & UInt32(0xff))
        m8[2e - 1] = m; m8[2e] = m
    end
    return
end

function _compact_warp_kernel!(cnt, cjw, m8w, xidw, sciw, counts, cjo, m8, xido, scio, bcx, bcy, bcz, sbcx, sbcy, sbcz,
                               box::Union{NTuple{3,T},AbstractVector{T}}, ::Val{FILL}) where {T,FILL}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    S = (blockIdx().x - Int32(1)) * (blockDim().x >> Int32(5)) + ((threadIdx().x - Int32(1)) >> Int32(5)) + Int32(1)
    @inbounds (S > counts[CL_NS] || counts[CL_ERR] != Int32(0)) && return
    @inbounds begin
        e0 = scio[S]; e1 = scio[S + 1]
        out = FILL ? sciw[S] : Int32(0)
        n = Int32(0)
        hx = T(0.5) * Lx; hy = T(0.5) * Ly; hz = T(0.5) * Lz
        cSx = sbcx[S]; cSy = sbcy[S]; cSz = sbcz[S]
        base = e0
        while base < e1
            e = base + lane + Int32(1)
            a = UInt8(0); b = UInt8(0)
            if e <= e1
                a = m8[2e - 1]; b = m8[2e]
            end
            live = (a | b) != UInt8(0)
            bal = vote_ballot_sync(0xffffffff, live)
            if FILL && live
                pos = out + n + Int32(count_ones(bal & ((UInt32(1) << lane) - UInt32(1)))) + Int32(1)
                ent = cjo[e]
                cj = ent >> 9
                c = Int32(cj) + Int32(1)
                ix = Int32(1) + _shift_of(cSx - bcx[c], hx)
                iy = Int32(1) + _shift_of(cSy - bcy[c], hy)
                iz = Int32(1) + _shift_of(cSz - bcz[c], hz)
                sh = UInt32(ix + Int32(3) * iy + Int32(9) * iz)
                cjw[pos] = (cj << 6) | (sh << 1) | ((ent >> 8) & UInt32(1))
                m8w[2pos - 1] = a; m8w[2pos] = b
                xidw[pos] = xido[e]
            end
            n += Int32(count_ones(bal))
            base += Int32(32)
        end
        (!FILL && lane == Int32(0)) && (cnt[S] = n)
    end
    return
end

# sci = [0; inclusive scan]
function _offsets_kernel!(sci, incl)
    S = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    S > length(incl) && return
    @inbounds begin
        S == 1 && (sci[1] = Int32(0))
        sci[S + 1] = incl[S]
    end
    return
end

# ---------- trigger ----------
# One kernel per evaluation: per particle the displacement ratios since the
# build (÷ half skin) and since the last prune (÷ half prune buffer), reduced
# per block and combined with an integer atomic max (non-negative floats order
# like their bit patterns) into flags[1] = max of both ratios, flags[2] =
# build ratio. The host reads the two words from pinned memory.
@inline _ratio(d2::T, thr2::T) where {T} = thr2 > zero(T) ? d2 / thr2 : ifelse(d2 > zero(T), T(Inf), zero(T))

@inline function _warp_max(v::Float32)
    for off in (16, 8, 4, 2, 1)
        v = max(v, shfl_down_sync(0xffffffff, v, off))
    end
    return v
end

function _displacement_kernel!(flags, rx, ry, rz, rx0, ry0, rz0, px0, py0, pz0, box::Union{NTuple{3,T},AbstractVector{T}},
                               tb2::T, tp2::T) where {T}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    rb = 0.0f0; rp = 0.0f0
    @inbounds if i <= length(rx)
        x = rx[i]; y = ry[i]; z = rz[i]
        dx = _mic(x - rx0[i], Lx); dy = _mic(y - ry0[i], Ly); dz = _mic(z - rz0[i], Lz)
        rb = Float32(_ratio(dx * dx + dy * dy + dz * dz, tb2))
        isfinite(rb) || (rb = Inf32)       # non-finite motion: rebuild (which flags it)
        dx = _mic(x - px0[i], Lx); dy = _mic(y - py0[i], Ly); dz = _mic(z - pz0[i], Lz)
        rp = Float32(_ratio(dx * dx + dy * dy + dz * dz, tp2))
    end
    ra = _warp_max(max(rb, rp)); rb = _warp_max(rb)
    sh = CuStaticSharedArray(Float32, (2, 32))
    lane = (threadIdx().x - Int32(1)) & Int32(31); w = (threadIdx().x - Int32(1)) >> Int32(5)
    if lane == Int32(0)
        @inbounds sh[1, w + 1] = ra
        @inbounds sh[2, w + 1] = rb
    end
    sync_threads()
    if w == Int32(0)
        nw = (blockDim().x + Int32(31)) >> Int32(5)
        a = lane < nw ? (@inbounds sh[1, lane + 1]) : 0.0f0
        c = lane < nw ? (@inbounds sh[2, lane + 1]) : 0.0f0
        a = _warp_max(a); c = _warp_max(c)
        if lane == Int32(0)
            CUDA.atomic_max!(pointer(flags, 1), reinterpret(UInt32, a))
            CUDA.atomic_max!(pointer(flags, 2), reinterpret(UInt32, c))
        end
    end
    return
end

# Fills and copies as kernels: build and prune run inside conditional graph bodies,
# where memset/memcpy nodes are avoided (the ELL engine's bodies are kernels only;
# Nsight's per-node graph tracing failed on the memset/memcpy form, 13 Oct).
function _fill_kernel!(a, v)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= length(a) && (@inbounds a[i] = v)
    return
end
function _copy_kernel!(dst, src)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= length(dst) && (@inbounds dst[i] = src[i])
    return
end
kfill!(a, v) = (@cuda threads=256 blocks=cld(max(length(a), 1), 256) _fill_kernel!(a, convert(eltype(a), v)); a)
kcopy!(dst, src) = (@cuda threads=256 blocks=cld(max(length(dst), 1), 256) _copy_kernel!(dst, src); dst)

# ---------- storage ----------

"""
    ClusterPairList(N, box, T) -> ClusterPairList

Empty list; [`setup!`](@ref) sizes it for a box and [`build!`](@ref) fills it.
"""
function ClusterPairList(N::Integer, box::NTuple{3,<:Real}, ::Type{T}) where {T}
    e(::Type{S}) where {S} = CuVector{S}(undef, 0)
    es = DeviceScan(1)
    z = Int32(0)
    return ClusterPairList{T}(Int(N), T.(box), CuVector{T}(collect(T.(box))), zero(T), z, z, z, z, z, 0, 0, 0, 0, 0, 0, 0,
                              CUDA.zeros(Int32, CL_LEN),
                              e(Int32), e(Int32), e(Int32), e(Int32), e(Int32), e(Int32), e(Int32), e(Int32), e(Int32),
                              e(Int32), e(Int32), e(Int32), e(Int32), e(T), e(Int32),
                              e(Int32), CuVector{Int32}(undef, N), CuVector{Int32}(undef, N),
                              e(T), e(T), e(T), e(T), e(Int32), e(T), e(T), e(T), e(T), e(T),
                              e(T), e(T), e(T), e(T), e(T), e(T), e(T), e(T), e(T), e(T), e(T), e(T),
                              e(Int32), e(Int32), e(UInt32), e(Int32), e(Int32), e(UInt64), e(UInt8), e(UInt8),
                              e(UInt32), e(UInt8), e(Int32), e(Int32),
                              e(Int32), e(Int32), e(UInt32), e(UInt8), e(Int32),
                              e(Int32), e(Int32), e(T),
                              es, es, es, es, es,
                              CuVector{T}(undef, N), CuVector{T}(undef, N), CuVector{T}(undef, N),
                              CuVector{T}(undef, N), CuVector{T}(undef, N), CuVector{T}(undef, N),
                              CUDA.zeros(UInt32, 2), CUDA.pin(zeros(UInt32, 2)), false, 0, 0, Int32(0), Int32(0), false,
                              Int32(0), es)
end

"""
    setup!(list, box, rlist, npmax) -> Bool

Size every buffer for this box and list radius (host boundary; allocates when
sizes change) and request a rebuild. Returns whether storage was reallocated
(graphs that captured it must be recaptured; a box change needs a fresh capture
of its geometry scalars either way).
"""
function setup!(L::ClusterPairList{T}, box::NTuple{3,<:Real}, rlist::Real, npmax::Integer) where {T}
    N = L.N
    Lx, Ly, Lz = T.(box)
    min(Lx, Ly, Lz) > 2 * rlist ||
        throw(ArgumentError("cluster-pair list radius $(rlist) needs every box length above $(2rlist)."))
    same = getfield(L, :nslot) > 0 && L.box == (Lx, Ly, Lz) && L.rlist == T(rlist) && L.npmax == max(Int(npmax), 0)
    same && return false
    rho = N / (Float64(Lx) * Float64(Ly) * Float64(Lz))
    b = cbrt(SUPER / rho)
    ncx = Int32(max(1, floor(Int, Lx / b))); ncy = Int32(max(1, floor(Int, Ly / b)))
    ncol = Int(ncx) * Int(ncy)
    nzb = Int32(max(1, round(Int, Float64(Lz) * rho * (Float64(Lx) / ncx) * (Float64(Ly) / ncy) / 4)))
    nbin = ncol * Int(nzb)
    nbin < typemax(Int32) || throw(ArgumentError("cluster-pair binning exceeds Int32 range."))
    NSmax = cld(N, SUPER) + ncol
    nslot = SUPER * NSmax
    # Box changes that keep every buffer size (barostat events) only update the
    # geometry scalars; graphs that captured them are re-instantiated by the owner.
    keep = getfield(L, :nslot) > 0 && (ncx, ncy, nzb) == (L.ncx, L.ncy, L.nzb) && L.npmax == max(Int(npmax), 0)
    L.box = (Lx, Ly, Lz); copyto!(L.geom, T[Lx, Ly, Lz])
    L.rlist = T(rlist)
    L.ncx = ncx; L.ncy = ncy; L.nzb = nzb; L.ncol = ncol; L.nbin = nbin
    # column search span with a 2% shrink allowance: the device box may change between
    # host setups (device barostat); a larger change is caught on the device
    L.mx = Int32(floor(rlist / (GEOMETRY_SHRINK * Lx / ncx)) + 1)
    L.my = Int32(floor(rlist / (GEOMETRY_SHRINK * Ly / ncy)) + 1)
    L.ncs = Int32(min(2 * L.mx + 1, ncx) * min(2 * L.my + 1, ncy))
    # a wider column span than the count buffers hold needs new buffers (and graphs)
    keep = keep && Int(L.ncs) * L.NSmax <= length(L.cntS)
    if keep
        request_build!(L)
        return false
    end
    L.NSmax = NSmax; L.nslot = nslot
    L.bin = CuVector{Int32}(undef, N); L.col = CuVector{Int32}(undef, N)
    L.bcnt = CUDA.zeros(Int32, nbin); L.cursor = CUDA.zeros(Int32, nbin); L.border = CuVector{Int32}(undef, N)
    L.cnt = CUDA.zeros(Int32, ncol); L.nsc = CUDA.zeros(Int32, ncol)
    L.colstart = CUDA.zeros(Int32, ncol); L.col_sc_off = CUDA.zeros(Int32, ncol + 1)
    L.sc_first = CUDA.zeros(Int32, NSmax); L.sc_n = CUDA.zeros(Int32, NSmax); L.sc_col = CUDA.zeros(Int32, NSmax)
    L.ones = CUDA.ones(Int32, N); L.dummy = CuVector{T}(undef, 1); L.occ = CUDA.zeros(Int32, nslot)
    L.perm = CUDA.zeros(Int32, nslot)
    L.xs = CuVector{T}(undef, nslot); L.ys = similar(L.xs); L.zs = similar(L.xs); L.qs = similar(L.xs)
    L.ts = CuVector{Int32}(undef, nslot); L.s6 = similar(L.xs); L.s12 = similar(L.xs)
    L.xw0 = similar(L.xs); L.yw0 = similar(L.xs); L.zw0 = similar(L.xs)
    NC = CLUSTER * NSmax
    L.bcx = CuVector{T}(undef, NC); L.bcy = similar(L.bcx); L.bcz = similar(L.bcx)
    L.bhx = similar(L.bcx); L.bhy = similar(L.bcx); L.bhz = similar(L.bcx)
    L.sbcx = CuVector{T}(undef, NSmax); L.sbcy = similar(L.sbcx); L.sbcz = similar(L.sbcx)
    L.sbhx = similar(L.sbcx); L.sbhy = similar(L.sbcx); L.sbhz = similar(L.sbcx)
    # one more column per side than this box needs, so small barostat changes keep the buffers
    ncscap = min(2 * Int(L.mx) + 3, Int(ncx)) * min(2 * Int(L.my) + 3, Int(ncy))
    L.cntS = CUDA.zeros(Int32, NSmax * ncscap); L.scan_scs = DeviceScan(NSmax * ncscap)
    L.scio = CUDA.zeros(Int32, NSmax + 1)
    L.scif = CUDA.zeros(Int32, NSmax + 1); L.cntc = CUDA.zeros(Int32, NSmax); L.sciw = CUDA.zeros(Int32, NSmax + 1)
    L.npmax = max(Int(npmax), 0)
    np1 = max(L.npmax, 1)
    L.pi = CUDA.zeros(Int32, np1); L.pj = CUDA.zeros(Int32, np1); L.pc = CUDA.zeros(T, np1)
    L.scan_bin = DeviceScan(nbin); L.scan_col = DeviceScan(ncol); L.scan_sc = DeviceScan(NSmax)
    L.scan_slot = DeviceScan(nslot)
    _alloc_entries!(L, max(L.necap, 1))
    L.nbuild = 0
    request_build!(L)
    return true
end

# Outer/working entry storage for capacity `necap`.
function _alloc_entries!(L::ClusterPairList{T}, necap::Int) where {T}
    L.necap = necap
    L.nexcap = L.npmax > 0 ? min(necap, L.npmax) : 1
    L.cjo = CUDA.zeros(UInt32, necap); L.xido = CUDA.zeros(Int32, necap); L.flag = CUDA.zeros(Int32, necap)
    L.masks = CUDA.zeros(UInt64, CLUSTER * L.nexcap)
    L.m8o = CUDA.zeros(UInt8, 2necap); L.full = CUDA.zeros(UInt8, 2necap)
    L.cjf = CUDA.zeros(UInt32, necap); L.m8f = CUDA.zeros(UInt8, 2necap); L.xidf = CUDA.zeros(Int32, necap)
    L.cjw = CUDA.zeros(UInt32, necap); L.m8w = CUDA.zeros(UInt8, 2necap); L.xidw = CUDA.zeros(Int32, necap)
    L.scan_e = DeviceScan(necap)
    return L
end

"""
    ensure_capacity!(list) -> Bool

Host boundary check after a build: when the outer list overflowed, grow the
capacity to 1.25× the required entries (plus 1024) and return true (the
caller rebuilds and recaptures anything that captured the old storage).
"""
function ensure_capacity!(L::ClusterPairList)
    c = Array(getfield(L, :counts))
    c[CL_ERR] == 0 && return false
    c[CL_ERR] == CL_ERR_NONFINITE && throw(ArgumentError("Cluster-pair list: non-finite coordinates"))
    # The first sizing usually sees the most compact clusters a run will have (a lattice
    # or a minimized start), so the first allocation leaves 1.5× headroom; later growth 1.25×.
    growth = L.necap <= 1 ? 1.5 : 1.25
    _alloc_entries!(L, ceil(Int, growth * c[CL_REQ]) + 1024)
    request_build!(L)
    return true
end

"""
    build!(list, rx, ry, rz; exo, exj, exs)

Cluster the particles, search cluster pairs within the list radius, map the
exception CSR rows (storage indices) to bit masks, write the full-list masks
and record the reference coordinates. Device work only, at the geometry of
the last [`setup!`](@ref); an outer-list overflow latches `counts[CL_ERR]`.
"""
function build!(L::ClusterPairList{T}, rx, ry, rz; exo=nothing, exj=nothing, exs=nothing) where {T}
    N = L.N
    g = L.geom   # device box
    c = L.counts
    th, bl = _launch1(N)
    # counting sort by (column, z bin)
    @cuda threads=1 blocks=1 _fill_kernel!(view(c, CL_ERR:CL_ERR), Int32(0))
    @cuda threads=th blocks=bl _column_key_kernel!(L.bin, L.col, c, rx, ry, rz, g, L.ncx, L.ncy, L.nzb)
    kfill!(L.bcnt, Int32(0))
    @cuda threads=th blocks=bl _column_histogram_kernel!(L.bcnt, L.bin)
    incl = scan!(L.scan_bin, L.bcnt)
    tb, bb = _launch1(L.nbin)
    @cuda threads=tb blocks=bb _exclusive_kernel!(L.cursor, incl, L.bcnt)
    @cuda threads=th blocks=bl _bin_scatter_kernel!(L.border, L.cursor, L.bin)
    # columns → super-clusters
    kfill!(L.cnt, Int32(0))
    @cuda threads=th blocks=bl _column_histogram_kernel!(L.cnt, L.col)
    tc, bc = _launch1(L.ncol)
    incl = scan!(L.scan_col, L.cnt)
    @cuda threads=tc blocks=bc _exclusive_kernel!(L.colstart, incl, L.cnt)
    @cuda threads=tc blocks=bc _super_count_kernel!(L.nsc, L.cnt)
    incl = scan!(L.scan_col, L.nsc)
    @cuda threads=tc blocks=bc _super_offsets_kernel!(L.col_sc_off, incl, c)
    @cuda threads=tc blocks=bc _super_meta_kernel!(L.sc_first, L.sc_n, L.sc_col, L.colstart, L.cnt, L.col_sc_off)
    kfill!(L.perm, Int32(0))
    @cuda threads=SUPER blocks=L.NSmax _cluster_assign_kernel!(L.perm, L.slot_of, L.border, L.sc_first, L.sc_n, c,
                                                               rx, ry, rz, g)
    ts, bs = _launch1(L.nslot)
    @cuda threads=ts blocks=bs _occupancy_kernel!(L.occ, L.perm)
    rank = scan!(L.scan_slot, L.occ)
    @cuda threads=ts blocks=bs _order_kernel!(L.order, L.perm, rank)
    d = L.dummy
    @cuda threads=ts blocks=bs _pack_kernel!(L.xs, L.ys, L.zs, L.qs, L.ts, L.s6, L.s12, L.perm, rx, ry, rz, d,
                                             L.ones, d, d, d, d, d, d, d, d, g, Val(false), Val(false), Val(true),
                                             Val(false))
    kcopy!(L.xw0, L.xs); kcopy!(L.yw0, L.ys); kcopy!(L.zw0, L.zs)
    t4, b4 = _launch1(CLUSTER * L.NSmax)
    @cuda threads=t4 blocks=b4 _bbox_kernel!(L.bcx, L.bcy, L.bcz, L.bhx, L.bhy, L.bhz, L.xs, L.ys, L.zs)
    t5, b5 = _launch1(L.NSmax, 128)
    @cuda threads=t5 blocks=b5 _super_bbox_kernel!(L.sbcx, L.sbcy, L.sbcz, L.sbhx, L.sbhy, L.sbhz,
                                                   L.bcx, L.bcy, L.bcz, L.bhx, L.bhy, L.bhz)
    # search: counts, offsets (capacity check), fill, sort
    rl2 = L.rlist^2
    sargs = (c, L.sc_col, L.col_sc_off, L.ncx, L.ncy, L.mx, L.my, L.bcx, L.bcy, L.bcz, L.bhx, L.bhy, L.bhz,
             L.sbcx, L.sbcy, L.sbcz, L.sbhx, L.sbhy, L.sbhz, g, rl2)
    wblocks = cld(L.NSmax * Int(L.ncs), 4)       # one warp per (super-cluster, column slot), 4 per block
    kfill!(L.cntS, Int32(0))
    @cuda threads=128 blocks=wblocks _search_warp_kernel!(L.cntS, L.cntS, L.cjo, L.ncs, sargs..., Val(false))
    incl = scan!(L.scan_scs, L.cntS)
    t6, b6 = _launch1(L.NSmax)
    @cuda threads=t6 blocks=b6 _entry_offsets_kernel!(L.scio, incl, c, Int32(L.necap), L.ncs)
    @cuda threads=128 blocks=wblocks _search_warp_kernel!(L.cntS, incl, L.cjo, L.ncs, sargs..., Val(true))
    @cuda threads=256 blocks=L.NSmax _sort_entries_kernel!(L.cjo, L.scio, c)
    # exclusions
    te, be = _launch1(L.necap)
    if exo !== nothing && L.npmax > 0
        kfill!(L.flag, Int32(0)); kfill!(L.masks, UInt64(0))
        @cuda threads=th blocks=bl _exclusion_kernel!(L.flag, L.masks, L.xido, c, L.pi, L.pj, L.pc, exo, exj, exs,
                                                      L.slot_of, L.scio, L.cjo, Val(1))
        kcopy!(L.xido, scan!(L.scan_e, L.flag))
        @cuda threads=te blocks=be _flag_entries_kernel!(L.cjo, L.flag, L.xido, c)
        @cuda threads=th blocks=bl _exclusion_kernel!(L.flag, L.masks, L.xido, c, L.pi, L.pj, L.pc, exo, exj, exs,
                                                      L.slot_of, L.scio, L.cjo, Val(2))
    else
        kfill!(L.xido, Int32(0))
    end
    # the whole outer list in working format, every tile live
    @cuda threads=te blocks=be _duplicate_masks_kernel!(L.full, L.cjo, c)
    compact!(L, L.full, L.scif, L.cjf, L.m8f, L.xidf)
    L.prune_pending = true
    kcopy!(L.rx0, rx); kcopy!(L.ry0, ry); kcopy!(L.rz0, rz)
    return L
end

"""
    compact!(list, m8, sci, cj, m8w, xid)

Working-format list from the outer list and per-warp masks `m8` into the
given buffers (entries whose masks are both zero are dropped).
"""
function compact!(L::ClusterPairList{T}, m8, sci, cj, m8w, xid) where {T}
    blocks = cld(L.NSmax, 4)
    args = (L.counts, L.cjo, m8, L.xido, L.scio, L.bcx, L.bcy, L.bcz, L.sbcx, L.sbcy, L.sbcz, L.geom)
    kfill!(L.cntc, Int32(0))
    @cuda threads=128 blocks=blocks _compact_warp_kernel!(L.cntc, cj, m8w, xid, sci, args..., Val(false))
    incl = scan!(L.scan_sc, L.cntc)
    t, b = _launch1(L.NSmax)
    @cuda threads=t blocks=b _offsets_kernel!(sci, incl)
    @cuda threads=128 blocks=blocks _compact_warp_kernel!(L.cntc, cj, m8w, xid, sci, args..., Val(true))
    return nothing
end

"""
    request_build!(list)

Make the next device decision ([`enqueue_decide!`](@ref)) rebuild the list
(host boundary: setup, box change, capacity growth).
"""
request_build!(L::ClusterPairList) = (copyto!(view(getfield(L, :counts), CL_REQUEST:CL_REQUEST), Int32[1]); L)

"""
    request_prune!(list)

Make the next force pass run on the whole outer list and prune it (the
working list is replaced by the pruned list afterwards).
"""
function request_prune!(L::ClusterPairList)
    L.prune_pending = true
    return L
end

"""
    finish_prune!(list)

Replace the working list by the outer list compacted with the masks the
pruning force pass wrote.
"""
function finish_prune!(L::ClusterPairList)
    compact!(L, L.m8o, L.sciw, L.cjw, L.m8w, L.xidw)
    L.prune_pending = false
    L.nprune += 1
    return L
end

"""
    list_action(list, rx, ry, rz, half_skin, half_prune_buffer) -> :kept, :pruned or :built

`:built` when a particle moved more than `half_skin` since the build,
`:pruned` when one moved more than `half_prune_buffer` since the last prune.
"""
function list_action(L::ClusterPairList{T}, rx, ry, rz, half_skin::Real, half_prune::Real) where {T}
    th, bl = _launch1(L.N)
    fill!(L.flags, UInt32(0))
    @cuda threads=th blocks=bl _displacement_kernel!(L.flags, rx, ry, rz, L.rx0, L.ry0, L.rz0,
                                                     L.px0, L.py0, L.pz0, L.geom, T(half_skin)^2, T(half_prune)^2)
    copyto!(L.flags_host, L.flags)
    any_ratio = reinterpret(Float32, L.flags_host[1])
    any_ratio > 1.0f0 || return :kept
    return reinterpret(Float32, L.flags_host[2]) > 1.0f0 ? :built : :pruned
end

mark_pruned!(L::ClusterPairList, rx, ry, rz) = (kcopy!(L.px0, rx); kcopy!(L.py0, ry); kcopy!(L.pz0, rz); L)
