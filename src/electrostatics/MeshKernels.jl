# =========================
#   Mesh kernels: spread, spectral solve, gathers
# =========================
# Grids are column-major `(K₁, K₂, K₃)` arrays addressed through a flat view:
# node (n₁, n₂, n₃) (0-based) lives at n₁ + K₁ (n₂ + K₂ n₃) + 1.

# Non-finite coordinates map to node 0, so spread/gather stay in bounds; the run
# is invalid and the neighbor/cluster list check reports it at the next boundary.
@inline function _grid_coordinate(x::T, L::T, K::Int32) where {T<:AbstractFloat}
    s = x / L
    s -= floor(s)
    u = s * T(K)
    isfinite(u) || return zero(T)
    return u >= T(K) ? u - T(K) : u
end

@inline function _wrap_node(n::Int32, K::Int32)
    n < Int32(0) && return n + K
    n >= K && return n - K
    return n
end

# `order[t]` is the particle handled by thread t: the neighbor list's
# cell-sorted order, so that a warp deposits into (and reads from) nearby,
# cache-friendly grid nodes regardless of the particle storage order.
function _spread_kernel!(grid::CuDeviceVector{T}, rx, ry, rz, q, order,
                         box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                         stencil::S) where {T,S}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    t > length(rx) && return
    @inbounds begin
        i = order[t]
        qi = q[i]
        qi == zero(T) && return
        k1, w1, _ = window_stencil(stencil, _grid_coordinate(rx[i], Lx, K1))
        k2, w2, _ = window_stencil(stencil, _grid_coordinate(ry[i], Ly, K2))
        k3, w3, _ = window_stencil(stencil, _grid_coordinate(rz[i], Lz, K3))
        P = length(w1)
        for j3 in 1:P
            n3 = _wrap_node(k3 + Int32(j3 - 1), K3)
            q3 = qi * w3[j3]
            for j2 in 1:P
                n2 = _wrap_node(k2 + Int32(j2 - 1), K2)
                q23 = q3 * w2[j2]
                base = K1 * (n2 + K2 * n3)
                for j1 in 1:P
                    n1 = _wrap_node(k1 + Int32(j1 - 1), K1)
                    Definitions.@red grid[base + n1 + Int32(1)] += q23 * w1[j1]
                end
            end
        end
    end
    return
end

# Tiled spreading: a block takes `blockDim` consecutive particles of a
# spatially sorted `order`, finds the bounding box of their stencils in grid
# indices (relative to the block's first particle, periodic), accumulates into
# shared memory, and flushes the box to the global grid with one atomic per
# nonzero node. Blocks whose box exceeds `TILE` nodes (or half the grid in a
# dimension) deposit directly with global atomics. Measured at 1M SPC/E
# atoms, order 4: 128 threads / 4096 nodes 0.71 ms, 256 / 8192 0.61 ms,
# 512 / 11264 0.56 ms, against 1.12 ms for one global atomic per node.

@inline function _rel_index(k::Int32, kref::Int32, K::Int32)
    d = k - kref
    h = K >> 1
    d >= h && (d -= K)
    d < -h && (d += K)
    return d
end

@inline function _mod_node(n::Int32, K::Int32)
    n = n % K
    return n < Int32(0) ? n + K : n
end

function _spread_tiled_kernel!(grid::CuDeviceVector{T}, rx, ry, rz, q, order,
                               box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                               stencil::S, ::Val{TILE}) where {T,S,TILE}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    tid = threadIdx().x
    t = (blockIdx().x - Int32(1)) * blockDim().x + tid
    tile = CuStaticSharedArray(T, TILE)
    box = CuStaticSharedArray(Int32, 9)      # kref(3), min(3), max(3)
    N = length(order)
    @inbounds begin
        active = t <= N
        i = active ? order[t] : order[N]
        qi = active ? q[i] : zero(T)
        k1, w1, _ = window_stencil(stencil, _grid_coordinate(rx[i], Lx, K1))
        k2, w2, _ = window_stencil(stencil, _grid_coordinate(ry[i], Ly, K2))
        k3, w3, _ = window_stencil(stencil, _grid_coordinate(rz[i], Lz, K3))
        P = Int32(length(w1))
        if tid == Int32(1)
            box[1] = k1; box[2] = k2; box[3] = k3
            box[4] = typemax(Int32); box[5] = typemax(Int32); box[6] = typemax(Int32)
            box[7] = typemin(Int32); box[8] = typemin(Int32); box[9] = typemin(Int32)
        end
        sync_threads()
        r1 = _rel_index(k1, box[1], K1); r2 = _rel_index(k2, box[2], K2); r3 = _rel_index(k3, box[3], K3)
        if active
            CUDA.atomic_min!(pointer(box, 4), r1); CUDA.atomic_min!(pointer(box, 5), r2); CUDA.atomic_min!(pointer(box, 6), r3)
            CUDA.atomic_max!(pointer(box, 7), r1); CUDA.atomic_max!(pointer(box, 8), r2); CUDA.atomic_max!(pointer(box, 9), r3)
        end
        sync_threads()
        n1 = box[7] - box[4] + P; n2 = box[8] - box[5] + P; n3 = box[9] - box[6] + P
        tiled = (n1 * n2 * n3 <= Int32(TILE)) & (Int32(2) * n1 <= K1) & (Int32(2) * n2 <= K2) & (Int32(2) * n3 <= K3)
        if tiled
            nt = n1 * n2 * n3
            m = tid
            while m <= nt
                tile[m] = zero(T)
                m += blockDim().x
            end
            sync_threads()
            if active & (qi != zero(T))
                o1 = r1 - box[4]; o2 = r2 - box[5]; o3 = r3 - box[6]
                for j3 in Int32(0):(P - Int32(1))
                    q3 = qi * w3[j3 + 1]
                    for j2 in Int32(0):(P - Int32(1))
                        q23 = q3 * w2[j2 + 1]
                        base = n1 * ((o2 + j2) + n2 * (o3 + j3)) + o1 + Int32(1)
                        for j1 in Int32(0):(P - Int32(1))
                            Definitions.@red tile[base + j1] += q23 * w1[j1 + 1]
                        end
                    end
                end
            end
            sync_threads()
            m = tid
            g1 = box[1] + box[4]; g2 = box[2] + box[5]; g3 = box[3] + box[6]
            while m <= nt
                v = tile[m]
                if v != zero(T)
                    a = m - Int32(1)
                    a1 = a % n1; a = a ÷ n1
                    a2 = a % n2; a3 = a ÷ n2
                    node = _mod_node(g1 + a1, K1) + K1 * (_mod_node(g2 + a2, K2) + K2 * _mod_node(g3 + a3, K3))
                    Definitions.@red grid[node + Int32(1)] += v
                end
                m += blockDim().x
            end
        elseif active & (qi != zero(T))
            for j3 in 1:P
                n3g = _wrap_node(k3 + Int32(j3 - 1), K3)
                q3 = qi * w3[j3]
                for j2 in 1:P
                    n2g = _wrap_node(k2 + Int32(j2 - 1), K2)
                    q23 = q3 * w2[j2]
                    base = K1 * (n2g + K2 * n3g)
                    for j1 in 1:P
                        n1g = _wrap_node(k1 + Int32(j1 - 1), K1)
                        Definitions.@red grid[base + n1g + Int32(1)] += q23 * w1[j1]
                    end
                end
            end
        end
    end
    return
end

@inline function _warp_sum(v::Float64)
    v += shfl_down_sync(0xffffffff, v, 16)
    v += shfl_down_sync(0xffffffff, v, 8)
    v += shfl_down_sync(0xffffffff, v, 4)
    v += shfl_down_sync(0xffffffff, v, 2)
    v += shfl_down_sync(0xffffffff, v, 1)
    return v
end

# Block-level reduction of 7 Float64 partials into `acc[1:7]` with one atomic
# per component per block. Every thread of the block must call it.
@inline function _block_accumulate7!(acc, vals::NTuple{7,Float64})
    shared = CuStaticSharedArray(Float64, (7, 32))
    lane = (threadIdx().x - Int32(1)) % Int32(32) + Int32(1)
    warp = (threadIdx().x - Int32(1)) ÷ Int32(32) + Int32(1)
    nwarps = (blockDim().x + Int32(31)) ÷ Int32(32)
    @inbounds for c in 1:7
        s = _warp_sum(vals[c])
        lane == Int32(1) && (shared[c, warp] = s)
    end
    sync_threads()
    if warp == Int32(1)
        @inbounds for c in 1:7
            s = lane <= nwarps ? shared[c, lane] : 0.0
            s = _warp_sum(s)
            lane == Int32(1) && Definitions.@red acc[c] += s
        end
    end
    return nothing
end

# Multiply the transformed charge grid by G in place. With ENERGY, also reduce
# the mesh energy ½ Σ w |b̂|² G (acc[1]) and the six virial components
# ½ Σ w |b̂|² Vc[ab] (acc[2:7]); w is the Hermitian weight of the R2C axis.
function _solve_kernel!(spec::CuDeviceVector{Complex{T}}, G::CuDeviceVector{T},
                        Vc::CuDeviceMatrix{T}, acc::CuDeviceVector{Float64},
                        K1::Int32, K1h::Int32, nmodes::Int32, ::Val{ENERGY}) where {T,ENERGY}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    e = 0.0; v1 = 0.0; v2 = 0.0; v3 = 0.0; v4 = 0.0; v5 = 0.0; v6 = 0.0
    if idx <= nmodes
        @inbounds begin
            z = spec[idx]
            Gm = G[idx]
            if ENERGY
                m1 = (idx - Int32(1)) % K1h
                w = (m1 == Int32(0)) | (iseven(K1) & (m1 == K1 ÷ Int32(2))) ? 0.5 : 1.0
                a = w * Float64(abs2(z))
                e = a * Float64(Gm)
                v1 = a * Float64(Vc[1, idx]); v2 = a * Float64(Vc[2, idx]); v3 = a * Float64(Vc[3, idx])
                v4 = a * Float64(Vc[4, idx]); v5 = a * Float64(Vc[5, idx]); v6 = a * Float64(Vc[6, idx])
            end
            spec[idx] = z * Gm
        end
    end
    if ENERGY
        _block_accumulate7!(acc, (e, v1, v2, v3, v4, v5, v6))
    end
    return
end

# Analytic-differentiation gather: F_i = −q_i Σ c_ℓ ∇W, E_i += ½ q_i u_i + bg q_i.
function _gather_ad_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, order, grid::CuDeviceVector{T},
                            box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                            stencil::S, bg::T, ::Val{ENERGY}) where {T,S,ENERGY}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    t > length(rx) && return
    @inbounds begin
        i = order[t]
        qi = q[i]
        qi == zero(T) && return
        k1, w1, d1 = window_stencil(stencil, _grid_coordinate(rx[i], Lx, K1))
        k2, w2, d2 = window_stencil(stencil, _grid_coordinate(ry[i], Ly, K2))
        k3, w3, d3 = window_stencil(stencil, _grid_coordinate(rz[i], Lz, K3))
        P = length(w1)
        phi = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        for j3 in 1:P
            n3 = _wrap_node(k3 + Int32(j3 - 1), K3)
            for j2 in 1:P
                n2 = _wrap_node(k2 + Int32(j2 - 1), K2)
                base = K1 * (n2 + K2 * n3)
                w23 = w2[j2] * w3[j3]
                d23y = d2[j2] * w3[j3]
                d23z = w2[j2] * d3[j3]
                sx = zero(T); s0 = zero(T)
                for j1 in 1:P
                    n1 = _wrap_node(k1 + Int32(j1 - 1), K1)
                    c = grid[base + n1 + Int32(1)]
                    s0 += w1[j1] * c
                    sx += d1[j1] * c
                end
                phi += w23 * s0
                gx += w23 * sx
                gy += d23y * s0
                gz += d23z * s0
            end
        end
        fx[i] -= qi * gx * (T(K1) / Lx)
        fy[i] -= qi * gy * (T(K2) / Ly)
        fz[i] -= qi * gz * (T(K3) / Lz)
        if ENERGY
            Epot[i] += T(0.5) * qi * phi + bg * qi
        end
    end
    return
end

# ik: out = −i k_d spec (Nyquist component of the derivative operator zeroed).
function _ik_field_kernel!(out::CuDeviceVector{Complex{T}}, spec::CuDeviceVector{Complex{T}},
                           K1::Int32, K2::Int32, K3::Int32, box::Union{NTuple{3,T},AbstractVector{T}}, dim::Int32) where {T}
    L = Definitions.box_lengths(box, Val(3))[dim]   # tuple or device geometry
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    K1h = K1 ÷ Int32(2) + Int32(1)
    idx > K1h * K2 * K3 && return
    @inbounds begin
        m1 = (idx - Int32(1)) % K1h
        rest = (idx - Int32(1)) ÷ K1h
        m2 = rest % K2
        m3 = rest ÷ K2
        m, K = dim == Int32(1) ? (m1, K1) : (dim == Int32(2) ? (m2, K2) : (m3, K3))
        s = (iseven(K) & (m == K ÷ Int32(2))) ? Int32(0) : (m <= K ÷ Int32(2) ? m : m - K)
        kd = T(2pi) * T(s) / L
        z = spec[idx]
        out[idx] = Complex{T}(imag(z) * kd, -real(z) * kd)
    end
    return
end

# ik gather: F_i = q_i Σ W E_ℓ; with ENERGY, E_i += ½ q_i u_i + bg q_i.
function _gather_ik_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, order,
                            ex::CuDeviceVector{T}, ey::CuDeviceVector{T}, ez::CuDeviceVector{T},
                            pot::CuDeviceVector{T},
                            box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                            stencil::S, bg::T, ::Val{ENERGY}) where {T,S,ENERGY}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))   # tuple or device geometry
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    t > length(rx) && return
    @inbounds begin
        i = order[t]
        qi = q[i]
        qi == zero(T) && return
        k1, w1, _ = window_stencil(stencil, _grid_coordinate(rx[i], Lx, K1))
        k2, w2, _ = window_stencil(stencil, _grid_coordinate(ry[i], Ly, K2))
        k3, w3, _ = window_stencil(stencil, _grid_coordinate(rz[i], Lz, K3))
        P = length(w1)
        sx = zero(T); sy = zero(T); sz = zero(T); phi = zero(T)
        for j3 in 1:P
            n3 = _wrap_node(k3 + Int32(j3 - 1), K3)
            for j2 in 1:P
                n2 = _wrap_node(k2 + Int32(j2 - 1), K2)
                base = K1 * (n2 + K2 * n3)
                w23 = w2[j2] * w3[j3]
                for j1 in 1:P
                    n1 = _wrap_node(k1 + Int32(j1 - 1), K1)
                    l = base + n1 + Int32(1)
                    w = w23 * w1[j1]
                    sx += w * ex[l]
                    sy += w * ey[l]
                    sz += w * ez[l]
                    ENERGY && (phi += w * pot[l])
                end
            end
        end
        fx[i] += qi * sx
        fy[i] += qi * sy
        fz[i] += qi * sz
        if ENERGY
            Epot[i] += T(0.5) * qi * phi + bg * qi
        end
    end
    return
end

# =========================
#   Line kernels: 16 threads per particle (GROMACS-like)
# =========================
# Lane l of a particle's 16-lane group takes the (x, y) node pairs l, l+16, … of its P×P
# stencil (x fastest: lanes of one line touch adjacent grid words, a few sectors per warp
# access, as GROMACS's spread) and walks the P nodes along z. Spread: the P atomics of a
# line in parallel with the other lines (one thread per particle issued P³ sequential
# atomics). Gather: line partial sums reduced over the group with width-16 shuffles.
# Each lane needs all P z weights, but only its x and y node weights.
const MESH_LANES = 16
"""Use the 16-lane spread/gather kernels (default) or the one-thread-per-particle ones."""
const MESH_LINES = Ref(true)

@inline _stencil_order(::BSplineStencil{P}) where {P} = P
@inline _stencil_order(::ProlateStencil{P}) where {P} = P

# Element j (0-based, lane-dependent) of a tuple by an unrolled select: a dynamic tuple
# index would place the tuple in local memory (96 bytes per thread in the gather).
@inline function _tsel(t::NTuple{K,T}, j) where {K,T}
    r = t[1]
    for k in 2:K
        r = ifelse(j == k - 1, t[k], r)
    end
    return r
end
# weight and derivative of node j (0-based) of the stencil at u: (k0, w, dw)
@inline function _window_node(s::BSplineStencil{P}, u::T, j::Integer) where {P,T}
    k0, w, dw = window_stencil(s, u)
    return k0, _tsel(w, j), _tsel(dw, j)
end
# The P z nodes of one (x, y) line, unrolled (static tuple indices): spread q12·w3 into
# them; gather (Σ w3·c, Σ d3·c).
@inline _zspread!(grid, base, stride, k3, K3, q12, w3::NTuple{P}) where {P} =
    (ntuple(j -> (Definitions.@red grid[base + stride * _wrap_node(k3 + Int32(j - 1), K3)] += q12 * w3[j]; nothing), Val(P)); nothing)
@inline function _zgather(grid, base, stride, k3, K3, w3::NTuple{P,T}, d3::NTuple{P,T}) where {P,T}
    c = ntuple(j -> (@inbounds grid[base + stride * _wrap_node(k3 + Int32(j - 1), K3)]), Val(P))
    return sum(ntuple(j -> w3[j] * c[j], Val(P))), sum(ntuple(j -> d3[j] * c[j], Val(P)))
end
@inline function _window_node(s::ProlateStencil{P,T}, u::T, j::Integer) where {P,T}
    k0 = iseven(P) ? unsafe_trunc(Int32, floor(u)) - Int32(P ÷ 2 - 1) :
                     unsafe_trunc(Int32, floor(u + T(0.5))) - Int32((P - 1) ÷ 2)
    scale = T(2) / T(P)
    y = (T(k0 + Int32(j)) - u) * scale
    inside = abs(y) <= one(T)
    return k0, inside ? clenshaw(s.c, y) : zero(T), inside ? -scale * clenshaw(s.dc, y) : zero(T)
end

@inline function _spread_lines_generic!(grid::CuDeviceVector{T}, rx, ry, rz, q, order,
                               box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                               stencil::S, xq=nothing) where {T,S}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    P = _stencil_order(stencil)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    t > length(order) && return
    @inbounds begin
        i = order[t]
        qi = q[i]
        # packed (x, y, z, q) for the device engine's pair kernel (it follows the spread)
        (xq !== nothing && lane == Int32(0)) && (xq[i] = (VecElement(rx[i]), VecElement(ry[i]), VecElement(rz[i]), VecElement(qi)))
        qi == zero(T) && return
        u1 = _grid_coordinate(rx[i], Lx, K1); u2 = _grid_coordinate(ry[i], Ly, K2); u3 = _grid_coordinate(rz[i], Lz, K3)
        k3, w3, _ = window_stencil(stencil, u3)
        l = lane
        while l < Int32(P * P)
            j1 = l % Int32(P); j2 = l ÷ Int32(P)
            k1, wx, _ = _window_node(stencil, u1, j1)
            k2, wy, _ = _window_node(stencil, u2, j2)
            n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
            q12 = qi * wx * wy
            base = n1 + K1 * n2 + Int32(1)
            _zspread!(grid, base, K1 * K2, k3, K3, q12, w3)
            l += Int32(MESH_LANES)
        end
    end
    return
end

@inline function _sum16(v)
    for off in (8, 4, 2, 1)
        v += shfl_down_sync(0xffffffff, v, off, 16)
    end
    return v
end

# Lane partial sums of one particle's gather (16-lane group): (t, lane, i, qi, phi, gx, gy, gz,
# Lx, Ly, Lz); the caller reduces them over the group (`_sum16`). Shared by the gather kernel
# and the device engine's fused gather + kick.
@inline function _gather_ad_lines_generic(rx, ry, rz, q, order, grid::CuDeviceVector{T},
                                       box, K1::Int32, K2::Int32, K3::Int32, stencil::S) where {T,S}
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    P = _stencil_order(stencil)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    phi = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
    i = Int32(0); qi = zero(T)
    @inbounds if t <= length(order)
        i = order[t]
        qi = q[i]
        if qi != zero(T)
            u1 = _grid_coordinate(rx[i], Lx, K1); u2 = _grid_coordinate(ry[i], Ly, K2); u3 = _grid_coordinate(rz[i], Lz, K3)
            k3, w3, d3 = window_stencil(stencil, u3)
            l = lane
            while l < Int32(P * P)
                j1 = l % Int32(P); j2 = l ÷ Int32(P)
                k1, wx, dx = _window_node(stencil, u1, j1)
                k2, wy, dy = _window_node(stencil, u2, j2)
                n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
                base = n1 + K1 * n2 + Int32(1)
                s0, sz = _zgather(grid, base, K1 * K2, k3, K3, w3, d3)
                w12 = wx * wy
                phi = muladd(w12, s0, phi); gz = muladd(w12, sz, gz)
                gx = muladd(dx * wy, s0, gx); gy = muladd(wx * dy, s0, gy)
                l += Int32(MESH_LANES)
            end
        end
    end
    return t, lane, i, qi, phi, gx, gy, gz, Lx, Ly, Lz
end

# Shared-weight lines (15 Oct 2026). Each of the first 3P lanes of a particle's group evaluates
# one window weight (and derivative): node j of dimension d on lane d·P + j; the lines read
# theirs by warp shuffles. The prolate window costs a polynomial per weight, and every lane
# evaluated up to P + 4 of them (spread 2.6×, gather 2× the B-spline cost at P = 5). Needs
# 3P ≤ MESH_LANES (P ≤ 5); larger orders use the per-lane evaluation (`*_generic`). No lane
# leaves before the shuffles: groups past N (or uncharged particles) shuffle and skip the work.
@inline _window_base(::BSplineStencil{P}, u::T) where {P,T} = unsafe_trunc(Int32, floor(u)) - Int32(P - 1)
@inline _window_base(::ProlateStencil{P}, u::T) where {P,T} =
    iseven(P) ? unsafe_trunc(Int32, floor(u)) - Int32(P ÷ 2 - 1) : unsafe_trunc(Int32, floor(u + T(0.5))) - Int32((P - 1) ÷ 2)
@inline _lane_shfl(x, src) = shfl_sync(0xffffffff, x, src, MESH_LANES)   # src: 1-based lane of the group

function _spread_lines_kernel!(grid::CuDeviceVector{T}, rx, ry, rz, q, order,
                               box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                               stencil::S, xq=nothing) where {T,S}
    P = _stencil_order(stencil)
    if 3P > MESH_LANES
        @inline _spread_lines_generic!(grid, rx, ry, rz, q, order, box, K1, K2, K3, stencil, xq)
        return
    end
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    active = t <= length(order)
    @inbounds begin
        i = order[active ? t : Int32(1)]
        qi = active ? q[i] : zero(T)
        xi = rx[i]; yi = ry[i]; zi = rz[i]
        # packed (x, y, z, q) for the device engine's pair kernel (it follows the spread)
        (xq !== nothing && active && lane == Int32(0)) && (xq[i] = (VecElement(xi), VecElement(yi), VecElement(zi), VecElement(qi)))
        u1 = _grid_coordinate(xi, Lx, K1); u2 = _grid_coordinate(yi, Ly, K2); u3 = _grid_coordinate(zi, Lz, K3)
        d = lane ÷ Int32(P); j = lane - d * Int32(P)
        _, wl, _ = _window_node(stencil, d == Int32(0) ? u1 : d == Int32(1) ? u2 : u3, min(j, Int32(P - 1)))
        k1 = _window_base(stencil, u1); k2 = _window_base(stencil, u2); k3 = _window_base(stencil, u3)
        w3 = ntuple(k -> _lane_shfl(wl, 2P + k), Val(P))
        for r in 0:cld(P * P, MESH_LANES) - 1
            l = lane + Int32(r * MESH_LANES)
            j1 = l % Int32(P); j2 = min(l ÷ Int32(P), Int32(P - 1))
            wx = _lane_shfl(wl, j1 + Int32(1)); wy = _lane_shfl(wl, Int32(P) + j2 + Int32(1))
            if active && qi != zero(T) && l < Int32(P * P)
                n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
                _zspread!(grid, n1 + K1 * n2 + Int32(1), K1 * K2, k3, K3, qi * wx * wy, w3)
            end
        end
    end
    return
end

@inline function _gather_ad_lines_sums(rx, ry, rz, q, order, grid::CuDeviceVector{T},
                                       box, K1::Int32, K2::Int32, K3::Int32, stencil::S) where {T,S}
    P = _stencil_order(stencil)
    3P > MESH_LANES && return _gather_ad_lines_generic(rx, ry, rz, q, order, grid, box, K1, K2, K3, stencil)
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    active = t <= length(order)
    phi = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
    @inbounds begin
        i = order[active ? t : Int32(1)]
        qi = active ? q[i] : zero(T)
        u1 = _grid_coordinate(rx[i], Lx, K1); u2 = _grid_coordinate(ry[i], Ly, K2); u3 = _grid_coordinate(rz[i], Lz, K3)
        d = lane ÷ Int32(P); j = lane - d * Int32(P)
        _, wl, dl = _window_node(stencil, d == Int32(0) ? u1 : d == Int32(1) ? u2 : u3, min(j, Int32(P - 1)))
        k1 = _window_base(stencil, u1); k2 = _window_base(stencil, u2); k3 = _window_base(stencil, u3)
        w3 = ntuple(k -> _lane_shfl(wl, 2P + k), Val(P)); d3 = ntuple(k -> _lane_shfl(dl, 2P + k), Val(P))
        for r in 0:cld(P * P, MESH_LANES) - 1
            l = lane + Int32(r * MESH_LANES)
            j1 = l % Int32(P); j2 = min(l ÷ Int32(P), Int32(P - 1))
            wx = _lane_shfl(wl, j1 + Int32(1)); dx = _lane_shfl(dl, j1 + Int32(1))
            wy = _lane_shfl(wl, Int32(P) + j2 + Int32(1)); dy = _lane_shfl(dl, Int32(P) + j2 + Int32(1))
            if active && qi != zero(T) && l < Int32(P * P)
                n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
                s0, sz = _zgather(grid, n1 + K1 * n2 + Int32(1), K1 * K2, k3, K3, w3, d3)
                w12 = wx * wy
                phi = muladd(w12, s0, phi); gz = muladd(w12, sz, gz)
                gx = muladd(dx * wy, s0, gx); gy = muladd(wx * dy, s0, gy)
            end
        end
    end
    return t, lane, i, qi, phi, gx, gy, gz, Lx, Ly, Lz
end

function _gather_ad_lines_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, order, grid::CuDeviceVector{T},
                                  box::Union{NTuple{3,T},AbstractVector{T}}, K1::Int32, K2::Int32, K3::Int32,
                                  stencil::S, bg::T, ::Val{ENERGY}, ::Val{SET}=Val(false)) where {T,S,ENERGY,SET}
    t, lane, i, qi, phi, gx, gy, gz, Lx, Ly, Lz = _gather_ad_lines_sums(rx, ry, rz, q, order, grid, box, K1, K2, K3, stencil)
    gx = _sum16(gx); gy = _sum16(gy); gz = _sum16(gz)
    ENERGY && (phi = _sum16(phi))
    # SET: the mesh force (and energy) replaces the slot, for every particle of `order`
    # (zero when uncharged); the pair scatter then adds to it
    @inbounds if SET && lane == Int32(0) && t <= length(order)
        fx[i] = -(qi * gx * (T(K1) / Lx))
        fy[i] = -(qi * gy * (T(K2) / Ly))
        fz[i] = -(qi * gz * (T(K3) / Lz))
        ENERGY && (Epot[i] = T(0.5) * qi * phi + bg * qi)
    elseif !SET && lane == Int32(0) && t <= length(order) && qi != zero(T)
        fx[i] -= qi * gx * (T(K1) / Lx)
        fy[i] -= qi * gy * (T(K2) / Ly)
        fz[i] -= qi * gz * (T(K3) / Lz)
        ENERGY && (Epot[i] += T(0.5) * qi * phi + bg * qi)
    end
    return
end
