# =========================
#   Influence functions and virial coefficients (GPU setup kernels)
# =========================
# All coefficients live on the R2C half spectrum `(K₁÷2+1, K₂, K₃)` and are
# computed in Float64 on the device, then stored in the solver precision.
# `G[m]` multiplies the transformed charge grid (it contains C/V) and
# `Vc[ab, m] = −∂G/∂ε_ab` at fixed fractional coordinates, so the global
# reciprocal virial is `W_ab = ½ Σ_m w_m |b̂_m|² Vc[ab, m]`.
#
# Off-diagonal virial coefficients that involve a Nyquist index are set to zero
# (the ±K/2 choices are equivalent modes, and their average carries no shear).

@inline function _mode_indices(idx::Int, K1h::Int, K2::Int)
    m1 = (idx - 1) % K1h
    rest = (idx - 1) ÷ K1h
    m2 = rest % K2
    m3 = rest ÷ K2
    return m1, m2, m3
end

@inline _signed(m::Int, K::Int) = m <= K ÷ 2 ? m : m - K
@inline _is_nyquist(m::Int, K::Int) = iseven(K) && m == K ÷ 2

@inline function _store_virial!(Vc, idx, Gm, c, kx, ky, kz, ny1::Bool, ny2::Bool, ny3::Bool)
    Vc[1, idx] = Gm + c * kx * kx
    Vc[2, idx] = Gm + c * ky * ky
    Vc[3, idx] = Gm + c * kz * kz
    Vc[4, idx] = (ny1 | ny2) ? zero(eltype(Vc)) : c * kx * ky
    Vc[5, idx] = (ny1 | ny3) ? zero(eltype(Vc)) : c * kx * kz
    Vc[6, idx] = (ny2 | ny3) ? zero(eltype(Vc)) : c * ky * kz
    return nothing
end

# SPME / ESP: G = (C/V) g(k) D₁D₂D₃, −∂G/∂ε_ab = G δ_ab + (C/V) D g′(k)/k k_a k_b.
function _influence_ewald_kernel!(G, Vc, D1, D2, D3, K1::Int, K2::Int, K3::Int,
                                  box, C::Float64, spectral)
    # box: host tuple or the device geometry of a run (refreshed inside step graphs)
    L1, L2, L3 = map(Float64, Definitions.box_lengths(box, Val(3)))
    CoverV = C / (L1 * L2 * L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    K1h = K1 ÷ 2 + 1
    idx > K1h * K2 * K3 && return
    @inbounds begin
        m1, m2, m3 = _mode_indices(idx, K1h, K2)
        s1 = m1
        s2 = _signed(m2, K2)
        s3 = _signed(m3, K3)
        if (s1 == 0) & (s2 == 0) & (s3 == 0)
            G[idx] = zero(eltype(G))
            for c in 1:6
                Vc[c, idx] = zero(eltype(Vc))
            end
            return
        end
        kx = 2pi * s1 / L1
        ky = 2pi * s2 / L2
        kz = 2pi * s3 / L3
        k = sqrt(kx * kx + ky * ky + kz * kz)
        g, gpk = spectral_g(spectral, k)
        D = D1[m1 + 1] * D2[m2 + 1] * D3[m3 + 1]
        Gm = CoverV * g * D
        G[idx] = Gm
        _store_virial!(Vc, idx, Gm, CoverV * D * gpk, kx, ky, kz,
                       _is_nyquist(m1, K1), _is_nyquist(m2, K2), _is_nyquist(m3, K3))
    end
    return
end

# Optimal influence functions (Hockney–Eastwood; Ballenegger, Cerdà & Holm
# 2012) with alias sums |n_d| ≤ nmax, accumulated in R (the solver's precision on device
# box changes: Float64 sums of 125 alias terms per mode cost ≈2 ms at 36³ on GA102, whose FP64
# rate is 1/64 of FP32). The window transform is separable:
# U(k_n) = Π_d U_d[n_d, m_d] (tables from `window_alias_table`), so the same
# kernel serves B-spline (P3M) and prolate (ESP) windows, and any split.
#   AD: G = Σ U² g k_n² / (Σ U² · Σ U² k_n²)
#   ik: G = Σ U² g (D̃·k_n) / (|D̃|² (Σ U²)²),  D̃ = k with Nyquist components zeroed
function _influence_alias_kernel!(G, Vc, U1t, U2t, U3t, K1::Int, K2::Int, K3::Int,
                                  box, C::Float64, spectral, nmax::Int, ik::Bool,
                                  ::Type{R}=Float64) where {R}
    L1, L2, L3 = map(R, Definitions.box_lengths(box, Val(3)))
    CoverV = R(C) / (L1 * L2 * L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    K1h = K1 ÷ 2 + 1
    idx > K1h * K2 * K3 && return
    @inbounds begin
        m1, m2, m3 = _mode_indices(idx, K1h, K2)
        s1 = m1
        s2 = _signed(m2, K2)
        s3 = _signed(m3, K3)
        ny1 = _is_nyquist(m1, K1); ny2 = _is_nyquist(m2, K2); ny3 = _is_nyquist(m3, K3)
        kx = R(2pi) * s1 / L1
        ky = R(2pi) * s2 / L2
        kz = R(2pi) * s3 / L3
        # ik differentiation operator
        dx = ny1 ? zero(R) : kx
        dy = ny2 ? zero(R) : ky
        dz = ny3 ? zero(R) : kz
        d2 = dx * dx + dy * dy + dz * dz
        zero_mode = (s1 == 0) & (s2 == 0) & (s3 == 0)
        if zero_mode | (ik & (d2 == zero(R)))
            G[idx] = zero(eltype(G))
            for c in 1:6
                Vc[c, idx] = zero(eltype(Vc))
            end
            return
        end
        A = zero(R); B = zero(R); Cc = zero(R)
        a1 = zero(R); a2 = zero(R); a3 = zero(R); a4 = zero(R); a5 = zero(R); a6 = zero(R)     # −∂A/∂ε_ab
        c1 = zero(R); c2 = zero(R); c3 = zero(R); c4 = zero(R); c5 = zero(R); c6 = zero(R)     # −∂C/∂ε_ab (AD)
        for n1 in -nmax:nmax
            t1 = s1 + K1 * n1
            U1 = R(U1t[n1 + nmax + 1, m1 + 1])
            q1 = R(2pi) * t1 / L1
            for n2 in -nmax:nmax
                t2 = s2 + K2 * n2
                U2 = R(U2t[n2 + nmax + 1, m2 + 1])
                q2 = R(2pi) * t2 / L2
                for n3 in -nmax:nmax
                    t3 = s3 + K3 * n3
                    U3 = R(U3t[n3 + nmax + 1, m3 + 1])
                    q3 = R(2pi) * t3 / L3
                    U2tot = (U1 * U2 * U3)^2
                    B += U2tot
                    qq = q1 * q1 + q2 * q2 + q3 * q3
                    qq == zero(R) && continue
                    g, gpk = spectral_g(spectral, sqrt(qq))
                    if ik
                        dq = dx * q1 + dy * q2 + dz * q3
                        A += U2tot * g * dq
                        h = U2tot * gpk * dq
                        a1 += h * q1 * q1 + U2tot * g * R(2) * dx * q1
                        a2 += h * q2 * q2 + U2tot * g * R(2) * dy * q2
                        a3 += h * q3 * q3 + U2tot * g * R(2) * dz * q3
                        a4 += h * q1 * q2 + U2tot * g * (dx * q2 + q1 * dy)
                        a5 += h * q1 * q3 + U2tot * g * (dx * q3 + q1 * dz)
                        a6 += h * q2 * q3 + U2tot * g * (dy * q3 + q2 * dz)
                    else
                        A += U2tot * g * qq
                        Cc += U2tot * qq
                        h = U2tot * (gpk * qq + R(2) * g)
                        a1 += h * q1 * q1; a2 += h * q2 * q2; a3 += h * q3 * q3
                        a4 += h * q1 * q2; a5 += h * q1 * q3; a6 += h * q2 * q3
                        h2 = R(2) * U2tot
                        c1 += h2 * q1 * q1; c2 += h2 * q2 * q2; c3 += h2 * q3 * q3
                        c4 += h2 * q1 * q2; c5 += h2 * q1 * q3; c6 += h2 * q2 * q3
                    end
                end
            end
        end
        if ik
            den = d2 * B * B
            Gm = CoverV * A / den
            f = CoverV / den
            r = CoverV * A / (B * B) * R(2) / (d2 * d2)
            v1 = Gm + f * a1 - r * dx * dx
            v2 = Gm + f * a2 - r * dy * dy
            v3 = Gm + f * a3 - r * dz * dz
            v4 = f * a4 - r * dx * dy
            v5 = f * a5 - r * dx * dz
            v6 = f * a6 - r * dy * dz
        else
            Gm = CoverV * A / (B * Cc)
            f = CoverV / (B * Cc)
            r = CoverV * A / (B * Cc * Cc)
            v1 = Gm + f * a1 - r * c1
            v2 = Gm + f * a2 - r * c2
            v3 = Gm + f * a3 - r * c3
            v4 = f * a4 - r * c4
            v5 = f * a5 - r * c5
            v6 = f * a6 - r * c6
        end
        G[idx] = Gm
        Vc[1, idx] = v1
        Vc[2, idx] = v2
        Vc[3, idx] = v3
        Vc[4, idx] = (ny1 | ny2) ? zero(R) : v4
        Vc[5, idx] = (ny1 | ny3) ? zero(R) : v5
        Vc[6, idx] = (ny2 | ny3) ? zero(R) : v6
    end
    return
end

"""
    compute_influence!(G, Vc, spec, mesh, box, C)

Fill the influence function `G` and virial coefficients `Vc` (Float64 device
arrays on the half spectrum) for the given box. Called at setup and after any
box change; never during ordinary stepping.
"""
function compute_influence!(G::CuArray{Float64}, Vc::CuArray{Float64,2},
                            spec::ElectrostaticsSpec, mesh::NTuple{3,Int},
                            box::NTuple{3,Float64}, C::Float64)
    K1, K2, K3 = mesh
    L1, L2, L3 = box
    V = L1 * L2 * L3
    nmodes = (K1 ÷ 2 + 1) * K2 * K3
    threads = 128
    blocks = cld(nmodes, threads)
    keep = Any[]                       # device arrays referenced by the launched kernel
    spectral = if spec.split isa GaussianSplit
        GaussianSpectral(spec.split.alpha)
    else
        sp = spec.split::ProlateSplit
        xq = CuArray(sp.xq); wchi = CuArray(sp.wchi)
        push!(keep, xq, wchi)
        ProlateSpectral(sp.cutoff, cudaconvert(xq), cudaconvert(wchi))
    end
    if spec.influence === :spme || spec.influence === :esp_deconvolution
        D1 = CuArray(window_deconvolution(spec.window, K1))
        D2 = CuArray(window_deconvolution(spec.window, K2))
        D3 = CuArray(window_deconvolution(spec.window, K3))
        push!(keep, D1, D2, D3)
        @cuda threads=threads blocks=blocks _influence_ewald_kernel!(G, Vc, D1, D2, D3, K1, K2, K3,
                                                                    box, C, spectral)
    else
        nmax = spec.alias_order
        U1 = CuArray(window_alias_table(spec.window, K1, nmax))
        U2 = CuArray(window_alias_table(spec.window, K2, nmax))
        U3 = CuArray(window_alias_table(spec.window, K3, nmax))
        push!(keep, U1, U2, U3)
        @cuda threads=threads blocks=blocks _influence_alias_kernel!(G, Vc, U1, U2, U3, K1, K2, K3,
                                                                    box, C, spectral, nmax,
                                                                    spec.influence === :p3m_ik)
    end
    CUDA.synchronize()                 # keep the referenced arrays alive until the kernel has run
    return nothing
end

# ---------------------------------------------------------------------------
# Device-resident refresh (device-resident barostat): the influence function
# and virial coefficients at the box held in a run's device geometry vector, written
# straight into the solver's G and Vc. The window tables and the prolate quadrature
# live in a per-solver cache built once, so a refresh allocates nothing, never touches
# the host and can be captured in a step graph.
const _INFLUENCE_CACHE = WeakKeyDict{Any,Any}()

function _influence_tables(s)
    get!(_INFLUENCE_CACHE, s) do
        spec = s.spec; K1, K2, K3 = s.ws.mesh
        keep = Any[]
        spectral = if spec.split isa GaussianSplit
            GaussianSpectral(spec.split.alpha)
        else
            sp = spec.split::ProlateSplit
            xq = CuArray(sp.xq); wchi = CuArray(sp.wchi); push!(keep, xq, wchi)
            exact = ProlateSpectral(sp.cutoff, cudaconvert(xq), cudaconvert(wchi))
            # tabulate ĉ up to the largest wavevector (alias images included) of a box 10%
            # smaller than the current one; the exact quadrature covers anything beyond
            nal = spec.influence === :spme || spec.influence === :esp_deconvolution ? 0 : spec.alias_order
            kmax = 2pi * sqrt(sum(((s.ws.mesh[d] ÷ 2 + nal * s.ws.mesh[d]) / (0.9 * s.box[d]))^2 for d in 1:3))
            du = 0.05; n = ceil(Int, sp.cutoff * kmax / du) + 3
            ct = CUDA.zeros(Float64, n); dct = CUDA.zeros(Float64, n); push!(keep, ct, dct)
            @cuda threads=128 blocks=cld(n, 128) _prolate_table_kernel!(ct, dct, xq, wchi, du)
            CUDA.synchronize()
            TabulatedSpectral(sp.cutoff, du, cudaconvert(ct), cudaconvert(dct), exact)
        end
        tables = if spec.influence === :spme || spec.influence === :esp_deconvolution
            (CuArray(window_deconvolution(spec.window, K1)), CuArray(window_deconvolution(spec.window, K2)),
             CuArray(window_deconvolution(spec.window, K3)))
        else
            nmax = spec.alias_order
            (CuArray(window_alias_table(spec.window, K1, nmax)), CuArray(window_alias_table(spec.window, K2, nmax)),
             CuArray(window_alias_table(spec.window, K3, nmax)))
        end
        (spectral=spectral, tables=tables, keep=keep)
    end
end

"""
    device_influence!(solver, geom; launch=true)

Influence function and virial coefficients of `solver` at the box in the device
vector `geom`, written into the solver's mesh workspace on the device. With
`launch=false` only compiles the kernel (before capture).
"""
function device_influence!(s, geom; launch::Bool=true)
    c = _influence_tables(s); spec = s.spec; K1, K2, K3 = s.ws.mesh
    nmodes = (K1 ÷ 2 + 1) * K2 * K3; threads = 128; blocks = cld(nmodes, threads)
    D1, D2, D3 = c.tables
    if spec.influence === :spme || spec.influence === :esp_deconvolution
        k = @cuda launch=false _influence_ewald_kernel!(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, Float64(s.C), c.spectral)
    else
        k = @cuda launch=false _influence_alias_kernel!(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, Float64(s.C),
                                                         c.spectral, spec.alias_order, spec.influence === :p3m_ik, eltype(s.ws.G))
    end
    launch || return nothing
    if spec.influence === :spme || spec.influence === :esp_deconvolution
        k(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, Float64(s.C), c.spectral; threads, blocks)
    else
        k(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, Float64(s.C), c.spectral, spec.alias_order,
          spec.influence === :p3m_ik, eltype(s.ws.G); threads, blocks)
    end
    return nothing
end
