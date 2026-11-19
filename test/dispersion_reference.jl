"""
Independent Float64 CPU Ewald reference for periodic `1/r⁶` lattice dispersion tests.

The oracle shares no code with the GPU solver: it uses its own erfc, picks its
own splitting parameter `β`, and converges real-space images and reciprocal
wavevectors far beyond production tolerances. Supports general triclinic `3×3`
cell matrices `H` (for finite-difference strain checks of all 6 virial components)
and intramolecular excluded/scaled pairs `(i, j, scale)` at minimum-image distance.

Hamiltonian (attractive dispersion):
    U = −½ Σ'_{i,j,n} s6_i s6_j / |r_ij + H n|⁶ + Σ_{(i,j,s) ∈ exc} (1 − s) s6_i s6_j / r_ij,mic⁶
"""
module DispersionReference

using LinearAlgebra

export dispersion_reference

function _erfc_ref(x::Float64)
    x < 0 && return 2.0 - _erfc_ref(-x)
    if x < 2.0
        term = x
        acc = x
        n = 0
        while term > 1e-18 * acc
            n += 1
            term *= 2.0 * x * x / (2n + 1)
            acc += term
        end
        return 1.0 - 2.0 / sqrt(pi) * exp(-x * x) * acc
    end
    t = x
    for n in 400:-1:1
        t = x + (n / 2) / t
    end
    return exp(-x * x) / sqrt(pi) / t
end

@inline function _g_ess(x2::Float64)
    return exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2)
end

@inline function _phi6_hat_gauss(k::Float64, beta::Float64)
    u = k / (2.0 * beta)
    u2 = u * u
    eu = exp(-u2)
    erfc_u = _erfc_ref(u)
    pref = (pi^1.5) * (beta^3) / 3.0
    g = pref * ((1.0 - 2.0 * u2) * eu + 2.0 * (u^3) * sqrt(pi) * erfc_u)
    gpk = 0.5 * (pi^1.5) * beta * (u * sqrt(pi) * erfc_u - eu)
    return g, gpk
end

"""
    dispersion_reference(R, s6, H; exceptions=(), beta=nothing, digits=15) -> (energy, forces, virial)

`R` is `3×N` Cartesian coordinates, `s6` length-`N` dispersion weights (`√c6_ii ≥ 0`),
and `H` the `3×3` cell matrix (columns = box vectors). Returns `(E, F, W)` with
configurational virial `W_ab = Σ r_a F_b = −∂U/∂ε_ab`.
"""
function dispersion_reference(R::AbstractMatrix, s6::AbstractVector, H::AbstractMatrix;
                              exceptions=(), beta=nothing, digits::Real=15)
    R = Float64.(R)
    s6 = Float64.(s6)
    H = Float64.(H)
    N = size(R, 2)
    V = abs(det(H))
    Hinv = inv(H)
    widths = [V / norm(cross(H[:, mod1(d + 1, 3)], H[:, mod1(d + 2, 3)])) for d in 1:3]
    b = beta === nothing ? 3.4 / minimum(widths) : Float64(beta)
    x = sqrt(digits * log(10.0)) + 1.2
    rcut = x / b
    kmax = 2.0 * b * x

    E = 0.0
    F = zeros(3, N)
    W = zeros(3, 3)

    # Real space: -½ c6 g_ess(b r) / r⁶
    nimg = [ceil(Int, rcut / widths[d]) + 1 for d in 1:3]
    for i in 1:N, j in 1:N
        c6 = s6[i] * s6[j]
        c6 == 0.0 && continue
        for n1 in -nimg[1]:nimg[1], n2 in -nimg[2]:nimg[2], n3 in -nimg[3]:nimg[3]
            (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
            d = R[:, i] - R[:, j] + H * [n1, n2, n3]
            r2 = dot(d, d)
            r2 < rcut^2 || continue
            x2 = b * b * r2
            g = _g_ess(x2)
            ir2 = 1.0 / r2
            ir6 = ir2^3
            E -= 0.5 * c6 * g * ir6
            fmag = -c6 * (6.0 * g + (x2^3) * exp(-x2)) * ir6 * ir2
            F[:, i] .+= fmag .* d
            W .+= 0.5 * fmag .* (d * d')
        end
    end

    # Reciprocal space (k ≠ 0):
    B = 2pi * transpose(Hinv)
    mmax = [ceil(Int, kmax / norm(B[:, d])) + 1 for d in 1:3]
    for m1 in -mmax[1]:mmax[1], m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
        (m1 == 0 && m2 == 0 && m3 == 0) && continue
        kv = B * [m1, m2, m3]
        k2 = dot(kv, kv)
        k2 < kmax^2 || continue
        k = sqrt(k2)
        gk, gpk = _phi6_hat_gauss(k, b)
        S = 0.0 + 0.0im
        for i in 1:N
            S += s6[i] * cis(dot(kv, R[:, i]))
        end
        s2 = abs2(S)
        Ek = -(0.5 / V) * gk * s2
        E += Ek
        for i in 1:N
            F[:, i] .-= (1.0 / V) * gk * s6[i] * imag(cis(dot(kv, R[:, i])) * conj(S)) .* kv
        end
        W .+= 0.5 * s2 .* (-(gk / V) .* I(3) .- (gpk / V) .* (kv * kv'))
    end

    # Self removal (+½ (b⁶/6) Σ s6²) and mandatory k = 0 mode (-½ Φ̂₆(0)/V (Σ s6)²):
    E += 0.5 * (b^6 / 6.0) * sum(abs2, s6)
    g0, _ = _phi6_hat_gauss(0.0, b)
    Ek0 = -(0.5 / V) * g0 * (sum(s6)^2)
    E += Ek0
    W .+= Ek0 .* I(3)

    # Intramolecular exceptions at minimum-image distance: +(1 - scale) s6_i s6_j / r⁶
    for item in exceptions
        i, j = Int(item[1]), Int(item[2])
        sc = length(item) == 3 ? Float64(item[3]) : 0.0
        d = R[:, i] - R[:, j]
        d .-= H * round.(Hinv * d)
        r2 = dot(d, d)
        ir2 = 1.0 / r2
        ir6 = ir2^3
        c6 = (1.0 - sc) * s6[i] * s6[j]
        E += c6 * ir6
        fmag = 6.0 * c6 * ir6 * ir2
        F[:, i] .+= fmag .* d
        F[:, j] .-= fmag .* d
        W .+= fmag .* (d * d')
    end

    return E, F, W
end

end # module DispersionReference
