"""
Independent Float64 CPU Ewald reference for the electrostatics tests.

The oracle shares no code with the GPU solver: it has its own erfc, picks its
own splitting parameter, and converges real-space images and reciprocal
vectors far beyond the production tolerances. The cell is a general matrix
`H` whose columns are the box vectors, so homogeneous shear strains can be
applied in finite-difference checks of the virial.

Hamiltonian: periodic Coulomb `C Σ' qᵢqⱼ/|rᵢⱼ + Hn|` with tin-foil boundary
conditions, optionally a uniform neutralising background, minus
`(1 − s) C qᵢqⱼ/r` for each exception pair at its minimum-image distance.
"""
module EwaldReference

using LinearAlgebra

export ewald_reference

function _erfc_ref(x::Float64)
    x < 0 && return 2.0 - _erfc_ref(-x)
    if x < 2.0
        # erfc(x) = 1 − 2/√π e^{−x²} Σ 2ⁿ x^{2n+1}/(1·3·…·(2n+1))   (positive terms)
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
    # erfc(x) = e^{−x²}/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …)))), evaluated backwards
    t = x
    for n in 400:-1:1
        t = x + (n / 2) / t
    end
    return exp(-x * x) / sqrt(pi) / t
end

"""
    ewald_reference(R, q, H; C=1.0, exceptions=(), background=false,
                    alpha=nothing, digits=16) -> (energy, forces, virial)

`R` is `3×N` (Cartesian), `q` length `N`, `H` the `3×3` cell matrix. Returns
the energy, `3×N` forces and the `3×3` configurational virial
`W_ab = Σ r_a F_b = −∂U/∂ε_ab`.
"""
function ewald_reference(R::AbstractMatrix, q::AbstractVector, H::AbstractMatrix;
                         C::Real=1.0, exceptions=(), background::Bool=false,
                         alpha=nothing, digits::Real=16)
    R = Float64.(R)
    q = Float64.(q)
    H = Float64.(H)
    C = Float64(C)
    N = size(R, 2)
    V = abs(det(H))
    Hinv = inv(H)
    widths = [V / norm(cross(H[:, mod1(d + 1, 3)], H[:, mod1(d + 2, 3)])) for d in 1:3]
    a = alpha === nothing ? 3.6 / minimum(widths) : Float64(alpha)
    x = sqrt(digits * log(10.0))            # erfc(x) ≈ e^{−x²}
    rcut = x / a
    kmax = 2a * x
    F = zeros(3, N)
    W = zeros(3, 3)
    E = 0.0
    # real space
    nimg = [ceil(Int, rcut / widths[d]) + 1 for d in 1:3]
    for i in 1:N, j in 1:N
        qq = C * q[i] * q[j]
        qq == 0 && continue
        for n1 in -nimg[1]:nimg[1], n2 in -nimg[2]:nimg[2], n3 in -nimg[3]:nimg[3]
            i == j && n1 == 0 && n2 == 0 && n3 == 0 && continue
            d = R[:, i] - R[:, j] + H * [n1, n2, n3]
            r = norm(d)
            r < rcut || continue
            ec = _erfc_ref(a * r)
            E += 0.5 * qq * ec / r
            fmag = qq * (ec / r + 2a / sqrt(pi) * exp(-a * a * r * r)) / (r * r)
            F[:, i] .+= fmag .* d
            W .+= 0.5 * fmag .* (d * d')
        end
    end
    # reciprocal space
    B = 2pi * transpose(Hinv)               # columns: reciprocal vectors
    mmax = [ceil(Int, kmax / norm(B[:, d]) * 1.0) + 1 for d in 1:3]
    for m1 in -mmax[1]:mmax[1], m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
        m1 == 0 && m2 == 0 && m3 == 0 && continue
        k = B * [m1, m2, m3]
        k2 = dot(k, k)
        k2 < kmax^2 || continue
        g = 4pi * exp(-k2 / (4a * a)) / k2
        S = 0.0 + 0.0im
        for i in 1:N
            S += q[i] * cis(dot(k, R[:, i]))
        end
        Ek = C / (2V) * g * abs2(S)
        E += Ek
        for i in 1:N
            F[:, i] .+= (C / V) * g * q[i] * imag(cis(dot(k, R[:, i])) * conj(S)) .* k
        end
        W .+= Ek .* (I - 2.0 .* (k * k') ./ k2 .* (1.0 + k2 / (4a * a)))
    end
    # self and background
    E -= C * a / sqrt(pi) * sum(abs2, q)
    if background
        Ebg = -C * pi * sum(q)^2 / (2 * a * a * V)
        E += Ebg
        W .+= Ebg .* I(3)
    end
    # exceptions at the minimum-image distance
    for item in exceptions
        i, j = Int(item[1]), Int(item[2])
        s = length(item) == 3 ? Float64(item[3]) : 0.0
        d = R[:, i] - R[:, j]
        d .-= H * round.(Hinv * d)
        r = norm(d)
        qq = (s - 1.0) * C * q[i] * q[j]
        E += qq / r
        f = qq / r^3 .* d
        F[:, i] .+= f
        F[:, j] .-= f
        W .+= qq / r^3 .* (d * d')
    end
    return E, F, W
end

end # module EwaldReference
