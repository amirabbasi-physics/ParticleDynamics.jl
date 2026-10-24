# =========================
#   Pair physics shared by all cluster-pair kernels
# =========================
# Coulomb: e = C qᵢqⱼ (1/r + P_e(r²)), f/r = C qᵢqⱼ (1/r³ − P_f(r²)) inside the
# cutoff, with the polynomials evaluated by Horner in x = r²·sx − 1 ∈ [−1, 1].
#   Ewald / prolate split: P_e = −corr, P_f = fcorr (long-range correction)
#   reaction field:        P_e = k r² − c, P_f = 2k
# An excluded pair drops the 1/r term and keeps P (the mesh, or the
# reaction-field medium, still sees it); beyond the cutoff it contributes
# −1/r when a mesh carries the full 1/r, otherwise nothing.
#
# Lennard-Jones: e = c12/r¹² − c6/r⁶ − shift, plain truncation at r_cut.

"""
    CoulombPolynomial{T,NE,NF}

Device description of the short-range Coulomb kernel: prefactor `C`, cutoff
`rc2`, energy shift `eshift`, whether a mesh carries the full `1/r`
(`has_mesh`), and monomial coefficients of the energy and force polynomials.
"""
struct CoulombPolynomial{T,NE,NF,MONO}
    C::T
    rc2::T
    sx::T
    eshift::T
    has_mesh::Bool
    pe::NTuple{NE,T}
    pf::NTuple{NF,T}
end

# MONO: monomial coefficients evaluated by Horner (Float32, low degree);
# otherwise Chebyshev coefficients evaluated by Clenshaw (Float64 accuracy).
@inline poly_eval(c::CoulombPolynomial{T,NE,NF,true}, p::NTuple, x::T) where {T,NE,NF} = horner(p, x)
@inline poly_eval(c::CoulombPolynomial{T,NE,NF,false}, p::NTuple, x::T) where {T,NE,NF} = ES.clenshaw(p, x)

# Chebyshev series in x → monomial coefficients in x.
function _chebyshev_to_monomial(c::AbstractVector{<:Real})
    n = length(c)
    t0 = zeros(n); t0[1] = 1.0
    m = c[1] .* t0
    n == 1 && return m
    t1 = zeros(n); t1[2] = 1.0
    m .+= c[2] .* t1
    for k in 3:n
        t2 = zeros(n)
        t2[2:end] .= 2.0 .* t1[1:(end - 1)]
        t2 .-= t0
        m .+= c[k] .* t2
        t0, t1 = t1, t2
    end
    return m
end

"""
    coulomb_polynomial(split, C; T=Float32, cutoff=nothing, eshift=0, tol=nothing, energy_tol=nothing)

Fit the long-range correction of an Ewald or prolate `split` on
`[0, cutoff²]` (the split's own cutoff for prolates). Tolerances are relative
to the largest value of each function. The Float32 defaults (`1e-7` for the
force polynomial, `1e-8` for the energy polynomial, which only energy
evaluations use) keep the pair force within ~1e-6 of `1/r³`, below the
resolution of the summed Float32 forces; Float64 uses `1e-13` (Gaussian) and
`1e-11` (prolate).
"""
function coulomb_polynomial(split::ES.AbstractSplit, C::Real; T::Type=Float32, cutoff=nothing,
                            eshift::Real=0.0, tol=nothing, energy_tol=nothing)
    # Float64: 1e-13 for the Gaussian split; the prolate correction is
    # evaluated by quadrature to ~1e-12, so its fit stops at 1e-11
    f64tol = split isa ES.ProlateSplit ? 1e-11 : 1e-13
    ftol = tol === nothing ? (T === Float64 ? f64tol : 1e-7) : Float64(tol)
    etol = energy_tol === nothing ? (T === Float64 ? f64tol : 1e-8) : Float64(energy_tol)
    rc = cutoff === nothing ? (split isa ES.ProlateSplit ? split.cutoff : throw(ArgumentError("cutoff required"))) :
         Float64(cutoff)
    rc2 = rc^2
    s_of(x) = (x + 1.0) * rc2 / 2
    src = split isa ES.ProlateSplit ? ES.ProlateCorrection(split) : split
    ce = ES.chebyshev_fit_adaptive(x -> -ES.long_range_correction(src, s_of(x))[1]; tol=etol)
    cf = ES.chebyshev_fit_adaptive(x -> ES.long_range_correction(src, s_of(x))[2]; tol=ftol)
    mono = T === Float32 && max(length(ce), length(cf)) <= 20
    pe = mono ? _chebyshev_to_monomial(ce) : ce
    pf = mono ? _chebyshev_to_monomial(cf) : cf
    return CoulombPolynomial{T,length(pe),length(pf),mono}(T(C), T(rc2), T(2 / rc2), T(eshift), true,
                                                           Tuple(T.(pe)), Tuple(T.(pf)))
end

"""
    coulomb_reaction_field(C, cutoff; eps_r=1, eps_rf=Inf, T=Float32)

Reaction-field Coulomb (`eps_rf = Inf` or `0` is the conducting medium) with
the energy shifted to zero at the cutoff:
`e = C qᵢqⱼ/εᵣ (1/r + k r² − c)`, `k = (ε_rf − εᵣ)/((2ε_rf + εᵣ) r_c³)`,
`c = 1/r_c + k r_c²`.
"""
function coulomb_reaction_field(C::Real, cutoff::Real; eps_r::Real=1.0, eps_rf::Real=Inf, T::Type=Float32)
    rc = Float64(cutoff)
    k = (isinf(eps_rf) || eps_rf == 0) ? 1 / (2rc^3) : (eps_rf - eps_r) / ((2eps_rf + eps_r) * rc^3)
    c = 1 / rc + k * rc^2
    rc2 = rc^2
    pe = (k * rc2 / 2 - c, k * rc2 / 2)     # k s − c with s = (x + 1) rc²/2
    pf = (2k,)
    return CoulombPolynomial{T,2,1,true}(T(C / eps_r), T(rc2), T(2 / rc2), zero(T), false, T.(pe), T.(pf))
end

@inline function horner(c::NTuple{N,T}, x::T) where {N,T}
    acc = c[N]
    @inbounds for k in (N - 1):-1:1
        acc = muladd(acc, x, c[k])
    end
    return acc
end

# LJ kinds: 0 none, 1 type table in shared memory, 2 separable (per-atom
# √c6, √c12, one cutoff), 3 type table read through the read-only cache.
const LJ_NONE = 0
const LJ_TABLE_SHARED = 1
const LJ_SEPARABLE = 2
const LJ_TABLE_GLOBAL = 3
const LJ_MIXED = 4   # per-particle σ: σij = (σi + σj)/2, rc = factor·σij, one ε
const MAX_SHARED_TABLE = 256

"""
    pair_force(r2, rinv, qq, c6, c12, lrc2, esh, excluded, coul, Val(LJK), Val(LJEXCL), Val(COUL), Val(E))
        -> (f, e)

Scalar force factor `f` (force on i = `f · (rᵢ − rⱼ)`) and pair energy `e`.
"""
@inline function pair_force(r2::T, rinv::T, qq::T, c6::T, c12::T, lrc2::T, esh::T, excluded::Bool,
                            coul::CoulombPolynomial{T}, ::Val{LJK}, ::Val{LJEXCL}, ::Val{COUL},
                            ::Val{E}) where {T,LJK,LJEXCL,COUL,E}
    f = zero(T)
    e = zero(T)
    rinv2 = rinv * rinv
    if LJK != LJ_NONE
        use = (r2 < lrc2) & !(LJEXCL & excluded)
        ir6 = rinv2 * rinv2 * rinv2
        f += ifelse(use, (T(12) * c12 * ir6 - T(6) * c6) * ir6 * rinv2, zero(T))
        E && (e += ifelse(use, (c12 * ir6 - c6) * ir6 - esh, zero(T)))
    end
    if COUL
        inr = r2 < coul.rc2
        x = muladd(r2, coul.sx, -one(T))
        pf = poly_eval(coul, coul.pf, x)
        rinv3 = rinv * rinv2
        fx = ifelse(excluded, ifelse(inr, -pf, ifelse(coul.has_mesh, -rinv3, zero(T))),
                    ifelse(inr, rinv3 - pf, zero(T)))
        cq = coul.C * qq
        f += cq * fx
        if E
            pe = poly_eval(coul, coul.pe, x)
            ex = ifelse(excluded, ifelse(inr, pe, ifelse(coul.has_mesh, -rinv, zero(T))),
                        ifelse(inr, rinv + pe - coul.eshift, zero(T)))
            e += cq * ex
        end
    end
    return f, e
end

"""
    lj_type_table(sigma, epsilon, rcut; T=Float32, shift=false) -> Vector{NTuple{4,T}}

Row-major `(c6, c12, r_cut², e_shift)` entries per type pair from `ntypes×ntypes`
σ, ε and cutoff matrices (`c6 = 4εσ⁶`, `c12 = 4εσ¹²`); `e_shift` is the pair
energy at the cutoff with `shift` (potential-shift modifier), −ε with `wca`
(WCA: LJ shifted by +ε, cut at its minimum), otherwise 0.
"""
function lj_type_table(sigma::AbstractMatrix, epsilon::AbstractMatrix, rcut::AbstractMatrix; T::Type=Float32,
                       shift::Bool=false, wca::Bool=false)
    n = size(sigma, 1)
    v = Vector{NTuple{4,T}}(undef, n * n)
    for a in 1:n, b in 1:n
        s6 = Float64(sigma[a, b])^6
        e = Float64(epsilon[a, b])
        rc = Float64(rcut[a, b])
        # potential shift: the pair energy at the cutoff, c12/rc¹² − c6/rc⁶
        esh = wca ? -e : (shift && rc > 0) ? 4e * s6 * (s6 / rc^12 - 1 / rc^6) : 0.0   # WCA: u + ε
        v[(a - 1) * n + b] = (T(4e * s6), T(4e * s6^2), T(rc^2), T(esh))
    end
    return v
end
