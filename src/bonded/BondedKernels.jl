# ------------------------------------------------------------------
# Typed bonded families: per-interaction kernels and particle gather
# ------------------------------------------------------------------
#
# Pass 1 (one thread per unique interaction) computes the local forces once
# and stores them, with the interaction energy and full virial tensor, in
# interaction-local scratch. Pass 2 (one thread per particle) gathers them
# through the CSR incidence list and adds onto the force/energy/virial
# buffers; each participant receives U/K and W/K, so particle sums contain
# every interaction exactly once. No atomics, no competing writes.

"""
    BondedFamily

One concrete potential kind with its interaction table, coefficient arrays
(a tuple of device vectors indexed by parameter id) and preallocated scratch.
"""
struct BondedFamily{P<:BondedPotential,K,C,T}
    kind::P
    table::InteractionTable{K}
    coeffs::C
    lfx::CuVector{T}
    lfy::CuVector{T}
    lfz::CuVector{T}
    lE::CuVector{T}
    lW::CuMatrix{T}
    name::Symbol
    source_index::Vector{Int}   # topology index of each interaction (messages)
end

function BondedFamily(kind::BondedPotential, table::InteractionTable{K}, coeffs::Tuple,
                      ::Type{T}, D::Integer; name::Symbol, source_index=collect(1:length(table))) where {K,T<:AbstractFloat}
    K == arity(kind) || throw(ArgumentError("$(name): table arity $(K) does not match $(typeof(kind))"))
    D in (2, 3) || throw(ArgumentError("bonded families support 2D or 3D"))
    K == 4 && D != 3 && throw(ArgumentError("$(name): torsions and impropers require 3D"))
    n = length(table)
    lfz = D == 3 ? CUDA.zeros(T, n * K) : CUDA.zeros(T, 0)
    return BondedFamily{typeof(kind),K,typeof(coeffs),T}(kind, table, coeffs,
        CUDA.zeros(T, n * K), CUDA.zeros(T, n * K), lfz,
        CUDA.zeros(T, n), CUDA.zeros(T, n, D == 3 ? 6 : 3), name, collect(Int, source_index))
end

"""
    BondedTerms

All typed bonded families of a simulation plus a reusable device status
vector (first invalid interaction per family).
"""
struct BondedTerms{F<:Tuple}
    families::F
    status::CuVector{Int32}
    status_host::Vector{Int32}
    dim::Int
end

function BondedTerms(families::Tuple, D::Integer)
    nf = length(families)
    return BondedTerms{typeof(families)}(families, CUDA.fill(typemax(Int32), max(nf, 1)),
                                         fill(typemax(Int32), max(nf, 1)), Int(D))
end

Base.isempty(terms::BondedTerms) = isempty(terms.families)

# ---------------------------- device helpers ----------------------------

@inline _load(rx, ry, rz, i, ::Val{2}) = @inbounds SVector(rx[i], ry[i])
@inline _load(rx, ry, rz, i, ::Val{3}) = @inbounds SVector(rx[i], ry[i], rz[i])

@inline _mic_vec(d::SVector{2,T}, L::NTuple{2,T}, h::NTuple{2,T}) where {T} =
    SVector(mic_fast(d[1], h[1], L[1]), mic_fast(d[2], h[2], L[2]))
@inline _mic_vec(d::SVector{3,T}, L::NTuple{3,T}, h::NTuple{3,T}) where {T} =
    SVector(mic_fast(d[1], h[1], L[1]), mic_fast(d[2], h[2], L[2]), mic_fast(d[3], h[3], L[3]))

@inline _coeffs_at(coeffs::Tuple, p) = map(c -> @inbounds(c[p]), coeffs)

# Locally unwrapped chain: p1 = 0, p(a+1) = p(a) + mic(r(a+1) - r(a)).
@inline function _chain_positions(r::NTuple{2}, L, h)
    p1 = zero(r[1]); p2 = _mic_vec(r[2] - r[1], L, h)
    return (p1, p2)
end
@inline function _chain_positions(r::NTuple{3}, L, h)
    p1 = zero(r[1]); p2 = _mic_vec(r[2] - r[1], L, h); p3 = p2 + _mic_vec(r[3] - r[2], L, h)
    return (p1, p2, p3)
end
@inline function _chain_positions(r::NTuple{4}, L, h)
    p1 = zero(r[1]); p2 = _mic_vec(r[2] - r[1], L, h)
    p3 = p2 + _mic_vec(r[3] - r[2], L, h); p4 = p3 + _mic_vec(r[4] - r[3], L, h)
    return (p1, p2, p3, p4)
end

@inline function _local_forces(kind::Union{HarmonicBondKind,FENEBondKind}, coeffs, p, pos::NTuple{2})
    F1, F2, U, ok = pair_forces(kind, _coeffs_at(coeffs, p), pos[2] - pos[1])
    return (F1, F2), U, ok
end
@inline function _local_forces(kind::Union{HarmonicAngleKind,CosineSquaredAngleKind,UreyBradleyKind}, coeffs, p, pos::NTuple{3})
    Fi, Fj, Fk, U, ok = angle_forces(kind, _coeffs_at(coeffs, p), pos[1] - pos[2], pos[3] - pos[2])
    return (Fi, Fj, Fk), U, ok
end
@inline function _local_forces(kind::Union{HarmonicImproperKind,RyckaertBellemansKind}, coeffs, p, pos::NTuple{4})
    F1, F2, F3, F4, U, ok = torsion_forces(kind, _coeffs_at(coeffs, p), pos[2] - pos[1], pos[3] - pos[2], pos[4] - pos[3])
    return (F1, F2, F3, F4), U, ok
end
@inline function _local_forces(::FourierTorsionKind, coeffs, p, pos::NTuple{4})
    offset, count, kf, nf, δf = coeffs
    first = Int(@inbounds offset[p]) + 1
    cnt = Int(@inbounds count[p])
    F1, F2, F3, F4, U, ok = fourier_torsion_forces(kf, nf, δf, first, cnt,
                                                   pos[2] - pos[1], pos[3] - pos[2], pos[4] - pos[3])
    return (F1, F2, F3, F4), U, ok
end

# Full tensor W = Σ_a p_a ⊗ F_a in virial_components order.
@inline function _local_virial(pos::NTuple{K,SVector{2,T}}, F::NTuple{K,SVector{2,T}}) where {K,T}
    xx = zero(T); yy = zero(T); xy = zero(T)
    for a in 1:K
        xx += pos[a][1] * F[a][1]; yy += pos[a][2] * F[a][2]; xy += pos[a][1] * F[a][2]
    end
    return (xx, yy, xy)
end
@inline function _local_virial(pos::NTuple{K,SVector{3,T}}, F::NTuple{K,SVector{3,T}}) where {K,T}
    xx = zero(T); yy = zero(T); zz = zero(T); xy = zero(T); xz = zero(T); yz = zero(T)
    for a in 1:K
        xx += pos[a][1] * F[a][1]; yy += pos[a][2] * F[a][2]; zz += pos[a][3] * F[a][3]
        xy += pos[a][1] * F[a][2]; xz += pos[a][1] * F[a][3]; yz += pos[a][2] * F[a][3]
    end
    return (xx, yy, zz, xy, xz, yz)
end

function _bonded_interaction_kernel!(kind, coeffs, atoms::NTuple{K,<:Any}, ptype,
                                     rx, ry, rz, L::NTuple{D,T}, h::NTuple{D,T},
                                     lfx, lfy, lfz, lE, lW, status, fam::Int,
                                     ::Val{EV}, slot_of) where {K,D,T,EV}
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t > length(ptype) && return
    @inbounds begin
        p = ptype[t]
        r = ntuple(a -> _load(rx, ry, rz, _bslot(atoms[a][t], slot_of), Val(D)), Val(K))
        pos = _chain_positions(r, L, h)
        F, U, ok = _local_forces(kind, coeffs, p, pos)
        ok || CUDA.atomic_min!(pointer(status, fam), Int32(t))
        base = (t - 1) * K
        for a in 1:K
            lfx[base + a] = F[a][1]
            lfy[base + a] = F[a][2]
            D == 3 && (lfz[base + a] = F[a][D])
        end
        if EV
            lE[t] = U
            W = _local_virial(pos, F)
            for c in 1:length(W)
                lW[t, c] = W[c]
            end
        end
    end
    return
end

function _bonded_gather_kernel!(fx, fy, fz, E, V, offsets, entries,
                                lfx, lfy, lfz, lE, lW, ::Val{K}, ::Val{D}, ::Val{EV}, tag) where {K,D,EV}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(offsets) - 1 && return
    T = eltype(fx)
    @inbounds begin
        ax = zero(T); ay = zero(T); az = zero(T); e_acc = zero(T)
        w1 = zero(T); w2 = zero(T); w3 = zero(T); w4 = zero(T); w5 = zero(T); w6 = zero(T)
        row = _brow(i, tag)
        for idx in (offsets[row] + 1):offsets[row + 1]
            e = entries[idx]
            ax += lfx[e]; ay += lfy[e]
            D == 3 && (az += lfz[e])
            if EV
                t = (e - Int32(1)) ÷ Int32(K) + Int32(1)
                e_acc += lE[t]
                w1 += lW[t, 1]; w2 += lW[t, 2]; w3 += lW[t, 3]
                if D == 3
                    w4 += lW[t, 4]; w5 += lW[t, 5]; w6 += lW[t, 6]
                end
            end
        end
        fx[i] += ax; fy[i] += ay
        D == 3 && (fz[i] += az)
        if EV
            invK = inv(T(K))
            E[i] += e_acc * invK
            V[i, 1] += w1 * invK; V[i, 2] += w2 * invK; V[i, 3] += w3 * invK
            if D == 3
                V[i, 4] += w4 * invK; V[i, 5] += w5 * invK; V[i, 6] += w6 * invK
            end
        end
    end
    return
end

function _evaluate_family!(fam::BondedFamily{P,K,C,T}, status, idx::Int,
                           rx, ry, rz, L::NTuple{D,T}, fx, fy, fz, E, V, ev::Val) where {P,K,C,T,D}
    n = length(fam.table)
    n == 0 && return nothing
    h = map(x -> T(0.5) * x, L)
    tbl = fam.table
    k1 = CUDA.@cuda launch=false _bonded_interaction_kernel!(fam.kind, fam.coeffs, tbl.atoms, tbl.ptype,
        rx, ry, rz, L, h, fam.lfx, fam.lfy, fam.lfz, fam.lE, fam.lW, status, idx, ev, nothing)
    threads = min(256, n)
    k1(fam.kind, fam.coeffs, tbl.atoms, tbl.ptype, rx, ry, rz, L, h,
       fam.lfx, fam.lfy, fam.lfz, fam.lE, fam.lW, status, idx, ev, nothing; threads, blocks=cld(n, threads))
    N = length(rx)
    k2 = CUDA.@cuda launch=false _bonded_gather_kernel!(fx, fy, fz, E, V, tbl.offsets, tbl.entries,
        fam.lfx, fam.lfy, fam.lfz, fam.lE, fam.lW, Val(K), Val(D), ev, nothing)
    threads2 = min(256, N)
    k2(fx, fy, fz, E, V, tbl.offsets, tbl.entries, fam.lfx, fam.lfy, fam.lfz, fam.lE, fam.lW,
       Val(K), Val(D), ev, nothing; threads=threads2, blocks=cld(N, threads2))
    return nothing
end

_evaluate_families!(::Tuple{}, status, idx, args...) = nothing
function _evaluate_families!(fams::Tuple, status, idx, args...)
    _evaluate_family!(first(fams), status, idx, args...)
    return _evaluate_families!(Base.tail(fams), status, idx + 1, args...)
end

"""
    evaluate_bonded!(terms, rx, ry, rz, box, fx, fy, fz, E, V, compute_ev; check=true)

Add all typed bonded families onto the force buffers (and, with
`compute_ev`, onto the per-particle energy and virial buffers). In 2D pass
`rz`/`fz` as any same-typed placeholder. With `check=true` an invalid
geometry raises `DomainError` naming the family and topology index.
"""
function evaluate_bonded!(terms::BondedTerms, rx::CuVector{T}, ry::CuVector{T}, rz::CuVector{T},
                          box::NTuple{D,T}, fx::CuVector{T}, fy::CuVector{T}, fz::CuVector{T},
                          E::CuVector{T}, V::CuMatrix{T}, compute_ev::Bool; check::Bool=true) where {T<:AbstractFloat,D}
    isempty(terms) && return nothing
    fill!(terms.status, typemax(Int32))
    ev = compute_ev ? Val(true) : Val(false)
    _evaluate_families!(terms.families, terms.status, 1, rx, ry, rz, box, fx, fy, fz, E, V, ev)
    check && check_bonded_status(terms)
    return nothing
end

"""
    bonded_status(terms) -> Union{Nothing,Tuple{Symbol,Int}}

First invalid interaction of the last evaluation as `(family, topology index)`.
"""
function bonded_status(terms::BondedTerms)
    copyto!(terms.status_host, terms.status)
    for (f, fam) in enumerate(terms.families)
        t = terms.status_host[f]
        t == typemax(Int32) && continue
        return (fam.name, fam.source_index[Int(t)])
    end
    return nothing
end

function check_bonded_status(terms::BondedTerms)
    bad = bonded_status(terms)
    bad === nothing && return nothing
    name, index = bad
    throw(DomainError(index, "$(name) interaction $(index) has invalid geometry (zero length, collinear or out-of-domain); forces are not clamped."))
end

"""
    family_energies(terms) -> Vector{Pair{Symbol,Float64}}

Total energy of each family from its interaction-local scratch. Valid only
after an evaluation with energies requested.
"""
family_energies(terms::BondedTerms) =
    [fam.name => (length(fam.table) == 0 ? 0.0 : CUDA.mapreduce(Float64, +, fam.lE)) for fam in terms.families]
