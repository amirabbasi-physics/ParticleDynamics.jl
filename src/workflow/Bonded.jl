using ..BondedForces

# ------------------------------------------------------------------
# Typed bonded force descriptors (angles, torsions, impropers)
# ------------------------------------------------------------------
#
# Every descriptor applies to the topology tuples whose label equals its
# `type` (`:default` when the topology section carries no labels). Angles are
# in radians; atom order is the topology order (angle vertex = middle atom).

"""
    HarmonicAngleForce(; k, theta0, type=:default)

`U = k(θ-θ0)²/2` on `topology.angles` (vertex = middle atom). `θ0 = π` (linear
equilibrium) is handled exactly at collinear geometry; any other `θ0` reports
collinear geometry as a domain error.
"""
@kwdef struct HarmonicAngleForce <: Force
    k
    theta0
    type = :default
end

"""
    CosineSquaredAngleForce(; k, theta0, type=:default)

`U = k(cosθ - cosθ0)²/2` on `topology.angles`.
"""
@kwdef struct CosineSquaredAngleForce <: Force
    k
    theta0
    type = :default
end

"""
    UreyBradleyAngleForce(; k_theta, theta0, k_r, r13, type=:default)

`U = kθ(θ-θ0)²/2 + k_r(r13 - r13⁰)²/2` on `topology.angles`; the 1–3 distance
term is part of this force and is not also counted as a bond.
"""
@kwdef struct UreyBradleyAngleForce <: Force
    k_theta
    theta0
    k_r
    r13
    type = :default
end

"""
    PeriodicDihedralForce(; k, n, phase, type=:default)

`U = Σ_m k_m[1 + cos(n_m φ - δ_m)]` on `topology.dihedrals`, with the IUPAC
signed dihedral `φ` of the ordered quartet. `k`, `n`, `phase` are scalars or
equal-length vectors (one Fourier term each); `n_m` are positive integers.
"""
@kwdef struct PeriodicDihedralForce <: Force
    k
    n
    phase
    type = :default
end

"""
    OPLSDihedralForce(; K1, K2, K3, K4=0, type=:default)

`U = ½[K1(1+cosφ) + K2(1-cos2φ) + K3(1+cos3φ) + K4(1-cos4φ)]` (LAMMPS `opls`
convention), compiled into the Fourier evaluator.
"""
@kwdef struct OPLSDihedralForce <: Force
    K1
    K2
    K3
    K4 = 0
    type = :default
end

"""
    RyckaertBellemansDihedralForce(; C, type=:default)

`U = Σ_{n=0}^{5} C_n cos(ψ)^n` with `ψ = φ - π` (polymer convention: trans
is `ψ = 0`). `C` holds six coefficients `C0..C5`.
"""
@kwdef struct RyckaertBellemansDihedralForce <: Force
    C
    type = :default
end

"""
    HarmonicImproperForce(; k, chi0, type=:default)

`U = k d²/2`, `d = wrap(χ - χ0)` into (-π, π], where `χ` is the signed dihedral
of the ordered improper quartet (torsion-style improper). Import adapters
must map their central-atom convention onto this order.
"""
@kwdef struct HarmonicImproperForce <: Force
    k
    chi0
    type = :default
end

"""
    PeriodicImproperForce(; k, n, phase, type=:default)

Fourier form of [`PeriodicDihedralForce`](@ref) applied to `topology.impropers`.
"""
@kwdef struct PeriodicImproperForce <: Force
    k
    n
    phase
    type = :default
end

const BondSectionForce = Union{HarmonicBondForce,FENEBondForce}
const AngleSectionForce = Union{HarmonicAngleForce,CosineSquaredAngleForce,UreyBradleyAngleForce}
const DihedralSectionForce = Union{PeriodicDihedralForce,OPLSDihedralForce,RyckaertBellemansDihedralForce}
const ImproperSectionForce = Union{HarmonicImproperForce,PeriodicImproperForce}
const BondedSectionForce = Union{BondSectionForce,AngleSectionForce,DihedralSectionForce,ImproperSectionForce}

_section_of(::BondSectionForce) = :bonds
_section_of(::AngleSectionForce) = :angles
_section_of(::DihedralSectionForce) = :dihedrals
_section_of(::ImproperSectionForce) = :impropers

# Concrete device kind of each descriptor (Fourier forms share one kind).
_bonded_kind(::HarmonicBondForce) = BondedForces.HarmonicBondKind()
_bonded_kind(::FENEBondForce) = BondedForces.FENEBondKind()
_bonded_kind(::HarmonicAngleForce) = BondedForces.HarmonicAngleKind()
_bonded_kind(::CosineSquaredAngleForce) = BondedForces.CosineSquaredAngleKind()
_bonded_kind(::UreyBradleyAngleForce) = BondedForces.UreyBradleyKind()
_bonded_kind(::Union{PeriodicDihedralForce,OPLSDihedralForce,PeriodicImproperForce}) = BondedForces.FourierTorsionKind()
_bonded_kind(::RyckaertBellemansDihedralForce) = BondedForces.RyckaertBellemansKind()
_bonded_kind(::HarmonicImproperForce) = BondedForces.HarmonicImproperKind()

function _finite_coeff(x, label)
    v = Float64(x)
    isfinite(v) || throw(ArgumentError("$(label) must be finite; got $(x)."))
    return v
end
function _positive_coeff(x, label)
    v = _finite_coeff(x, label)
    v > 0 || throw(ArgumentError("$(label) must be positive; got $(x)."))
    return v
end

# Exact linear equilibrium keeps the finite collinear limit (see BondedGeometry).
_angle_eq(x) = (v = _finite_coeff(x, "theta0"); abs(v - π) <= 1e-12 ? π : v)

# Host coefficient rows: one tuple of Float64 per descriptor.
_coeff_row(f::HarmonicBondForce) = (_finite_coeff(f.k, "HarmonicBondForce k"), _finite_coeff(f.r0, "HarmonicBondForce r0"))
_coeff_row(f::FENEBondForce) = (_finite_coeff(f.k, "FENEBondForce k"), _positive_coeff(f.R0, "FENEBondForce R0"))
_coeff_row(f::HarmonicAngleForce) = (_finite_coeff(f.k, "HarmonicAngleForce k"), _angle_eq(f.theta0))
_coeff_row(f::CosineSquaredAngleForce) = (_finite_coeff(f.k, "CosineSquaredAngleForce k"), cos(_finite_coeff(f.theta0, "theta0")))
_coeff_row(f::UreyBradleyAngleForce) = (_finite_coeff(f.k_theta, "k_theta"), _angle_eq(f.theta0),
                                        _finite_coeff(f.k_r, "k_r"), _finite_coeff(f.r13, "r13"))
_coeff_row(f::HarmonicImproperForce) = (_finite_coeff(f.k, "HarmonicImproperForce k"), _finite_coeff(f.chi0, "chi0"))
function _coeff_row(f::RyckaertBellemansDihedralForce)
    length(f.C) == 6 || throw(ArgumentError("RyckaertBellemansDihedralForce needs six coefficients C0..C5."))
    return ntuple(i -> _finite_coeff(f.C[i], "C$(i - 1)"), 6)
end

# Fourier term lists: vectors (k_m, n_m, δ_m).
function _fourier_terms(f::Union{PeriodicDihedralForce,PeriodicImproperForce})
    ks = f.k isa Number ? [f.k] : collect(f.k)
    ns = f.n isa Number ? [f.n] : collect(f.n)
    ps = f.phase isa Number ? [f.phase] : collect(f.phase)
    length(ks) == length(ns) == length(ps) && !isempty(ks) ||
        throw(ArgumentError("$(nameof(typeof(f))): k, n and phase need equal, nonzero lengths."))
    for n in ns
        (n isa Integer || (isfinite(n) && isinteger(n))) && n >= 1 ||
            throw(ArgumentError("$(nameof(typeof(f))): multiplicities must be positive integers; got $(n)."))
    end
    return [_finite_coeff(k, "k") for k in ks], Float64.(ns), [_finite_coeff(p, "phase") for p in ps]
end
function _fourier_terms(f::OPLSDihedralForce)
    K = (_finite_coeff(f.K1, "K1"), _finite_coeff(f.K2, "K2"), _finite_coeff(f.K3, "K3"), _finite_coeff(f.K4, "K4"))
    # ½K1(1+cosφ), ½K2(1+cos(2φ-π)), ½K3(1+cos3φ), ½K4(1+cos(4φ-π))
    return [K[1] / 2, K[2] / 2, K[3] / 2, K[4] / 2], [1.0, 2.0, 3.0, 4.0], [0.0, π, 0.0, π]
end

function _section_tuples(topology, section::Symbol)
    section === :bonds && return topology.bonds, topology.bond_types
    section === :angles && return topology.angles, topology.angle_types
    section === :dihedrals && return topology.dihedrals, topology.dihedral_types
    return topology.impropers, topology.improper_types
end

"""
Host description of the typed bonded families (also stored in the compiled
metadata as `:bonded_families`).
"""
struct BondedFamilySpec
    name::Symbol
    kind::Any
    tuples::Vector
    ptype::Vector{Int32}
    source_index::Vector{Int}
    rows::Vector      # coefficient rows (or Fourier term lists) per parameter id
end

function _compile_section(topology, section::Symbol, forces::Vector)
    tuples, types = _section_tuples(topology, section)
    isempty(tuples) && throw(ArgumentError("$(nameof(typeof(first(forces)))) requires `topology.$(section)` to be populated."))
    labels = isempty(types) ? fill(:default, length(tuples)) : types
    by_label = Dict{Symbol,Any}()
    for f in forces
        lbl = Symbol(f.type)
        haskey(by_label, lbl) &&
            throw(ArgumentError("Two $(section) forces are assigned to type $(lbl); combine them into one descriptor."))
        by_label[lbl] = f
    end
    unassigned = setdiff(unique(labels), keys(by_label))
    isempty(unassigned) || throw(ArgumentError("No force is assigned to $(section) type(s) $(join(string.(unassigned), ", "))."))

    specs = BondedFamilySpec[]
    # One family per concrete device kind; parameter ids index its descriptors.
    kinds = unique(typeof(_bonded_kind(f)) for f in forces)
    for K in kinds
        fam_forces = [f for f in forces if _bonded_kind(f) isa K]
        pid = Dict(Symbol(f.type) => Int32(i) for (i, f) in enumerate(fam_forces))
        sel = [t for t in eachindex(tuples) if haskey(pid, labels[t])]
        isempty(sel) && continue
        rows = K <: BondedForces.FourierTorsionKind ? [_fourier_terms(f) for f in fam_forces] :
                                                     [_coeff_row(f) for f in fam_forces]
        push!(specs, BondedFamilySpec(Symbol(section, :_, nameof(K)), K(), tuples[sel],
                                      Int32[pid[labels[t]] for t in sel], sel, rows))
    end
    return specs
end

function _device_coeffs(spec::BondedFamilySpec, ::Type{T}) where {T<:AbstractFloat}
    if spec.kind isa BondedForces.FourierTorsionKind
        counts = Int32[length(r[1]) for r in spec.rows]
        offsets = Int32[0; cumsum(counts)[1:end-1]]
        return (CuArray(offsets), CuArray(counts),
                CuArray(T.(reduce(vcat, (r[1] for r in spec.rows)))),
                CuArray(T.(reduce(vcat, (r[2] for r in spec.rows)))),
                CuArray(T.(reduce(vcat, (r[3] for r in spec.rows)))))
    end
    ncoef = length(first(spec.rows))
    return ntuple(c -> CuArray(T[row[c] for row in spec.rows]), ncoef)
end

"""
    build_bonded_terms(specs, N, D, T) -> BondedForces.BondedTerms

Upload compiled family specifications to the device.
"""
function build_bonded_terms(specs::Vector{BondedFamilySpec}, N::Integer, D::Integer, ::Type{T}) where {T<:AbstractFloat}
    fams = map(specs) do spec
        tbl = BondedForces.build_interaction_table(N, spec.tuples, spec.ptype)
        BondedForces.BondedFamily(spec.kind, tbl, _device_coeffs(spec, T), T, D;
                                  name=spec.name, source_index=spec.source_index)
    end
    return BondedForces.BondedTerms(Tuple(fams), D)
end

"""
    _compile_bonded_forces(system, forces, T) -> (build_kwargs, post_build or nothing, metadata)

Bond sections with a single untyped harmonic/FENE descriptor keep the legacy
uniform kernels; every other combination compiles to typed families.
"""
function _compile_bonded_forces(system, forces::Vector, ::Type{T}) where {T<:AbstractFloat}
    kwargs = Dict{Symbol,Any}()
    meta = Dict{Symbol,Any}()
    topology = system.topology
    by_section = Dict{Symbol,Vector{Any}}()
    for f in forces
        push!(get!(by_section, _section_of(f), Any[]), f)
    end

    specs = BondedFamilySpec[]
    bond_forces = get(by_section, :bonds, Any[])
    if length(bond_forces) == 1 && Symbol(only(bond_forces).type) === :default &&
       length(unique(topology.bond_types)) <= 1
        bkw, bmeta = _compile_bonded(system, only(bond_forces), T)
        merge!(kwargs, bkw); merge!(meta, bmeta)
    elseif !isempty(bond_forces)
        append!(specs, _compile_section(topology, :bonds, bond_forces))
        kwargs[:bonds] = topology.bonds      # exclusions and unwrapping
        meta[:bond_style] = :typed
    end
    for section in (:angles, :dihedrals, :impropers)
        fs = get(by_section, section, Any[])
        isempty(fs) || append!(specs, _compile_section(topology, section, fs))
    end
    for section in (:angles, :dihedrals, :impropers)
        tuples, _ = _section_tuples(topology, section)
        if !isempty(tuples) && !haskey(by_section, section)
            @warn "Topology has $(length(tuples)) $(section) but no $(section) force; they are not evaluated." maxlog=1
        end
    end
    any(s -> length(first(s.tuples)) == 4, specs) && length(system.box) != 3 &&
        throw(ArgumentError("Dihedral and improper forces require a 3D ParticleSystem."))
    meta[:bonded_families] = specs

    post = nothing
    if !isempty(specs)
        N = length(system); D = length(system.box)
        post = st -> SimulationCore.attach_bonded_terms!(st, build_bonded_terms(specs, N, D, T))
    end
    return kwargs, post, meta
end
