# Included in ParticleDynamics.DeviceEngine.
# Device pair models for the cooperative traversal. Each model is isbits; per-type
# or per-particle coefficient arrays travel separately in a `data` tuple so CUDA.jl
# converts them to device arrays without Adapt rules. Formulas copy the staged
# kernels exactly (src/nonbonded/*), so the staged golden records stay valid:
#   LJ   f/r = 24ε(2s¹² − s⁶)/r², u = 4ε(s¹² − s⁶) − u_c   (u_c = u(r_c) with the
#        potential-shift modifier, 0 otherwise; forces never change)
#   WCA  same force, u = 4ε(s¹² − s⁶) + ε, rc² = 1.2599211·σ²  (staged constant)
#   SR   f/r = (ε/σ)(1 − r/σ)/r, u = ½ε(1 − r/σ)², rc = σ     (harmonic repulsion)
# Mixed σ: σij = (σi + σj)/2, rc = factor·σij. Pair matrix: σ, ε, rc by typeid.

abstract type PairModel end
struct UniformLJ{T} <: PairModel; ε::T; σ::T; rc2::T; eshift::T; end
struct UniformWCA{T} <: PairModel; ε::T; σ::T; rc2::T; end
struct UniformSoftRepulsive{T} <: PairModel; ε::T; σ::T; rc2::T; end
struct MixedSigmaLJ{T} <: PairModel; ε::T; rcut_factor::T; eshift::T; end   # σ/rc is constant: one shift
struct MixedSigmaWCA{T} <: PairModel; ε::T; rcut_factor::T; end
struct PairMatrixLJ{SHIFT} <: PairModel end   # SHIFT: u(rc) per pair from the σ, ε, rc matrices
struct PairMatrixWCA <: PairModel end
struct NoPairs <: PairModel end   # no short-range term (bonded-only systems): pair forces are zero

# C²-smooth short-range LJ switch models (:force_switch = 1, :potential_switch = 2)
struct UniformSwitchedLJ{T,MODE} <: PairModel
    c6::T; c12::T; rsw::T; rsw2::T; rc2::T; inv_d::T
    A6::T; B6::T; C6::T; A12::T; B12::T; C12::T
end
struct PairMatrixSwitchedLJ{T,MODE} <: PairModel
    rsw::T; rsw2::T
end

function _fswitch_ab_c(p::Int, rsw::Float64, rc::Float64)
    d = rc - rsw
    rcp2 = rc^(p + 2)
    A = ((p + 4) * rc - (p + 1) * rsw) / (rcp2 * d^2)
    B = -((p + 3) * rc - (p + 1) * rsw) / (rcp2 * d^3)
    C = rc^(-p) + p * (A * d^3 / 3.0 + B * d^4 / 4.0)
    return A, B, C
end

function UniformSwitchedLJ(::Type{T}, ε::Real, σ::Real, rsw::Real, rc::Real, modifier::Symbol) where {T}
    s6 = Float64(σ)^6
    c6 = 4.0 * Float64(ε) * s6
    c12 = c6 * s6
    rswf = Float64(rsw); rcf = Float64(rc)
    if modifier === :force_switch
        A6, B6, C6 = _fswitch_ab_c(6, rswf, rcf)
        A12, B12, C12 = _fswitch_ab_c(12, rswf, rcf)
        return UniformSwitchedLJ{T,1}(T(c6), T(c12), T(rswf), T(rswf^2), T(rcf^2), T(1.0 / (rcf - rswf)),
                                      T(A6), T(B6), T(C6), T(A12), T(B12), T(C12))
    else
        return UniformSwitchedLJ{T,2}(T(c6), T(c12), T(rswf), T(rswf^2), T(rcf^2), T(1.0 / (rcf - rswf)),
                                      zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
    end
end

UniformWCA(ε::T,σ::T) where {T}=UniformWCA{T}(ε,σ,T(1.2599211)*(σ*σ))
UniformLJ{T}(ε,σ,rc2) where {T}=UniformLJ{T}(ε,σ,rc2,zero(T))
MixedSigmaLJ{T}(ε,rcut_factor) where {T}=MixedSigmaLJ{T}(ε,rcut_factor,zero(T))
PairMatrixLJ()=PairMatrixLJ{false}()
"""Pair energy at the cutoff, 4ε((σ/rc)¹² − (σ/rc)⁶) (potential-shift modifier)."""
@inline function lj_cutoff_energy(σ::T,ε::T,rc::T) where {T}
    rc>zero(T) || return zero(T)
    s2=(σ*σ)/(rc*rc); s6=s2*s2*s2
    return T(4)*ε*(s6*s6-s6)
end
UniformSoftRepulsive(ε::T,σ::T) where {T}=UniformSoftRepulsive{T}(ε,σ,σ*σ)

# Per-atom context hoisted out of the neighbor loop.
@inline pair_context(::Union{UniformLJ,UniformSwitchedLJ,UniformWCA,UniformSoftRepulsive},data,i)=nothing
@inline pair_context(::Union{MixedSigmaLJ,MixedSigmaWCA},data,i)=(@inbounds data[1][i])
@inline pair_context(::Union{PairMatrixLJ,PairMatrixSwitchedLJ,PairMatrixWCA},data,i)=(@inbounds data[1][i])

@inline function _lj_terms(r2::T,σ::T,ε::T) where {T}
    invr2=one(T)/r2; s2=(σ*σ)*invr2; s6=s2*s2*s2; s12=s6*s6
    return T(24)*ε*(T(2)*s12-s6)*invr2, T(4)*ε*(s12-s6)
end

# Returns (inside cutoff, force/r, pair energy).
@inline function pair_eval(m::UniformLJ{T},data,ctx,j,r2::T) where {T}
    inside=(r2>zero(T)) & (r2<m.rc2)
    f,u=_lj_terms(r2,m.σ,m.ε)
    return inside,f,u-m.eshift
end
@inline function pair_eval(m::UniformWCA{T},data,ctx,j,r2::T) where {T}
    inside=(r2>zero(T)) & (r2<m.rc2)
    f,u=_lj_terms(r2,m.σ,m.ε)
    return inside,f,u+m.ε
end
@inline function pair_eval(m::UniformSoftRepulsive{T},data,ctx,j,r2::T) where {T}
    inside=(r2>zero(T)) & (r2<m.rc2)
    r=sqrt(r2); x=one(T)-r/m.σ
    return inside,(m.ε/m.σ)*x/r,T(0.5)*m.ε*x*x
end
@inline function pair_eval(m::MixedSigmaLJ{T},data,σi,j,r2::T) where {T}
    σij=T(0.5)*(σi+(@inbounds data[1][j])); rc=m.rcut_factor*σij
    f,u=_lj_terms(r2,σij,m.ε)
    return (r2>zero(T)) & (r2<rc*rc),f,u-m.eshift
end
@inline function pair_eval(m::MixedSigmaWCA{T},data,σi,j,r2::T) where {T}
    σij=T(0.5)*(σi+(@inbounds data[1][j])); rc=m.rcut_factor*σij # staged: factor, not 2^(1/6)
    f,u=_lj_terms(r2,σij,m.ε)
    return (r2>zero(T)) & (r2<rc*rc),f,u+m.ε
end
@inline function pair_eval(::PairMatrixLJ{SHIFT},data,ti,j,r2::T) where {SHIFT,T}
    @inbounds begin
        tj=data[1][j]; σij=data[2][ti,tj]; εij=data[3][ti,tj]; rc=data[4][ti,tj]
    end
    f,u=_lj_terms(r2,σij,εij)
    SHIFT && (u-=lj_cutoff_energy(σij,εij,rc))
    return (r2>zero(T)) & (r2<rc*rc),f,u
end
@inline function pair_eval(::PairMatrixWCA,data,ti,j,r2::T) where {T}
    @inbounds begin
        tj=data[1][j]; σij=data[2][ti,tj]; εij=data[3][ti,tj]; rc=data[4][ti,tj]
    end
    f,u=_lj_terms(r2,σij,εij)
    return (r2>zero(T)) & (r2<rc*rc),f,u+εij
end

@inline function _lj_fswitch_eval(r2::T, c6::T, c12::T, rsw::T, rsw2::T, A6::T, B6::T, C6::T, A12::T, B12::T, C12::T) where {T}
    invr2 = one(T) / r2
    ir6 = invr2 * invr2 * invr2
    f = (T(12) * c12 * ir6 - T(6) * c6) * ir6 * invr2
    u = c12 * (ir6 * ir6 - C12) - c6 * (ir6 - C6)
    if r2 > rsw2
        r = sqrt(r2)
        x = r - rsw
        x2 = x * x; x3 = x2 * x; x4 = x2 * x2
        df = c12 * T(12) * muladd(B12, x3, A12 * x2) - c6 * T(6) * muladd(B6, x3, A6 * x2)
        f -= df / r
        u += c12 * T(12) * muladd(B12 / T(4), x4, (A12 / T(3)) * x3) - c6 * T(6) * muladd(B6 / T(4), x4, (A6 / T(3)) * x3)
    end
    return f, u
end

@inline function _lj_pswitch_eval(r2::T, c6::T, c12::T, rsw::T, rsw2::T, inv_d::T) where {T}
    invr2 = one(T) / r2
    ir6 = invr2 * invr2 * invr2
    f0 = (T(12) * c12 * ir6 - T(6) * c6) * ir6 * invr2
    u0 = (c12 * ir6 - c6) * ir6
    if r2 > rsw2
        r = sqrt(r2)
        t = (r - rsw) * inv_d
        t2 = t * t; t3 = t2 * t
        sw = one(T) + t3 * muladd(t, muladd(T(-6), t, T(15)), T(-10))
        om = one(T) - t
        dsw = T(-30) * t2 * om * om * inv_d
        return sw * f0 - (dsw / r) * u0, sw * u0
    end
    return f0, u0
end

@inline function pair_eval(m::UniformSwitchedLJ{T,1},data,ctx,j,r2::T) where {T}
    inside=(r2>zero(T)) & (r2<m.rc2)
    f,u=inside ? _lj_fswitch_eval(r2,m.c6,m.c12,m.rsw,m.rsw2,m.A6,m.B6,m.C6,m.A12,m.B12,m.C12) : (zero(T),zero(T))
    return inside,f,u
end
@inline function pair_eval(m::UniformSwitchedLJ{T,2},data,ctx,j,r2::T) where {T}
    inside=(r2>zero(T)) & (r2<m.rc2)
    f,u=inside ? _lj_pswitch_eval(r2,m.c6,m.c12,m.rsw,m.rsw2,m.inv_d) : (zero(T),zero(T))
    return inside,f,u
end
@inline function pair_eval(m::PairMatrixSwitchedLJ{T,MODE},data,ti,j,r2::T) where {T,MODE}
    @inbounds begin
        tj=data[1][j]; σij=data[2][ti,tj]; εij=data[3][ti,tj]; rc=data[4][ti,tj]
    end
    rc2=rc*rc
    inside=(r2>zero(T)) & (r2<rc2) & (εij>zero(T))
    inside || return false,zero(T),zero(T)
    s2=σij*σij; s6=s2*s2*s2; c6=T(4)*εij*s6; c12=c6*s6
    if MODE==1
        d=rc-m.rsw; d2=d*d; d3=d2*d
        rc4=rc2*rc2; rc8=rc4*rc4; rc14=rc8*rc4*rc2; rc6=rc4*rc2; rc12=rc6*rc6
        A6=(T(10)*rc-T(7)*m.rsw)/(rc8*d2); B6=-(T(9)*rc-T(7)*m.rsw)/(rc8*d3)
        C6=one(T)/rc6+T(6)*(A6*d3/T(3)+B6*d2*d2/T(4))
        A12=(T(16)*rc-T(13)*m.rsw)/(rc14*d2); B12=-(T(15)*rc-T(13)*m.rsw)/(rc14*d3)
        C12=one(T)/rc12+T(12)*(A12*d3/T(3)+B12*d2*d2/T(4))
        f,u=_lj_fswitch_eval(r2,c6,c12,m.rsw,m.rsw2,A6,B6,C6,A12,B12,C12)
        return true,f,u
    else
        f,u=_lj_pswitch_eval(r2,c6,c12,m.rsw,m.rsw2,one(T)/(rc-m.rsw))
        return true,f,u
    end
end

# Lennard-Jones (any model above) plus real-space Coulomb (charged systems on the dense list,
# 15 Oct 2026): the cluster backend's Ewald/prolate correction polynomial, so forces and
# energies equal the cluster path's. data = (LJ data..., charges); ctx = (LJ ctx, qᵢ).
# No Coulomb exceptions (excluded pairs run on the cluster backend).
struct ChargedPairs{M<:PairModel,CPL,T} <: PairModel
    lj::M
    coul::CPL
    halfself::T     # ½ C·self coefficient: the self energy −halfself·qᵢ² per particle
end
@inline pair_context(m::ChargedPairs,data,i)=(pair_context(m.lj,data,i),@inbounds data[end][i])
@inline pair_eval(m::ChargedPairs,data,ctx,j,r2)=(qj=@inbounds data[end][j]; _charged_eval(m,data,ctx,j,r2,qj))
# With packed coordinates (x, y, z, w): w is the charge qⱼ (`_pair_cooperative_kernel!`)
@inline pair_eval_w(m::ChargedPairs,data,ctx,j,r2,w)=_charged_eval(m,data,ctx,j,r2,w)
@inline function _charged_eval(m::ChargedPairs,data,ctx,j,r2::T,qj) where {T}
    ins,f,u=pair_eval(m.lj,data,ctx[1],j,r2)
    f=ifelse(ins,f,zero(T)); u=ifelse(ins,u,zero(T))
    c=m.coul
    inr=(r2>zero(T)) & (r2<c.rc2)
    cq=c.C*ctx[2]*qj
    x=muladd(r2,c.sx,-one(T))
    rinv=CUDA.rsqrt(r2); rinv3=rinv*rinv*rinv
    fc=cq*(rinv3-ClusterPairs.poly_eval(c,c.pf,x))
    uc=cq*(rinv+ClusterPairs.poly_eval(c,c.pe,x)-c.eshift)
    return ins | inr,f+ifelse(inr,fc,zero(T)),u+ifelse(inr,uc,zero(T))
end
# Packed coordinates' 4th component: unused by the uncharged models.
@inline pair_eval_w(m::PairModel,data,ctx,j,r2,w)=pair_eval(m,data,ctx,j,r2)
# Energy added once per particle (Coulomb self term); zero for every other model.
@inline pair_self_energy(m::PairModel,data,i,::Type{T}) where {T}=zero(T)
@inline pair_self_energy(m::ChargedPairs,data,i,::Type{T}) where {T}=(q=@inbounds data[end][i]; -m.halfself*q*q)
