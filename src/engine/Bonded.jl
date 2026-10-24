# Features: bonded interactions and bonded exclusions on the device engine.
# Staged order (`evaluate_forces_into_f!`): nonbonded pairs (bonded pairs skipped
# when `exclude_bonded`) → legacy harmonic/FENE bonds → typed bonded families →
# freeze spring → virial combination. The registry lists these features before
# `FreezeFeature`, so the after-force launches keep that order. Sorted bonded states
# Topology tables keep build-time particle ids; the shared kernels read rows
# through `tag` (slot → id) and partners through `slot_of` (id → slot), which the
# ordering refreshes at every sort (`nothing`: unsorted, slot = id, the staged path).
# Staged domain checks (FENE r ≥ R0, invalid typed geometry) read device flags on
# the host and throw. A graph cannot, so the flags are evaluated on the device:
# a failure latches the engine error state (every guarded kernel then skips, so
# the flags keep the failing values) and the feature's `at_boundary!` throws the
# staged error at the next host boundary.

function _latch_kernel!(state,flags,ok)
    if threadIdx().x==1 && blockIdx().x==1
        @inbounds for k in 1:length(flags)
            flags[k]!=ok && state[NL.DN_ERROR]==0 && (state[NL.DN_ERROR]=NL.DN_FEATURE_ERROR)
        end
    end
    return nothing
end
function _fill_kernel!(a,value)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i<=length(a) && (@inbounds a[i]=value)
    return nothing
end

"""Guarded launch over `n` threads (compile only with `launch=false`)."""
function _feature_launch!(run,kernel,args,n;launch::Bool,threads::Int=256)
    s=run.neighbors.state
    k=@cuda launch=false NL._device_if_valid!(s,kernel,args...)
    if launch && n>0
        t=min(threads,n)
        k(s,kernel,args...;threads=t,blocks=cld(n,t))
    end
    return nothing
end

"""Bonded configurations the engine runs: legacy harmonic/FENE bonds, typed
families and bonded exclusions, sorted or not."""
bonded_supported(st)=(st.bonds===nothing && st.bonded===nothing) ||
    ((st.bonding===nothing || st.bonding isa Union{Definitions.HarmonicBond,Definitions.FENEBond}) &&
     (st.bonded===nothing || st.bonded isa BondedForces.BondedTerms))

"""`(tag, slot_of)` for kernels of a sorted state with topology, else `(nothing, nothing)`."""
_topology_map(st)=st.tag===nothing || !SC._has_topology(st) ? (nothing,nothing) : (st.tag,SC._slot_of!(st))
# host-only (checked every step): pointers, no device work
_map_signature(st)=st.tag===nothing || !SC._has_topology(st) ? nothing :
    (pointer(st.tag),st.slot_of===nothing ? nothing : pointer(st.slot_of))

"""Engine observables: virial component `c` of particle `i`, plus the bonded rows
when a bonded feature writes them (staged `_combine_virial!` adds them too)."""
@inline obs_virial(fx::Tuple,virial,i,c)=_obs_virial(_find(fx,Val(:vbond)),virial,i,c)+_obs_cvir(_find(fx,Val(:vcons)),i,c)
@inline _obs_cvir(::Nothing,i,c)=0.0
@inline _obs_cvir(p,i,c)=Float64(@inbounds p.vcons[i,c])   # rigid-water constraint virial
@inline _obs_virial(::Nothing,virial,i,c)=Float64(@inbounds virial[i,c])
@inline _obs_virial(p,virial,i,c)=Float64(@inbounds virial[i,c])+Float64(@inbounds p.vbond[i,c])

# ---------------------------------------------------------------------------
# Bonded exclusions: bonded pairs feel no nonbonded potential (staged
# `BondExclusions`, per pair through the symmetric bond list).
struct BondExclusionFeature{I,G} <: EngineFeature
    index::I
    flat::I
    counts::I
    tag::G          # slot → particle id (sorted states) or nothing
end
BondExclusionFeature(index,flat,counts)=BondExclusionFeature(index,flat,counts,nothing)   # unsorted (slot = id)
make_feature(::Type{BondExclusionFeature},st,spec)=st.bonds!==nothing && st.exclude_bonded && st.nonbonded_backend!==:cluster ?
    BondExclusionFeature(st.bonds.index,st.bonds.flat,st.bonds.counts,_topology_map(st)[1]) : nothing
feature_signature(::Type{BondExclusionFeature},st,spec)=st.bonds===nothing ? nothing :
    (st.exclude_bonded,map(pointer,(st.bonds.index,st.bonds.flat,st.bonds.counts)),_map_signature(st))
device_part(f::BondExclusionFeature)=(bidx=f.index,bflat=f.flat,bcnt=f.counts,btag=f.tag)
generic_pairs(::BondExclusionFeature)=true   # the tuned uniform-LJ kernels have no exclusion hook

"""Pair kernels: is the pair (i, j) excluded from the nonbonded potential?"""
@inline excluded(fx::Tuple,i,j)=_excluded(_find(fx,Val(:bidx)),i,j)
@inline _excluded(::Nothing,i,j)=false
@inline function _excluded(p,i,j)
    r=BondedForces._brow(i,p.btag); idj=BondedForces._brow(j,p.btag)   # rows and entries are particle ids
    base=@inbounds p.bidx[r]; n=@inbounds p.bcnt[r]
    t=zero(n)
    while t<n
        (@inbounds p.bflat[base+t+1])==idj && return true
        t+=one(t)
    end
    return false
end

# ---------------------------------------------------------------------------
# Legacy uniform bonds (`st.bonds` + `st.bonding`): harmonic or FENE, the staged
# kernels. With energies they add the bond energy and overwrite the bonded virial
# rows. FENE first checks every bond for r ≥ R0 or a nonfinite length.
struct BondFeature{T,B,V,G,S} <: EngineFeature
    bonds::B
    fene::Bool
    k::T
    r0::T          # r0 (harmonic) or R0 (FENE)
    invalid::CuArray{Int32,1}
    virial::V      # st.virial_bonded
    tag::G         # slot → id, or nothing (unsorted)
    slot_of::S     # id → slot, or nothing
end
function make_feature(::Type{BondFeature},st::SC.SimulationState{T},spec) where {T}
    (st.bonds===nothing || st.bonding===nothing) && return nothing
    p=st.bonding.params; fene=st.bonding isa Definitions.FENEBond
    if fene   # staged parameter checks (`_require_fene_domain!`), once per capture
        R02=p.R0*p.R0
        isfinite(p.k) && p.k>=0 || throw(ArgumentError("FENE k must be finite and nonnegative"))
        isfinite(p.R0) && p.R0>0 && isfinite(R02) && R02>0 ||
            throw(ArgumentError("FENE R0 and its square must be finite and positive"))
    end
    return BondFeature(st.bonds,fene,T(p.k),T(fene ? p.R0 : p.r0),CUDA.fill(typemax(Int32),1),st.virial_bonded,
        _topology_map(st)...)
end
feature_signature(::Type{BondFeature},st,spec)=st.bonds===nothing || st.bonding===nothing ? nothing :
    (typeof(st.bonding),st.bonding.params,map(pointer,(st.bonds.index,st.bonds.flat,st.bonds.counts)),_map_signature(st))
device_part(f::BondFeature)=(vbond=f.virial,)

_bond_kernel(fene::Bool,::Val{2},energy::Bool)=fene ? (energy ? BondedForces._fene2_EV! : BondedForces._fene2_noE!) :
    (energy ? BondedForces._harmonic2_EV! : BondedForces._harmonic2_noE!)
_bond_kernel(fene::Bool,::Val{3},energy::Bool)=fene ? (energy ? BondedForces._fene3_EV! : BondedForces._fene3_noE!) :
    (energy ? BondedForces._harmonic3_EV! : BondedForces._harmonic3_noE!)

function after_force!(run,f::BondFeature;energy::Bool,launch::Bool=true)
    T=eltype(run.r[1]); N=length(run.r[1]); D=length(run.r); box=run.neighbors.box
    half=map(L->T(0.5)*L,box); b=f.bonds; threads=BondedForces._bond_threads(N)
    if f.fene && !isempty(b.flat)   # staged `_require_fene_domain!`, before any bond force
        _feature_launch!(run,_fill_kernel!,(f.invalid,typemax(Int32)),1;launch)
        _feature_launch!(run,BondedForces._fene_domain_kernel!,(f.invalid,run.r,b.index,b.flat,b.counts,box,f.r0*f.r0,
            f.tag,f.slot_of),N;launch)
        _feature_launch!(run,_latch_kernel!,(run.neighbors.state,f.invalid,typemax(Int32)),1;launch)
    end
    kernel=_bond_kernel(f.fene,Val(D),energy)
    args=energy ? (run.r...,run.f...,run.energy,f.virial,b.index,b.flat,b.counts,box...,half...,f.k,f.r0,f.tag,f.slot_of) :
                  (run.r...,run.f...,b.index,b.flat,b.counts,box...,half...,f.k,f.r0,f.tag,f.slot_of)
    _feature_launch!(run,kernel,args,N;launch,threads)
    return nothing
end

function at_boundary!(f::BondFeature)
    f.fene || return nothing
    first_invalid=only(Array(f.invalid))
    first_invalid==typemax(Int32) || throw(DomainError(first_invalid,
        "FENE bond incident on particle $first_invalid has nonfinite length or r >= R0=$(f.r0). Reduce the timestep or correct the configuration; forces are not clamped."))
    return nothing
end

# ---------------------------------------------------------------------------
# Typed bonded families (`st.bonded::BondedForces.BondedTerms`), the staged
# two-pass kernels. Without legacy bonds the staged path zeroes the bonded virial
# rows on energy evaluations before the families add to them.
struct BondedTermsFeature{B,V,G,S} <: EngineFeature
    terms::B
    zero_virial::Bool   # no legacy bonds
    virial::V           # st.virial_bonded
    tag::G              # slot → id, or nothing (unsorted)
    slot_of::S          # id → slot, or nothing
end
make_feature(::Type{BondedTermsFeature},st,spec)=st.bonded===nothing ? nothing :
    BondedTermsFeature(st.bonded,st.bonds===nothing || st.bonding===nothing,st.virial_bonded,_topology_map(st)...)
feature_signature(::Type{BondedTermsFeature},st,spec)=st.bonded===nothing ? nothing : (objectid(st.bonded),_map_signature(st))
device_part(f::BondedTermsFeature)=f.zero_virial ? (vbond=f.virial,) : nothing   # else BondFeature exposes the rows

_families_launch!(run,f,::Tuple{},idx,ev;launch)=nothing
function _families_launch!(run,f,fams::Tuple,idx,ev;launch)
    fam=first(fams); n=length(fam.table)
    if n>0
        T=eltype(run.r[1]); D=length(run.r); box=run.neighbors.box; h=map(x->T(0.5)*x,box)
        rz=D==3 ? run.r[3] : run.r[1]; fz=D==3 ? run.f[3] : run.f[1]   # 2D: placeholders, as staged
        tbl=fam.table; K=length(tbl.atoms)
        _feature_launch!(run,BondedForces._bonded_interaction_kernel!,(fam.kind,fam.coeffs,tbl.atoms,tbl.ptype,
            run.r[1],run.r[2],rz,box,h,fam.lfx,fam.lfy,fam.lfz,fam.lE,fam.lW,f.terms.status,idx,ev,f.slot_of),n;launch)
        _feature_launch!(run,BondedForces._bonded_gather_kernel!,(run.f[1],run.f[2],fz,run.energy,f.virial,
            tbl.offsets,tbl.entries,fam.lfx,fam.lfy,fam.lfz,fam.lE,fam.lW,Val(K),Val(D),ev,f.tag),length(run.r[1]);launch)
    end
    return _families_launch!(run,f,Base.tail(fams),idx+1,ev;launch)
end

function after_force!(run,f::BondedTermsFeature;energy::Bool,launch::Bool=true)
    energy && f.zero_virial && _feature_launch!(run,_fill_kernel!,(f.virial,zero(eltype(f.virial))),length(f.virial);launch)
    terms=f.terms
    isempty(terms) && return nothing
    _feature_launch!(run,_fill_kernel!,(terms.status,typemax(Int32)),length(terms.status);launch)
    _families_launch!(run,f,terms.families,1,energy ? Val(true) : Val(false);launch)
    _feature_launch!(run,_latch_kernel!,(run.neighbors.state,terms.status,typemax(Int32)),1;launch)
    return nothing
end

at_boundary!(f::BondedTermsFeature)=isempty(f.terms) ? nothing : BondedForces.check_bonded_status(f.terms)
