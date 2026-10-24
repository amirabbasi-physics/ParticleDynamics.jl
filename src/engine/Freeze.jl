# Feature: frozen particles (SimulationCore `Freeze.jl`).
#   hold   — after every position update, masked particles return to their
#            anchors; unwrapped coordinates move back by the same minimum-image
#            displacement, so they stay continuous (for the Euler–Heun midpoint
#            only the midpoint positions, as staged).
#   spring — every force evaluation adds f −= k·d, d = mic(r − anchor), for
#            masked particles and, with energy and `freeze_include_energy`,
#            Epot += k|d|²/2 (as staged).
#   dof    — CSVR/NHC degrees of freedom exclude masked particles while a freeze
#            is active (kinetic energy still includes them, as staged).
# Expiry (`freeze_until`) is a host decision before each step
# (`SC._freeze_active!`); it changes the fingerprint and recaptures. The mask and
# anchors are sorted with the particles whenever they exist, active or not.

const FREEZE_HOLD_ENGINE=SC.FREEZE_HOLD
const FREEZE_SPRING_ENGINE=SC.FREEZE_SPRING

struct FreezeFeature{T,M,A,U} <: EngineFeature
    active::Bool
    mode::UInt8
    mask::M           # UInt8 per particle (state-owned)
    anchors::A        # NTuple{D} anchor positions (state-owned)
    unwrap::U         # unwrapped coordinates held with the positions, or nothing
    k::T
    energy::Bool      # spring energy included in Epot on energy evaluations
end

_freeze_active(st)=st.freeze_mode!=SC.FREEZE_NONE && st.freeze_mask!==nothing
_freeze_anchors(st)=st.rz===nothing ? (st.freeze_rx,st.freeze_ry) : (st.freeze_rx,st.freeze_ry,st.freeze_rz)

function make_feature(::Type{FreezeFeature},st::SC.SimulationState{T},spec) where {T}
    st.freeze_mask===nothing && return nothing
    unwrap=st.rx_unwrap===nothing ? nothing : _unwrap_arrays(st)
    return FreezeFeature(_freeze_active(st),st.freeze_mode,st.freeze_mask,_freeze_anchors(st),unwrap,T(st.freeze_k),
        st.freeze_include_energy)
end
feature_signature(::Type{FreezeFeature},st,spec)=(st.freeze_mode,st.freeze_k,st.freeze_include_energy,
    st.freeze_mask===nothing ? nothing : map(pointer,(st.freeze_mask,_freeze_anchors(st)...)))
before_step!(::Type{FreezeFeature},st)=(SC._freeze_active!(st); nothing) # expiry, as staged
after_step!(::Type{FreezeFeature},st)=(st.force_freeze_spring=_freeze_active(st) && st.freeze_mode==SC.FREEZE_SPRING; nothing)
device_part(f::FreezeFeature)=f.active ? (frozen=f.mask,) : nothing
sorted_real(f::FreezeFeature)=f.anchors
sorted_int(f::FreezeFeature)=(f.mask,)
feature_active(f::FreezeFeature)=f.active
fusable(f::FreezeFeature)=!f.active          # an expired freeze does not block the fused NVE path

"""Thermostat dof: is particle `i` counted (not frozen)?"""
@inline counted(fx::Tuple,i)=_counted(_find(fx,Val(:frozen)),i)
@inline _counted(::Nothing,i)=true
@inline _counted(p,i)=@inbounds p.frozen[i]==0

@inline _mic_tuple(d::NTuple{D},box::NTuple{D},half::NTuple{D}) where {D}=map((x,L,h)->NL.mic_fast(x,h,L),d,box,half)

function _hold_kernel!(r::NTuple{D,<:CuDeviceVector{T}},unwrap,mask,anchors::NTuple{D},box,half) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds if mask[i]!=0
        a=_get(anchors,i)
        unwrap===nothing || _put!(unwrap,i,map(+,_get(unwrap,i),_mic_tuple(map(-,a,_get(r,i)),box,half)))
        _put!(r,i,a)
    end
    return nothing
end

function _spring_kernel!(r::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},energy,mask,
                         anchors::NTuple{D},k::T,box,half) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds if mask[i]!=0
        d=_mic_tuple(map(-,_get(r,i),_get(anchors,i)),box,half)
        _put!(f,i,map((a,b)->a-k*b,_get(f,i),d))
        energy===nothing || (energy[i]+=T(0.5)*k*_dot(d,d))
    end
    return nothing
end

"""Hold: masked particles back to their anchors after a position update
(the Heun midpoint holds positions only)."""
function after_position!(run,fz::FreezeFeature;midpoint::Bool=false,launch::Bool=true)
    (fz.active && fz.mode==FREEZE_HOLD_ENGINE) || return nothing
    s=run.neighbors.state; N=length(run.r[1]); u=midpoint ? nothing : fz.unwrap
    box=run.neighbors.box; half=map(L->eltype(box)(0.5)*L,box)
    k=@cuda launch=false NL._device_if_valid!(s,_hold_kernel!,run.r,u,fz.mask,fz.anchors,box,half)
    launch && k(s,_hold_kernel!,run.r,u,fz.mask,fz.anchors,box,half;threads=256,blocks=cld(N,256))
    return nothing
end

"""Spring: tether force (and energy) after a force evaluation."""
function after_force!(run,fz::FreezeFeature;energy::Bool,launch::Bool=true)
    (fz.active && fz.mode==FREEZE_SPRING_ENGINE) || return nothing
    s=run.neighbors.state; N=length(run.r[1])
    e=energy && fz.energy ? run.energy : nothing
    box=run.neighbors.box; half=map(L->eltype(box)(0.5)*L,box)
    k=@cuda launch=false NL._device_if_valid!(s,_spring_kernel!,run.r,run.f,e,fz.mask,fz.anchors,fz.k,box,half)
    launch && k(s,_spring_kernel!,run.r,run.f,e,fz.mask,fz.anchors,fz.k,box,half;threads=256,blocks=cld(N,256))
    return nothing
end
