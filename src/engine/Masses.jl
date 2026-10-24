# Feature: per-particle masses (`mass_particle`, `inv_mass_particle`).
# Kernels ask for mass-dependent coefficients through the functions below; without
# this feature they return the uniform-mass expressions the kernels used before,
# so uniform runs are unchanged bit for bit. Per-mass forms follow the staged
# kernels (`inv_mass_particle`).

struct MassFeature{V} <: EngineFeature
    m::V
    im::V
end

make_feature(::Type{MassFeature},st,spec)=st.mass_particle===nothing ? nothing :
    MassFeature(st.mass_particle,st.inv_mass_particle)
feature_signature(::Type{MassFeature},st,spec)=st.mass_particle===nothing ? nothing :
    map(pointer,(st.mass_particle,st.inv_mass_particle))
device_part(f::MassFeature)=(m=f.m,im=f.im)
sorted_real(f::MassFeature)=(f.m,f.im)
generic_integration(::MassFeature)=true
generic_kick(::MassFeature)=true

@inline _mp(fx::Tuple)=_find(fx,Val(:m))

@inline _pm(::Nothing,mass,i)=mass
@inline _pm(mp,mass,i)=@inbounds mp.m[i]
@inline _pim(::Nothing,mass,i)=inv(mass)
@inline _pim(mp,mass,i)=@inbounds mp.im[i]

"""Mass of particle `i` (T)."""
@inline mass_of(fx::Tuple,mass,i)=_pm(_mp(fx),mass,i)

"""NVE kick coefficient dt/(2m): uniform `c` or (dt/2)·m_i⁻¹."""
@inline kick_coef(fx::Tuple,c,dt::T,mass,i) where {T}=_kick_coef(_mp(fx),c,dt,mass,i)
@inline _kick_coef(::Nothing,c,dt::T,mass,i) where {T}=c
@inline _kick_coef(mp,c,dt::T,mass,i) where {T}=(dt/T(2))*_pim(mp,mass,i)

"""1/m in Float64 for the Langevin O step and B kicks."""
@inline inv_mass64(fx::Tuple,mA::Float64,mass,i)=_inv_mass64(_mp(fx),mA,mass,i)
@inline _inv_mass64(::Nothing,mA,mass,i)=1/mA
@inline _inv_mass64(mp,mA,mass,i)=Float64(_pim(mp,mass,i))

"""Half-kick factor dt/(2m) in Float64 (BAOAB B2; dt = 0 for BAOA)."""
@inline half_kick64(fx::Tuple,dtA::Float64,mA::Float64,mass,i)=_half_kick64(_mp(fx),dtA,mA,mass,i)
@inline _half_kick64(::Nothing,dtA,mA,mass,i)=dtA/(2mA)
@inline _half_kick64(mp,dtA,mA,mass,i)=dtA*Float64(_pim(mp,mass,i))/2

"""GJF position coefficients (b, b·dt/(2m))."""
@inline gjf_pos_coefs(fx::Tuple,g::T,dt::T,mass::T,i) where {T}=_gjf_pos_coefs(_mp(fx),g,dt,mass,i)
@inline function _gjf_pos_coefs(::Nothing,g::T,dt::T,mass::T,i) where {T}
    q=g*dt/(T(2)*mass); b=one(T)/(one(T)+q)
    return b,b*dt/(T(2)*mass)
end
@inline function _gjf_pos_coefs(mp,g::T,dt::T,mass::T,i) where {T}
    m⁻¹=_pim(mp,mass,i); q=g*dt*m⁻¹/T(2); b=one(T)/(one(T)+q)
    return b,b*dt*m⁻¹/T(2)
end

"""GJF velocity damping parameter q = γ dt/(2m) (T)."""
@inline gjf_q(fx::Tuple,g::T,dt::T,mass::T,i) where {T}=_gjf_q(_mp(fx),g,dt,mass,i)
@inline _gjf_q(::Nothing,g::T,dt::T,mass::T,i) where {T}=g*dt/(T(2)*mass)
@inline _gjf_q(mp,g::T,dt::T,mass::T,i) where {T}=g*dt*_pim(mp,mass,i)/T(2)

"""GJF velocity coefficients (dt/(2m), b/m) in Float64."""
@inline gjf_vel_coefs(fx::Tuple,dtA,mA,bA,mass,i)=_gjf_vel_coefs(_mp(fx),dtA,mA,bA,mass,i)
@inline _gjf_vel_coefs(::Nothing,dtA,mA,bA,mass,i)=(dtA/(2*mA),bA/mA)
@inline function _gjf_vel_coefs(mp,dtA,mA,bA,mass,i)
    imA=Float64(_pim(mp,mass,i))
    return dtA*imA/2.0,bA*imA
end

"""CSVR/NHC kick coefficient: uniform `coef` = dt/(2m) or dt/(2m_i)."""
@inline thermo_kick_coef(fx::Tuple,coef,mass,i)=_thermo_kick_coef(_mp(fx),coef,mass,i)
@inline _thermo_kick_coef(::Nothing,coef,mass,i)=coef
@inline _thermo_kick_coef(mp,coef,mass,i)=(coef*mass)*_pim(mp,mass,i)

"""Engine observables: (m_i, dt/(2m_i)) in Float64 from the uniform pair."""
@inline obs_mass(fx::Tuple,mass,h,i)=_obs_mass(_mp(fx),mass,h,i)
@inline _obs_mass(::Nothing,mass,h,i)=(mass,h)
@inline _obs_mass(mp,mass,h,i)=(Float64(@inbounds mp.m[i]),h*mass*Float64(@inbounds mp.im[i]))
