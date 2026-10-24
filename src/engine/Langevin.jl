# BAOAB (and GSM, whose staged stages are identical)
# inside the captured step graph. One kernel runs B1·A1·O·A2 per particle (no
# force is needed between them); B2 follows the force evaluation. Noise is
# Philox keyed on the particle tag and the device step, so it is graph-safe and
# independent of spatial sorting.
#
# Staged bookkeeping is reproduced exactly (LangevinIntegrators `_baoab_*`):
#   B (both halves): v += dt/(2m)·f;  dU += v·f (power, after the kick)
#   O: c = exp(-γ dt/m), β = noise·ξ (noise = √(2γT·dt_spec)),
#      v' = c v + √((1-c²)/(2γ dt m))·β,  v̄ = (v+v')/2,
#      dq += -γ|v̄|² + v̄·β/dt
#   B2 also writes Ekin = m|v|²/2.
# Intermediates are Float64, as in the staged kernels.
# BAOA kicks B1 with the full dt (staged passes 2dt to the B kernel); its
# `power` + `kinetic_refresh` stages equal a B2 with zero kick (dU += v·f, Ekin).
# VV/GJF (Grønbech-Jensen–Farago, `_vv_pos*`/`_vv_vel*`): positions from v, the
# old force and β; the old force is saved into the state's f0 slots (sorted with
# the particles) before the force kernel overwrites f; the velocity update uses
# f0, f and the same β (Philox redrawn), with the staged heat sign dq −= ….

const LANGEVIN_BAOAB=UInt8(1) # also GSM
const LANGEVIN_BAOA=UInt8(2)
const LANGEVIN_GJF=UInt8(3)

struct LangevinData{T,V<:CuArray{T,1},I,F}
    scheme::UInt8
    f0::F           # GJF: the state's old-force slots (NTuple); otherwise nothing
    gamma::V        # spec-owned, permuted in place by sorting
    noise::V        # spec-owned noise scale √(2γT dt_spec), permuted in place
    dq::V
    dU::V
    Ekin::V
    tags::I         # Int32 particle ids (SimulationState.tag) or `nothing` (slot = id)
    seed::UInt64
end

@inline _noise_id(tags::Nothing,i)=i
@inline _noise_id(tags,i)=@inbounds tags[i]

@inline _get(t::NTuple{D},i) where {D}=ntuple(d->@inbounds(t[d][i]),Val(D))
@inline function _put!(t::NTuple{D},i,x) where {D}
    for d in 1:D; @inbounds t[d][i]=x[d]; end
    return nothing
end
@inline _dot(a::NTuple{D},b::NTuple{D}) where {D}=sum(ntuple(d->a[d]*b[d],Val(D)))
@inline function _wrap(x::T,L::T) where {T}
    x-=(x>=T(0.5)*L)*L; x+=(x< -T(0.5)*L)*L
    return x
end

function _baoa_kernel!(r::NTuple{D,<:CuDeviceVector{T}},v::NTuple{D,<:CuDeviceVector{T}},
                       f::NTuple{D,<:CuDeviceVector{T}},gamma,noise,dq,dU,tags,state,
                       seed::UInt64,dt::T,mass::T,box::NTuple{D,T},kick::T,fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        dtA=Float64(dt); mA=Float64(mass)
        imA=inv_mass64(fx,mA,mass,i)
        h=Float64(kick)*imA/2                                      # kick = dt (BAOAB) or 2dt (BAOA)
        fi=map(Float64,_get(f,i))
        v1=map((a,b)->a+h*b,map(Float64,_get(v,i)),fi)          # B1
        dU[i]+=T(_dot(v1,fi))
        x=map((a,b)->Float64(a)+0.5dtA*b,_get(r,i),v1)           # A1
        g=Float64(gamma[i]); s=Float64(noise[i])
        c=exp(-g*dtA*imA); rr=sqrt((1.0-c*c)*imA/(2.0*g*dtA))      # staged per-mass form
        step=UInt64(state[NL.DN_STEP]); id=_noise_id(tags,i)
        xi=philox_normals(T,Val(D),seed,step,id,UInt32(0)) # state precision, as staged randn(T)
        beta=map(Float64,noise_add(fx,map(z->T(s)*z,xi),T,Val(D),seed,step,id,i,true))
        v2=map((a,b)->c*a+rr*b,v1,beta)                           # O
        vbar=map((a,b)->0.5*(a+b),v1,v2)
        dq[i]+=T(-g*_dot(vbar,vbar)+_dot(vbar,beta)/dtA)
        x2=map((a,b)->a+0.5dtA*b,x,v2)                            # A2
        _put!(v,i,map(T,v2))
        r0=_get(r,i)
        displace!(fx,i,map((a,b)->T(a)-b,x2,r0))
        _put!(r,i,map((a,L)->_wrap(T(a),L),x2,box))
    end
    return nothing
end

# Split BAOAB/BAOA: when a feature samples positions after A1 (collision
# counting, as staged), B1·A1 and O·A2 run as two kernels with a neighbor check
# and a sample in between. Positions and velocities are rounded to T between
# the halves (staged rounds after every stage as well); the noise draw is the
# same (Philox counter per step and particle).
_split_baoa(run,th::LangevinData)=th.scheme!=LANGEVIN_GJF && any_half_drift_sample(run)

function _ba_kernel!(r::NTuple{D,<:CuDeviceVector{T}},v::NTuple{D,<:CuDeviceVector{T}},
                     f::NTuple{D,<:CuDeviceVector{T}},dU,dt::T,mass::T,box::NTuple{D,T},kick::T,fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        dtA=Float64(dt); mA=Float64(mass)
        imA=inv_mass64(fx,mA,mass,i)
        h=Float64(kick)*imA/2
        fi=map(Float64,_get(f,i))
        v1=map((a,b)->a+h*b,map(Float64,_get(v,i)),fi)          # B1
        dU[i]+=T(_dot(v1,fi))
        r0=_get(r,i)
        x=map((a,b)->Float64(a)+0.5dtA*b,r0,v1)                  # A1
        displace!(fx,i,map((a,b)->T(a)-b,x,r0))
        _put!(v,i,map(T,v1))
        _put!(r,i,map((a,L)->_wrap(T(a),L),x,box))
    end
    return nothing
end

function _oa_kernel!(r::NTuple{D,<:CuDeviceVector{T}},v::NTuple{D,<:CuDeviceVector{T}},gamma,noise,dq,tags,state,
                     seed::UInt64,dt::T,mass::T,box::NTuple{D,T},fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        dtA=Float64(dt); mA=Float64(mass)
        imA=inv_mass64(fx,mA,mass,i)
        v1=map(Float64,_get(v,i))
        g=Float64(gamma[i]); s=Float64(noise[i])
        c=exp(-g*dtA*imA); rr=sqrt((1.0-c*c)*imA/(2.0*g*dtA))
        step=UInt64(state[NL.DN_STEP]); id=_noise_id(tags,i)
        xi=philox_normals(T,Val(D),seed,step,id,UInt32(0))
        beta=map(Float64,noise_add(fx,map(z->T(s)*z,xi),T,Val(D),seed,step,id,i,true))
        v2=map((a,b)->c*a+rr*b,v1,beta)                           # O
        vbar=map((a,b)->0.5*(a+b),v1,v2)
        dq[i]+=T(-g*_dot(vbar,vbar)+_dot(vbar,beta)/dtA)
        r0=_get(r,i)
        x2=map((a,b)->Float64(a)+0.5dtA*b,r0,v2)                 # A2
        displace!(fx,i,map((a,b)->T(a)-b,x2,r0))
        _put!(v,i,map(T,v2))
        _put!(r,i,map((a,L)->_wrap(T(a),L),x2,box))
    end
    return nothing
end

_ba_args(run,th::LangevinData)=(run.r,run.v,run.f,th.dU,run.dt,run.mass,run.neighbors.box,
    th.scheme==LANGEVIN_BAOA ? 2*run.dt : run.dt,run.fx)
_oa_args(run,th::LangevinData)=(run.r,run.v,th.gamma,th.noise,th.dq,th.tags,run.neighbors.state,th.seed,run.dt,run.mass,
    run.neighbors.box,run.fx)
function enqueue_ba!(run,th::LangevinData)
    s=run.neighbors.state; N=length(run.r[1])
    @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(s,_ba_kernel!,_ba_args(run,th)...)
    return nothing
end
function enqueue_oa!(run,th::LangevinData)
    s=run.neighbors.state; N=length(run.r[1])
    @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(s,_oa_kernel!,_oa_args(run,th)...)
    return nothing
end

function _b2_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},
                     dU,Ekin,dt::T,mass::T,fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(v[1]) && return nothing
    @inbounds begin
        dtA=Float64(dt); mA=Float64(mass)
        h=half_kick64(fx,dtA,mA,mass,i)                            # dt = 0: BAOA power + refresh
        fi=map(Float64,_get(f,i))
        v1=map((a,b)->a+h*b,map(Float64,_get(v,i)),fi)
        dU[i]+=T(_dot(v1,fi))
        _put!(v,i,map(T,v1))
        Ekin[i]=T(0.5)*mass_of(fx,mass,i)*T(_dot(v1,v1))
    end
    return nothing
end

# GJF positions; saves the current force as the old force for the velocity update.
function _gjf_pos_kernel!(r::NTuple{D,<:CuDeviceVector{T}},v::NTuple{D,<:CuDeviceVector{T}},
                          f::NTuple{D,<:CuDeviceVector{T}},f0::NTuple{D,<:CuDeviceVector{T}},gamma,noise,tags,
                          state,seed::UInt64,dt::T,mass::T,box::NTuple{D,T},fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(r[1]) && return nothing
    @inbounds begin
        g=gamma[i]
        b,coef=gjf_pos_coefs(fx,g,dt,mass,i)    # single assignment: no boxed closure captures
        s=noise[i]; step=UInt64(state[NL.DN_STEP]); id=_noise_id(tags,i)
        β=noise_add(fx,map(z->s*z,philox_normals(T,Val(D),seed,step,id,UInt32(0))),T,Val(D),seed,step,id,i,true)
        fi=_get(f,i)
        dp=map((u,c,e)->b*dt*u+coef*(dt*c+e),_get(v,i),fi,β)
        x=map((a,p,L)->_bd_wrap(a+p,L),_get(r,i),dp,box)
        displace!(fx,i,dp)
        _put!(f0,i,fi); _put!(r,i,x)
    end
    return nothing
end

function _gjf_vel_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f0::NTuple{D,<:CuDeviceVector{T}},
                          f::NTuple{D,<:CuDeviceVector{T}},gamma,noise,dq,dU,Ekin,tags,state,
                          seed::UInt64,dt::T,mass::T,fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(v[1]) && return nothing
    @inbounds begin
        g=gamma[i]
        q=gjf_q(fx,g,dt,mass,i)
        a=(one(T)-q)/(one(T)+q); b=one(T)/(one(T)+q)
        s=noise[i]; step=UInt64(state[NL.DN_STEP]); id=_noise_id(tags,i)
        # same β as the position kernel: white redrawn, OU states already advanced
        β=map(Float64,noise_add(fx,map(z->s*z,philox_normals(T,Val(D),seed,step,id,UInt32(0))),T,Val(D),seed,step,id,i,false))
        aA=Float64(a); bA=Float64(b); dtA=Float64(dt); gA=Float64(g); mA=Float64(mass)
        vp=map(Float64,_get(v,i)); f0i=map(Float64,_get(f0,i)); fi=map(Float64,_get(f,i))
        kf,nf=gjf_vel_coefs(fx,dtA,mA,bA,mass,i)
        vn=map((u,c0,c,e)->aA*u+kf*(aA*c0+c)+nf*e,vp,f0i,fi,β)
        v2=_dot(vn,vn); vbar=map((x,y)->0.5*(x+y),vp,vn)
        P_diss=-gA*v2; P_sto=_dot(vbar,map(e->e/bA,β))/dtA; P_cons=_dot(vn,fi)
        Ekin[i]=T(0.5)*mass_of(fx,mass,i)*T(v2)
        dq[i]=dq[i]-T(P_diss+P_sto)
        dU[i]=dU[i]+T(P_cons)
        _put!(v,i,map(T,vn))
    end
    return nothing
end

function _drift_args(run,th::LangevinData)
    th.scheme==LANGEVIN_GJF && return (run.r,run.v,run.f,th.f0,th.gamma,th.noise,th.tags,run.neighbors.state,
        th.seed,run.dt,run.mass,run.neighbors.box,run.fx)
    kick=th.scheme==LANGEVIN_BAOA ? 2*run.dt : run.dt
    return (run.r,run.v,run.f,th.gamma,th.noise,th.dq,th.dU,th.tags,run.neighbors.state,th.seed,run.dt,run.mass,
        run.neighbors.box,kick,run.fx)
end
function _kick_args(run,th::LangevinData)
    th.scheme==LANGEVIN_GJF && return (run.v,th.f0,run.f,th.gamma,th.noise,th.dq,th.dU,th.Ekin,th.tags,
        run.neighbors.state,th.seed,run.dt,run.mass,run.fx)
    return (run.v,run.f,th.dU,th.Ekin,th.scheme==LANGEVIN_BAOA ? zero(run.dt) : run.dt,run.mass,run.fx)
end
_drift_kernel(th::LangevinData)=th.scheme==LANGEVIN_GJF ? _gjf_pos_kernel! : _baoa_kernel!
_b2_kernel(th::LangevinData)=th.scheme==LANGEVIN_GJF ? _gjf_vel_kernel! : _b2_kernel!
