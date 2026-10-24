# CSVR (Bussi) thermostat inside the captured graph.
# Half-step form: the thermostat acts on the leapfrog
# velocities v(t+h/2), after the kick-drift and the position constraints, and the
# velocity scale enters the second kick:
#   drift   — the NVE kick-drift kernel (v(t) + h/2m·f(t), then the drift), position
#             constraints (their correction/h added to the velocities);
#   half    — K of v(t+h/2) and particle counts per bath: warp shuffles, one shared
#             atomic per warp, one Float64 global atomic per block and bath;
#   bath    — one thread per bath: dof = D·count − constraints, the Bussi update with
#             Philox draws (Marsaglia–Tsang gamma), K/heat/scale written to the spec
#             workspace, accumulators cleared for the next step;
#   kick    — v(t+h) = s_b·v(t+h/2) + h/2m·f(t+h) (then the velocity constraints).
# Why the half step: thermostatting the full-step velocity-Verlet kinetic energy samples
# stiff modes too hot — for a harmonic mode the configurational temperature is
# T/(1 − ω²h²/4). Rigid SPC/E water (librations ~500–1000 cm⁻¹) at h = 2 fs ran
# +0.11 kJ/mol per water and +28 bar against GROMACS at the same density (−1.25 kg/m³
# in NPT); at 1 fs both agreed. The kinetic energy of v(t+h/2) is what GROMACS's
# leap-frog v-rescale controls (the average of the half-step kinetic energies), and for
# a harmonic mode it makes the sampled configurational temperature exact.
# The bath arithmetic is Float64; everything else is in the state precision.

const CSVR_MAX_BATHS=16

struct CSVRData{T,V<:CuArray{T,1}}
    bath_id::CuArray{Int32,1}      # spec workspace, permuted with the particles
    target::V                      # per-bath target temperature (spec workspace)
    tau::V
    kinetic::V                     # spec workspace: K after the thermostat (observables)
    exchange::V                    # spec workspace: cumulative K_old − K_new
    scale::V                       # spec workspace: last velocity scale
    dof::V                         # spec workspace: D·count per bath
    Ekin::V
    kacc::CuArray{Float64,1}       # engine: per-bath K accumulated by kick2
    nacc::CuArray{Int32,1}         # engine: per-bath particle count
    seed::UInt64
    cdof::CuArray{Float64,1}       # holonomic constraints per bath, removed from its dof (3 per rigid water)
    kfull::CuArray{Float64,1}      # engine: per-bath K of the completed step (full step, reported)
    nfull::CuArray{Int32,1}
    interval::Int32                # coupling interval (steps): the Bussi update on steps ≡ 0 (mod interval)
end

nbaths(c::CSVRData)=length(c.target)

# Kick plus K and particle counts per bath (NHC's kick, and CSVR's full-step measurement with
# a zero kick): a shared-memory reduction per block, one Float64 atomic per bath and block.
function _csvr_kick_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},Ekin,bath_id,
                            kacc,nacc,coef::T,mass::T,::Val{B},fx) where {D,T,B}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    t=threadIdx().x
    ks=CuStaticSharedArray(Float64,CSVR_MAX_BATHS)
    ns=CuStaticSharedArray(Int32,CSVR_MAX_BATHS)
    if t<=B
        @inbounds ks[t]=0.0; @inbounds ns[t]=Int32(0)
    end
    sync_threads()
    if i<=length(v[1])
        @inbounds begin
            c=thermo_kick_coef(fx,coef,mass,i)                     # dt/(2m) or dt/(2m_i)
            u=map((a,b)->muladd(c,b,a),_get(v,i),_get(f,i))
            _put!(v,i,u)
            ek=T(0.5)*mass_of(fx,mass,i)*_dot(u,u)
            Ekin[i]=ek
            b=Int(bath_id[i])
            if 1<=b<=B
                Definitions.@red ks[b]+=Float64(ek)
                # staged: K includes frozen particles, dof excludes them
                counted(fx,i) && (Definitions.@red ns[b]+=Int32(1))
            end
        end
    end
    sync_threads()
    if t<=B
        @inbounds begin
            if ns[t]>0
                Definitions.@red kacc[t]+=ks[t]
                Definitions.@red nacc[t]+=ns[t]
            end
        end
    end
    return nothing
end

function _csvr_half_kernel!(v::NTuple{D,<:CuDeviceVector{T}},bath_id,kacc,nacc,mass::T,::Val{B},fx,
                            state=nothing,interval::Int32=Int32(1)) where {D,T,B}
    # coupling interval > 1: only coupling steps need the half-step K (uniform exit)
    (state!==nothing && interval>Int32(1) && UInt64(@inbounds state[NL.DN_STEP])%UInt64(interval)!=UInt64(0)) && return nothing
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    t=threadIdx().x
    ks=CuStaticSharedArray(Float64,CSVR_MAX_BATHS)
    ns=CuStaticSharedArray(Int32,CSVR_MAX_BATHS)
    if t<=B
        @inbounds ks[t]=0.0; @inbounds ns[t]=Int32(0)
    end
    sync_threads()
    ek=0.0; b=0; n=Int32(0)
    if i<=length(v[1])
        @inbounds begin
            u=_get(v,i)
            ek=Float64(T(0.5)*mass_of(fx,mass,i)*_dot(u,u))
            b=Int(bath_id[i])
            # staged: K includes frozen particles, dof excludes them
            n=counted(fx,i) ? Int32(1) : Int32(0)
        end
    end
    lane=(t-1)&31
    for bb in 1:B
        x=ifelse(b==bb,ek,0.0); c=ifelse(b==bb,n,Int32(0))
        for off in (16,8,4,2,1)
            x+=CUDA.shfl_down_sync(0xffffffff,x,off)
            c+=CUDA.shfl_down_sync(0xffffffff,c,off)
        end
        if lane==0 && (x!=0.0 || c!=Int32(0))
            Definitions.@red ks[bb]+=x
            Definitions.@red ns[bb]+=c
        end
    end
    sync_threads()
    if t<=B
        @inbounds begin
            if ks[t]!=0.0 || ns[t]!=Int32(0)
                Definitions.@red kacc[t]+=ks[t]
                Definitions.@red nacc[t]+=ns[t]
            end
        end
    end
    return nothing
end

# Scaled kick (optional) and the per-bath K of the result, Ekin per particle: warp shuffles,
# one shared atomic per warp and bath, one Float64 global atomic per block and bath.
# KICK=false only measures (velocity-constrained runs: after RATTLE).
function _csvr_kick_measure_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},Ekin,bath_id,
                                    scale,kacc,nacc,coef::T,mass::T,::Val{B},fx,::Val{KICK},
                                    ::Val{REPORT}=Val(true)) where {D,T,B,KICK,REPORT}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    t=threadIdx().x
    ks=CuStaticSharedArray(Float64,CSVR_MAX_BATHS)
    ns=CuStaticSharedArray(Int32,CSVR_MAX_BATHS)
    if REPORT && t<=B
        @inbounds ks[t]=0.0; @inbounds ns[t]=Int32(0)
    end
    REPORT && sync_threads()
    ek=0.0; b=0; n=Int32(0)
    if i<=length(v[1])
        @inbounds begin
            b=Int(bath_id[i])
            u=_get(v,i)
            if KICK
                s=(1<=b<=length(scale)) ? T(scale[b]) : one(T)
                kc=thermo_kick_coef(fx,coef,mass,i)
                u=map((a,g)->muladd(kc,g,s*a),u,_get(f,i))
                _put!(v,i,u)
            end
            e=T(0.5)*mass_of(fx,mass,i)*_dot(u,u)
            Ekin[i]=e; ek=Float64(e)
            n=counted(fx,i) ? Int32(1) : Int32(0)
        end
    end
    REPORT || return nothing   # keep Ekin current; no bath reduction on an ordinary step
    lane=(t-1)&31
    for bb in 1:B
        x=ifelse(b==bb,ek,0.0); cnt=ifelse(b==bb,n,Int32(0))
        for off in (16,8,4,2,1)
            x+=CUDA.shfl_down_sync(0xffffffff,x,off)
            cnt+=CUDA.shfl_down_sync(0xffffffff,cnt,off)
        end
        if lane==0 && (x!=0.0 || cnt!=Int32(0))
            Definitions.@red ks[bb]+=x
            Definitions.@red ns[bb]+=cnt
        end
    end
    sync_threads()
    if t<=B
        @inbounds begin
            if ks[t]!=0.0 || ns[t]!=Int32(0)
                Definitions.@red kacc[t]+=ks[t]
                Definitions.@red nacc[t]+=ns[t]
            end
        end
    end
    return nothing
end

function _csvr_scaled_kick_kernel!(v::NTuple{D,<:CuDeviceVector{T}},f::NTuple{D,<:CuDeviceVector{T}},bath_id,scale,
                                   coef::T,mass::T,fx) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(v[1]) && return nothing
    @inbounds begin
        b=Int(bath_id[i])
        s=(1<=b<=length(scale)) ? T(scale[b]) : one(T)
        c=thermo_kick_coef(fx,coef,mass,i)                     # dt/(2m) or dt/(2m_i)
        _put!(v,i,map((a,g)->muladd(c,g,s*a),_get(v,i),_get(f,i)))
    end
    return nothing
end

# Philox draws for bath b at `step` (counter word 1 ≥ 0x10000, disjoint from the
# Langevin streams): draw k gives one normal (Box–Muller from w0,w1,w2) and one
# independent uniform (w3), all in (0,1).
@inline function _csvr_draw(seed::UInt64,step::UInt64,b::Integer,k::Integer)
    w0,w1,w2,w3=philox4x32(b%UInt32,UInt32(0x00010000)+k%UInt32,step%UInt32,(step>>32)%UInt32,
        seed%UInt32,(seed>>32)%UInt32)
    z,_=_box_muller(_open01(Float64,w0,w1),(Float64(w2)+0.5)*2.0^-32)
    return z,(Float64(w3)+0.5)*2.0^-32
end

# Gamma(shape, 1) by Marsaglia–Tsang (staged `_csvr_gamma_sample`), Float64.
@inline function _csvr_gamma(shape::Float64,seed,step,b,k0::Int)
    shape>0 || return 0.0,k0
    k=k0; boost=1.0
    if shape<1
        _,u=_csvr_draw(seed,step,b,k); k+=1
        boost=max(u,eps())^(1/shape); shape+=1
    end
    d=shape-1/3; c=1/sqrt(9d)
    while true
        x,u=_csvr_draw(seed,step,b,k); k+=1
        w=1+c*x
        w<=0 && continue
        w3=w*w*w; x2=x*x
        (u<1-0.0331*x2*x2 || log(max(u,eps()))<0.5x2+d*(1-w3+log(w3))) && return boost*d*w3,k
    end
end

function _csvr_bath_kernel!(kacc,nacc,target,tau,kinetic,exchange,scale,dof,state,seed::UInt64,dt::Float64,dims::Float64,cdof,
                            interval::Int32=Int32(1))
    b=threadIdx().x
    b>length(target) && return nothing
    _csvr_bath_update!(b,kacc,nacc,target,tau,kinetic,exchange,scale,dof,state,seed,dt,dims,cdof,nothing,interval)
    return nothing
end

# The Bussi update of bath b from the accumulated K and particle count (device function: the
# bath kernel, and the last block of the fused CSVR prefix).
# The Bussi draws of bath b at `step` for `ndof` degrees of freedom: (r1, Σ of ndof−1 squared normals).
@inline function _csvr_draws(seed::UInt64,step::UInt64,b,ndof::Int)
    r1,_=_csvr_draw(seed,step,b,0)
    r2=ndof>1 ? 2*first(_csvr_gamma((ndof-1)/2,seed,step,b,1)) : 0.0
    return r1,r2
end

@inline function _csvr_bath_update!(b,kacc,nacc,target,tau,kinetic,exchange,scale,dof,state,seed::UInt64,
                                    dt::Float64,dims::Float64,cdof,draws=nothing,interval::Int32=Int32(1))
    @inbounds begin
        step=UInt64(state[NL.DN_STEP])
        # coupling steps only (GROMACS nsttcouple): the thermostat integrated over interval·dt;
        # between them the scale is 1 and the reported K, exchange and dof keep their values
        coupled=interval==Int32(1) || step%UInt64(interval)==UInt64(0)
        if !coupled
            kacc[b]=0.0; nacc[b]=Int32(0); scale[b]=1.0
            return nothing
        end
        Kold=kacc[b]; d=dims*Float64(nacc[b])-cdof[b]; dof[b]=d
        kacc[b]=0.0; nacc[b]=Int32(0)
        s=1.0; Knew=Kold
        Tt=Float64(target[b]); τ=Float64(tau[b])
        if d>0 && Kold>eps(Float32) && Tt>0 && τ>0
            ndof=max(1,floor(Int,d+0.5))
            Kbar=0.5*d*Tt; c1=exp(-(dt*Float64(interval))/τ); c2=(1-c1)*Kbar/(Kold*ndof)
            # precomputed draws (fused prefix: computed by another block meanwhile) when they
            # belong to this step and dof, otherwise drawn here
            r1,r2=(draws!==nothing && draws[3]==Float64(step) && draws[4]==Float64(ndof)) ? (draws[1],draws[2]) :
                _csvr_draws(seed,step,b,ndof)
            s2=max(c1+c2*(r1*r1+r2)+2r1*sqrt(max(c1*c2,0.0)),0.0)
            s=sqrt(s2); Knew=Kold*s2
        end
        kinetic[b]=Knew; exchange[b]+=Kold-Knew; scale[b]=s
    end
    return nothing
end

# Velocity and Ekin scaling per bath (NHC's chain scaling).
function _csvr_scale_kernel!(v::NTuple{D,<:CuDeviceVector{T}},Ekin,bath_id,scale) where {D,T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(v[1]) && return nothing
    @inbounds begin
        b=Int(bath_id[i])
        if 1<=b<=length(scale)
            s=scale[b]
            _put!(v,i,map(x->x*s,_get(v,i)))
            Ekin[i]*=s*s
        end
    end
    return nothing
end

"""Half-step kinetic energy and the Bussi update of every bath, after the drift and the
position constraints (`enqueue_drift!`); the scale enters the next kick."""
function enqueue_csvr_half!(run,c::CSVRData;launch::Bool=true)
    ws=run.neighbors; N=length(run.r[1]); D=length(run.r); B=nbaths(c)
    a1=(run.v,c.bath_id,c.kacc,c.nacc,run.mass,Val(B),run.fx,ws.state,c.interval)
    a2=(c.kacc,c.nacc,c.target,c.tau,c.kinetic,c.exchange,c.scale,c.dof,ws.state,c.seed,Float64(run.dt),Float64(D),c.cdof,c.interval)
    k1=@cuda launch=false NL._device_if_valid!(ws.state,_csvr_half_kernel!,a1...)
    k2=@cuda launch=false NL._device_if_valid!(ws.state,_csvr_bath_kernel!,a2...)
    if launch
        k1(ws.state,_csvr_half_kernel!,a1...;threads=256,blocks=cld(N,256))
        k2(ws.state,_csvr_bath_kernel!,a2...;threads=32,blocks=1)
    end
    return nothing
end

# K of the completed step per bath into the spec workspace (what observables read as the
# thermostat's kinetic energy); the accumulators are cleared for the next step.
function _csvr_report_kernel!(kfull,nfull,kinetic)
    b=threadIdx().x
    b>length(kinetic) && return nothing
    @inbounds begin
        nfull[b]>0 && (kinetic[b]=kfull[b])
        kfull[b]=0.0; nfull[b]=Int32(0)
    end
    return nothing
end

# Run integration: CSVR keeps the NVE drift (followed by `enqueue_csvr_half!`) and replaces
# the final kick by the scaled one. Without velocity constraints one kernel kicks and measures
# (Ekin per particle, K per bath of the completed step: reported, not thermostatted); with them
# the kick, the velocity constraints (also run again by the feature's after-kick: idempotent;
# host-force steps have no after-kick hook) and the measurement.
function _csvr_kick!(run,c::CSVRData;launch::Bool=true,advance::Bool=false,report::Bool=true)
    T=eltype(run.r[1]); ws=run.neighbors; N=length(run.r[1]); B=nbaths(c)
    coef=(run.dt/T(2))/run.mass
    constrained=any(constrains_velocities,run.features)
    a2=(c.kfull,c.nfull,c.kinetic)
    w=advance ? NL._device_if_valid_advance! : NL._device_if_valid!   # the report kernel is the last one
    k2=@cuda launch=false w(ws.state,_csvr_report_kernel!,a2...)
    if !constrained
        a1=(run.v,run.f,c.Ekin,c.bath_id,c.scale,c.kfull,c.nfull,coef,run.mass,Val(B),run.fx,Val(true),Val(report))
        wk=report ? NL._device_if_valid! : w
        k1=@cuda launch=false wk(ws.state,_csvr_kick_measure_kernel!,a1...)
        if launch
            k1(ws.state,_csvr_kick_measure_kernel!,a1...;threads=256,blocks=cld(N,256))
            report && k2(ws.state,_csvr_report_kernel!,a2...;threads=32,blocks=1)
        end
        return advance
    end
    a0=(run.v,run.f,c.bath_id,c.scale,coef,run.mass,run.fx)
    a1=(run.v,run.f,c.Ekin,c.bath_id,c.scale,c.kfull,c.nfull,coef,run.mass,Val(B),run.fx,Val(false))
    k0=@cuda launch=false NL._device_if_valid!(ws.state,_csvr_scaled_kick_kernel!,a0...)
    k1=@cuda launch=false NL._device_if_valid!(ws.state,_csvr_kick_measure_kernel!,a1...)
    foreach(f->constrain_velocities!(run,f;launch=false),run.features)
    if launch
        k0(ws.state,_csvr_scaled_kick_kernel!,a0...;threads=256,blocks=cld(N,256))
        foreach(f->constrain_velocities!(run,f;launch=true),run.features)
        k1(ws.state,_csvr_kick_measure_kernel!,a1...;threads=256,blocks=cld(N,256))
        k2(ws.state,_csvr_report_kernel!,a2...;threads=32,blocks=1)
    end
    return advance
end

# Refresh a lazy full-step bath total at a host boundary without a second kick.
function enqueue_csvr_report!(run,c::CSVRData)
    N=length(run.r[1]); ws=run.neighbors
    @cuda threads=256 blocks=cld(N,256) NL._device_if_valid!(ws.state,_csvr_kick_measure_kernel!,
        run.v,run.f,c.Ekin,c.bath_id,c.scale,c.kfull,c.nfull,zero(run.dt),run.mass,Val(nbaths(c)),run.fx,Val(false))
    @cuda threads=32 blocks=1 NL._device_if_valid!(ws.state,_csvr_report_kernel!,c.kfull,c.nfull,c.kinetic)
    return nothing
end
