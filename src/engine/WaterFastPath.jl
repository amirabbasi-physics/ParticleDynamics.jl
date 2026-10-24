# Included in ParticleDynamics.DeviceEngine. Fused integration kernels for cluster-backend
# runs whose particles are all in
# rigid three-site waters, with per-particle masses (and unwrapped coordinates) as the only
# other features, under NVE or CSVR. One thread per water.
#   W1 `_rw_drift_kernel!` — kick v(t)+h/2m·f(t), drift, SHAKE against the entry positions
#      (r(t); no saved reference), velocity correction, unwrapped coordinates, the CSVR
#      half-step kinetic energy, and the cluster-list displacement ratios (build/prune
#      triggers, as `_displacement_kernel!`). Replaces kick-drift, SHAKE, its status latch,
#      the CSVR half-step reduction and the displacement kernel.
#   W2 `_rw_kick_kernel!` — v(t+h) = s·v(t+h/2) + h/2m·f(t+h), RATTLE, Ekin and the
#      full-step kinetic energy per bath (reported). Replaces the scaled kick, two RATTLE
#      passes, the full-step measurement and the reference save.
# Same arithmetic as the generic path (`shake_water`, `rattle_water`, the CSVR bath kernel);
# reductions are warp shuffles plus one atomic per warp and block.

const RW_FAST_FEATURES=(UnwrapFeature,MassFeature,RigidWaterFeature)
"""Use the fast path when it applies (diagnostics: `RW_FAST[] = false` for the generic path)."""
const RW_FAST=Ref(true)

"""Does `run` take the rigid-water fast path (cluster pairs, only rigid water + masses
(+ unwrapping), every particle in a water, NVE or CSVR)?"""
function rw_fast_path(run)
    RW_FAST[] || return false
    run.cluster===nothing && return false
    th=run.thermostat
    (th===nothing || th isa CSVRData) || return false
    i=findfirst(f->f isa RigidWaterFeature,run.features)
    i===nothing && return false
    all(f->any(F->f isa F,RW_FAST_FEATURES),run.features) || return false
    3*length(run.features[i].ws.atoms)==length(run.r[1]) || return false
    return true
end
_rw_feature(run)=run.features[findfirst(f->f isa RigidWaterFeature,run.features)]

# Block reductions (blockDim a multiple of 32): warp shuffles, one shared atomic per warp
# into the caller's slot, every thread returns the block total. Distinct calls use distinct
# slots, so a slot is never cleared while another thread still reads it.
@inline function _rw_block_sum(x::Float64,sh,slot::Int)
    threadIdx().x==1 && (@inbounds sh[slot]=0.0)
    sync_threads()
    for off in (16,8,4,2,1)
        x+=CUDA.shfl_down_sync(0xffffffff,x,off)
    end
    ((threadIdx().x-1)&31)==0 && x!=0.0 && Definitions.@red sh[slot]+=x
    sync_threads()
    return @inbounds sh[slot]
end
# Max of non-negative Float32 values as their UInt32 bit patterns (ordered alike).
@inline function _rw_block_max(x::Float32,sh,slot::Int)
    threadIdx().x==1 && (@inbounds sh[slot]=UInt32(0))
    sync_threads()
    for off in (16,8,4,2,1)
        x=max(x,CUDA.shfl_down_sync(0xffffffff,x,off))
    end
    ((threadIdx().x-1)&31)==0 && CUDA.atomic_max!(pointer(sh,slot),reinterpret(UInt32,x))
    sync_threads()
    return @inbounds sh[slot]
end
@inline function _rw_latch!(state)
    @inbounds state[NL.DN_ERROR]==0 && (state[NL.DN_ERROR]=NL.DN_FEATURE_ERROR)
    return nothing
end

function _rw_drift_kernel!(state,atoms,weights,status,r,v,f,unwrap,geom,lengths2,tol::T,maxiter::Int32,dt::T,
                           thermo,flags,ref0,pref0,tb2::T,tp2::T,::Val{CSVR},::Val{UNWRAP},
                           ::Val{FIXED}=Val(false)) where {T,CSVR,UNWRAP,FIXED}
    rx,ry,rz=r; vx,vy,vz=v; fx,fy,fz=f; bath_id,kacc,nacc=thermo
    rx0,ry0,rz0=ref0; px0,py0,pz0=pref0
    w=(blockIdx().x-1)*blockDim().x+threadIdx().x
    ek=0.0; rb=0.0f0; rp=0.0f0; b=0
    if w<=length(atoms)
        @inbounds begin
            box=Definitions.box_lengths(geom,Val(3))
            ids=atoms[w]; im=weights[w]
            raw=RWC._load_water(rx,ry,rz,ids)
            origin=raw[1]
            old=RWC.local_water(raw,origin,box)
            v0=RWC._load_water(vx,vy,vz,ids); f0=RWC._load_water(fx,fy,fz,ids)
            h=dt/T(2)
            vh=(v0[1]+(h*im[1])*f0[1],v0[2]+(h*im[2])*f0[2],v0[3]+(h*im[3])*f0[3])
            trial=(old[1]+dt*vh[1],old[2]+dt*vh[2],old[3]+dt*vh[3])
            x,ok,_=RWC.shake_water(old,trial,im,lengths2,tol,maxiter)
            status[w]=ok ? Int32(0) : Int32(1)
            if !ok
                _rw_latch!(state)
            else
                for k in 1:3
                    j=ids[k]
                    vk=vh[k]+(x[k]-trial[k])/dt
                    p=RWC._image(origin+x[k],box)
                    rx[j]=p[1]; ry[j]=p[2]; rz[j]=p[3]
                    vx[j]=vk[1]; vy[j]=vk[2]; vz[j]=vk[3]
                    if UNWRAP
                        d=x[k]-old[k]
                        unwrap[1][j]+=d[1]; unwrap[2][j]+=d[2]; unwrap[3][j]+=d[3]
                    end
                    CSVR && (ek+=Float64(T(0.5)/im[k]*RWC._dot3(vk,vk)))
                    if FIXED   # fixed list intervals: the decision reads only non-finite motion
                        (isfinite(p[1]) & isfinite(p[2]) & isfinite(p[3])) || (rb=Inf32)
                    else
                        dx=ClusterPairs._mic(p[1]-rx0[j],box[1]); dy=ClusterPairs._mic(p[2]-ry0[j],box[2]); dz=ClusterPairs._mic(p[3]-rz0[j],box[3])
                        r=ClusterPairs._ratio(dx*dx+dy*dy+dz*dz,tb2); rb=max(rb,isfinite(r) ? Float32(r) : Inf32)
                        dx=ClusterPairs._mic(p[1]-px0[j],box[1]); dy=ClusterPairs._mic(p[2]-py0[j],box[2]); dz=ClusterPairs._mic(p[3]-pz0[j],box[3])
                        rp=max(rp,Float32(ClusterPairs._ratio(dx*dx+dy*dy+dz*dz,tp2)))
                    end
                end
                CSVR && (b=Int(bath_id[ids[1]]))
            end
        end
    end
    sh=CuStaticSharedArray(UInt32,2)
    ra=_rw_block_max(max(rb,rp),sh,1)
    rbb=_rw_block_max(rb,sh,2)
    if threadIdx().x==1
        CUDA.atomic_max!(pointer(flags,1),ra)
        CUDA.atomic_max!(pointer(flags,2),rbb)
    end
    if CSVR
        # one bath for every water of the fast path (checked on the host)
        sd=CuStaticSharedArray(Float64,2)
        e=_rw_block_sum(ek,sd,1)
        n=_rw_block_sum(b>0 ? 3.0 : 0.0,sd,2)
        if threadIdx().x==1 && n>0
            Definitions.@red kacc[1]+=e
            Definitions.@red nacc[1]+=Int32(n)
        end
    end
    return nothing
end

function _rw_kick_kernel!(state,atoms,weights,status,r,v,f,geom,dt::T,thermo,::Val{CSVR}) where {T,CSVR}
    rx,ry,rz=r; vx,vy,vz=v; fx,fy,fz=f; scale,Ekin,kfull,nfull=thermo
    w=(blockIdx().x-1)*blockDim().x+threadIdx().x
    ek=0.0; n=0.0
    if w<=length(atoms)
        @inbounds begin
            box=Definitions.box_lengths(geom,Val(3))
            ids=atoms[w]; im=weights[w]
            raw=RWC._load_water(rx,ry,rz,ids)
            x=RWC.local_water(raw,raw[1],box)
            v=RWC._load_water(vx,vy,vz,ids); f=RWC._load_water(fx,fy,fz,ids)
            s=CSVR ? T(scale[1]) : one(T)
            h=dt/T(2)
            u=(s*v[1]+(h*im[1])*f[1],s*v[2]+(h*im[2])*f[2],s*v[3]+(h*im[3])*f[3])
            u,ok=RWC.rattle_water(x,u,im)
            status[w]=ok ? Int32(0) : Int32(2)
            ok || _rw_latch!(state)
            for k in 1:3
                j=ids[k]
                vx[j]=u[k][1]; vy[j]=u[k][2]; vz[j]=u[k][3]
                if CSVR
                    e=T(0.5)/im[k]*RWC._dot3(u[k],u[k])
                    Ekin[j]=e
                    ek+=Float64(e)
                end
            end
            n=3.0
        end
    end
    if CSVR
        sd=CuStaticSharedArray(Float64,2)
        e=_rw_block_sum(ek,sd,1)
        c=_rw_block_sum(n,sd,2)
        if threadIdx().x==1 && c>0
            Definitions.@red kfull[1]+=e
            Definitions.@red nfull[1]+=Int32(c)
        end
    end
    return nothing
end

_rw_unwrap(run)=(i=findfirst(f->f isa UnwrapFeature,run.features); i===nothing ? nothing : run.features[i].arrays)

"""W1 (+ the CSVR bath update unless `bath=false`): the whole drift phase of a fast-path step."""
function enqueue_rw_drift!(run;launch::Bool=true,bath::Bool=true)
    T=eltype(run.r[1]); ws=run.neighbors; w=_rw_feature(run).ws; n=length(w.atoms)
    eng=run.cluster.eng; L=eng.list; th=run.thermostat; csvr=th isa CSVRData
    csvr && nbaths(th)!=1 && error("Rigid-water fast path: one CSVR bath")
    u=_rw_unwrap(run)
    un=u===nothing ? run.r : u                                   # unused without unwrapping
    bid,kacc,nacc=csvr ? (th.bath_id,th.kacc,th.nacc) : (w.status,L.flags,w.status)   # unused under NVE
    args=(ws.state,w.atoms,w.inverse_masses,w.status,run.r,run.v,run.f,un,ws.geom,w.lengths2,w.tolerance,
          w.maxiter,T(run.dt),(bid,kacc,nacc),L.flags,(L.rx0,L.ry0,L.rz0),(L.px0,L.py0,L.pz0),
          T(eng.skin/2)^2,T(eng.prune_buffer/2)^2,Val(csvr),Val(u!==nothing),Val(L.nstlist>0))
    k=@cuda launch=false NL._device_if_valid!(ws.state,_rw_drift_kernel!,args...)
    launch && k(ws.state,_rw_drift_kernel!,args...;threads=128,blocks=cld(n,128))
    if csvr && bath
        a2=_rw_bath_args(run)
        k2=@cuda launch=false NL._device_if_valid!(ws.state,_csvr_bath_kernel!,a2...)
        launch && k2(ws.state,_csvr_bath_kernel!,a2...;threads=32,blocks=1)
    end
    return nothing
end
_rw_bath_args(run)=(th=run.thermostat; (th.kacc,th.nacc,th.target,th.tau,th.kinetic,th.exchange,th.scale,th.dof,
    run.neighbors.state,th.seed,Float64(run.dt),3.0,th.cdof,th.interval))

"""The CSVR bath update of a fast-path step as its own graph branch (`RW_BATH_SIDE`): it only
produces the scale the final kick applies, so it runs beside the list decision, mesh and pair
force. Unguarded: it runs exactly once per step even when a list build fails in the same step
(the recovery graph resumes after the drift and does not repeat it)."""
function enqueue_rw_bath!(run;launch::Bool=true)
    a2=_rw_bath_args(run)
    k2=@cuda launch=false _csvr_bath_kernel!(a2...)
    launch && k2(a2...;threads=32,blocks=1)
    return nothing
end

"""W2 (+ the CSVR report): the final kick of a fast-path step."""
# `advance`: the last kernel also advances the device step counter (returns true)
function enqueue_rw_kick!(run;launch::Bool=true,advance::Bool=false)
    T=eltype(run.r[1]); ws=run.neighbors; w=_rw_feature(run).ws; n=length(w.atoms)
    th=run.thermostat; csvr=th isa CSVRData
    sc,ek,kf,nf=csvr ? (th.scale,th.Ekin,th.kfull,th.nfull) : (run.r[1],run.r[1],run.r[1],w.status)   # unused under NVE
    args=(ws.state,w.atoms,w.inverse_masses,w.status,run.r,run.v,run.f,ws.geom,T(run.dt),(sc,ek,kf,nf),Val(csvr))
    wk=(advance && !csvr) ? NL._device_if_valid_advance! : NL._device_if_valid!
    k=@cuda launch=false wk(ws.state,_rw_kick_kernel!,args...)
    launch && k(ws.state,_rw_kick_kernel!,args...;threads=128,blocks=cld(n,128))
    if csvr
        a2=(th.kfull,th.nfull,th.kinetic)
        wr=advance ? NL._device_if_valid_advance! : NL._device_if_valid!
        k2=@cuda launch=false wr(ws.state,_csvr_report_kernel!,a2...)
        launch && k2(ws.state,_csvr_report_kernel!,a2...;threads=32,blocks=1)
    end
    return advance
end
