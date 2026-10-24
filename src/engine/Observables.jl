# Included in ParticleDynamics.DeviceEngine. Fixed-cell, unconstrained, uniform mass only.
# U and W belong to r(t); v is the completed velocity-Verlet velocity v(t).
# GROMACS md uses <K(t)> = (K(v-hF/m) + K(v+hF/m))/2, h=dt/2.
# Thus its kinetic tensor is m/2 * (v⊗v + (hF/m)⊗(hF/m)).
# This is NOT K of the averaged velocity, nor an average of cadence samples.
const OBS_COMPONENTS = (:xx,:yy,:zz,:xy,:xz,:yz)
const OBS_CONVENTION = "U,W at r(t); K half-step averaged from v(t) +/- dt*F(t)/(2m); no COM subtraction; D·N DOF; reduced units kB=1"

struct ObservableWorkspace
    partial::CuArray{Float64,2} # blocks × (U, W[6], K_lf[6], K_vv)
    values::CuArray{Float64,1}  # 1:14 above; 15 K_lf,16 T,17 E_lf,18 P,19:24 P tensor
    stamp::CuArray{Int64,1}    # last sampled completed step, number of samples
end
ObservableWorkspace(N::Integer) = ObservableWorkspace(CUDA.zeros(Float64,cld(N,256),14),
    CUDA.fill(NaN,24),CuArray(Int64[-1,0]))

@inline function _observable_block_sum!(scratch, values)
    t=threadIdx().x
    @inbounds for q in 1:14; scratch[t,q]=values[q]; end
    sync_threads()
    stride=Int32(128)
    while stride>0
        if t<=stride
            @inbounds for q in 1:14; scratch[t,q]+=scratch[t+stride,q]; end
        end
        sync_threads()
        stride>>=1
    end
    return nothing
end

function _observable_partial_kernel!(partial,energy,virial,vx,vy,vz,fx,fy,fz,
                                     mass::Float64,h_over_m0::Float64,feat)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    values=ntuple(_->0.0,Val(14))
    if i<=length(energy)
        @inbounds begin
            mi,h_over_m=obs_mass(feat,mass,h_over_m0,i)  # feature bundle (masses)
            x=Float64(vx[i]); y=Float64(vy[i]); z=Float64(vz[i])
            a=h_over_m*Float64(fx[i]); b=h_over_m*Float64(fy[i]); c=h_over_m*Float64(fz[i])
            m=0.5mi
            values=(Float64(energy[i]),obs_virial(feat,virial,i,1),obs_virial(feat,virial,i,2),
                obs_virial(feat,virial,i,3),obs_virial(feat,virial,i,4),obs_virial(feat,virial,i,5),
                obs_virial(feat,virial,i,6),
                m*(x*x+a*a),m*(y*y+b*b),m*(z*z+c*c),m*(x*y+a*b),m*(x*z+a*c),m*(y*z+b*c),
                m*(x*x+y*y+z*z))
        end
    end
    scratch=CuStaticSharedArray(Float64,(256,14))
    _observable_block_sum!(scratch,values)
    if threadIdx().x==1
        @inbounds for q in 1:14; partial[blockIdx().x,q]=scratch[1,q]; end
    end
    return nothing
end

# 2D: same 14 slots with z components zero; 2D virial columns are (xx, yy, xy).
function _observable_partial2_kernel!(partial,energy,virial,vx,vy,fx,fy,
                                      mass::Float64,h_over_m0::Float64,feat)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    values=ntuple(_->0.0,Val(14))
    if i<=length(energy)
        @inbounds begin
            mi,h_over_m=obs_mass(feat,mass,h_over_m0,i)  # feature bundle (masses)
            x=Float64(vx[i]); y=Float64(vy[i])
            a=h_over_m*Float64(fx[i]); b=h_over_m*Float64(fy[i])
            m=0.5mi
            values=(Float64(energy[i]),obs_virial(feat,virial,i,1),obs_virial(feat,virial,i,2),0.0,
                obs_virial(feat,virial,i,3),0.0,0.0,
                m*(x*x+a*a),m*(y*y+b*b),0.0,m*(x*y+a*b),0.0,0.0,
                m*(x*x+y*y))
        end
    end
    scratch=CuStaticSharedArray(Float64,(256,14))
    _observable_block_sum!(scratch,values)
    if threadIdx().x==1
        @inbounds for q in 1:14; partial[blockIdx().x,q]=scratch[1,q]; end
    end
    return nothing
end

function _observable_finish_kernel!(out,stamp,partial,state,box,::Val{D},dof::Float64,dims::Float64,vglobal) where {D}
    # the volume from the device box: captured observable graphs stay valid when the box changes
    volume=prod(map(Float64,Definitions.box_lengths(box,Val(D))))
    t=threadIdx().x
    values=ntuple(Val(14)) do q
        s=0.0; j=t
        @inbounds while j<=size(partial,1)
            s+=partial[j,q]; j+=blockDim().x
        end
        s
    end
    scratch=CuStaticSharedArray(Float64,(256,14))
    _observable_block_sum!(scratch,values)
    if t==1
        @inbounds begin
            for q in 1:14; out[q]=scratch[1,q]; end
            # global virial terms without a per-particle split (the mesh tensor)
            if vglobal!==nothing
                for q in 1:6; scratch[1,1+q]+=Float64(vglobal[q]); out[1+q]=scratch[1,1+q]; end
            end
            k=scratch[1,8]+scratch[1,9]+scratch[1,10]
            out[15]=k; out[16]=2k/dof; out[17]=scratch[1,1]+k
            out[18]=(2k+scratch[1,2]+scratch[1,3]+scratch[1,4])/(dims*volume)
            for q in 1:6; out[18+q]=(2scratch[1,7+q]+scratch[1,1+q])/volume; end
            stamp[1]=state[NL.DN_STEP]; stamp[2]+=1
        end
    end
    return nothing
end

function observable_arguments(run)
    o=run.observables; D=length(run.v)
    # Velocity constraints (rigid water): the half-step velocities v ∓ dt·F/(2m) carry the
    # bond components the constraint forces would cancel, so K comes from the projected v(t).
    constrained=hasproperty(run,:features) && any(constrains_velocities,run.features)
    partial=(o.partial,run.energy,run.virial,run.v...,run.f...,
        Float64(run.mass),constrained ? 0.0 : Float64(run.dt)/(2Float64(run.mass)),hasproperty(run,:fx) ? run.fx : ())
    # Volume is the box area in 2D; the pressure trace divides by D·V.
    vg=hasproperty(run,:cluster) && run.cluster!==nothing ? run.cluster.Vglobal :
       (hasproperty(run,:mesh) && run.mesh!==nothing ? run.mesh.Vglobal : nothing)   # ELL charged runs
    finish=(o.values,o.stamp,o.partial,run.neighbors.state,
        hasproperty(run.neighbors,:geom) ? run.neighbors.geom : run.neighbors.box,Val(D),
        Float64(D*length(run.energy)-_constrained_dof(run)),Float64(D),vg)
    return partial,finish
end
_partial_kernel(run)=length(run.v)==3 ? _observable_partial_kernel! : _observable_partial2_kernel!
_constrained_dof(run)=hasproperty(run,:features) ? features_constrained_dof(run) : 0

"""Reduce current U/W/v/F on CUDA. Call only after an energy force and final kick.

No allocation/readback. Graphs invoke this after the completed-step counter.
Errors guard both stages, leaving the previous sample and stamp intact.
"""
function enqueue_observables!(run)
    a,b=observable_arguments(run)
    @cuda threads=256 blocks=size(run.observables.partial,1) NL._device_if_valid!(
        run.neighbors.state,_partial_kernel(run),a...)
    @cuda threads=256 blocks=1 NL._device_if_valid!(run.neighbors.state,_observable_finish_kernel!,b...)
    return nothing
end

function compile_observables!(run)
    a,b=observable_arguments(run)
    @cuda launch=false NL._device_if_valid!(run.neighbors.state,_partial_kernel(run),a...)
    @cuda launch=false NL._device_if_valid!(run.neighbors.state,_observable_finish_kernel!,b...)
    return nothing
end

"""Explicit reporting boundary; rejects unsampled/stale results and latched errors."""
function observable_status(run;expected_step=nothing)
    state=NL.device_neighbor_status(run.neighbors)
    stamp=Array(run.observables.stamp)
    stamp[2]>0 || error("No global observable sample is available")
    expected_step===nothing || stamp[1]==expected_step || error("Stale global observable sample")
    stamp[1]<=state.step || error("Observable stamp exceeds completed step")
    a=Array(run.observables.values)
    all(isfinite,a) || error("Nonfinite global observables")
    return (;step=stamp[1],samples=stamp[2],potential=a[1],virial=a[2:7],
        kinetic_tensor=a[8:13],kinetic_full=a[14],kinetic=a[15],temperature=a[16],
        total=a[17],pressure=a[18],pressure_tensor=a[19:24],convention=OBS_CONVENTION)
end
