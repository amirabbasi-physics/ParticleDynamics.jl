# Feature: rigid three-site water (SimulationCore `RigidWaterCoupling.jl`, kernels in
# `RigidWaterConstraints`). Staged NVE order: kick1 · save reference · drift · SHAKE
# (position correction, correction/dt added to velocities and unwrapped coordinates) ·
# force · kick2 · RATTLE (velocity projection). On the engine:
#   after the (fused) kick-drift — SHAKE against the reference saved at the end of the
#                                  previous step (positions do not move in between);
#   after the final kick          — RATTLE, then the reference for the next drift; on
#                                  sampled steps the constraint virial of each water
#                                  (written on its first atom's row) for the observables.
# Constraint failures (staged `check_status`, a host read per call) latch the engine
# error state on the device; `at_boundary!` throws the staged DomainError. The kinetic
# temperature counts 3 constraints per water (`kinetic_dof`). Rigid water states are not
# sorted (`attach_rigid_water!` drops the tags), so atom indices are storage slots.

const RWC=RigidWaterConstraints

struct RigidWaterFeature{W,U,V} <: EngineFeature
    ws::W        # RigidWaterConstraints.WaterWorkspace (state-owned)
    unwrap::U    # unwrapped coordinates (state-owned) or nothing
    vcons::V     # N×6 constraint virial rows (first atom of each water), engine-owned
end

function make_feature(::Type{RigidWaterFeature},st::SC.SimulationState{T},spec) where {T}
    st.rigid_water===nothing && return nothing
    ws=st.rigid_water.workspace
    RWC.save_reference!(ws,st.rx,st.ry,st.rz)   # the first drift's reference (current positions)
    unwrap=st.rx_unwrap===nothing ? nothing : (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)
    return RigidWaterFeature(ws,unwrap,CUDA.zeros(T,length(st.rx),6))
end
feature_signature(::Type{RigidWaterFeature},st,spec)=st.rigid_water===nothing ? nothing :
    (pointer(st.rigid_water.workspace.atoms),st.rigid_water.workspace.lengths2,st.rigid_water.workspace.tolerance,
     st.rigid_water.workspace.maxiter)
device_part(f::RigidWaterFeature)=(vcons=f.vcons,)
constrained_dof(f::RigidWaterFeature)=3*length(f.ws.atoms)

function _status_latch_kernel!(state,status)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    @inbounds if i<=length(status) && status[i]!=0 && state[NL.DN_ERROR]==0
        state[NL.DN_ERROR]=NL.DN_FEATURE_ERROR
    end
    return nothing
end
# Not guarded by list validity: a failed constraint must latch even when nothing else runs.
_latch_status!(run,f;launch::Bool)=(k=@cuda launch=false _status_latch_kernel!(run.neighbors.state,f.ws.status);
    launch && k(run.neighbors.state,f.ws.status;threads=256,blocks=cld(length(f.ws.status),256)); nothing)

# the device box: captured SHAKE/RATTLE/virial kernels read the current lengths
_box3(run)=run.neighbors.geom

function after_position!(run,f::RigidWaterFeature;midpoint::Bool=false,launch::Bool=true)
    midpoint && return nothing
    T=eltype(run.r[1]); ws=f.ws; n=length(ws.atoms)
    u=f.unwrap===nothing ? run.r : f.unwrap
    args=(ws.atoms,ws.inverse_masses,ws.reference,ws.status,run.r...,run.v...,u...,_box3(run),ws.lengths2,
          ws.tolerance,ws.maxiter,T(run.dt),Val(f.unwrap!==nothing))
    _feature_launch!(run,RWC._shake_water_kernel!,args,n;launch,threads=128)
    _latch_status!(run,f;launch)
    return nothing
end

constrains_velocities(::RigidWaterFeature)=true
function constrain_velocities!(run,f::RigidWaterFeature;launch::Bool=true)
    ws=f.ws
    _feature_launch!(run,RWC._rattle_water_kernel!,(ws.atoms,ws.inverse_masses,ws.status,run.r...,run.v...,_box3(run)),
        length(ws.atoms);launch,threads=128)
    _latch_status!(run,f;launch)
    return nothing
end

function after_kick!(run,f::RigidWaterFeature;energy::Bool,launch::Bool=true)
    ws=f.ws; n=length(ws.atoms); box=_box3(run)
    # the fast path's W2 already applied RATTLE, and its W1 reads the entry positions
    if !rw_fast_path(run)
        constrain_velocities!(run,f;launch)
        _feature_launch!(run,RWC._save_water_kernel!,(ws.reference,ws.atoms,run.r...),n;launch,threads=128)
    end
    if energy   # constraint virial at the completed step (staged: on demand at r, v, f)
        _feature_launch!(run,_constraint_virial_rows_kernel!,(f.vcons,ws.atoms,ws.inverse_masses,ws.status,
            run.r...,run.v...,run.f...,box),n;launch,threads=128)
        _latch_status!(run,f;launch)
    end
    return nothing
end

function _constraint_virial_rows_kernel!(out,atoms,weights,status,rx,ry,rz,vx,vy,vz,fx,fy,fz,box)
    box=Definitions.box_lengths(box,Val(3))
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(atoms) && return nothing
    @inbounds begin
        ids=atoms[i]
        raw=RWC._load_water(rx,ry,rz,ids)
        x=RWC.local_water(raw,raw[1],box)
        c,ok=RWC.constraint_force_water(x,RWC._load_water(vx,vy,vz,ids),RWC._load_water(fx,fy,fz,ids),weights[i])
        status[i]=ok ? Int32(0) : Int32(3)
        t=RWC.constraint_tensor_water(x,c)
        j=ids[1]
        out[j,1]=t[1]; out[j,2]=t[2]; out[j,3]=t[3]; out[j,4]=t[4]; out[j,5]=t[5]; out[j,6]=t[6]
    end
    return nothing
end

function at_boundary!(f::RigidWaterFeature)
    maximum(f.ws.status)==0 ||
        throw(DomainError(:rigid_water,"Rigid water constraints failed; configuration is not valid"))
    return nothing
end
