# Nosé–Hoover chains inside the captured graph.
# Staged step: thermostat(dt/2) · kick1 · drift · force · kick2 · thermostat(dt/2),
# where each thermostat stage recounts per-bath dof, copies K to the stage start,
# propagates the chain (legacy / GROMACS / LAMMPS kernels), adds the exchanged
# energy and scales v and Ekin per bath. The staged chain kernels read and write
# only device arrays, so the engine launches them unchanged inside the graph.
# Engine changes: kick2 reduces K and particle counts per bath with the CSVR
# block reduction (Float64, no N-way atomic contention); the post-stage uses the
# counts of that same kick. The pre-stage uses the dof of the previous kick2
# (staged recounts at every stage; they differ only if bath ids are edited in
# place between steps, which then takes effect half a step later).

struct NHCData{S,T,V<:CuArray{T,1}}
    spec::S                       # NHCSpec (workspace arrays are captured by address)
    Ekin::V
    kacc::CuArray{Float64,1}
    nacc::CuArray{Int32,1}
end
nbaths(c::NHCData)=length(c.spec.workspace.kinetic_total_per_bath)

function _nhc_stage_start_kernel!(start,K)
    b=threadIdx().x
    b<=length(K) && (@inbounds start[b]=K[b])
    return nothing
end
function _nhc_stage_end_kernel!(exchange,start,K)
    b=threadIdx().x
    b<=length(K) && (@inbounds exchange[b]+=start[b]-K[b])
    return nothing
end
function _nhc_kinetic_kernel!(K,dof,kacc,nacc,dims)
    b=threadIdx().x
    if b<=length(K)
        @inbounds begin
            T=eltype(K); K[b]=T(kacc[b]); dof[b]=T(dims)*T(nacc[b])
            kacc[b]=0.0; nacc[b]=Int32(0)
        end
    end
    return nothing
end

function _nhc_chain_args(spec,stage_dt)
    p=spec.params; ws=spec.workspace; ns=Int32(p.substeps)
    common=(ws.kinetic_total_per_bath,ws.thermostat_kinetic_per_bath,ws.thermostat_potential_per_bath,
        ws.last_velocity_scale_per_bath,ws.dof_per_bath,ws.target_temperature,stage_dt,ns)
    p.propagator==SC.NHC_PROPAGATOR_LEGACY && return SC._nhc_chain_stage_legacy_kernel!,(ws.xi,ws.eta,ws.chain_force,ws.chain_masses,common...)
    p.propagator==SC.NHC_PROPAGATOR_GROMACS && return SC._nhc_chain_stage_gromacs_kernel!,(ws.xi,ws.eta,ws.chain_masses,common...)
    p.propagator==SC.NHC_PROPAGATOR_LAMMPS && return SC._nhc_chain_stage_lammps_kernel!,(ws.xi,ws.eta,ws.chain_force,ws.chain_masses,common...)
    throw(ArgumentError("Unsupported NHC propagator id $(p.propagator)"))
end

"""One thermostat half-stage: stage start, chain, exchange, velocity/Ekin scale."""
function _nhc_stage!(run,c::NHCData;launch::Bool=true)
    T=eltype(run.r[1]); ws=run.neighbors; s=run.neighbors.state; w=c.spec.workspace
    N=length(run.r[1]); B=nbaths(c)
    chain,cargs=_nhc_chain_args(c.spec,run.dt/T(2))
    scale_args=(run.v,c.Ekin,w.particle_bath_id,w.last_velocity_scale_per_bath)
    k1=@cuda launch=false NL._device_if_valid!(s,_nhc_stage_start_kernel!,w.kinetic_stage_start_per_bath,w.kinetic_total_per_bath)
    k2=@cuda launch=false NL._device_if_valid!(s,chain,cargs...)
    k3=@cuda launch=false NL._device_if_valid!(s,_nhc_stage_end_kernel!,w.cumulative_energy_exchange_per_bath,
        w.kinetic_stage_start_per_bath,w.kinetic_total_per_bath)
    k4=@cuda launch=false NL._device_if_valid!(s,_csvr_scale_kernel!,scale_args...)
    if launch
        k1(s,_nhc_stage_start_kernel!,w.kinetic_stage_start_per_bath,w.kinetic_total_per_bath;threads=32)
        k2(s,chain,cargs...;threads=32,blocks=cld(B,32))
        k3(s,_nhc_stage_end_kernel!,w.cumulative_energy_exchange_per_bath,w.kinetic_stage_start_per_bath,
            w.kinetic_total_per_bath;threads=32)
        k4(s,_csvr_scale_kernel!,scale_args...;threads=256,blocks=cld(N,256))
    end
    return nothing
end

"""Kick2 with the per-bath K/count reduction, K/dof finalization, then the post-stage."""
function _nhc_kick!(run,c::NHCData;launch::Bool=true)
    T=eltype(run.r[1]); s=run.neighbors.state; w=c.spec.workspace
    N=length(run.r[1]); D=length(run.r); B=nbaths(c)
    coef=(run.dt/T(2))/run.mass
    a1=(run.v,run.f,c.Ekin,w.particle_bath_id,c.kacc,c.nacc,coef,run.mass,Val(B),run.fx)
    a2=(w.kinetic_total_per_bath,w.dof_per_bath,c.kacc,c.nacc,D)
    k1=@cuda launch=false NL._device_if_valid!(s,_csvr_kick_kernel!,a1...)
    k2=@cuda launch=false NL._device_if_valid!(s,_nhc_kinetic_kernel!,a2...)
    if launch
        k1(s,_csvr_kick_kernel!,a1...;threads=256,blocks=cld(N,256))
        k2(s,_nhc_kinetic_kernel!,a2...;threads=32)
    end
    _nhc_stage!(run,c;launch)
    return nothing
end
