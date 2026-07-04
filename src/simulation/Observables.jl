# -----------------------------------------------------------------------------
# Observables and thermostat metadata
# -----------------------------------------------------------------------------

"""
    thermostatted_particle_mask(st, spec)

Return a mask of particles thermostatted by `spec`. `nothing` means all
particles are currently thermostatted.
"""
function thermostatted_particle_mask(st::SimulationState,
                                     spec::IntegratorSpec)
    if st.freeze_mask === nothing || st.freeze_mode == FREEZE_NONE
        return nothing
    end
    return ifelse.(st.freeze_mask .== UInt8(0), UInt8(1), UInt8(0))
end

function thermostatted_particle_mask(st::SimulationState,
                                     spec::NHCSpec)
    ws = spec.workspace
    active = ifelse.(ws.particle_bath_id .> Int32(0), UInt8(1), UInt8(0))
    if st.freeze_mask === nothing || st.freeze_mode == FREEZE_NONE
        return active
    end
    return ifelse.((ws.particle_bath_id .> Int32(0)) .& (st.freeze_mask .== UInt8(0)), UInt8(1), UInt8(0))
end

function thermostatted_particle_mask(st::SimulationState,
                                     spec::CSVRSpec)
    ws = spec.workspace
    active = ifelse.(ws.particle_bath_id .> Int32(0), UInt8(1), UInt8(0))
    if st.freeze_mask === nothing || st.freeze_mode == FREEZE_NONE
        return active
    end
    return ifelse.((ws.particle_bath_id .> Int32(0)) .& (st.freeze_mask .== UInt8(0)), UInt8(1), UInt8(0))
end

"""
    thermostatted_dof(st, spec) -> Int

Return the physical degrees of freedom acted on by the thermostat.
"""
function thermostatted_dof(st::SimulationState,
                           spec::IntegratorSpec)
    D = _is_3d(st) ? 3 : 2
    mask = thermostatted_particle_mask(st, spec)
    if mask === nothing
        return D * length(st.rx)
    end
    ntherm = Int(CUDA.sum(Int32.(mask)))
    return D * ntherm
end

function thermostatted_dof(st::SimulationState,
                           spec::NHCSpec)
    D = _is_3d(st) ? 3 : 2
    mask = thermostatted_particle_mask(st, spec)
    ntherm = Int(CUDA.sum(Int32.(mask)))
    return D * ntherm
end

function thermostatted_dof(st::SimulationState,
                           spec::CSVRSpec)
    D = _is_3d(st) ? 3 : 2
    mask = thermostatted_particle_mask(st, spec)
    ntherm = Int(CUDA.sum(Int32.(mask)))
    return D * ntherm
end

thermostatted_dof(st::SimulationState, spec::NVESpec) = 0

collect_integrator_observables(spec::NVESpec, st::SimulationState) =
    (thermostat_kind=:none,)

function collect_integrator_observables(spec::NHCSpec{T},
                                        st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    _nhc_update_dof_per_bath!(spec, st)
    Ekin_total = T(CUDA.sum(st.Ekin))
    Epot_total = T(CUDA.sum(st.Epot))
    dof_b = Array(ws.dof_per_bath)
    K_b = Array(ws.kinetic_total_per_bath)
    target_b = p.target_temperature
    tau_b = p.tau
    scale_b = Array(ws.last_velocity_scale_per_bath)
    total_dof = sum(dof_b)

    weighted_target = zero(T)
    weighted_error = zero(T)
    if total_dof > zero(T)
        weighted_target = sum(target_b .* dof_b) / total_dof
        @inbounds for b in eachindex(target_b)
            dof = dof_b[b]
            if dof > zero(T)
                Tinst = T(2) * K_b[b] / dof
                weighted_error += (Tinst - target_b[b]) * dof
            end
        end
        weighted_error /= total_dof
    end

    thermostat_kin = T(sum(Array(ws.thermostat_kinetic_per_bath)))
    thermostat_pot = T(sum(Array(ws.thermostat_potential_per_bath)))
    if total_dof > zero(T)
        acc = zero(T)
        @inbounds for b in eachindex(scale_b)
            if dof_b[b] > zero(T)
                acc += log(max(scale_b[b], eps(T))) * dof_b[b]
            end
        end
        stage_scale = exp(acc / total_dof)
    else
        stage_scale = one(T)
    end

    ext_h = Epot_total + Ekin_total + thermostat_kin + thermostat_pot
    return (thermostat_kind=:nhc,
            target_temperature=weighted_target,
            thermostat_timescale=(isempty(tau_b) ? zero(T) : T(sum(tau_b) / length(tau_b))),
            chain_length=p.chain_length,
            chain_substeps=p.substeps,
            nhc_propagator=_nhc_propagator_name(p.propagator),
            nhc_num_baths=length(target_b),
            nhc_dof_total=total_dof,
            physical_kinetic=Ekin_total,
            thermostat_kinetic=thermostat_kin,
            thermostat_potential=thermostat_pot,
            extended_hamiltonian=ext_h,
            thermostat_temperature_error=weighted_error,
            nhc_velocity_scale=stage_scale)
end

function collect_integrator_observables(spec::CSVRSpec{T},
                                        st::SimulationState{T}) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    _csvr_update_dof_per_bath!(spec, st)
    _ensure_csvr_kinetic_initialized!(spec, st)
    Ekin_total = T(CUDA.sum(st.Ekin))
    Epot_total = T(CUDA.sum(st.Epot))
    dof_b = Array(ws.dof_per_bath)
    K_b = Array(ws.kinetic_total_per_bath)
    target_b = p.target_temperature
    tau_b = p.tau
    scale_b = Array(ws.last_velocity_scale_per_bath)
    total_dof = sum(dof_b)

    weighted_target = zero(T)
    weighted_error = zero(T)
    if total_dof > zero(T)
        weighted_target = sum(target_b .* dof_b) / total_dof
        @inbounds for b in eachindex(target_b)
            dof = dof_b[b]
            if dof > zero(T)
                Tinst = T(2) * K_b[b] / dof
                weighted_error += (Tinst - target_b[b]) * dof
            end
        end
        weighted_error /= total_dof
    end

    thermostat_energy = T(sum(Array(ws.cumulative_energy_exchange_per_bath)))
    if total_dof > zero(T)
        acc = zero(T)
        @inbounds for b in eachindex(scale_b)
            if dof_b[b] > zero(T)
                acc += log(max(scale_b[b], eps(T))) * dof_b[b]
            end
        end
        stage_scale = exp(acc / total_dof)
    else
        stage_scale = one(T)
    end

    ext_h = Epot_total + Ekin_total + thermostat_energy
    return (thermostat_kind=:csvr,
            target_temperature=weighted_target,
            thermostat_timescale=(isempty(tau_b) ? zero(T) : T(sum(tau_b) / length(tau_b))),
            csvr_num_baths=length(target_b),
            csvr_dof_total=total_dof,
            physical_kinetic=Ekin_total,
            thermostat_energy=thermostat_energy,
            extended_hamiltonian=ext_h,
            thermostat_temperature_error=weighted_error,
            csvr_velocity_scale=stage_scale)
end

@inline _langevin_bath_heat_sign(::VVSpec) = 1
@inline _langevin_bath_heat_sign(::Union{BAOABSpec,BAOASpec,GSMSpec}) = -1

function _langevin_inverse_temperature_host(gamma::CuArray{T,1},
                                            noise_scale::CuArray{T,1},
                                            dt::T) where {T<:AbstractFloat}
    gamma_host = Array(gamma)
    scale_host = Array(noise_scale)
    invT = zeros(T, length(scale_host))
    @inbounds for i in eachindex(scale_host)
        gi = gamma_host[i]
        si = scale_host[i]
        if gi > zero(T) && si > zero(T)
            invT[i] = (T(2) * gi * dt) / (si * si)
        end
    end
    return invT
end

function _bath_entropy_per_bath(heat_b::AbstractVector{T},
                                target_b::AbstractVector{T}) where {T<:AbstractFloat}
    entropy_b = similar(heat_b, T)
    @inbounds for b in eachindex(heat_b)
        Tb = target_b[b]
        entropy_b[b] = Tb > zero(T) ? heat_b[b] / Tb : zero(T)
    end
    return entropy_b
end

function _collect_bath_observables(spec::Union{VVSpec{T},BAOABSpec{T},BAOASpec{T},GSMSpec{T}},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    dt = spec.params.dt
    invT = _langevin_inverse_temperature_host(spec.params.gamma, spec.params.noise_scale, dt)
    sign = T(_langevin_bath_heat_sign(spec))
    dq_host = Array(st.dq)
    heat_total = sign * T(sum(dq_host)) * dt
    entropy_total = sign * T(sum(dq_host .* invT)) * dt
    return (bath_heat_total=heat_total,
            bath_entropy_total=entropy_total)
end

function _collect_bath_observables(spec::NHCSpec{T},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    heat_b = Array(spec.workspace.cumulative_energy_exchange_per_bath)
    temp_b = copy(spec.params.target_temperature)
    entropy_b = _bath_entropy_per_bath(heat_b, temp_b)
    return (bath_heat_total=T(sum(heat_b)),
            bath_entropy_total=T(sum(entropy_b)),
            bath_heat_per_bath=heat_b,
            bath_entropy_per_bath=entropy_b,
            bath_temperature_per_bath=temp_b)
end

function _collect_bath_observables(spec::CSVRSpec{T},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    heat_b = Array(spec.workspace.cumulative_energy_exchange_per_bath)
    temp_b = copy(spec.params.target_temperature)
    entropy_b = _bath_entropy_per_bath(heat_b, temp_b)
    return (bath_heat_total=T(sum(heat_b)),
            bath_entropy_total=T(sum(entropy_b)),
            bath_heat_per_bath=heat_b,
            bath_entropy_per_bath=entropy_b,
            bath_temperature_per_bath=temp_b)
end

_collect_bath_observables(spec::IntegratorSpec, st::SimulationState) = NamedTuple()

function _reset_particle_exchange_buffers!(st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(st.dq, zero(T))
    fill!(st.dU, zero(T))
    return nothing
end

"""
    reset_bath_exchange_accumulators!(st, spec)

Reset the accumulators used for bath heat and entropy diagnostics. This leaves
positions, velocities, and thermostat internal state untouched.
"""
function reset_bath_exchange_accumulators!(st::SimulationState{T},
                                           spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    return _reset_particle_exchange_buffers!(st)
end

function reset_bath_exchange_accumulators!(st::SimulationState{T},
                                           spec::NHCSpec{T}) where {T<:AbstractFloat}
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    return _reset_particle_exchange_buffers!(st)
end

function reset_bath_exchange_accumulators!(st::SimulationState{T},
                                           spec::CSVRSpec{T}) where {T<:AbstractFloat}
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    return _reset_particle_exchange_buffers!(st)
end

"""
    _ensure_sample_buffers!(st, spec)

Bring lazily-maintained observable buffers up to date before sampling. The
deterministic NVE hot loop does not touch `st.Ekin`; refresh it from the
velocities here instead of paying for it every timestep.
"""
_ensure_sample_buffers!(st::SimulationState, spec::IntegratorSpec) = nothing

_ensure_sample_buffers!(st::SimulationState{T}, spec::NVESpec{T}) where {T<:AbstractFloat} =
    _refresh_kinetic_buffer!(st)

"""
    collect_step_observables(st, spec) -> NamedTuple

Collect universal diagnostics, integrator-specific diagnostics, and bath heat /
entropy totals accumulated since the last reset of the relevant buffers.
"""
function collect_step_observables(st::SimulationState{T},
                                  spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    _ensure_sample_buffers!(st, spec)
    Epot_total = T(CUDA.sum(st.Epot))
    Ekin_total = T(CUDA.sum(st.Ekin))
    virial_total = T(CUDA.sum(st.virial))
    dq_total = T(CUDA.sum(st.dq))
    dU_total = T(CUDA.sum(st.dU))

    base = (step=st.step,
            integrator=integrator_name(spec),
            Epot_total=Epot_total,
            Ekin_total=Ekin_total,
            Etot=Epot_total + Ekin_total,
            virial_total=virial_total,
            dq_total=dq_total,
            dU_total=dU_total,
            Qtot=dq_total,
            thermostatted_dof=thermostatted_dof(st, spec))

    return merge(base,
                 collect_integrator_observables(spec, st),
                 _collect_bath_observables(spec, st))
end
