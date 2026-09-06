"""
    _swap_force_slots!(st)

Swap the active force slot (`f`) and reference force slot (`f0`).
"""
function _swap_force_slots!(st::SimulationState)
    st.f0x, st.fx = st.fx, st.f0x
    st.f0y, st.fy = st.fy, st.f0y
    if _is_3d(st)
        st.f0z, st.fz = st.fz, st.f0z
    end
    return nothing
end

"""
    _swap_midpoint_position_slots!(st)

Swap physical coordinates (`r`) with midpoint scratch coordinates (`v`) used by
Brownian midpoint-style integrators.
"""
function _swap_midpoint_position_slots!(st::SimulationState)
    st.rx, st.vx = st.vx, st.rx
    st.ry, st.vy = st.vy, st.ry
    if _is_3d(st)
        st.rz, st.vz = st.vz, st.rz
    end
    return nothing
end

"""
    prepare_previous_force_buffers!(st, spec)

Prepare any force-buffer reuse needed before stage execution.
"""
function prepare_previous_force_buffers!(st::SimulationState,
                                         spec::IntegratorSpec)
    return nothing
end


"""
    ensure_reference_forces_ready!(st, spec, compute_energy, freeze_spring)

Ensure the required reference force buffers for the selected integrator are
initialized before stage execution.
"""
function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::IntegratorSpec,
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    return nothing
end

function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::Union{VVSpec,BAOABSpec,BAOASpec,GSMSpec,NVESpec,NHCSpec,CSVRSpec},
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    if st.force_valid && st.force_freeze_spring == freeze_spring
        _swap_force_slots!(st)
        st.force_valid = false
    else
        evaluate_forces_into_f0!(st, compute_energy; freeze_spring=freeze_spring)
    end
    return nothing
end

function ensure_reference_forces_ready!(st::SimulationState,
                                        spec::Union{BrownianSpec,EMSpec},
                                        compute_energy::Bool,
                                        freeze_spring::Bool)
    if !st.force_valid || st.force_freeze_spring != freeze_spring
        evaluate_forces_into_f!(st, compute_energy; freeze_spring=freeze_spring)
    end
    return nothing
end

"""
    finalize_step_accounting!(st, spec, compute_energy)

Perform integrator-independent end-of-step bookkeeping.
"""
function finalize_step_accounting!(st::SimulationState,
                                   spec::IntegratorSpec,
                                   compute_energy::Bool)
    return nothing
end

function finalize_step_accounting!(st::SimulationState{T},
                                   spec::Union{NVESpec{T},NHCSpec{T},CSVRSpec{T}},
                                   compute_energy::Bool) where {T<:AbstractFloat}
    # Deterministic thermostatted MD paths do not define per-particle
    # stochastic heat/work channels, and none of their kernels write `dq`/`dU`.
    # The buffers only need clearing when the previous step was taken by an
    # integrator that populated them; clearing every step costs two full-N
    # sweeps in the hot loop.
    if st.last_integrator != integrator_id(spec)
        fill!(st.dq, zero(T))
        fill!(st.dU, zero(T))
    end
    return nothing
end

"""
    finalize_step_counter!(st, integrator_id)

Finalize step counters shared across integrators.
"""
function finalize_step_counter!(st::SimulationState,
                                id::UInt8)
    st.last_integrator = id
    st.step += 1
    return nothing
end

"""
    run_integrator_step!(st, spec, dt; compute_energy=true)

Shared stage-driven step engine. Integrator-independent orchestration lives
here, while stage-specific work is delegated through the integrator protocol.
"""
function run_integrator_step!(st::SimulationState{T},
                              spec::IntegratorSpec{T},
                              dt::T;
                              compute_energy::Bool=true) where {T<:AbstractFloat}
    validate_integrator_inputs!(spec, st, dt)

    try
        freeze_active = _freeze_active!(st)
        freeze_hold = freeze_active && st.freeze_mode == FREEZE_HOLD
        freeze_spring = freeze_active && st.freeze_mode == FREEZE_SPRING

        rebuild_needed = _spatial_reorder_active(st, spec) || plan_neighbor_rebuild!(st, dt)
        apply_neighbor_rebuild_if_needed!(st, spec, rebuild_needed)

        prepare_previous_force_buffers!(st, spec)
        ensure_reference_forces_ready!(st, spec, compute_energy, freeze_spring)
        ensure_integrator_workspace!(spec, st)

        for stage_tag in stage_sequence(spec)
            _trace_integrator_stage!(st, spec, stage_tag;
                                     force_evaluated=(stage_tag == :force),
                                     rebuild_applied=rebuild_needed)
            execute_integrator_stage!(spec, st, dt, stage_tag;
                                      compute_energy=compute_energy,
                                      freeze_hold=freeze_hold,
                                      freeze_spring=freeze_spring)
        end

        finalize_step_accounting!(st, spec, compute_energy)
        finalize_step_counter!(st, integrator_id(spec))
    catch
        invalidate_forces!(st)
        rethrow()
    end
    return nothing
end

"""
    step!(st, spec, dt; compute_energy=true)

Canonical explicit stepping API.
"""
function step!(st::SimulationState{T},
               spec::IntegratorSpec{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return run_integrator_step!(st, spec, T(dt); compute_energy=compute_energy)
end

"""
    step_graph!(st, spec, dt; compute_energy=true)

Graph-entry stepping API with explicit integrator selection. It currently
shares the same stage-driven engine as [`step!`](@ref).
"""
function step_graph!(st::SimulationState{T},
                     spec::IntegratorSpec{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return run_integrator_step!(st, spec, T(dt); compute_energy=compute_energy)
end

"""
    step!(st, vv, dt; compute_energy=true)

Compatibility wrapper for explicit `VVParams` stepping.
"""
function step!(st::SimulationState{T},
               vv::LangevinIntegrators.VVParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, VVSpec(vv), dt; compute_energy=compute_energy)
end

"""
    step!(st, bao, dt; compute_energy=true)

Compatibility wrapper for explicit `BAOABParams` stepping.
"""
function step!(st::SimulationState{T},
               bao::LangevinIntegrators.BAOABParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, BAOABSpec(bao), dt; compute_energy=compute_energy)
end

"""
    step!(st, bp, dt; compute_energy=true)

Compatibility wrapper for explicit Brownian midpoint stepping.
"""
function step!(st::SimulationState{T},
               bp::BrownianIntegrators.BrownianParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, BrownianSpec(bp), dt; compute_energy=compute_energy)
end

"""
    step!(st, em, dt; compute_energy=true)

Compatibility wrapper for explicit Euler-Maruyama stepping.
"""
function step!(st::SimulationState{T},
               em::BrownianIntegrators.EMParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, EMSpec(em), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, vv, dt; compute_energy=true)

Compatibility wrapper for explicit `VVParams` graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     vv::LangevinIntegrators.VVParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, VVSpec(vv), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, bao, dt; compute_energy=true)

Compatibility wrapper for explicit `BAOABParams` graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     bao::LangevinIntegrators.BAOABParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, BAOABSpec(bao), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, bp, dt; compute_energy=true)

Compatibility wrapper for explicit Brownian midpoint graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     bp::BrownianIntegrators.BrownianParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, BrownianSpec(bp), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, em, dt; compute_energy=true)

Compatibility wrapper for explicit Euler-Maruyama graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     em::BrownianIntegrators.EMParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, EMSpec(em), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, nhc, dt; compute_energy=true)

Compatibility wrapper for explicit Nose-Hoover chain graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     nhc::NHCParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, NHCSpec(nhc), dt; compute_energy=compute_energy)
end

"""
    step_graph!(st, csvr_params, dt; compute_energy=true)

Compatibility wrapper for explicit CSVR graph stepping.
"""
function step_graph!(st::SimulationState{T},
                     csvr_params::CSVRParams{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step_graph!(st, CSVRSpec(csvr_params), dt; compute_energy=compute_energy)
end

"""
    step!(st, nhc, dt; compute_energy=true)

Compatibility wrapper for explicit Nose-Hoover Chain parameter stepping.
"""
function step!(st::SimulationState{T},
               nhc::NHCParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st,
                 NHCSpec{T}(nhc, _new_nhc_workspace(Backends.storage_backend(st), T,
                                                    nhc.chain_length,
                                                    length(nhc.target_temperature),
                                                    length(st.rx))),
                 dt;
                 compute_energy=compute_energy)
end

"""
    step!(st, csvr_params, dt; compute_energy=true)

Compatibility wrapper for explicit CSVR parameter stepping.
"""
function step!(st::SimulationState{T},
               csvr_params::CSVRParams{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st,
                 CSVRSpec{T}(csvr_params,
                             _new_csvr_workspace(Backends.storage_backend(st), T,
                                                 length(csvr_params.target_temperature),
                                                 length(st.rx))),
                 dt;
                 compute_energy=compute_energy)
end

"""
    step_bd!(st, dt, bp; compute_energy=true)

Deprecated thin wrapper. Use `step!(st, bp, dt; ...)`.
"""
function step_bd!(st::SimulationState{T},
                  dt::Real,
                  bp::BrownianIntegrators.BrownianParams{T};
                  compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, bp, dt; compute_energy=compute_energy)
end
