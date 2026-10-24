# =========================
#   Force evaluation
# =========================
# Every force evaluation runs on the device engine (`ParticleDynamics.DeviceEngine`):
# the engine that owns the state, or an evaluation-only engine created for it. An
# attached external provider is the force field itself and is called directly. The
# staged force driver (pair kernels per potential, bond application, virial
# combination) was removed with the staged step engine.

"""
Forces from an attached external provider (the user's code, not a force-field term)
plus the freeze tether. Restraints are user biases on top of the provider's surface:
the freeze tether survives; force-field terms do not
combine with a provider (`attach_external_potential!`).
"""
function _provider_evaluate!(st::SimulationState, compute_energy::Bool, freeze_spring::Bool)
    external_forces!(st.external_potential, st, compute_energy)
    if freeze_spring
        E = compute_energy ? st.Epot : nothing
        if _is_3d(st)
            _apply_freeze_spring!(st, st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, E, compute_energy)
        else
            _apply_freeze_spring!(st, st.rx, st.ry, st.fx, st.fy, E, compute_energy)
        end
    end
    st.force_valid = true
    st.energy_valid = compute_energy
    st.force_freeze_spring = freeze_spring
    return nothing
end

"""
    evaluate_forces_into_f!(st, compute_energy; freeze_spring=false)

A fresh evaluation at the current coordinates (forces invalidated first, then
[`evaluate_current_configuration!`](@ref)): forces, and with `compute_energy`
energies and virials, on the device engine that owns the state, or on an
evaluation-only engine (the staged force driver was removed).
"""
function evaluate_forces_into_f!(st::SimulationState, compute_energy::Bool; freeze_spring::Bool=false)
    invalidate_forces!(st)
    evaluate_current_configuration!(st, compute_energy; freeze_spring=freeze_spring, moved=true)
    return nothing
end

"""
    ensure_energies!(st; freeze_spring=st.force_freeze_spring) -> st

Refresh the energy and virial buffers for the current coordinates if they are
stale (`!st.energy_valid`), by evaluating forces with `compute_energy=true`
into the active slot. Forces at unchanged coordinates are reproduced, so this
is safe at step boundaries.
"""
function ensure_energies!(st::SimulationState; freeze_spring::Bool=st.force_freeze_spring)
    st.energy_valid && return st
    evaluate_current_configuration!(st, true; freeze_spring=freeze_spring)
    return st
end

# Device-engine hook (configuration evaluation): forces at the current configuration
# from the engine that owns the state, without handing it back to the staged
# path. `DeviceEngine` adds a method for its engine; `false` means not taken.
_engine_evaluate!(engine, st, compute_energy::Bool, freeze_spring::Bool; moved::Bool=false, create::Bool=false) = false

"""
    evaluate_current_configuration!(st, compute_energy; freeze_spring=false, moved=false, create=true)

Forces (and energies/virials) at the current configuration for the public
evaluation entry points. A state owned by the device engine is evaluated there
(storage order unchanged); otherwise an evaluation engine is created when
`create=true`. Unsupported configurations raise an error.
`moved=true`: the caller only moved coordinates in place since the last
evaluation (forces are invalid for that reason alone); `create=true` (default): an engine-supported state without an engine gets an evaluation-only
engine, kept for later evaluations.
"""
function evaluate_current_configuration!(st::SimulationState, compute_energy::Bool;
                                         freeze_spring::Bool=false, moved::Bool=false, create::Bool=true)
    _engine_evaluate!(st.engine, st, compute_energy, freeze_spring; moved, create) && return nothing
    if st.external_potential !== nothing && st.engine === nothing
        _provider_evaluate!(st, compute_energy, freeze_spring)   # the provider is the force field
        return nothing
    end
    # No staged fallback: an engine that cannot take this evaluation (forces
    # invalidated by an edit, another tether) gives way to an evaluation-only engine.
    if st.engine !== nothing
        _drop_engine!(st.engine, st)
        st.external_potential === nothing || (_provider_evaluate!(st, compute_energy, freeze_spring); return nothing)
        _engine_evaluate!(nothing, st, compute_energy, freeze_spring; moved=true, create=true) && return nothing
    end
    throw(ArgumentError("The device engine does not evaluate this state: " * _evaluation_refusal(st, freeze_spring)))
end
