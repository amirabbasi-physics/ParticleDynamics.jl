module IntegratorInterfaces

export AbstractIntegratorSpec,
       validate_integrator_inputs!,
       ensure_integrator_workspace!,
       integrator_id,
       integrator_name,
       collect_integrator_observables

"""
    AbstractIntegratorSpec{T}

Base type of the integrator specs (VV, BAOAB, BAOA, GSM, Brownian, EM, NHC, CSVR, NVE).
The device engine (`ParticleDynamics.DeviceEngine`) steps every spec in captured
CUDA graphs; the protocol here validates inputs and sizes per-spec workspaces.
"""
abstract type AbstractIntegratorSpec{T<:AbstractFloat} end

"""
    validate_integrator_inputs!(spec, st, dt)

Validate runtime preconditions for one step. Integrators should check constraints
such as positive friction where required.
"""
function validate_integrator_inputs!(spec::AbstractIntegratorSpec, st, dt)
    return nothing
end

"""
    ensure_integrator_workspace!(spec, st)

Ensure any lazily allocated integrator-local work buffers exist and are shaped
for the provided simulation state.
"""
function ensure_integrator_workspace!(spec::AbstractIntegratorSpec, st)
    return nothing
end

"""
    integrator_id(spec) -> UInt8

Small numeric id used for compatibility flags (e.g. writer behavior).
"""
function integrator_id(spec::AbstractIntegratorSpec)
    throw(MethodError(integrator_id, (spec,)))
end

"""
    integrator_name(spec) -> Symbol

Stable symbolic name used in diagnostics and stage traces.
"""
function integrator_name(spec::AbstractIntegratorSpec)
    throw(MethodError(integrator_name, (spec,)))
end



"""
    collect_integrator_observables(spec, st) -> NamedTuple

Optional integrator-specific observables merged into generic writer/logging
pipelines.
"""
function collect_integrator_observables(spec::AbstractIntegratorSpec, st)
    return NamedTuple()
end

end # module IntegratorInterfaces
