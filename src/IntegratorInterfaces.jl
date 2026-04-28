module IntegratorInterfaces

export AbstractIntegratorSpec,
       validate_integrator_inputs!,
       ensure_integrator_workspace!,
       integrator_id,
       integrator_name,
       stage_sequence,
       execute_integrator_stage!,
       collect_integrator_observables

"""
    AbstractIntegratorSpec{T}

Base protocol type for stage-driven time integrators. Concrete specs (VV, BAOAB,
Brownian, EM, NHC) should subtype this and provide protocol methods used
by the shared stepping engine.
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
    stage_sequence(spec) -> Tuple

Return ordered stage tags executed by the shared engine for one step.
"""
function stage_sequence(spec::AbstractIntegratorSpec)
    throw(MethodError(stage_sequence, (spec,)))
end

"""
    execute_integrator_stage!(spec, st, dt, stage_tag; compute_energy=true,
                              freeze_hold=false, freeze_spring=false)

Execute a single integrator stage. The shared engine drives stage iteration and
passes common runtime flags.
"""
function execute_integrator_stage!(spec::AbstractIntegratorSpec, st, dt, stage_tag;
                                   compute_energy::Bool=true,
                                   freeze_hold::Bool=false,
                                   freeze_spring::Bool=false)
    throw(MethodError(execute_integrator_stage!, (spec, st, dt, stage_tag)))
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
