import ..SimulationCore: minimize!

_mobile_indices(sim, ::Nothing) = nothing
_mobile_indices(sim, group::Group) = _selection_indices(sim.system, group.selection)
_mobile_indices(sim, selection::Union{AllSelection,TypeSelection,IndexSelection}) = _selection_indices(sim.system, selection)
_mobile_indices(sim, indices::AbstractVector{<:Integer}) = collect(Int, indices)

"""
    minimize!(sim::Simulation; method, force_tolerance, max_displacement,
              max_iterations=10_000, max_evaluations=20_000, mobile=nothing,
              velocities=:zero) -> MinimizationResult

Minimize the workflow's declared forces at fixed box (see the low-level
[`minimize!`](@ref)). The integrator only supplies forces: `methods` may be
empty and no thermostat runs. `mobile` is `nothing` (all particles), a `Group`,
a selection or particle indices; other coordinates stay pinned but keep their
interactions. MSD/VACF references are reset to the minimized configuration;
thermalization is a separate step.
"""
function minimize!(sim::Simulation; mobile=nothing, kwargs...)
    sim.prepared || prepare!(sim)
    result = minimize!(sim.state::SimulationState; mobile=_mobile_indices(sim, mobile), kwargs...)
    reset_observables!(sim)
    sim.metadata[:last_minimization] = result
    return result
end
