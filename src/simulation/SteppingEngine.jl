include("NVEFusion.jl")









# Device-engine hooks; `ParticleDynamics.DeviceEngine` adds methods for its engine.
# `_engine_boundary!` checks device status at an explicit host boundary.
_engine_boundary!(::Nothing, st) = nothing
# ELL rows `(flat, counts)` of the engine that owns the state (`flat[t*N + i]`), or nothing.
_engine_neighbor_rows(engine) = nothing
# Box change for the engine that owns the state (`set_box!`); no-op without one.
_engine_set_box!(engine, st, box) = nothing
# Any force invalidation: the engine no longer knows why forces are invalid.
_engine_invalidated!(engine) = nothing
# Settle and discard the engine that owns the state (its arrays are the state's).
_drop_engine!(::Nothing, st) = nothing
# Why no engine evaluates the state (DeviceEngine adds the reasons).
_evaluation_refusal(st, freeze_spring) = "unsupported configuration"

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
    step!(st, spec, dt; compute_energy=true)

Canonical explicit stepping API. Every step runs on the device engine
(`ParticleDynamics.DeviceEngine` adds the methods for each integrator); there is
no staged fallback.
"""
function step!(st::SimulationState{T},
               spec::IntegratorSpec{T},
               dt::Real;
               compute_energy::Bool=true) where {T<:AbstractFloat}
    throw(ArgumentError("integrator $(nameof(typeof(spec))) is not available on the device engine"))
end

"""
    step_graph!(st, spec, dt; compute_energy=true)

Same as [`step!`](@ref): every step is a captured CUDA-graph replay on the device engine.
"""
function step_graph!(st::SimulationState{T},
                     spec::IntegratorSpec{T},
                     dt::Real;
                     compute_energy::Bool=true) where {T<:AbstractFloat}
    return step!(st, spec, dt; compute_energy=compute_energy)
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
