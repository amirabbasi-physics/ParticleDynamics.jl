function _csvr_update_dof_per_bath!(spec::CSVRSpec{T},
                                    st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _ensure_csvr_kinetic_initialized!(spec::CSVRSpec{T},
                                           st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

@inline function _csvr_gamma_sample(shape::T) where {T<:AbstractFloat}
    shape > zero(T) || return zero(T)
    if shape < one(T)
        u = max(rand(T), eps(T))
        return _csvr_gamma_sample(shape + one(T)) * u^(inv(shape))
    end

    d = shape - T(1) / T(3)
    c = inv(sqrt(T(9) * d))
    while true
        x = randn(T)
        v = one(T) + c * x
        v <= zero(T) && continue
        v3 = v * v * v
        u = rand(T)
        x2 = x * x
        if u < one(T) - T(0.0331) * x2 * x2
            return d * v3
        end
        if log(max(u, eps(T))) < T(0.5) * x2 + d * (one(T) - v3 + log(v3))
            return d * v3
        end
    end
end
