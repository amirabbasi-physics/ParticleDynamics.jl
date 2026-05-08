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

@inline function _csvr_chisq(::Type{T}, dof::Int) where {T<:AbstractFloat}
    dof <= 0 && return zero(T)
    return T(2) * _csvr_gamma_sample(T(dof) / T(2))
end

function _csvr_thermostat_stage_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                        cumulative_energy_exchange_per_bath::CuDeviceVector{T},
                                        last_velocity_scale_per_bath::CuDeviceVector{T},
                                        dof_per_bath::CuDeviceVector{T},
                                        target_temperature::CuDeviceVector{T},
                                        tau::CuDeviceVector{T},
                                        stage_dt::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            Kold = kinetic_total_per_bath[b]
            Ttarget = target_temperature[b]
            τ = tau[b]
            if !(Kold > eps(T)) || !(Ttarget > zero(T)) || !(τ > zero(T))
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            ndof = max(1, Int(floor(dof + T(0.5))))
            Kbar = T(0.5) * dof * Ttarget
            c1 = exp(-stage_dt / τ)
            c2 = ((one(T) - c1) * Kbar) / (Kold * T(ndof))
            r1 = randn(T)
            r2 = _csvr_chisq(T, ndof - 1)
            cross = sqrt(max(c1 * c2, zero(T)))
            scale2 = c1 + c2 * (r1 * r1 + r2) + T(2) * r1 * cross
            scale2 = max(scale2, zero(T))
            scale = sqrt(scale2)
            Knew = Kold * scale2

            kinetic_total_per_bath[b] = Knew
            cumulative_energy_exchange_per_bath[b] += Kold - Knew
            last_velocity_scale_per_bath[b] = scale
        end
    end
    return nothing
end

function _run_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                     stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    k = CUDA.@cuda launch=false _csvr_thermostat_stage_kernel!(ws.kinetic_total_per_bath,
                                                               ws.cumulative_energy_exchange_per_bath,
                                                               ws.last_velocity_scale_per_bath,
                                                               ws.dof_per_bath,
                                                               ws.target_temperature,
                                                               ws.tau,
                                                               stage_dt)
    k(ws.kinetic_total_per_bath,
      ws.cumulative_energy_exchange_per_bath,
      ws.last_velocity_scale_per_bath,
      ws.dof_per_bath,
      ws.target_temperature,
      ws.tau,
      stage_dt;
      threads,
      blocks)
    return nothing
end

function _apply_csvr_thermostat_stage!(spec::CSVRSpec{T},
                                       st::SimulationState{T},
                                       stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _csvr_update_dof_per_bath!(spec, st)
    _ensure_csvr_kinetic_initialized!(spec, st)
    _run_csvr_thermostat_stage!(spec, stage_dt)
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end
