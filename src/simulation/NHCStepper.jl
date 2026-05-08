function _nhc_active_bath_counts_kernel!(bath_counts::CuDeviceVector{Int32},
                                         particle_bath_id::CuDeviceVector{Int32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(bath_counts)
                CUDA.@atomic bath_counts[b] += Int32(1)
            end
        end
    end
    return nothing
end

function _nhc_active_bath_counts_with_freeze_kernel!(bath_counts::CuDeviceVector{Int32},
                                                     particle_bath_id::CuDeviceVector{Int32},
                                                     freeze_mask::CuDeviceVector{UInt8})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(particle_bath_id)
        @inbounds begin
            if freeze_mask[i] == UInt8(0)
                b = Int(particle_bath_id[i])
                if 1 <= b <= length(bath_counts)
                    CUDA.@atomic bath_counts[b] += Int32(1)
                end
            end
        end
    end
    return nothing
end

function _nhc_finalize_dof_kernel!(dof_per_bath::CuDeviceVector{T},
                                   bath_counts::CuDeviceVector{Int32},
                                   dim::T) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if b <= length(dof_per_bath)
        @inbounds dof_per_bath[b] = dim * T(bath_counts[b])
    end
    return nothing
end

function _update_bath_dof!(bath_counts::CuArray{Int32,1},
                           dof_per_bath::CuArray{T,1},
                           particle_bath_id::CuArray{Int32,1},
                           st::SimulationState{T}) where {T<:AbstractFloat}
    fill!(bath_counts, Int32(0))
    fill!(dof_per_bath, zero(T))

    N = length(particle_bath_id)
    if N > 0
        threads = min(256, N)
        blocks = cld(N, threads)
        if st.freeze_mode != FREEZE_NONE && st.freeze_mask !== nothing
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_with_freeze_kernel!(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1})
            k(bath_counts, particle_bath_id, st.freeze_mask::CuArray{UInt8,1}; threads, blocks)
        else
            k = CUDA.@cuda launch=false _nhc_active_bath_counts_kernel!(bath_counts, particle_bath_id)
            k(bath_counts, particle_bath_id; threads, blocks)
        end
    end

    B = length(dof_per_bath)
    if B > 0
        threads = min(256, B)
        blocks = cld(B, threads)
        dim = T(_is_3d(st) ? 3 : 2)
        k = CUDA.@cuda launch=false _nhc_finalize_dof_kernel!(dof_per_bath, bath_counts, dim)
        k(dof_per_bath, bath_counts, dim; threads, blocks)
    end
    return nothing
end

function _nhc_update_dof_per_bath!(spec::NHCSpec{T},
                                   st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    _update_bath_dof!(ws.bath_counts, ws.dof_per_bath, ws.particle_bath_id, st)
    ws.dof_dirty = false
    return nothing
end

function _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath::CuDeviceVector{T},
                                             Ekin::CuDeviceVector{T},
                                             particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Ekin)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += Ekin[i]
            end
        end
    end
    return nothing
end

function _nhc_reduce_kinetic_by_bath!(kinetic_total_per_bath::CuArray{T,1},
                                      Ekin::CuArray{T,1},
                                      particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(Ekin)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _nhc_reduce_kinetic_by_bath_kernel!(kinetic_total_per_bath, Ekin, particle_bath_id)
    k(kinetic_total_per_bath, Ekin, particle_bath_id; threads, blocks)
    return nothing
end

"""
    _ensure_nhc_kinetic_initialized!(spec, st)

Initialize per-bath kinetic totals and `st.Ekin` once before the first NHC
thermostat stage if no prior kick/thermostat stage has populated them.
"""
function _ensure_nhc_kinetic_initialized!(spec::NHCSpec{T},
                                          st::SimulationState{T}) where {T<:AbstractFloat}
    ws = spec.workspace
    if !ws.kinetic_initialized
        _refresh_kinetic_buffer!(st, spec.params.mass)
        _nhc_reduce_kinetic_by_bath!(ws.kinetic_total_per_bath, st.Ekin, ws.particle_bath_id)
        ws.kinetic_initialized = true
    end
    return nothing
end

function _nhc_apply_stage_scale2_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

function _nhc_apply_stage_scale3_by_bath_kernel!(vx::CuDeviceVector{T},
                                                 vy::CuDeviceVector{T},
                                                 vz::CuDeviceVector{T},
                                                 Ekin::CuDeviceVector{T},
                                                 stage_scale_per_bath::CuDeviceVector{T},
                                                 particle_bath_id::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(stage_scale_per_bath)
                s = stage_scale_per_bath[b]
                s2 = s * s
                vx[i] *= s
                vy[i] *= s
                vz[i] *= s
                Ekin[i] *= s2
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_stage_scale!(st, stage_scale_per_bath, particle_bath_id)

Apply the net thermostat scale per bath for one NHC stage to velocities and to
`st.Ekin` in a single GPU pass.
"""
function _nhc_apply_stage_scale!(st::SimulationState{T},
                                 stage_scale_per_bath::CuArray{T,1},
                                 particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    N = length(st.vx)
    N == 0 && return nothing
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.vz::CuArray{T,1}, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_apply_stage_scale2_by_bath_kernel!(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id)
        k(st.vx, st.vy, st.Ekin, stage_scale_per_bath, particle_bath_id; threads, blocks)
    end
    return nothing
end

function _nhc_half_kick2_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new)
            vx[i] = vx_new
            vy[i] = vy_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

function _nhc_half_kick3_by_bath_kernel!(vx::CuDeviceVector{T},
                                         vy::CuDeviceVector{T},
                                         vz::CuDeviceVector{T},
                                         fx::CuDeviceVector{T},
                                         fy::CuDeviceVector{T},
                                         fz::CuDeviceVector{T},
                                         Ekin::CuDeviceVector{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         particle_bath_id::CuDeviceVector{Int32},
                                         coef::T,
                                         mass::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(vx)
        @inbounds begin
            vx_new = vx[i] + coef * fx[i]
            vy_new = vy[i] + coef * fy[i]
            vz_new = vz[i] + coef * fz[i]
            ek = T(0.5) * mass * (vx_new * vx_new + vy_new * vy_new + vz_new * vz_new)
            vx[i] = vx_new
            vy[i] = vy_new
            vz[i] = vz_new
            Ekin[i] = ek
            b = Int(particle_bath_id[i])
            if 1 <= b <= length(kinetic_total_per_bath)
                CUDA.@atomic kinetic_total_per_bath[b] += ek
            end
        end
    end
    return nothing
end

"""
    _nhc_apply_half_kick!(st, fx, fy, fz, dt, mass, kinetic_total_per_bath, particle_bath_id)

Apply a deterministic half-force kick `v <- v + (dt / (2m)) f`, refresh
`st.Ekin`, and reduce per-bath kinetic energies into `kinetic_total_per_bath`.
"""
function _nhc_apply_half_kick!(st::SimulationState{T},
                               fx::CuArray{T,1},
                               fy::CuArray{T,1},
                               fz::Union{Nothing,CuArray{T,1}},
                               dt::T,
                               mass::T,
                               kinetic_total_per_bath::CuArray{T,1},
                               particle_bath_id::CuArray{Int32,1}) where {T<:AbstractFloat}
    fill!(kinetic_total_per_bath, zero(T))
    N = length(st.vx)
    N == 0 && return nothing
    coef = dt / (T(2) * mass)
    threads = min(256, N)
    blocks = cld(N, threads)
    if _is_3d(st)
        k = CUDA.@cuda launch=false _nhc_half_kick3_by_bath_kernel!(st.vx, st.vy, st.vz::CuArray{T,1},
                                                                     fx, fy, fz::CuArray{T,1},
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy, st.vz::CuArray{T,1},
          fx, fy, fz::CuArray{T,1},
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    else
        k = CUDA.@cuda launch=false _nhc_half_kick2_by_bath_kernel!(st.vx, st.vy,
                                                                     fx, fy,
                                                                     st.Ekin, kinetic_total_per_bath,
                                                                     particle_bath_id,
                                                                     coef, mass)
        k(st.vx, st.vy,
          fx, fy,
          st.Ekin, kinetic_total_per_bath,
          particle_bath_id,
          coef, mass; threads, blocks)
    end
    return nothing
end

"""
    _nhc_drift_positions!(st, dt)

Advance positions by one full deterministic drift under periodic boundaries.
Implemented through the existing BAOAB A-kernel using `2dt` so the effective
drift is exactly `dt`.
"""
function _nhc_drift_positions!(st::SimulationState{T}, dt::T) where {T<:AbstractFloat}
    drift_dt = T(2) * dt
    if _is_3d(st)
        LangevinIntegrators.baoab_A_3d!(st.rx, st.ry, st.rz,
                                        st.vx, st.vy, st.vz,
                                        drift_dt, st.box3::Definitions.Box3;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap,
                                        unwrapped_z=st.rz_unwrap)
    else
        LangevinIntegrators.baoab_A_2d!(st.rx, st.ry,
                                        st.vx, st.vy,
                                        drift_dt, st.box2::Definitions.Box2;
                                        unwrapped_x=st.rx_unwrap,
                                        unwrapped_y=st.ry_unwrap)
    end
    return nothing
end

@inline function _nhc_gromacs_sy_weight(::Type{T}, idx::Int32) where {T<:AbstractFloat}
    if idx == Int32(3)
        return T(-0.186929716880426)
    elseif idx >= Int32(1) && idx <= Int32(5)
        return T(0.2967324292201065)
    end
    return zero(T)
end

function _nhc_chain_stage_legacy_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                scale = exp(-h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                if M == 1
                    g1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]
                    xi[1, b] += half_h * g1
                else
                    chain_force[1, b] = (T(2) * K - dof * Ttarget) / chain_masses[1, b] - xi[1, b] * xi[2, b]
                    for j in 2:(M - 1)
                        chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b] - xi[j, b] * xi[j + 1, b]
                    end
                    chain_force[M, b] = (chain_masses[M - 1, b] * xi[M - 1, b] * xi[M - 1, b] - Ttarget) / chain_masses[M, b]
                    for j in 1:M
                        xi[j, b] += half_h * chain_force[j, b]
                    end
                end

                for j in 1:M
                    eta[j, b] += h * xi[j, b]
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_gromacs_kernel!(xi::CuDeviceMatrix{T},
                                          eta::CuDeviceMatrix{T},
                                          chain_masses::CuDeviceMatrix{T},
                                          kinetic_total_per_bath::CuDeviceVector{T},
                                          thermostat_kinetic_per_bath::CuDeviceVector{T},
                                          thermostat_potential_per_bath::CuDeviceVector{T},
                                          last_velocity_scale_per_bath::CuDeviceVector{T},
                                          dof_per_bath::CuDeviceVector{T},
                                          target_temperature::CuDeviceVector{T},
                                          stage_dt::T,
                                          substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for _ in 1:ns
                for sy_idx in Int32(1):Int32(5)
                    time_step = stage_dt * _nhc_gromacs_sy_weight(T, sy_idx) / T(ns)

                    for j in M:-1:1
                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end

                    system_scale = exp(-time_step * xi[1, b])
                    total_scale *= system_scale
                    K *= system_scale * system_scale

                    for j in 1:M
                        eta[j, b] += time_step * xi[j, b]

                        kinetic2 = if j == 1
                            T(2) * K
                        else
                            chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b]
                        end
                        num_dof = j == 1 ? dof : one(T)
                        xi_accel = (kinetic2 - num_dof * Ttarget) / chain_masses[j, b]
                        local_scale = if j < M
                            exp(-T(0.25) * time_step * xi[j + 1, b])
                        else
                            one(T)
                        end
                        xi[j, b] = local_scale * (xi[j, b] * local_scale + T(0.5) * time_step * xi_accel)
                    end
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _nhc_chain_stage_lammps_kernel!(xi::CuDeviceMatrix{T},
                                         eta::CuDeviceMatrix{T},
                                         chain_force::CuDeviceMatrix{T},
                                         chain_masses::CuDeviceMatrix{T},
                                         kinetic_total_per_bath::CuDeviceVector{T},
                                         thermostat_kinetic_per_bath::CuDeviceVector{T},
                                         thermostat_potential_per_bath::CuDeviceVector{T},
                                         last_velocity_scale_per_bath::CuDeviceVector{T},
                                         dof_per_bath::CuDeviceVector{T},
                                         target_temperature::CuDeviceVector{T},
                                         stage_dt::T,
                                         substeps::Int32) where {T<:AbstractFloat}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nbaths = length(kinetic_total_per_bath)
    if b <= nbaths
        @inbounds begin
            dof = dof_per_bath[b]
            if dof <= zero(T)
                kinetic_total_per_bath[b] = zero(T)
                thermostat_kinetic_per_bath[b] = zero(T)
                thermostat_potential_per_bath[b] = zero(T)
                last_velocity_scale_per_bath[b] = one(T)
                return nothing
            end

            M = size(xi, 1)
            ns = Int(substeps)
            h = stage_dt / T(ns)
            quarter_h = h / T(4)
            eighth_h = h / T(8)
            half_h = h / T(2)
            Ttarget = target_temperature[b]
            K = kinetic_total_per_bath[b]
            total_scale = one(T)

            for j in 2:M
                chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
            end

            for _ in 1:ns
                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in M:-1:2
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end

                expfac1 = M > 1 ? exp(-eighth_h * xi[2, b]) : one(T)
                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                scale = exp(-half_h * xi[1, b])
                total_scale *= scale
                K *= scale * scale

                accel1 = (T(2) * K - dof * Ttarget) / chain_masses[1, b]

                for j in 1:M
                    eta[j, b] += half_h * xi[j, b]
                end

                xi[1, b] *= expfac1
                xi[1, b] += accel1 * quarter_h
                xi[1, b] *= expfac1

                for j in 2:M
                    expfac = j < M ? exp(-eighth_h * xi[j + 1, b]) : one(T)
                    xi[j, b] *= expfac
                    chain_force[j, b] = (chain_masses[j - 1, b] * xi[j - 1, b] * xi[j - 1, b] - Ttarget) / chain_masses[j, b]
                    xi[j, b] += chain_force[j, b] * quarter_h
                    xi[j, b] *= expfac
                end
            end

            kinetic_total_per_bath[b] = K
            last_velocity_scale_per_bath[b] = total_scale

            therm_kin = zero(T)
            for j in 1:M
                therm_kin += T(0.5) * chain_masses[j, b] * xi[j, b] * xi[j, b]
            end
            thermostat_kinetic_per_bath[b] = therm_kin

            therm_pot = dof * Ttarget * eta[1, b]
            for j in 2:M
                therm_pot += Ttarget * eta[j, b]
            end
            thermostat_potential_per_bath[b] = therm_pot
        end
    end
    return nothing
end

function _run_nhc_chain_stage!(spec::NHCSpec{T},
                               stage_dt::T) where {T<:AbstractFloat}
    p = spec.params
    ws = spec.workspace
    B = length(ws.kinetic_total_per_bath)
    B == 0 && return nothing
    threads = min(256, B)
    blocks = cld(B, threads)
    if p.propagator == NHC_PROPAGATOR_LEGACY
        k = CUDA.@cuda launch=false _nhc_chain_stage_legacy_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_GROMACS
        k = CUDA.@cuda launch=false _nhc_chain_stage_gromacs_kernel!(ws.xi,
                                                                     ws.eta,
                                                                     ws.chain_masses,
                                                                     ws.kinetic_total_per_bath,
                                                                     ws.thermostat_kinetic_per_bath,
                                                                     ws.thermostat_potential_per_bath,
                                                                     ws.last_velocity_scale_per_bath,
                                                                     ws.dof_per_bath,
                                                                     ws.target_temperature,
                                                                     stage_dt,
                                                                     Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    elseif p.propagator == NHC_PROPAGATOR_LAMMPS
        k = CUDA.@cuda launch=false _nhc_chain_stage_lammps_kernel!(ws.xi,
                                                                    ws.eta,
                                                                    ws.chain_force,
                                                                    ws.chain_masses,
                                                                    ws.kinetic_total_per_bath,
                                                                    ws.thermostat_kinetic_per_bath,
                                                                    ws.thermostat_potential_per_bath,
                                                                    ws.last_velocity_scale_per_bath,
                                                                    ws.dof_per_bath,
                                                                    ws.target_temperature,
                                                                    stage_dt,
                                                                    Int32(p.substeps))
        k(ws.xi,
          ws.eta,
          ws.chain_force,
          ws.chain_masses,
          ws.kinetic_total_per_bath,
          ws.thermostat_kinetic_per_bath,
          ws.thermostat_potential_per_bath,
          ws.last_velocity_scale_per_bath,
          ws.dof_per_bath,
          ws.target_temperature,
          stage_dt,
          Int32(p.substeps);
          threads,
          blocks)
    else
        throw(ArgumentError("Unsupported NHC propagator id $(p.propagator)."))
    end
    return nothing
end

"""
    _apply_nhc_thermostat_stage!(spec, st, stage_dt)

Propagate the NHC chain and apply deterministic velocity scaling over
`stage_dt`, internally split into `spec.params.substeps`.
"""
function _apply_nhc_thermostat_stage!(spec::NHCSpec{T},
                                      st::SimulationState{T},
                                      stage_dt::T) where {T<:AbstractFloat}
    ws = spec.workspace
    _nhc_update_dof_per_bath!(spec, st)
    _ensure_nhc_kinetic_initialized!(spec, st)
    copyto!(ws.kinetic_stage_start_per_bath, ws.kinetic_total_per_bath)
    _run_nhc_chain_stage!(spec, stage_dt)
    @. ws.cumulative_energy_exchange_per_bath += ws.kinetic_stage_start_per_bath - ws.kinetic_total_per_bath
    _nhc_apply_stage_scale!(st, ws.last_velocity_scale_per_bath, ws.particle_bath_id)
    return nothing
end
