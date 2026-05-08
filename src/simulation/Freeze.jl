# -------------------------
# Freeze helpers
# -------------------------

@inline function _freeze_active!(st::SimulationState)
    if st.freeze_mode == FREEZE_NONE
        return false
    end
    if st.freeze_until >= 0 && st.step >= st.freeze_until
        st.freeze_mode = FREEZE_NONE
        st.freeze_until = -1
        return false
    end
    return true
end

function _freeze_hold2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
        end
    end
    return
end

function _freeze_hold3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                               mask::CuDeviceVector{UInt8},
                               ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            rx[i] = ax[i]
            ry[i] = ay[i]
            rz[i] = az[i]
        end
    end
    return
end

function _freeze_hold2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold2_kernel!(rx, ry, mask, ax, ay)
    k(rx, ry, mask, ax, ay; threads, blocks)
    return nothing
end

function _freeze_hold3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        mask::CuArray{UInt8,1},
                        ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _freeze_hold3_kernel!(rx, ry, rz, mask, ax, ay, az)
    k(rx, ry, rz, mask, ax, ay, az; threads, blocks)
    return nothing
end

function _freeze_spring2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
        end
    end
    return
end

function _freeze_spring2_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy)
        end
    end
    return
end

function _freeze_spring3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
        end
    end
    return
end

function _freeze_spring3_energy_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                        fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                        Epot::CuDeviceVector{T},
                                        mask::CuDeviceVector{UInt8},
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = rx[i] - ax[i]
            dy = ry[i] - ay[i]
            dz = rz[i] - az[i]
            fx[i] -= k * dx
            fy[i] -= k * dy
            fz[i] -= k * dz
            Epot[i] += T(0.5) * k * (dx * dx + dy * dy + dz * dz)
        end
    end
    return
end

function _freeze_spring2!(rx::CuArray{T,1}, ry::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_kernel!(rx, ry, fx, fy, mask, ax, ay, k)
    ker(rx, ry, fx, fy, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring2_energy!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring2_energy_kernel!(rx, ry, fx, fy, Epot, mask, ax, ay, k)
    ker(rx, ry, fx, fy, Epot, mask, ax, ay, k; threads, blocks)
    return nothing
end

function _freeze_spring3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_kernel!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _freeze_spring3_energy!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    ker = CUDA.@cuda launch=false _freeze_spring3_energy_kernel!(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k)
    ker(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k; threads, blocks)
    return nothing
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    return _freeze_hold2!(rx, ry, mask, ax, ay)
end

function _apply_freeze_hold!(st::SimulationState{T}, rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1}) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    return _freeze_hold3!(rx, ry, rz, mask, ax, ay, az)
end

function _apply_freeze_hold_unwrap!(st::SimulationState{T}) where {T<:AbstractFloat}
    rxu = st.rx_unwrap
    ryu = st.ry_unwrap
    if rxu === nothing || ryu === nothing
        return nothing
    end
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    if st.rz_unwrap === nothing
        return _freeze_hold2!(rxu, ryu, mask, ax, ay)
    end
    az = st.freeze_rz
    az === nothing && return nothing
    return _freeze_hold3!(rxu, ryu, st.rz_unwrap, mask, ax, ay, az)
end

function _apply_freeze_hold_positions!(st::SimulationState{T}) where {T<:AbstractFloat}
    if st.rz === nothing
        _apply_freeze_hold!(st, st.rx, st.ry)
    else
        _apply_freeze_hold!(st, st.rx, st.ry, st.rz)
    end
    _apply_freeze_hold_unwrap!(st)
    return nothing
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    if mask === nothing || ax === nothing || ay === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring2_energy!(rx, ry, fx, fy, E, mask, ax, ay, k)
    end
    return _freeze_spring2!(rx, ry, fx, fy, mask, ax, ay, k)
end

function _apply_freeze_spring!(st::SimulationState{T},
                               rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                               fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                               E::Union{Nothing,CuArray{T,1}}, compute_energy::Bool) where {T<:AbstractFloat}
    mask = st.freeze_mask
    ax = st.freeze_rx
    ay = st.freeze_ry
    az = st.freeze_rz
    if mask === nothing || ax === nothing || ay === nothing || az === nothing
        return nothing
    end
    k = st.freeze_k
    k <= zero(T) && return nothing
    if compute_energy && st.freeze_include_energy && E !== nothing
        return _freeze_spring3_energy!(rx, ry, rz, fx, fy, fz, E, mask, ax, ay, az, k)
    end
    return _freeze_spring3!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k)
end
