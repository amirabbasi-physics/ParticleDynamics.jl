# -------------------------
# Freeze helpers
# -------------------------

@inline function _freeze_active!(st::SimulationState)
    if st.freeze_mode == FREEZE_NONE
        return false
    end
    if st.freeze_until >= 0 && st.step >= st.freeze_until
        invalidate_forces!(st)
        st.freeze_mode = FREEZE_NONE
        st.freeze_until = -1
        return false
    end
    return true
end



# The tether displacement is the minimum image of r - anchor: positions are
# wrapped, so a tethered particle crossing a periodic face keeps a continuous
# force and energy (any physical tether keeps |r - anchor| < L/2).
function _freeze_spring2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                 fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                                 mask::CuDeviceVector{UInt8},
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T,
                                 L::NTuple{2,T}, h::NTuple{2,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = NeighborLists.mic_fast(rx[i] - ax[i], h[1], L[1])
            dy = NeighborLists.mic_fast(ry[i] - ay[i], h[2], L[2])
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
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, k::T,
                                        L::NTuple{2,T}, h::NTuple{2,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = NeighborLists.mic_fast(rx[i] - ax[i], h[1], L[1])
            dy = NeighborLists.mic_fast(ry[i] - ay[i], h[2], L[2])
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
                                 ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T,
                                 L::NTuple{3,T}, h::NTuple{3,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = NeighborLists.mic_fast(rx[i] - ax[i], h[1], L[1])
            dy = NeighborLists.mic_fast(ry[i] - ay[i], h[2], L[2])
            dz = NeighborLists.mic_fast(rz[i] - az[i], h[3], L[3])
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
                                        ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T}, k::T,
                                        L::NTuple{3,T}, h::NTuple{3,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        if mask[i] != 0
            dx = NeighborLists.mic_fast(rx[i] - ax[i], h[1], L[1])
            dy = NeighborLists.mic_fast(ry[i] - ay[i], h[2], L[2])
            dz = NeighborLists.mic_fast(rz[i] - az[i], h[3], L[3])
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
                          ax::CuArray{T,1}, ay::CuArray{T,1}, k::T, L::NTuple{2,T}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads); h = map(x -> T(0.5) * x, L)
    ker = CUDA.@cuda launch=false _freeze_spring2_kernel!(rx, ry, fx, fy, mask, ax, ay, k, L, h)
    ker(rx, ry, fx, fy, mask, ax, ay, k, L, h; threads, blocks)
    return nothing
end

function _freeze_spring2_energy!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, k::T, L::NTuple{2,T}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads); h = map(x -> T(0.5) * x, L)
    ker = CUDA.@cuda launch=false _freeze_spring2_energy_kernel!(rx, ry, fx, fy, Epot, mask, ax, ay, k, L, h)
    ker(rx, ry, fx, fy, Epot, mask, ax, ay, k, L, h; threads, blocks)
    return nothing
end

function _freeze_spring3!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                          fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                          mask::CuArray{UInt8,1},
                          ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T, L::NTuple{3,T}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads); h = map(x -> T(0.5) * x, L)
    ker = CUDA.@cuda launch=false _freeze_spring3_kernel!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k, L, h)
    ker(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k, L, h; threads, blocks)
    return nothing
end

function _freeze_spring3_energy!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 Epot::CuArray{T,1},
                                 mask::CuArray{UInt8,1},
                                 ax::CuArray{T,1}, ay::CuArray{T,1}, az::CuArray{T,1}, k::T, L::NTuple{3,T}) where {T<:AbstractFloat}
    N = length(rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads); h = map(x -> T(0.5) * x, L)
    ker = CUDA.@cuda launch=false _freeze_spring3_energy_kernel!(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k, L, h)
    ker(rx, ry, rz, fx, fy, fz, Epot, mask, ax, ay, az, k, L, h; threads, blocks)
    return nothing
end

# Held particles move their unwrapped coordinates back with them by the minimum
# image of anchor - r. Called before the positions are reset to the anchors, so
# unwrapped - wrapped stays a multiple of the box length and the unwrapped
# trajectory stays continuous (a held particle's unwrapped coordinate keeps its
# value from the freeze, image count included).
function _freeze_hold_unwrap2_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                                      ux::CuDeviceVector{T}, uy::CuDeviceVector{T},
                                      mask::CuDeviceVector{UInt8},
                                      ax::CuDeviceVector{T}, ay::CuDeviceVector{T},
                                      L::NTuple{2,T}, h::NTuple{2,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds if mask[i] != 0
        ux[i] += NeighborLists.mic_fast(ax[i] - rx[i], h[1], L[1])
        uy[i] += NeighborLists.mic_fast(ay[i] - ry[i], h[2], L[2])
    end
    return
end

function _freeze_hold_unwrap3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                      ux::CuDeviceVector{T}, uy::CuDeviceVector{T}, uz::CuDeviceVector{T},
                                      mask::CuDeviceVector{UInt8},
                                      ax::CuDeviceVector{T}, ay::CuDeviceVector{T}, az::CuDeviceVector{T},
                                      L::NTuple{3,T}, h::NTuple{3,T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds if mask[i] != 0
        ux[i] += NeighborLists.mic_fast(ax[i] - rx[i], h[1], L[1])
        uy[i] += NeighborLists.mic_fast(ay[i] - ry[i], h[2], L[2])
        uz[i] += NeighborLists.mic_fast(az[i] - rz[i], h[3], L[3])
    end
    return
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
    N = length(st.rx); N == 0 && return nothing
    threads = min(256, N); blocks = cld(N, threads)
    if st.rz_unwrap === nothing
        L2 = st.box2::Definitions.Box2{T}; h2 = map(x -> T(0.5) * x, L2)
        ker = CUDA.@cuda launch=false _freeze_hold_unwrap2_kernel!(st.rx, st.ry, rxu, ryu, mask, ax, ay, L2, h2)
        ker(st.rx, st.ry, rxu, ryu, mask, ax, ay, L2, h2; threads, blocks)
        return nothing
    end
    az = st.freeze_rz
    az === nothing && return nothing
    L3 = st.box3::Definitions.Box3{T}; h3 = map(x -> T(0.5) * x, L3)
    ker = CUDA.@cuda launch=false _freeze_hold_unwrap3_kernel!(st.rx, st.ry, st.rz, rxu, ryu, st.rz_unwrap, mask, ax, ay, az, L3, h3)
    ker(st.rx, st.ry, st.rz, rxu, ryu, st.rz_unwrap, mask, ax, ay, az, L3, h3; threads, blocks)
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
        return _freeze_spring2_energy!(rx, ry, fx, fy, E, mask, ax, ay, k, st.box2::Definitions.Box2{T})
    end
    return _freeze_spring2!(rx, ry, fx, fy, mask, ax, ay, k, st.box2::Definitions.Box2{T})
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
        return _freeze_spring3_energy!(rx, ry, rz, fx, fy, fz, E, mask, ax, ay, az, k, st.box3::Definitions.Box3{T})
    end
    return _freeze_spring3!(rx, ry, rz, fx, fy, fz, mask, ax, ay, az, k, st.box3::Definitions.Box3{T})
end
