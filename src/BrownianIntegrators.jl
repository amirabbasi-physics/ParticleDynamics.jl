"""
Overdamped (Brownian) integrators used by `examples/*BD*.jl`.
"""
module BrownianIntegrators

using CUDA
using ..Definitions

export BrownianParams,
       bd_midpoint_positions_2d!, bd_midpoint_positions_3d!,
       bd_prepare_noise_2d!, bd_prepare_noise_3d!,
       bd_finish_step_2d!, bd_finish_step_3d!,
       EMParams, em_step_2d!, em_step_3d!,
       em_apply_step_2d!, em_apply_step_3d!

"""
    BrownianParams(gamma, noise_scale; corr_time=nothing)

Parameters for the stochastic midpoint Brownian integrator. `gamma` and
`noise_scale` are CuArrays so filters can assign different temperatures to
different particle groups (as in `examples/TwoT_2D_BD_EH.jl`).
"""
struct BrownianParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
    function BrownianParams{T}(gamma::CuArray{T,1},
                               dt::T,
                               noise_scale::CuArray{T,1},
                               corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                               ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, dt, noise_scale, corr_time, ou)
    end
end

BrownianParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, one(T), noise_scale, nothing, nothing)
BrownianParams(gamma::CuArray{T,1},
               noise_scale::CuArray{T,1},
               corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, one(T), noise_scale, corr_time, nothing)
BrownianParams(gamma::CuArray{T,1},
               noise_scale::CuArray{T,1},
               corr_time::Union{Nothing,CuArray{T,1}},
               dt::T,
               ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, dt, noise_scale, corr_time, ou)

function BrownianParams{T}(gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    γ = T(gamma)
    Tval = T(temperature)
    Δt = T(dt)
    gamma_vec = CUDA.fill(γ, N)
    scale = sqrt(T(2) * γ * Tval * Δt)
    noise_vec = CUDA.fill(scale, N)
    return BrownianParams{T}(gamma_vec, Δt, noise_vec, nothing, nothing)
end

function BrownianParams(::Type{T}, gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    return BrownianParams{T}(gamma, temperature, dt, N)
end

# Euler–Maruyama parameters (reuse gamma/noise scale like BrownianParams)
"""
    EMParams(gamma, noise_scale; corr_time=nothing)

Parameter bundle for the Euler–Maruyama step (`em_step_*`). Shares the same
layout as [`BrownianParams`](@ref) so utilities such as
`Filters.set_temperature!` operate on either type.
"""
struct EMParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    dt::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    ou::Union{Nothing,Definitions.OUSpectrum{T}}
    function EMParams{T}(gamma::CuArray{T,1},
                         dt::T,
                         noise_scale::CuArray{T,1},
                         corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                         ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, dt, noise_scale, corr_time, ou)
    end
end

EMParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    EMParams{T}(gamma, one(T), noise_scale, nothing, nothing)
EMParams(gamma::CuArray{T,1},
         noise_scale::CuArray{T,1},
         corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} =
    EMParams{T}(gamma, one(T), noise_scale, corr_time, nothing)
EMParams(gamma::CuArray{T,1},
         noise_scale::CuArray{T,1},
         corr_time::Union{Nothing,CuArray{T,1}},
         dt::T,
         ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing) where {T<:AbstractFloat} =
    EMParams{T}(gamma, dt, noise_scale, corr_time, ou)

function BrownianParams(gamma::Real, temperature::Real, dt::Real, N::Integer)
    inferred = promote_type(float(typeof(gamma)), float(typeof(temperature)), float(typeof(dt)))
    T = inferred <: AbstractFloat ? inferred : Float32
    return BrownianParams{T}(gamma, temperature, dt, N)
end

@inline function _wrap_centered(x::T, L::T) where {T<:AbstractFloat}
    half = T(0.5)
    y = x + half * L
    y -= floor(y / L) * L
    return y - half * L
end

# ---------------------------------------------
# Noise preparation (standard normal components)
# ---------------------------------------------
function _noise2!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
                  noise_scale::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        ξx[i] = s * randn(T)
        ξy[i] = s * randn(T)
    end
    return
end

function _noise3!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
                  noise_scale::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        ξx[i] = s * randn(T)
        ξy[i] = s * randn(T)
        ξz[i] = s * randn(T)
    end
    return
end

function _apply_ou2!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
                     active_idx::CuDeviceVector{Int32},
                     coeff_a::CuDeviceMatrix{T}, coeff_c::CuDeviceMatrix{T},
                     state_x::CuDeviceMatrix{T}, state_y::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    K = length(active_idx); if j > K; return; end
    i = active_idx[j]
    M = size(coeff_a, 1)
    sumx = zero(T)
    sumy = zero(T)
    @inbounds for k in 1:M
        nx = coeff_a[k, j] * state_x[k, j] + coeff_c[k, j] * randn(T)
        ny = coeff_a[k, j] * state_y[k, j] + coeff_c[k, j] * randn(T)
        state_x[k, j] = nx
        state_y[k, j] = ny
        sumx += nx
        sumy += ny
    end
    @inbounds begin
        ξx[i] = sumx
        ξy[i] = sumy
    end
    return nothing
end

function _apply_ou3!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
                     active_idx::CuDeviceVector{Int32},
                     coeff_a::CuDeviceMatrix{T}, coeff_c::CuDeviceMatrix{T},
                     state_x::CuDeviceMatrix{T}, state_y::CuDeviceMatrix{T}, state_z::CuDeviceMatrix{T}) where {T<:AbstractFloat}
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    K = length(active_idx); if j > K; return; end
    i = active_idx[j]
    M = size(coeff_a, 1)
    sumx = zero(T)
    sumy = zero(T)
    sumz = zero(T)
    @inbounds for k in 1:M
        nx = coeff_a[k, j] * state_x[k, j] + coeff_c[k, j] * randn(T)
        ny = coeff_a[k, j] * state_y[k, j] + coeff_c[k, j] * randn(T)
        nz = coeff_a[k, j] * state_z[k, j] + coeff_c[k, j] * randn(T)
        state_x[k, j] = nx
        state_y[k, j] = ny
        state_z[k, j] = nz
        sumx += nx
        sumy += ny
        sumz += nz
    end
    @inbounds begin
        ξx[i] = sumx
        ξy[i] = sumy
        ξz[i] = sumz
    end
    return nothing
end

function bd_prepare_noise_2d!(ξx::CuArray{T,1}, ξy::CuArray{T,1};
                              noise_scale::Union{Nothing,CuArray{T,1}}=nothing,
                              ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing,
                              state_x::Union{Nothing,CuArray{T,2}}=nothing,
                              state_y::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    @assert length(ξx) == length(ξy)
    N = length(ξx)
    threads = min(256, N)
    blocks = cld(N, threads)
    @assert noise_scale !== nothing
    k = CUDA.@cuda launch=false _noise2!(ξx, ξy, noise_scale)
    k(ξx, ξy, noise_scale; threads, blocks)
    if ou !== nothing
        @assert state_x !== nothing && state_y !== nothing
        K = length(ou.active_idx)
        K == 0 && return nothing
        ou_threads = min(256, K)
        ou_blocks = cld(K, ou_threads)
        k = CUDA.@cuda launch=false _apply_ou2!(ξx, ξy, ou.active_idx, ou.coeff_a, ou.coeff_c, state_x, state_y)
        k(ξx, ξy, ou.active_idx, ou.coeff_a, ou.coeff_c, state_x, state_y; threads=ou_threads, blocks=ou_blocks)
    end
    return nothing
end

function bd_prepare_noise_3d!(ξx::CuArray{T,1}, ξy::CuArray{T,1}, ξz::CuArray{T,1};
                              noise_scale::Union{Nothing,CuArray{T,1}}=nothing,
                              ou::Union{Nothing,Definitions.OUSpectrum{T}}=nothing,
                              state_x::Union{Nothing,CuArray{T,2}}=nothing,
                              state_y::Union{Nothing,CuArray{T,2}}=nothing,
                              state_z::Union{Nothing,CuArray{T,2}}=nothing) where {T<:AbstractFloat}
    @assert length(ξx) == length(ξy) == length(ξz)
    N = length(ξx)
    threads = min(256, N)
    blocks = cld(N, threads)
    @assert noise_scale !== nothing
    k = CUDA.@cuda launch=false _noise3!(ξx, ξy, ξz, noise_scale)
    k(ξx, ξy, ξz, noise_scale; threads, blocks)
    if ou !== nothing
        @assert state_x !== nothing && state_y !== nothing && state_z !== nothing
        K = length(ou.active_idx)
        K == 0 && return nothing
        ou_threads = min(256, K)
        ou_blocks = cld(K, ou_threads)
        k = CUDA.@cuda launch=false _apply_ou3!(ξx, ξy, ξz, ou.active_idx, ou.coeff_a, ou.coeff_c, state_x, state_y, state_z)
        k(ξx, ξy, ξz, ou.active_idx, ou.coeff_a, ou.coeff_c, state_x, state_y, state_z; threads=ou_threads, blocks=ou_blocks)
    end
    return nothing
end

function _mid2!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
        fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
        rxm::CuDeviceVector{T}, rym::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + μ * ξx[i])
        dy = half * (μ * fy[i] * dt + μ * ξy[i])
        x = rx[i] + dx
        y = ry[i] + dy
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
    end
    return nothing
end


function _mid3!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
        fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
        rxm::CuDeviceVector{T}, rym::CuDeviceVector{T}, rzm::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        Lx::T, Ly::T, Lz::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + μ * ξx[i])
        dy = half * (μ * fy[i] * dt + μ * ξy[i])
        dz = half * (μ * fz[i] * dt + μ * ξz[i])
        x = rx[i] + dx
        y = ry[i] + dy
        z = rz[i] + dz
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
        rzm[i] = _wrap_centered(z, Lz)
    end
    return nothing
end


function _fin2!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        # Propose new positions
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fxm[i] * dt + μ * ξx[i]
        local Δy = μ * fym[i] * dt + μ * ξy[i]
        local x = x0 + Δx
        local y = y0 + Δy
        # Wrap into box
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        # Use minimum-image displacement for work consistency under PBC
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        # Conservative work over step (Stratonovich): w = f_mid · Δr_mic
        # Heat to bath δq = + w; potential change δU = - w
        local w = fxm[i] * dx_mic + fym[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _fin2_unwrap!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
        rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fxm[i] * dt + μ * ξx[i]
        local Δy = μ * fym[i] * dt + μ * ξy[i]
        rxu[i] += Δx
        ryu[i] += Δy
        local x = x0 + Δx
        local y = y0 + Δy
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local w = fxm[i] * dx_mic + fym[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _fin3!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T}, fzm::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T, Lz::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fxm[i] * dt + μ * ξx[i]
        local Δy = μ * fym[i] * dt + μ * ξy[i]
        local Δz = μ * fzm[i] * dt + μ * ξz[i]
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fxm[i] * dx_mic + fym[i] * dy_mic + fzm[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _fin3_unwrap!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
        rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T}, rzu::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T}, fzm::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
        gamma::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T, Lz::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fxm[i] * dt + μ * ξx[i]
        local Δy = μ * fym[i] * dt + μ * ξy[i]
        local Δz = μ * fzm[i] * dt + μ * ξz[i]
        rxu[i] += Δx
        ryu[i] += Δy
        rzu[i] += Δz
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fxm[i] * dx_mic + fym[i] * dy_mic + fzm[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

"""
    bd_midpoint_positions_2d!(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, noise_scale, dt, box)

Generate midpoint trial positions (`rxm`, `rym`) by combining deterministic
forces and stochastic kicks. Matches the workflow in
`examples/2D_soft_repulsive_BD.jl`, which subsequently evaluates forces at the
midpoints before calling [`bd_finish_step_2d!`](@ref).
"""
function bd_midpoint_positions_2d!(rx, ry, fx, fy, ξx, ξy, rxm, rym,
                                   gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                                   dt::Real, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    k = CUDA.@cuda launch=false _mid2!(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, dtT, Lx, Ly)
    k(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, dtT, Lx, Ly; threads, blocks)
    return nothing
end

 

"""
3D variant of [`bd_midpoint_positions_2d!`](@ref).
"""
function bd_midpoint_positions_3d!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm,
                                   gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                                   dt::Real, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    Lz = convert(T, box[3])
    k = CUDA.@cuda launch=false _mid3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, gamma, dtT, Lx, Ly, Lz)
    k(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, gamma, dtT, Lx, Ly, Lz; threads, blocks)
    return nothing
end

 

"""
    bd_finish_step_2d!(rx, ry, fxm, fym, ξx, ξy, gamma, noise_scale, dt, dq, dU, box)

Finalize the midpoint Brownian step by combining midpoint forces (`fxm`, `fym`)
with noise to update positions and the heat/energy buffers.
"""
function bd_finish_step_2d!(rx, ry, fxm, fym, ξx, ξy,
                            gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                            dt::Real, dq::CuArray{T,1}, dU::CuArray{T,1}, box::Definitions.Box2{T};
                            unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                            unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    if unwrapped_x === nothing || unwrapped_y === nothing
        k = CUDA.@cuda launch=false _fin2!(rx, ry, fxm, fym, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly)
        k(rx, ry, fxm, fym, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly; threads, blocks)
    else
        k = CUDA.@cuda launch=false _fin2_unwrap!(rx, ry, unwrapped_x, unwrapped_y, fxm, fym, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly)
        k(rx, ry, unwrapped_x, unwrapped_y, fxm, fym, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly; threads, blocks)
    end
    return nothing
end

"""
3D finishing step (see [`bd_finish_step_2d!`](@ref)).
"""
function bd_finish_step_3d!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz,
                            gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                            dt::Real, dq::CuArray{T,1}, dU::CuArray{T,1}, box::Definitions.Box3{T};
                            unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                            unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing,
                            unwrapped_z::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    Lz = convert(T, box[3])
    if unwrapped_x === nothing || unwrapped_y === nothing || unwrapped_z === nothing
        k = CUDA.@cuda launch=false _fin3!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    else
        k = CUDA.@cuda launch=false _fin3_unwrap!(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
                                                  fxm, fym, fzm, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
          fxm, fym, fzm, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    end
    return nothing
end



# -----------------------------
# Euler–Maruyama (overdamped)
# -----------------------------
function _em2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
               fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
               gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
               dt::T,
               dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
               Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        ξx = randn(T); ξy = randn(T)
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        local Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        local x = x0 + Δx
        local y = y0 + Δy
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local w = fx[i] * dx_mic + fy[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em2_unwrap!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                      rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T},
                      fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                      gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
                      dt::T,
                      dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                      Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        ξx = randn(T); ξy = randn(T)
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        local Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        rxu[i] += Δx
        ryu[i] += Δy
        local x = x0 + Δx
        local y = y0 + Δy
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local w = fx[i] * dx_mic + fy[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
               fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
               gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
               dt::T,
               dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
               Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        ξx = randn(T); ξy = randn(T); ξz = randn(T)
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        local Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        local Δz = μ * fz[i] * dt + sqrt2Ddt * ξz
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fx[i] * dx_mic + fy[i] * dy_mic + fz[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em3_unwrap!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                      rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T}, rzu::CuDeviceVector{T},
                      fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                      gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
                      dt::T,
                      dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                      Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        ξx = randn(T); ξy = randn(T); ξz = randn(T)
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        local Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        local Δz = μ * fz[i] * dt + sqrt2Ddt * ξz
        rxu[i] += Δx
        ryu[i] += Δy
        rzu[i] += Δz
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fx[i] * dx_mic + fy[i] * dy_mic + fz[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em_apply2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                     fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                     ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
                     gamma::CuDeviceVector{T},
                     dt::T,
                     dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                     Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = one(T) / g
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fx[i] * dt + μ * ξx[i]
        local Δy = μ * fy[i] * dt + μ * ξy[i]
        local x = x0 + Δx
        local y = y0 + Δy
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local w = fx[i] * dx_mic + fy[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em_apply2_unwrap!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                            rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T},
                            fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                            ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
                            gamma::CuDeviceVector{T},
                            dt::T,
                            dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                            Lx::T, Ly::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = one(T) / g
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fx[i] * dt + μ * ξx[i]
        local Δy = μ * fy[i] * dt + μ * ξy[i]
        rxu[i] += Δx
        ryu[i] += Δy
        local x = x0 + Δx
        local y = y0 + Δy
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        rx[i] = xw
        ry[i] = yw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local w = fx[i] * dx_mic + fy[i] * dy_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em_apply3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                     fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                     ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
                     gamma::CuDeviceVector{T},
                     dt::T,
                     dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                     Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = one(T) / g
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fx[i] * dt + μ * ξx[i]
        local Δy = μ * fy[i] * dt + μ * ξy[i]
        local Δz = μ * fz[i] * dt + μ * ξz[i]
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fx[i] * dx_mic + fy[i] * dy_mic + fz[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

function _em_apply3_unwrap!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                            rxu::CuDeviceVector{T}, ryu::CuDeviceVector{T}, rzu::CuDeviceVector{T},
                            fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                            ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
                            gamma::CuDeviceVector{T},
                            dt::T,
                            dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
                            Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = one(T) / g
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fx[i] * dt + μ * ξx[i]
        local Δy = μ * fy[i] * dt + μ * ξy[i]
        local Δz = μ * fz[i] * dt + μ * ξz[i]
        rxu[i] += Δx
        ryu[i] += Δy
        rzu[i] += Δz
        local x = x0 + Δx
        local y = y0 + Δy
        local z = z0 + Δz
        local xw = _wrap_centered(x, Lx)
        local yw = _wrap_centered(y, Ly)
        local zw = _wrap_centered(z, Lz)
        rx[i] = xw
        ry[i] = yw
        rz[i] = zw
        local dx_mic = _wrap_centered(xw - x0, Lx)
        local dy_mic = _wrap_centered(yw - y0, Ly)
        local dz_mic = _wrap_centered(zw - z0, Lz)
        local w = fx[i] * dx_mic + fy[i] * dy_mic + fz[i] * dz_mic
        dq[i] = dq[i] + w
        dU[i] = dU[i] + w
    end
    return nothing
end

"""
    em_step_2d!(rx, ry, fx, fy, params, dt, dq, dU, box)

Euler–Maruyama update for overdamped dynamics using the per-particle friction
and noise stored in `params`. Used in `examples/3D_BD.jl` via
`eulermaruyama(st; gamma=..., temperature=..., dt=...)`.
"""
function em_step_2d!(rx::CuArray{T,1}, ry::CuArray{T,1},
                     fx::CuArray{T,1}, fy::CuArray{T,1},
                     params::EMParams{T}, dt::Real,
                     dq::CuArray{T,1}, dU::CuArray{T,1},
                     box::Definitions.Box2{T};
                     unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                     unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2])
    if unwrapped_x === nothing || unwrapped_y === nothing
        k = CUDA.@cuda launch=false _em2!(rx, ry, fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly)
        k(rx, ry, fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly; threads, blocks)
    else
        k = CUDA.@cuda launch=false _em2_unwrap!(rx, ry, unwrapped_x, unwrapped_y,
                                                 fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly)
        k(rx, ry, unwrapped_x, unwrapped_y,
          fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly; threads, blocks)
    end
    return nothing
end

"""
3D Euler–Maruyama step (see [`em_step_2d!`](@ref)).
"""
function em_step_3d!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                     fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                     params::EMParams{T}, dt::Real,
                     dq::CuArray{T,1}, dU::CuArray{T,1},
                     box::Definitions.Box3{T};
                     unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                     unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing,
                     unwrapped_z::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2]); Lz = convert(T, box[3])
    if unwrapped_x === nothing || unwrapped_y === nothing || unwrapped_z === nothing
        k = CUDA.@cuda launch=false _em3!(rx, ry, rz, fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    else
        k = CUDA.@cuda launch=false _em3_unwrap!(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
                                                 fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
          fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    end
    return nothing
end

"""
    em_apply_step_2d!(rx, ry, fx, fy, ξx, ξy, gamma, dt, dq, dU, box)

Apply one Euler-Maruyama position update using precomputed stochastic
increments `ξx`, `ξy`.
"""
function em_apply_step_2d!(rx::CuArray{T,1}, ry::CuArray{T,1},
                           fx::CuArray{T,1}, fy::CuArray{T,1},
                           ξx::CuArray{T,1}, ξy::CuArray{T,1},
                           gamma::CuArray{T,1}, dt::Real,
                           dq::CuArray{T,1}, dU::CuArray{T,1},
                           box::Definitions.Box2{T};
                           unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                           unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2])
    if unwrapped_x === nothing || unwrapped_y === nothing
        k = CUDA.@cuda launch=false _em_apply2!(rx, ry, fx, fy, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly)
        k(rx, ry, fx, fy, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly; threads, blocks)
    else
        k = CUDA.@cuda launch=false _em_apply2_unwrap!(rx, ry, unwrapped_x, unwrapped_y,
                                                       fx, fy, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly)
        k(rx, ry, unwrapped_x, unwrapped_y,
          fx, fy, ξx, ξy, gamma, dtT, dq, dU, Lx, Ly; threads, blocks)
    end
    return nothing
end

"""
3D variant of [`em_apply_step_2d!`](@ref).
"""
function em_apply_step_3d!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                           fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                           ξx::CuArray{T,1}, ξy::CuArray{T,1}, ξz::CuArray{T,1},
                           gamma::CuArray{T,1}, dt::Real,
                           dq::CuArray{T,1}, dU::CuArray{T,1},
                           box::Definitions.Box3{T};
                           unwrapped_x::Union{Nothing,CuArray{T,1}}=nothing,
                           unwrapped_y::Union{Nothing,CuArray{T,1}}=nothing,
                           unwrapped_z::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2]); Lz = convert(T, box[3])
    if unwrapped_x === nothing || unwrapped_y === nothing || unwrapped_z === nothing
        k = CUDA.@cuda launch=false _em_apply3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    else
        k = CUDA.@cuda launch=false _em_apply3_unwrap!(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
                                                       fx, fy, fz, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz)
        k(rx, ry, rz, unwrapped_x, unwrapped_y, unwrapped_z,
          fx, fy, fz, ξx, ξy, ξz, gamma, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    end
    return nothing
end
end # module BrownianIntegrators
