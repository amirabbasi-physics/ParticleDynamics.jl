module BrownianIntegrators

using CUDA
using ..Definitions

export BrownianParams,
       bd_midpoint_positions_2d!, bd_midpoint_positions_3d!,
       bd_prepare_noise_2d!, bd_prepare_noise_3d!,
       bd_finish_step_2d!, bd_finish_step_3d!,
       EMParams, em_step_2d!, em_step_3d!

struct BrownianParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    function BrownianParams{T}(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}, corr_time::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, noise_scale, corr_time)
    end
end

BrownianParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, noise_scale, nothing)
BrownianParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}, corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, noise_scale, corr_time)

function BrownianParams{T}(gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    γ = T(gamma)
    Tval = T(temperature)
    Δt = T(dt)
    gamma_vec = CUDA.fill(γ, N)
    scale = sqrt(T(2) * γ * Tval * Δt)
    noise_vec = CUDA.fill(scale, N)
    return BrownianParams{T}(gamma_vec, noise_vec, nothing)
end

function BrownianParams(::Type{T}, gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    return BrownianParams{T}(gamma, temperature, dt, N)
end

# Euler–Maruyama parameters (reuse gamma/noise scale like BrownianParams)
struct EMParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
    function EMParams{T}(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}, corr_time::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        corr_time !== nothing && @assert length(corr_time) == length(gamma)
        new{T}(gamma, noise_scale, corr_time)
    end
end

EMParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} = EMParams{T}(gamma, noise_scale, nothing)
EMParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}, corr_time::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat} = EMParams{T}(gamma, noise_scale, corr_time)

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
function _noise2!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        ξx[i] = randn(T)
        ξy[i] = randn(T)
    end
    return
end

function _noise3!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        ξx[i] = randn(T)
        ξy[i] = randn(T)
        ξz[i] = randn(T)
    end
    return
end

function _noise2_ou!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
                     noise_scale::CuDeviceVector{T}, corr_time::CuDeviceVector{T},
                     state_x::CuDeviceVector{T}, state_y::CuDeviceVector{T},
                     dt::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        τ = corr_time[i]
        if τ <= zero(T)
            valx = randn(T); valy = randn(T)
            ξx[i] = valx; ξy[i] = valy
            state_x[i] = valx; state_y[i] = valy
        else
            a = exp(-dt / τ)
            b = sqrt(max(one(T) - a*a, zero(T)))
            nx = a * state_x[i] + b * randn(T)
            ny = a * state_y[i] + b * randn(T)
            ξx[i] = nx; ξy[i] = ny
            state_x[i] = nx; state_y[i] = ny
        end
    end
    return
end

function _noise3_ou!(ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
                     noise_scale::CuDeviceVector{T}, corr_time::CuDeviceVector{T},
                     state_x::CuDeviceVector{T}, state_y::CuDeviceVector{T}, state_z::CuDeviceVector{T},
                     dt::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(ξx); if i > N; return; end
    @inbounds begin
        τ = corr_time[i]
        if τ <= zero(T)
            valx = randn(T); valy = randn(T); valz = randn(T)
            ξx[i] = valx; ξy[i] = valy; ξz[i] = valz
            state_x[i] = valx; state_y[i] = valy; state_z[i] = valz
        else
            a = exp(-dt / τ)
            b = sqrt(max(one(T) - a*a, zero(T)))
            nx = a * state_x[i] + b * randn(T)
            ny = a * state_y[i] + b * randn(T)
            nz = a * state_z[i] + b * randn(T)
            ξx[i] = nx; ξy[i] = ny; ξz[i] = nz
            state_x[i] = nx; state_y[i] = ny; state_z[i] = nz
        end
    end
    return
end

function bd_prepare_noise_2d!(ξx::CuArray{T,1}, ξy::CuArray{T,1};
                              noise_scale::Union{Nothing,CuArray{T,1}}=nothing,
                              corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                              state_x::Union{Nothing,CuArray{T,1}}=nothing,
                              state_y::Union{Nothing,CuArray{T,1}}=nothing,
                              dt::Union{Nothing,T}=nothing) where {T<:AbstractFloat}
    @assert length(ξx) == length(ξy)
    N = length(ξx)
    threads = min(256, N)
    blocks = cld(N, threads)
    if corr_time === nothing
        k = CUDA.@cuda launch=false _noise2!(ξx, ξy)
        k(ξx, ξy; threads, blocks)
    else
        @assert noise_scale !== nothing && state_x !== nothing && state_y !== nothing
        @assert dt !== nothing "dt required for correlated noise"
        k = CUDA.@cuda launch=false _noise2_ou!(ξx, ξy, noise_scale, corr_time, state_x, state_y, dt::T)
        k(ξx, ξy, noise_scale, corr_time, state_x, state_y, dt::T; threads, blocks)
    end
    return nothing
end

function bd_prepare_noise_3d!(ξx::CuArray{T,1}, ξy::CuArray{T,1}, ξz::CuArray{T,1};
                              noise_scale::Union{Nothing,CuArray{T,1}}=nothing,
                              corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                              state_x::Union{Nothing,CuArray{T,1}}=nothing,
                              state_y::Union{Nothing,CuArray{T,1}}=nothing,
                              state_z::Union{Nothing,CuArray{T,1}}=nothing,
                              dt::Union{Nothing,T}=nothing) where {T<:AbstractFloat}
    @assert length(ξx) == length(ξy) == length(ξz)
    N = length(ξx)
    threads = min(256, N)
    blocks = cld(N, threads)
    if corr_time === nothing
        k = CUDA.@cuda launch=false _noise3!(ξx, ξy, ξz)
        k(ξx, ξy, ξz; threads, blocks)
    else
        @assert noise_scale !== nothing && state_x !== nothing && state_y !== nothing && state_z !== nothing
        @assert dt !== nothing "dt required for correlated noise"
        k = CUDA.@cuda launch=false _noise3_ou!(ξx, ξy, ξz, noise_scale, corr_time, state_x, state_y, state_z, dt::T)
        k(ξx, ξy, ξz, noise_scale, corr_time, state_x, state_y, state_z, dt::T; threads, blocks)
    end
    return nothing
end

function _mid2!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
        fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T},
        rxm::CuDeviceVector{T}, rym::CuDeviceVector{T},
        gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
        dt::T,
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + sqrt2Ddt * ξx[i])
        dy = half * (μ * fy[i] * dt + sqrt2Ddt * ξy[i])
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
        gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
        dt::T,
        Lx::T, Ly::T, Lz::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + sqrt2Ddt * ξx[i])
        dy = half * (μ * fy[i] * dt + sqrt2Ddt * ξy[i])
        dz = half * (μ * fz[i] * dt + sqrt2Ddt * ξz[i])
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
        gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        # Propose new positions
        local x0 = rx[i]; local y0 = ry[i]
        local Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        local Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
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

function _fin3!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T}, fzm::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
        gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T}, dU::CuDeviceVector{T},
        Lx::T, Ly::T, Lz::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = 1 / g
        sqrt2Ddt = μ * noise_scale[i]
        local x0 = rx[i]; local y0 = ry[i]; local z0 = rz[i]
        local Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        local Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        local Δz = μ * fzm[i] * dt + sqrt2Ddt * ξz[i]
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
    k = CUDA.@cuda launch=false _mid2!(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, noise_scale, dtT, Lx, Ly)
    k(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, noise_scale, dtT, Lx, Ly; threads, blocks)
    return nothing
end

 

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
    k = CUDA.@cuda launch=false _mid3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, gamma, noise_scale, dtT, Lx, Ly, Lz)
    k(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, gamma, noise_scale, dtT, Lx, Ly, Lz; threads, blocks)
    return nothing
end

 

function bd_finish_step_2d!(rx, ry, fxm, fym, ξx, ξy,
                            gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                            dt::Real, dq::CuArray{T,1}, dU::CuArray{T,1}, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    k = CUDA.@cuda launch=false _fin2!(rx, ry, fxm, fym, ξx, ξy, gamma, noise_scale, dtT, dq, dU, Lx, Ly)
    k(rx, ry, fxm, fym, ξx, ξy, gamma, noise_scale, dtT, dq, dU, Lx, Ly; threads, blocks)
    return nothing
end

function bd_finish_step_3d!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz,
                            gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                            dt::Real, dq::CuArray{T,1}, dU::CuArray{T,1}, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    Lz = convert(T, box[3])
    k = CUDA.@cuda launch=false _fin3!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, noise_scale, dtT, dq, dU, Lx, Ly, Lz)
    k(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, noise_scale, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
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

function em_step_2d!(rx::CuArray{T,1}, ry::CuArray{T,1},
                     fx::CuArray{T,1}, fy::CuArray{T,1},
                     params::EMParams{T}, dt::Real,
                     dq::CuArray{T,1}, dU::CuArray{T,1},
                     box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2])
    k = CUDA.@cuda launch=false _em2!(rx, ry, fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly)
    k(rx, ry, fx, fy, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly; threads, blocks)
    return nothing
end

function em_step_3d!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                     fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                     params::EMParams{T}, dt::Real,
                     dq::CuArray{T,1}, dU::CuArray{T,1},
                     box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1]); Ly = convert(T, box[2]); Lz = convert(T, box[3])
    k = CUDA.@cuda launch=false _em3!(rx, ry, rz, fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz)
    k(rx, ry, rz, fx, fy, fz, params.gamma, params.noise_scale, dtT, dq, dU, Lx, Ly, Lz; threads, blocks)
    return nothing
end
end # module BrownianIntegrators
