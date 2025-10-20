module BrownianIntegrators

using CUDA
using ..Definitions

export BrownianParams,
       bd_midpoint_positions_2d!, bd_midpoint_positions_3d!,
       bd_prepare_midpoint_2d!, bd_prepare_midpoint_3d!,
       bd_finish_step_2d!, bd_finish_step_3d!,
       EMParams, em_step_2d!, em_step_3d!

struct BrownianParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    noise_scale::CuArray{T,1}
    function BrownianParams{T}(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        new{T}(gamma, noise_scale)
    end
end

BrownianParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} =
    BrownianParams{T}(gamma, noise_scale)

function BrownianParams{T}(gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    γ = T(gamma)
    Tval = T(temperature)
    Δt = T(dt)
    gamma_vec = CUDA.fill(γ, N)
    scale = sqrt(T(2) * γ * Tval * Δt)
    noise_vec = CUDA.fill(scale, N)
    return BrownianParams{T}(gamma_vec, noise_vec)
end

function BrownianParams(::Type{T}, gamma::Real, temperature::Real, dt::Real, N::Integer) where {T<:AbstractFloat}
    return BrownianParams{T}(gamma, temperature, dt, N)
end

# Euler–Maruyama parameters (reuse gamma/noise scale like BrownianParams)
struct EMParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    noise_scale::CuArray{T,1}
    function EMParams{T}(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat}
        @assert length(gamma) == length(noise_scale)
        new{T}(gamma, noise_scale)
    end
end

EMParams(gamma::CuArray{T,1}, noise_scale::CuArray{T,1}) where {T<:AbstractFloat} = EMParams{T}(gamma, noise_scale)

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

function _prep2!(
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
        ξx_i = randn(T)
        ξy_i = randn(T)
        ξx[i] = ξx_i
        ξy[i] = ξy_i
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + sqrt2Ddt * ξx_i)
        dy = half * (μ * fy[i] * dt + sqrt2Ddt * ξy_i)
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

function _prep3!(
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
        ξx_i = randn(T)
        ξy_i = randn(T)
        ξz_i = randn(T)
        ξx[i] = ξx_i
        ξy[i] = ξy_i
        ξz[i] = ξz_i
        half = T(0.5)
        dx = half * (μ * fx[i] * dt + sqrt2Ddt * ξx_i)
        dy = half * (μ * fy[i] * dt + sqrt2Ddt * ξy_i)
        dz = half * (μ * fz[i] * dt + sqrt2Ddt * ξz_i)
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
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        x = rx[i] + Δx
        y = ry[i] + Δy
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        # Conservative work over step (power integrated): use midpoint force
        local w = fxm[i] * Δx + fym[i] * Δy
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
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        Δz = μ * fzm[i] * dt + sqrt2Ddt * ξz[i]
        x = rx[i] + Δx
        y = ry[i] + Δy
        z = rz[i] + Δz
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        rz[i] = _wrap_centered(z, Lz)
        local w = fxm[i] * Δx + fym[i] * Δy + fzm[i] * Δz
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

function bd_prepare_midpoint_2d!(rx, ry, fx, fy, ξx, ξy, rxm, rym,
                                 gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                                 dt::Real, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    k = CUDA.@cuda launch=false _prep2!(rx, ry, fx, fy, ξx, ξy, rxm, rym, gamma, noise_scale, dtT, Lx, Ly)
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

function bd_prepare_midpoint_3d!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm,
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
    k = CUDA.@cuda launch=false _prep3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, gamma, noise_scale, dtT, Lx, Ly, Lz)
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
        Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        x = rx[i] + Δx
        y = ry[i] + Δy
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        local w = fx[i] * Δx + fy[i] * Δy
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
        Δx = μ * fx[i] * dt + sqrt2Ddt * ξx
        Δy = μ * fy[i] * dt + sqrt2Ddt * ξy
        Δz = μ * fz[i] * dt + sqrt2Ddt * ξz
        x = rx[i] + Δx
        y = ry[i] + Δy
        z = rz[i] + Δz
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        rz[i] = _wrap_centered(z, Lz)
        local w = fx[i] * Δx + fy[i] * Δy + fz[i] * Δz
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
