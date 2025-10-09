module BrownianIntegrators

using CUDA
using ..Definitions

export BrownianParams,
       bd_midpoint_positions_2d!, bd_midpoint_positions_3d!,
       bd_prepare_midpoint_2d!, bd_prepare_midpoint_3d!,
       bd_finish_step_2d!, bd_finish_step_3d!

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

@inline _maybe_inv(g::T) where {T<:AbstractFloat} = g == zero(T) ? zero(T) : inv(g)
@inline _maybe_scale(noise::T, g::T) where {T<:AbstractFloat} = g == zero(T) ? zero(T) : noise / g

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
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
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
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
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
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
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
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
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
        dq::CuDeviceVector{T},
        Lx::T, Ly::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        x = rx[i] + Δx
        y = ry[i] + Δy
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        dq[i] = dq[i] + (fxm[i] * Δx + fym[i] * Δy)
    end
    return nothing
end

function _fin3!(
        rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
        fxm::CuDeviceVector{T}, fym::CuDeviceVector{T}, fzm::CuDeviceVector{T},
        ξx::CuDeviceVector{T}, ξy::CuDeviceVector{T}, ξz::CuDeviceVector{T},
        gamma::CuDeviceVector{T}, noise_scale::CuDeviceVector{T},
        dt::T,
        dq::CuDeviceVector{T},
        Lx::T, Ly::T, Lz::T
        ) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        μ = _maybe_inv(g)
        sqrt2Ddt = _maybe_scale(noise_scale[i], g)
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        Δz = μ * fzm[i] * dt + sqrt2Ddt * ξz[i]
        x = rx[i] + Δx
        y = ry[i] + Δy
        z = rz[i] + Δz
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        rz[i] = _wrap_centered(z, Lz)
        dq[i] = dq[i] + (fxm[i] * Δx + fym[i] * Δy + fzm[i] * Δz)
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
                            dt::Real, dq::CuArray{T,1}, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    k = CUDA.@cuda launch=false _fin2!(rx, ry, fxm, fym, ξx, ξy, gamma, noise_scale, dtT, dq, Lx, Ly)
    k(rx, ry, fxm, fym, ξx, ξy, gamma, noise_scale, dtT, dq, Lx, Ly; threads, blocks)
    return nothing
end

function bd_finish_step_3d!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz,
                            gamma::CuArray{T,1}, noise_scale::CuArray{T,1},
                            dt::Real, dq::CuArray{T,1}, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx)
    @assert length(gamma) == N == length(noise_scale)
    threads = min(256, N)
    blocks = cld(N, threads)
    dtT = convert(T, dt)
    Lx = convert(T, box[1])
    Ly = convert(T, box[2])
    Lz = convert(T, box[3])
    k = CUDA.@cuda launch=false _fin3!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, noise_scale, dtT, dq, Lx, Ly, Lz)
    k(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, gamma, noise_scale, dtT, dq, Lx, Ly, Lz; threads, blocks)
    return nothing
end

end # module BrownianIntegrators
