module BrownianIntegrators

using CUDA
using ..Definitions

export BrownianParams,
        bd_midpoint_positions_2d!, bd_midpoint_positions_3d!,
        bd_prepare_midpoint_2d!, bd_prepare_midpoint_3d!,
        bd_finish_step_2d!, bd_finish_step_3d!

struct BrownianParams{T}
    γ::T # friction
    kT::T # thermal energy (k_B T)
end

@inline _wrap_centered(x::Float32, L::Float32) = begin
    y = x + 0.5f0 * L
    y -= floor(y / L) * L
    y - 0.5f0 * L
end

# midpoint positions: r_mid = r + 0.5*(μ F dt + sqrt(2D dt) ξ)
function _mid2!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
        fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32},
        rxm::CuDeviceVector{Float32}, rym::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        Lx::Float32, Ly::Float32
        )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = 0.5f0 * (μ * fx[i] * dt + sqrt2Ddt * ξx[i])
        dy = 0.5f0 * (μ * fy[i] * dt + sqrt2Ddt * ξy[i])
        x = rx[i] + dx; y = ry[i] + dy
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
    end
    return nothing
end

# fused prepare: generate ξ and compute midpoint positions (store ξ for reuse)
function _prep2!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
        fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32},
        rxm::CuDeviceVector{Float32}, rym::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        Lx::Float32, Ly::Float32
        )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        # draw and store
        ξx_i = randn(Float32); ξy_i = randn(Float32)
        ξx[i] = ξx_i; ξy[i] = ξy_i
        # midpoint
        dx = 0.5f0 * (μ * fx[i] * dt + sqrt2Ddt * ξx_i)
        dy = 0.5f0 * (μ * fy[i] * dt + sqrt2Ddt * ξy_i)
        x = rx[i] + dx; y = ry[i] + dy
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
    end
    return nothing
end

function _mid3!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
        fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32}, ξz::CuDeviceVector{Float32},
        rxm::CuDeviceVector{Float32}, rym::CuDeviceVector{Float32}, rzm::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        Lx::Float32, Ly::Float32, Lz::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        dx = 0.5f0 * (μ * fx[i] * dt + sqrt2Ddt * ξx[i])
        dy = 0.5f0 * (μ * fy[i] * dt + sqrt2Ddt * ξy[i])
        dz = 0.5f0 * (μ * fz[i] * dt + sqrt2Ddt * ξz[i])
        x = rx[i] + dx; y = ry[i] + dy; z = rz[i] + dz
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
        rzm[i] = _wrap_centered(z, Lz)
    end
    return nothing
end

# fused prepare 3D
function _prep3!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
        fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32}, ξz::CuDeviceVector{Float32},
        rxm::CuDeviceVector{Float32}, rym::CuDeviceVector{Float32}, rzm::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        Lx::Float32, Ly::Float32, Lz::Float32
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        ξx_i = randn(Float32); ξy_i = randn(Float32); ξz_i = randn(Float32)
        ξx[i] = ξx_i; ξy[i] = ξy_i; ξz[i] = ξz_i
        dx = 0.5f0 * (μ * fx[i] * dt + sqrt2Ddt * ξx_i)
        dy = 0.5f0 * (μ * fy[i] * dt + sqrt2Ddt * ξy_i)
        dz = 0.5f0 * (μ * fz[i] * dt + sqrt2Ddt * ξz_i)
        x = rx[i] + dx; y = ry[i] + dy; z = rz[i] + dz
        rxm[i] = _wrap_centered(x, Lx)
        rym[i] = _wrap_centered(y, Ly)
        rzm[i] = _wrap_centered(z, Lz)
    end
    return nothing
end

# finish: Δr = μ F_mid dt + sqrt(2D dt) ξ; r += Δr; dq += F_mid ⋅ Δr
function _fin2!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
        fxm::CuDeviceVector{Float32}, fym::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        dq::CuDeviceVector{Float32},
        Lx::Float32, Ly::Float32
        )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        x = rx[i] + Δx; y = ry[i] + Δy
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        dq[i] = dq[i] + (fxm[i]*Δx + fym[i]*Δy)
    end
    return nothing
end

function _fin3!(
        rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
        fxm::CuDeviceVector{Float32}, fym::CuDeviceVector{Float32}, fzm::CuDeviceVector{Float32},
        ξx::CuDeviceVector{Float32}, ξy::CuDeviceVector{Float32}, ξz::CuDeviceVector{Float32},
        μ::Float32, sqrt2Ddt::Float32, dt::Float32,
        dq::CuDeviceVector{Float32},
        Lx::Float32, Ly::Float32, Lz::Float32
        )
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        Δx = μ * fxm[i] * dt + sqrt2Ddt * ξx[i]
        Δy = μ * fym[i] * dt + sqrt2Ddt * ξy[i]
        Δz = μ * fzm[i] * dt + sqrt2Ddt * ξz[i]
        x = rx[i] + Δx; y = ry[i] + Δy; z = rz[i] + Δz
        rx[i] = _wrap_centered(x, Lx)
        ry[i] = _wrap_centered(y, Ly)
        rz[i] = _wrap_centered(z, Lz)
        dq[i] = dq[i] + (fxm[i]*Δx + fym[i]*Δy + fzm[i]*Δz)
    end
    return nothing
end

function bd_midpoint_positions_2d!(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ::Float32, sqrt2Ddt::Float32, dt::Float32, box::Definitions.Box2)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _mid2!(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ, sqrt2Ddt, dt, box[1], box[2])
    k(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ, sqrt2Ddt, dt, box[1], box[2]; threads=t, blocks=b)
    return nothing
end

function bd_prepare_midpoint_2d!(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ::Float32, sqrt2Ddt::Float32, dt::Float32, box::Definitions.Box2)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _prep2!(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ, sqrt2Ddt, dt, box[1], box[2])
    k(rx, ry, fx, fy, ξx, ξy, rxm, rym, μ, sqrt2Ddt, dt, box[1], box[2]; threads=t, blocks=b)
    return nothing
end

function bd_midpoint_positions_3d!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ::Float32, sqrt2Ddt::Float32, dt::Float32, box::Definitions.Box3)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _mid3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ, sqrt2Ddt, dt, box[1], box[2], box[3])
    k(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ, sqrt2Ddt, dt, box[1], box[2], box[3]; threads=t, blocks=b)
    return nothing
end

function bd_prepare_midpoint_3d!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ::Float32, sqrt2Ddt::Float32, dt::Float32, box::Definitions.Box3)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _prep3!(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ, sqrt2Ddt, dt, box[1], box[2], box[3])
    k(rx, ry, rz, fx, fy, fz, ξx, ξy, ξz, rxm, rym, rzm, μ, sqrt2Ddt, dt, box[1], box[2], box[3]; threads=t, blocks=b)
    return nothing
end

function bd_finish_step_2d!(rx, ry, fxm, fym, ξx, ξy, μ::Float32, sqrt2Ddt::Float32, dt::Float32, dq, box::Definitions.Box2)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _fin2!(rx, ry, fxm, fym, ξx, ξy, μ, sqrt2Ddt, dt, dq, box[1], box[2])
    k(rx, ry, fxm, fym, ξx, ξy, μ, sqrt2Ddt, dt, dq, box[1], box[2]; threads=t, blocks=b)
    return nothing
end

function bd_finish_step_3d!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, μ::Float32, sqrt2Ddt::Float32, dt::Float32, dq, box::Definitions.Box3)
    N = length(rx); t = min(256, N); b = cld(N, t)
    k = CUDA.@cuda launch=false _fin3!(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, μ, sqrt2Ddt, dt, dq, box[1], box[2], box[3])
    k(rx, ry, rz, fxm, fym, fzm, ξx, ξy, ξz, μ, sqrt2Ddt, dt, dq, box[1], box[2], box[3]; threads=t, blocks=b)
    return nothing
end

end # module
