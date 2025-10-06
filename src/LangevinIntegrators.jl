module LangevinIntegrators

using CUDA
using ..Definitions

export VVParams,
       vv_prepare_noise!,
       vv_positions_soa!,
       vv_velocities_soa!,
       BAOABParams,
       baoab_BA_2d!, baoab_OU_2d!, baoab_A_2d!, baoab_B_2d!,
       baoab_BA_3d!, baoab_OU_3d!, baoab_A_3d!, baoab_B_3d!

struct VVParams{T}
    gamma::T
    mass::T
    noise_scale::CuArray{Float32,1}
end

# =============================
#   BAOAB parameters
# =============================
struct BAOABParams{T}
    gamma::T
    mass::T
    noise_scale::CuArray{Float32,1}
end

# Noise preparation
function _vv_noise2_kernel!(beta_x::CuDeviceVector{Float32},
                            beta_y::CuDeviceVector{Float32},
                            noise_scale::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        beta_x[i] = s * randn(Float32)
        beta_y[i] = s * randn(Float32)
    end
    return
end

function _vv_noise3_kernel!(beta_x::CuDeviceVector{Float32},
                            beta_y::CuDeviceVector{Float32},
                            beta_z::CuDeviceVector{Float32},
                            noise_scale::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        beta_x[i] = s * randn(Float32)
        beta_y[i] = s * randn(Float32)
        beta_z[i] = s * randn(Float32)
    end
    return
end

function vv_prepare_noise!(beta_x::CuArray{Float32,1},
                           beta_y::CuArray{Float32,1},
                           noise_scale::CuArray{Float32,1};
                           beta_z::Union{Nothing,CuArray{Float32,1}}=nothing)
    @assert length(beta_x) == length(noise_scale) == length(beta_y)
    N = length(beta_x)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    if beta_z === nothing
        k = CUDA.@cuda launch=false _vv_noise2_kernel!(beta_x, beta_y, noise_scale)
        k(beta_x, beta_y, noise_scale; threads, blocks)
    else
        @assert length(beta_z) == length(noise_scale)
        k = CUDA.@cuda launch=false _vv_noise3_kernel!(beta_x, beta_y, beta_z, noise_scale)
        k(beta_x, beta_y, beta_z, noise_scale; threads, blocks)
    end
    return nothing
end

# Positions
function _vv_pos2!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
                   vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
                   fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
                   beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32},
                   gamma::Float32, mass::Float32, dt::Float32,
                   Lx::Float32, Ly::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    q = gamma * dt / (2f0 * mass)
    b = 1f0 / (1f0 + q)
    @inbounds begin
        dpx = b*dt*vx[i] + (b*dt/(2f0*mass))*(dt*fx[i] + beta_x[i])
        dpy = b*dt*vy[i] + (b*dt/(2f0*mass))*(dt*fy[i] + beta_y[i])
        x = rx[i] + dpx; y = ry[i] + dpy
        x = (x + Lx/2); x -= floor(x/Lx)*Lx; x -= Lx/2
        y = (y + Ly/2); y -= floor(y/Ly)*Ly; y -= Ly/2
        rx[i] = x; ry[i] = y
    end
    return
end

function _vv_pos3!(rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
                   vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
                   fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
                   beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32}, beta_z::CuDeviceVector{Float32},
                   gamma::Float32, mass::Float32, dt::Float32,
                   Lx::Float32, Ly::Float32, Lz::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    q = gamma * dt / (2f0 * mass)
    b = 1f0 / (1f0 + q)
    @inbounds begin
        dpx = b*dt*vx[i] + (b*dt/(2f0*mass))*(dt*fx[i] + beta_x[i])
        dpy = b*dt*vy[i] + (b*dt/(2f0*mass))*(dt*fy[i] + beta_y[i])
        dpz = b*dt*vz[i] + (b*dt/(2f0*mass))*(dt*fz[i] + beta_z[i])
        x = rx[i] + dpx; y = ry[i] + dpy; z = rz[i] + dpz
        x = (x + Lx/2); x -= floor(x/Lx)*Lx; x -= Lx/2
        y = (y + Ly/2); y -= floor(y/Ly)*Ly; y -= Ly/2
        z = (z + Lz/2); z -= floor(z/Lz)*Lz; z -= Lz/2
        rx[i] = x; ry[i] = y; rz[i] = z
    end
    return
end

# Velocities
function _vv_vel2!(vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
                   f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32},
                   fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
                   beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32},
                   dq::CuDeviceVector{Float32}, Ekin::CuDeviceVector{Float32},
                   noise_scale::CuDeviceVector{Float32},
                   gamma::Float32, mass::Float32, dt::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    q = gamma * dt / (2f0 * mass)
    a = (1f0 - q) / (1f0 + q)
    b = 1f0 / (1f0 + q)
    vpx = vx[i]; vpy = vy[i]
    vnx = a*vpx + (dt/(2f0*mass))*(a*f0x[i] + fx[i]) + (b/mass)*beta_x[i]
    vny = a*vpy + (dt/(2f0*mass))*(a*f0y[i] + fy[i]) + (b/mass)*beta_y[i]
    v2 = vnx*vnx + vny*vny
    Ekin[i] = 0.5f0 * mass * v2
    s = noise_scale[i]
    Tbath = (s*s) / (2f0*gamma*dt)
    dq[i] = dq[i] + gamma * (v2 - 2f0 * Tbath / mass) * dt
    vx[i] = vnx; vy[i] = vny
    return
end

# =============================
#   BAOAB kernels (B/A and OU)
# =============================

function _baoab_BA2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32},
    mass::Float32, dt::Float32,
    Lx::Float32, Ly::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(2f0*mass)) * f0x[i]
        vyi = vy[i] + (dt/(2f0*mass)) * f0y[i]
        xi = rx[i] + 0.5f0*dt*vxi
        yi = ry[i] + 0.5f0*dt*vyi
        xi = (xi + Lx/2); xi -= floor(xi/Lx)*Lx; xi -= Lx/2
        yi = (yi + Ly/2); yi -= floor(yi/Ly)*Ly; yi -= Ly/2
        vx[i] = vxi; vy[i] = vyi
        rx[i] = xi;  ry[i] = yi
    end
    return
end

function _baoab_BA3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32}, f0z::CuDeviceVector{Float32},
    mass::Float32, dt::Float32,
    Lx::Float32, Ly::Float32, Lz::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(2f0*mass)) * f0x[i]
        vyi = vy[i] + (dt/(2f0*mass)) * f0y[i]
        vzi = vz[i] + (dt/(2f0*mass)) * f0z[i]
        xi = rx[i] + 0.5f0*dt*vxi
        yi = ry[i] + 0.5f0*dt*vyi
        zi = rz[i] + 0.5f0*dt*vzi
        xi = (xi + Lx/2); xi -= floor(xi/Lx)*Lx; xi -= Lx/2
        yi = (yi + Ly/2); yi -= floor(yi/Ly)*Ly; yi -= Ly/2
        zi = (zi + Lz/2); zi -= floor(zi/Lz)*Lz; zi -= Lz/2
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi
        rx[i] = xi;  ry[i] = yi;  rz[i] = zi
    end
    return
end

function _baoab_OU2!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    noise_scale::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32,
    dq::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        c = exp(-gamma*dt/mass)
        s = noise_scale[i]
        Tbath = (s*s) / (2f0*gamma*dt)
        sigma = sqrt((Tbath/mass) * (1f0 - c*c))
        vx[i] = c*vx[i] + sigma*randn(Float32)
        vy[i] = c*vy[i] + sigma*randn(Float32)
        # dq unchanged here (can be added later)
    end
    return
end

function _baoab_OU3!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    noise_scale::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32,
    dq::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        c = exp(-gamma*dt/mass)
        s = noise_scale[i]
        Tbath = (s*s) / (2f0*gamma*dt)
        sigma = sqrt((Tbath/mass) * (1f0 - c*c))
        vx[i] = c*vx[i] + sigma*randn(Float32)
        vy[i] = c*vy[i] + sigma*randn(Float32)
        vz[i] = c*vz[i] + sigma*randn(Float32)
    end
    return
end

function _baoab_A2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    dt::Float32, Lx::Float32, Ly::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        xi = rx[i] + 0.5f0*dt*vx[i]
        yi = ry[i] + 0.5f0*dt*vy[i]
        xi = (xi + Lx/2); xi -= floor(xi/Lx)*Lx; xi -= Lx/2
        yi = (yi + Ly/2); yi -= floor(yi/Ly)*Ly; yi -= Ly/2
        rx[i] = xi; ry[i] = yi
    end
    return
end

function _baoab_A3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    dt::Float32, Lx::Float32, Ly::Float32, Lz::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        xi = rx[i] + 0.5f0*dt*vx[i]
        yi = ry[i] + 0.5f0*dt*vy[i]
        zi = rz[i] + 0.5f0*dt*vz[i]
        xi = (xi + Lx/2); xi -= floor(xi/Lx)*Lx; xi -= Lx/2
        yi = (yi + Ly/2); yi -= floor(yi/Ly)*Ly; yi -= Ly/2
        zi = (zi + Lz/2); zi -= floor(zi/Lz)*Lz; zi -= Lz/2
        rx[i] = xi; ry[i] = yi; rz[i] = zi
    end
    return
end

function _baoab_B2!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    mass::Float32, dt::Float32,
    Ekin::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(2f0*mass)) * fx[i]
        vyi = vy[i] + (dt/(2f0*mass)) * fy[i]
        vx[i] = vxi; vy[i] = vyi
        v2 = vxi*vxi + vyi*vyi
        Ekin[i] = 0.5f0 * mass * v2
    end
    return
end

function _baoab_B3!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    mass::Float32, dt::Float32,
    Ekin::CuDeviceVector{Float32})
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(2f0*mass)) * fx[i]
        vyi = vy[i] + (dt/(2f0*mass)) * fy[i]
        vzi = vz[i] + (dt/(2f0*mass)) * fz[i]
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi
        v2 = vxi*vxi + vyi*vyi + vzi*vzi
        Ekin[i] = 0.5f0 * mass * v2
    end
    return
end

# Host launchers
function baoab_BA_2d!(rx, ry, vx, vy, f0x, f0y, params::BAOABParams{Float32}, dt::Float32, box::Definitions.Box2)
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA2!(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2])
    k(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function baoab_BA_3d!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params::BAOABParams{Float32}, dt::Float32, box::Definitions.Box3)
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA3!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

function baoab_OU_2d!(vx, vy, params::BAOABParams{Float32}, dt::Float32, dq)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU2!(vx, vy, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

function baoab_OU_3d!(vx, vy, vz, params::BAOABParams{Float32}, dt::Float32, dq)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU3!(vx, vy, vz, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, vz, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

function baoab_A_2d!(rx, ry, vx, vy, dt::Float32, box::Definitions.Box2)
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A2!(rx, ry, vx, vy, dt, box[1], box[2])
    k(rx, ry, vx, vy, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function baoab_A_3d!(rx, ry, rz, vx, vy, vz, dt::Float32, box::Definitions.Box3)
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A3!(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

function baoab_B_2d!(vx, vy, fx, fy, params::BAOABParams{Float32}, dt::Float32, Ekin)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B2!(vx, vy, fx, fy, params.mass, dt, Ekin)
    k(vx, vy, fx, fy, params.mass, dt, Ekin; threads, blocks)
    return nothing
end

function baoab_B_3d!(vx, vy, vz, fx, fy, fz, params::BAOABParams{Float32}, dt::Float32, Ekin)
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B3!(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin)
    k(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin; threads, blocks)
    return nothing
end

function _vv_vel3!(vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
                   f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32}, f0z::CuDeviceVector{Float32},
                   fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
                   beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32}, beta_z::CuDeviceVector{Float32},
                   dq::CuDeviceVector{Float32}, Ekin::CuDeviceVector{Float32},
                   noise_scale::CuDeviceVector{Float32},
                   gamma::Float32, mass::Float32, dt::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    q = gamma * dt / (2f0 * mass)
    a = (1f0 - q) / (1f0 + q)
    b = 1f0 / (1f0 + q)
    vpx = vx[i]; vpy = vy[i]; vpz = vz[i]
    vnx = a*vpx + (dt/(2f0*mass))*(a*f0x[i] + fx[i]) + (b/mass)*beta_x[i]
    vny = a*vpy + (dt/(2f0*mass))*(a*f0y[i] + fy[i]) + (b/mass)*beta_y[i]
    vnz = a*vpz + (dt/(2f0*mass))*(a*f0z[i] + fz[i]) + (b/mass)*beta_z[i]
    v2 = vnx*vnx + vny*vny + vnz*vnz
    Ekin[i] = 0.5f0 * mass * v2
    s = noise_scale[i]
    Tbath = (s*s) / (2f0*gamma*dt)
    dq[i] = dq[i] + gamma * (v2 - 3f0 * Tbath / mass) * dt
    vx[i] = vnx; vy[i] = vny; vz[i] = vnz
    return
end

# Public wrappers
function vv_positions_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                           vx::CuArray{Float32,1}, vy::CuArray{Float32,1},
                           fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                           beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1},
                           params::VVParams{Float32}, dt::Float32, box::Definitions.Box2)
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_pos2!(rx, ry, vx, vy, fx, fy, beta_x, beta_y, params.gamma, params.mass, dt, box[1], box[2])
    k(rx, ry, vx, vy, fx, fy, beta_x, beta_y, params.gamma, params.mass, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function vv_positions_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                           vx::CuArray{Float32,1}, vy::CuArray{Float32,1}, vz::CuArray{Float32,1},
                           fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                           beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1}, beta_z::CuArray{Float32,1},
                           params::VVParams{Float32}, dt::Float32, box::Definitions.Box3)
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_pos3!(rx, ry, rz, vx, vy, vz, fx, fy, fz, beta_x, beta_y, beta_z, params.gamma, params.mass, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, fx, fy, fz, beta_x, beta_y, beta_z, params.gamma, params.mass, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

function vv_velocities_soa!(vx::CuArray{Float32,1}, vy::CuArray{Float32,1},
                            f0x::CuArray{Float32,1}, f0y::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                            beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1},
                            dq::CuArray{Float32,1}, Ekin::CuArray{Float32,1},
                            params::VVParams{Float32}, dt::Float32)
    N = length(vx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel2!(vx, vy, f0x, f0y, fx, fy, beta_x, beta_y, dq, Ekin, params.noise_scale, params.gamma, params.mass, dt)
    k(vx, vy, f0x, f0y, fx, fy, beta_x, beta_y, dq, Ekin, params.noise_scale, params.gamma, params.mass, dt; threads, blocks)
    return nothing
end

function vv_velocities_soa!(vx::CuArray{Float32,1}, vy::CuArray{Float32,1}, vz::CuArray{Float32,1},
                            f0x::CuArray{Float32,1}, f0y::CuArray{Float32,1}, f0z::CuArray{Float32,1},
                            fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                            beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1}, beta_z::CuArray{Float32,1},
                            dq::CuArray{Float32,1}, Ekin::CuArray{Float32,1},
                            params::VVParams{Float32}, dt::Float32)
    N = length(vx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel3!(vx, vy, vz, f0x, f0y, f0z, fx, fy, fz, beta_x, beta_y, beta_z, dq, Ekin, params.noise_scale, params.gamma, params.mass, dt)
    k(vx, vy, vz, f0x, f0y, f0z, fx, fy, fz, beta_x, beta_y, beta_z, dq, Ekin, params.noise_scale, params.gamma, params.mass, dt; threads, blocks)
    return nothing
end

end # module
