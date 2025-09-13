module LangevinIntegrators

using CUDA
using ..Definitions

export VVParams,
       vv_prepare_noise!,
       vv_positions_soa!,
       vv_velocities_soa!,
       vv_step_fused_2d!, vv_step_fused_3d!

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

struct VVParams{T}
    gamma::T                 # friction coefficient
    mass::T                  # particle mass
    noise_scale::CuArray{Float32,1}   # per-particle noise scale sqrt(2*gamma*kT*dt)
end

# ------------------------------------------------------------------------------
# Random force preparation for VV (generate ONCE per step)
# rf_x, rf_y, rf_z := c3 .* randn()  (all on device)
# ------------------------------------------------------------------------------

"""
    vv_prepare_noise!(beta_x, beta_y, noise_scale; beta_z=nothing)

Fill `beta_x`, `beta_y` (and optionally `beta_z`) with noise_scale .* randn(Float32) on the GPU.
This generates β_n ~ N(0, 2*gamma*kT*dt) for the GJF integrator.
Call exactly once per time step and pass the same arrays to both
`vv_positions_soa!` and `vv_velocities_soa!`.
Fused implementation reduces kernel launches vs. randn! + scaling.
"""

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

# ------------------------------------------------------------------------------
# VV positions kernels (SoA) — use SAME rf_* that will be reused in velocity
# ------------------------------------------------------------------------------

function _vv_pos2!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32,
    Lx::Float32, Ly::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    # GJF coefficients: eq (21)
    # a = (1 - gamma*dt/(2*mass))/(1 + gamma*dt/(2*mass))
    # b = 1/(1 + gamma*dt/(2*mass))
    q = gamma * dt / (2f0 * mass)
    b = 1f0 / (1f0 + q)

    # Position update: r_{n+1} = r_n + b*dt*v_n + (b*dt/(2*mass))*(dt*f_n + β_n)
    dpx = b * dt * vx[i] + (b * dt / (2f0 * mass)) * (dt * fx[i] + beta_x[i])
    dpy = b * dt * vy[i] + (b * dt / (2f0 * mass)) * (dt * fy[i] + beta_y[i])

    x = rx[i] + dpx
    y = ry[i] + dpy

    # PBC (centered box)
    x = (x + Lx/2); x -= floor(x/Lx)*Lx; x -= Lx/2
    y = (y + Ly/2); y -= floor(y/Ly)*Ly; y -= Ly/2

    rx[i] = x; ry[i] = y
    return
end

function _vv_pos3!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32}, beta_z::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32,
    Lx::Float32, Ly::Float32, Lz::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    # GJF coefficients
    q = gamma * dt / (2f0 * mass)
    b = 1f0 / (1f0 + q)

    # Position update: r_{n+1} = r_n + b*dt*v_n + (b*dt/(2*mass))*(dt*f_n + β_n)
    dpx = b * dt * vx[i] + (b * dt / (2f0 * mass)) * (dt * fx[i] + beta_x[i])
    dpy = b * dt * vy[i] + (b * dt / (2f0 * mass)) * (dt * fy[i] + beta_y[i])
    dpz = b * dt * vz[i] + (b * dt / (2f0 * mass)) * (dt * fz[i] + beta_z[i])

    x = rx[i] + dpx
    y = ry[i] + dpy
    z = rz[i] + dpz

    x = (x + Lx/2); x -= floor(x/Lx)*Lx; x -= Lx/2
    y = (y + Ly/2); y -= floor(y/Ly)*Ly; y -= Ly/2
    z = (z + Lz/2); z -= floor(z/Lz)*Lz; z -= Lz/2

    rx[i] = x; ry[i] = y; rz[i] = z
    return
end

# Public 2D
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

# Public 3D
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

# ------------------------------------------------------------------------------
# VV velocities kernels (SoA) — reuse SAME rf_* generated before positions
# ------------------------------------------------------------------------------

function _vv_vel2!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32},
    f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32},  fy::CuDeviceVector{Float32},
    beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32},
    dq::CuDeviceVector{Float32},  Ekin::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end

    # GJF coefficients
    q = gamma * dt / (2f0 * mass)
    a = (1f0 - q) / (1f0 + q)
    b = 1f0 / (1f0 + q)

    vpx = vx[i]; vpy = vy[i]

    # Velocity update: v_{n+1} = a*v_n + (dt/(2*mass))*(a*f_n + f_{n+1}) + (b/mass)*β_n
    # Note: f0 is force at t, fx is force at t+dt after position update
    vnx = a * vpx + (dt / (2f0 * mass)) * (a * f0x[i] + fx[i]) + (b / mass) * beta_x[i]
    vny = a * vpy + (dt / (2f0 * mass)) * (a * f0y[i] + fy[i]) + (b / mass) * beta_y[i]


    Ekin[i] = 0.5f0 * mass * (vnx * vnx + vny * vny)
    #TODO We need to add temperature to the kernel to compute dq as  dq[i] = (gamma/m)* (2*temperature - Ekin[i])
    dq[i] = (2.f0 * gamma / mass) * (1.0f0 - Ekin[i])  # Placeholder for correct heat bookkeeping

    vx[i] = vnx; vy[i] = vny
    return
end

function _vv_vel3!(
    vx::CuDeviceVector{Float32}, vy::CuDeviceVector{Float32}, vz::CuDeviceVector{Float32},
    f0x::CuDeviceVector{Float32}, f0y::CuDeviceVector{Float32}, f0z::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32},  fy::CuDeviceVector{Float32},  fz::CuDeviceVector{Float32},
    beta_x::CuDeviceVector{Float32}, beta_y::CuDeviceVector{Float32}, beta_z::CuDeviceVector{Float32},
    dq::CuDeviceVector{Float32},  Ekin::CuDeviceVector{Float32},
    gamma::Float32, mass::Float32, dt::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end

    # GJF coefficients
    q = gamma * dt / (2f0 * mass)
    a = (1f0 - q) / (1f0 + q)
    b = 1f0 / (1f0 + q)

    vpx = vx[i]; vpy = vy[i]; vpz = vz[i]

    # Velocity update: v_{n+1} = a*v_n + (dt/(2*mass))*(a*f_n + f_{n+1}) + (b/mass)*β_n
    # Note: f0 is force at t, fx is force at t+dt after position update
    vnx = a * vpx + (dt / (2f0 * mass)) * (a * f0x[i] + fx[i]) + (b / mass) * beta_x[i]
    vny = a * vpy + (dt / (2f0 * mass)) * (a * f0y[i] + fy[i]) + (b / mass) * beta_y[i]
    vnz = a * vpz + (dt / (2f0 * mass)) * (a * f0z[i] + fz[i]) + (b / mass) * beta_z[i]


    Ekin[i] = 0.5f0 * mass * (vnx * vnx + vny * vny + vnz * vnz)
    #TODO We need to add temperature to the kernel to compute dq as  dq[i] = (gamma/m)* (2*temperature - Ekin[i])
    dq[i] = (2.f0 * gamma / mass) * (1.5f0 - Ekin[i])  # Placeholder for correct heat bookkeeping

    vx[i] = vnx; vy[i] = vny; vz[i] = vnz
    return
end

# Public 2D
function vv_velocities_soa!(
    vx::CuArray{Float32,1}, vy::CuArray{Float32,1},
    f0x::CuArray{Float32,1}, f0y::CuArray{Float32,1},
    fx::CuArray{Float32,1},  fy::CuArray{Float32,1},
    beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1},
    dq::CuArray{Float32,1},  Ekin::CuArray{Float32,1},
    params::VVParams{Float32}, dt::Float32
)
    N = length(vx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel2!(vx, vy, f0x, f0y, fx, fy, beta_x, beta_y, dq, Ekin, params.gamma, params.mass, dt)
    k(vx, vy, f0x, f0y, fx, fy, beta_x, beta_y, dq, Ekin, params.gamma, params.mass, dt; threads, blocks)
    return nothing
end

# Public 3D
function vv_velocities_soa!(
    vx::CuArray{Float32,1}, vy::CuArray{Float32,1}, vz::CuArray{Float32,1},
    f0x::CuArray{Float32,1}, f0y::CuArray{Float32,1}, f0z::CuArray{Float32,1},
    fx::CuArray{Float32,1},  fy::CuArray{Float32,1},  fz::CuArray{Float32,1},
    beta_x::CuArray{Float32,1}, beta_y::CuArray{Float32,1}, beta_z::CuArray{Float32,1},
    dq::CuArray{Float32,1},  Ekin::CuArray{Float32,1},
    params::VVParams{Float32}, dt::Float32
)
    N = length(vx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel3!(vx, vy, vz, f0x, f0y, f0z, fx, fy, fz, beta_x, beta_y, beta_z, dq, Ekin, params.gamma, params.mass, dt)
    k(vx, vy, vz, f0x, f0y, f0z, fx, fy, fz, beta_x, beta_y, beta_z, dq, Ekin, params.gamma, params.mass, dt; threads, blocks)
    return nothing
end

end # module
