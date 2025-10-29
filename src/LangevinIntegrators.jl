module LangevinIntegrators

using CUDA
using ..Definitions

export VVParams,
       vv_prepare_noise!,
       vv_positions_soa!,
       vv_velocities_soa!,
       BAOABParams,
        # BAOAB sub-steps
       baoab_BA_2d!, baoab_OU_2d!, baoab_A_2d!, baoab_B_2d!,
       baoab_BA_3d!, baoab_OU_3d!, baoab_A_3d!, baoab_B_3d!

# ------------------------------------------------------------------
# Parameter containers
# ------------------------------------------------------------------

struct VVParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    noise_scale::CuArray{T,1}
end

struct BAOABParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    noise_scale::CuArray{T,1}
end

# ------------------------------------------------------------------
# Langevin VV noise draw
# ------------------------------------------------------------------

function _vv_noise2_kernel!(beta_x::CuDeviceVector{T},
                            beta_y::CuDeviceVector{T},
                            noise_scale::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        beta_x[i] = s * randn(T)
        beta_y[i] = s * randn(T)
    end
    return
end

function _vv_noise3_kernel!(beta_x::CuDeviceVector{T},
                            beta_y::CuDeviceVector{T},
                            beta_z::CuDeviceVector{T},
                            noise_scale::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        beta_x[i] = s * randn(T)
        beta_y[i] = s * randn(T)
        beta_z[i] = s * randn(T)
    end
    return
end

function vv_prepare_noise!(beta_x::CuArray{T,1},
                           beta_y::CuArray{T,1},
                           noise_scale::CuArray{T,1};
                           beta_z::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat}
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

# ------------------------------------------------------------------
# Position updates (VV B step)
# ------------------------------------------------------------------

function _vv_pos2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                   vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                   fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                   beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T},
                   gamma::CuDeviceVector{T}, mass::T, dt::T,
                   Lx::T, Ly::T) where {T<:AbstractFloat}
    half = T(0.5); two = T(2)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    g = gamma[i]
    q = g * dt / (two * mass)
    b = one(T) / (one(T) + q)
    coef = b * dt / (two * mass)
    @inbounds begin
        dpx = b*dt*vx[i] + coef*(dt*fx[i] + beta_x[i])
        dpy = b*dt*vy[i] + coef*(dt*fy[i] + beta_y[i])
        x = rx[i] + dpx
        y = ry[i] + dpy
        x = (x + Lx*half); x -= floor(x/Lx)*Lx; x -= Lx*half
        y = (y + Ly*half); y -= floor(y/Ly)*Ly; y -= Ly*half
        rx[i] = x; ry[i] = y
    end
    return
end

function _vv_pos3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                   vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                   fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                   beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T}, beta_z::CuDeviceVector{T},
                   gamma::CuDeviceVector{T}, mass::T, dt::T,
                   Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    half = T(0.5); two = T(2)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    g = gamma[i]
    q = g * dt / (two * mass)
    b = one(T) / (one(T) + q)
    coef = b * dt / (two * mass)
    @inbounds begin
        dpx = b*dt*vx[i] + coef*(dt*fx[i] + beta_x[i])
        dpy = b*dt*vy[i] + coef*(dt*fy[i] + beta_y[i])
        dpz = b*dt*vz[i] + coef*(dt*fz[i] + beta_z[i])
        x = rx[i] + dpx
        y = ry[i] + dpy
        z = rz[i] + dpz
        x = (x + Lx*half); x -= floor(x/Lx)*Lx; x -= Lx*half
        y = (y + Ly*half); y -= floor(y/Ly)*Ly; y -= Ly*half
        z = (z + Lz*half); z -= floor(z/Lz)*Lz; z -= Lz*half
        rx[i] = x; ry[i] = y; rz[i] = z
    end
    return
end

function vv_positions_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                           vx::CuArray{T,1}, vy::CuArray{T,1},
                           fx::CuArray{T,1}, fy::CuArray{T,1},
                           beta_x::CuArray{T,1}, beta_y::CuArray{T,1},
                           params::VVParams{T}, dt::T, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_pos2!(rx, ry, vx, vy, fx, fy, beta_x, beta_y,
                                          params.gamma, params.mass, dt,
                                          box[1], box[2])
    k(rx, ry, vx, vy, fx, fy, beta_x, beta_y,
      params.gamma, params.mass, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function vv_positions_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                           vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1},
                           fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                           beta_x::CuArray{T,1}, beta_y::CuArray{T,1}, beta_z::CuArray{T,1},
                           params::VVParams{T}, dt::T, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_pos3!(rx, ry, rz, vx, vy, vz, fx, fy, fz,
                                          beta_x, beta_y, beta_z,
                                          params.gamma, params.mass, dt,
                                          box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, fx, fy, fz,
      beta_x, beta_y, beta_z,
      params.gamma, params.mass, dt,
      box[1], box[2], box[3]; threads, blocks)
    return nothing
end

# ------------------------------------------------------------------
# Velocity updates (VV O step)
# ------------------------------------------------------------------

function _vv_vel2!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                   f0x::CuDeviceVector{T}, f0y::CuDeviceVector{T},
                   fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                   beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T},
                   dq::CuDeviceVector{T}, dU::CuDeviceVector{T}, Ekin::CuDeviceVector{T},
                   noise_scale::CuDeviceVector{T},
                   gamma::CuDeviceVector{T}, mass::T, dt::T) where {T<:AbstractFloat}
    zeroT = zero(T); two = T(2); half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    g = gamma[i]
    q = g * dt / (two * mass)
    a = (one(T) - q) / (one(T) + q)
    b = one(T) / (one(T) + q)
    vpx = vx[i]; vpy = vy[i]
    rand_force_x = (b/mass)*beta_x[i]
    rand_force_y = (b/mass)*beta_y[i]
    vnx = a*vpx + (dt/(two*mass))*(a*f0x[i] + fx[i]) + rand_force_x 
    vny = a*vpy + (dt/(two*mass))*(a*f0y[i] + fy[i]) + rand_force_y 
    v2 = vnx*vnx + vny*vny
    #s = noise_scale[i]
    
    #Tbath = g > zeroT ? (s*s) / (two*g*dt) : zeroT
    dissipated_energy = - g * v2 * dt
    injected_energy = half*((vnx + vpx) * beta_x[i] + (vny + vpy) * beta_y[i]) / b
    

    Ekin[i] = half * mass * v2
    dq[i] = dq[i] - (dissipated_energy + injected_energy) / dt
    dU[i] = dU[i] + (vnx * fx[i] + vny * fy[i])
    vx[i] = vnx; vy[i] = vny
    return
end

function _vv_vel3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                   f0x::CuDeviceVector{T}, f0y::CuDeviceVector{T}, f0z::CuDeviceVector{T},
                   fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                   beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T}, beta_z::CuDeviceVector{T},
                   dq::CuDeviceVector{T}, dU::CuDeviceVector{T}, Ekin::CuDeviceVector{T},
                   noise_scale::CuDeviceVector{T},
                   gamma::CuDeviceVector{T}, mass::T, dt::T) where {T<:AbstractFloat}
    zeroT = zero(T); two = T(2); three = T(3); half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    g = gamma[i]
    q = g * dt / (two * mass)
    a = (one(T) - q) / (one(T) + q)
    b = one(T) / (one(T) + q)
    vpx = vx[i]; vpy = vy[i]; vpz = vz[i]

    rand_force_x = (b/mass)*beta_x[i]
    rand_force_y = (b/mass)*beta_y[i]
    rand_force_z = (b/mass)*beta_z[i]

    vnx = a*vpx + (dt/(two*mass))*(a*f0x[i] + fx[i]) + rand_force_x
    vny = a*vpy + (dt/(two*mass))*(a*f0y[i] + fy[i]) + rand_force_y
    vnz = a*vpz + (dt/(two*mass))*(a*f0z[i] + fz[i]) + rand_force_z

    v2 = vnx*vnx + vny*vny + vnz*vnz
    
    #s = noise_scale[i]
    #Tbath = g > zeroT ? (s*s) / (two*g*dt) : zeroT
    dissipated_energy = - g * v2 * dt
    injected_energy = half*((vnx + vpx) * beta_x[i] + (vny + vpy) * beta_y[i] + (vnz + vpz) * beta_z[i]) / b


    Ekin[i] = half * mass * v2
    dq[i] = dq[i] - (dissipated_energy + injected_energy) / dt
    dU[i] = dU[i] + (vnx * fx[i] + vny * fy[i] + vnz * fz[i])
    vx[i] = vnx; vy[i] = vny; vz[i] = vnz
    return
end

function vv_velocities_soa!(vx::CuArray{T,1}, vy::CuArray{T,1},
                            f0x::CuArray{T,1}, f0y::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            beta_x::CuArray{T,1}, beta_y::CuArray{T,1},
                            dq::CuArray{T,1}, dU::CuArray{T,1}, Ekin::CuArray{T,1},
                            params::VVParams{T}, dt::T) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel2!(vx, vy, f0x, f0y, fx, fy,
                                          beta_x, beta_y, dq, dU, Ekin,
                                          params.noise_scale, params.gamma,
                                          params.mass, dt)
    k(vx, vy, f0x, f0y, fx, fy,
      beta_x, beta_y, dq, dU, Ekin,
      params.noise_scale, params.gamma,
      params.mass, dt; threads, blocks)
    return nothing
end

function vv_velocities_soa!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1},
                            f0x::CuArray{T,1}, f0y::CuArray{T,1}, f0z::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            beta_x::CuArray{T,1}, beta_y::CuArray{T,1}, beta_z::CuArray{T,1},
                            dq::CuArray{T,1}, dU::CuArray{T,1}, Ekin::CuArray{T,1},
                            params::VVParams{T}, dt::T) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _vv_vel3!(vx, vy, vz, f0x, f0y, f0z,
                                          fx, fy, fz,
                                          beta_x, beta_y, beta_z,
                                          dq, dU, Ekin, params.noise_scale,
                                          params.gamma, params.mass, dt)
    k(vx, vy, vz, f0x, f0y, f0z,
      fx, fy, fz,
      beta_x, beta_y, beta_z,
      dq, dU, Ekin, params.noise_scale,
      params.gamma, params.mass, dt; threads, blocks)
    return nothing
end

# ------------------------------------------------------------------
# BAOAB helper kernels
# ------------------------------------------------------------------

function _baoab_BA2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                     vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                     f0x::CuDeviceVector{T}, f0y::CuDeviceVector{T},
                     mass::T, dt::T,
                     Lx::T, Ly::T) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(T(2)*mass)) * f0x[i]
        vyi = vy[i] + (dt/(T(2)*mass)) * f0y[i]
        xi = rx[i] + half*dt*vxi
        yi = ry[i] + half*dt*vyi
        xi = (xi + Lx*half); xi -= floor(xi/Lx)*Lx; xi -= Lx*half
        yi = (yi + Ly*half); yi -= floor(yi/Ly)*Ly; yi -= Ly*half
        vx[i] = vxi; vy[i] = vyi
        rx[i] = xi;  ry[i] = yi
    end
    return
end

function _baoab_BA3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                     vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                     f0x::CuDeviceVector{T}, f0y::CuDeviceVector{T}, f0z::CuDeviceVector{T},
                     mass::T, dt::T,
                     Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(T(2)*mass)) * f0x[i]
        vyi = vy[i] + (dt/(T(2)*mass)) * f0y[i]
        vzi = vz[i] + (dt/(T(2)*mass)) * f0z[i]
        xi = rx[i] + half*dt*vxi
        yi = ry[i] + half*dt*vyi
        zi = rz[i] + half*dt*vzi
        xi = (xi + Lx*half); xi -= floor(xi/Lx)*Lx; xi -= Lx*half
        yi = (yi + Ly*half); yi -= floor(yi/Ly)*Ly; yi -= Ly*half
        zi = (zi + Lz*half); zi -= floor(zi/Lz)*Lz; zi -= Lz*half
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi
        rx[i] = xi;  ry[i] = yi; rz[i] = zi
    end
    return
end

function _baoab_OU2!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                     beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T},
                     noise_scale::CuDeviceVector{T},
                     gamma::CuDeviceVector{T}, mass::T, dt::T,
                     dq::CuDeviceVector{T}) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        c = exp(-g*dt/mass)
        # scale pre-generated VV noise (β = s * N(0,1)) to OU sigma
        # ratio r satisfies: r * β ~ N(0, sigma^2) with sigma^2 = (T/m)*(1-c^2)
        # where T = s^2/(2 g dt)
        # ⇒ r = sqrt((1-c^2)/(2 g dt m))
        r = sqrt((one(T) - c*c) / (T(2)*g*dt*mass))
        vpx = vx[i]; vpy = vy[i]
        vnx = c*vpx + r*beta_x[i]
        vny = c*vpy + r*beta_y[i]
        # Heat increment during OU step: δQ = -ΔK (no conservative work in O)
        v2 = vnx*vnx + vny*vny
        
        #s = noise_scale[i]
        #Tbath = g > zeroT ? (s*s) / (two*g*dt) : zeroT
        dissipated_energy = - g * v2 * dt
        injected_energy = half*((vnx + vpx) * beta_x[i] + (vny + vpy) * beta_y[i])

        dq[i] = dq[i] - (dissipated_energy + injected_energy) / dt
        vx[i] = vnx; vy[i] = vny
    end
    return
end

function _baoab_OU3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                     beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T}, beta_z::CuDeviceVector{T},
                     noise_scale::CuDeviceVector{T},
                     gamma::CuDeviceVector{T}, mass::T, dt::T,
                     dq::CuDeviceVector{T}) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        c = exp(-g*dt/mass)
        r = sqrt((one(T) - c*c) / (T(2)*g*dt*mass))
        vpx = vx[i]; vpy = vy[i]; vpz = vz[i]
        vnx = c*vpx + r*beta_x[i]
        vny = c*vpy + r*beta_y[i]
        vnz = c*vpz + r*beta_z[i]
        
        v2 = vnx*vnx + vny*vny + vnz*vnz
    
        #s = noise_scale[i]
        #Tbath = g > zeroT ? (s*s) / (two*g*dt) : zeroT
        dissipated_energy = - g * v2 * dt
        injected_energy = half*((vnx + vpx) * beta_x[i] + (vny + vpy) * beta_y[i] + (vnz + vpz) * beta_z[i])

        dq[i] = dq[i] - (dissipated_energy + injected_energy) / dt
        vx[i] = vnx; vy[i] = vny; vz[i] = vnz
    end
    return
end

function _baoab_A2!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
                    vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                    dt::T, Lx::T, Ly::T) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        xi = rx[i] + half*dt*vx[i]
        yi = ry[i] + half*dt*vy[i]
        xi = (xi + Lx*half); xi -= floor(xi/Lx)*Lx; xi -= Lx*half
        yi = (yi + Ly*half); yi -= floor(yi/Ly)*Ly; yi -= Ly*half
        rx[i] = xi; ry[i] = yi
    end
    return
end

function _baoab_A3!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                    vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                    dt::T, Lx::T, Ly::T, Lz::T) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    @inbounds begin
        xi = rx[i] + half*dt*vx[i]
        yi = ry[i] + half*dt*vy[i]
        zi = rz[i] + half*dt*vz[i]
        xi = (xi + Lx*half); xi -= floor(xi/Lx)*Lx; xi -= Lx*half
        yi = (yi + Ly*half); yi -= floor(yi/Ly)*Ly; yi -= Ly*half
        zi = (zi + Lz*half); zi -= floor(zi/Lz)*Lz; zi -= Lz*half
        rx[i] = xi; ry[i] = yi; rz[i] = zi
    end
    return
end

function _baoab_B2!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                    mass::T, dt::T,
                    Ekin::CuDeviceVector{T}) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(T(2)*mass)) * fx[i]
        vyi = vy[i] + (dt/(T(2)*mass)) * fy[i]
        vx[i] = vxi; vy[i] = vyi
        v2 = vxi*vxi + vyi*vyi
        Ekin[i] = half * mass * v2
    end
    return
end

function _baoab_B3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                    mass::T, dt::T,
                    Ekin::CuDeviceVector{T}) where {T<:AbstractFloat}
    half = T(0.5)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        vxi = vx[i] + (dt/(T(2)*mass)) * fx[i]
        vyi = vy[i] + (dt/(T(2)*mass)) * fy[i]
        vzi = vz[i] + (dt/(T(2)*mass)) * fz[i]
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi
        v2 = vxi*vxi + vyi*vyi + vzi*vzi
        Ekin[i] = half * mass * v2
    end
    return
end

# ------------------------------------------------------------------
# Public BAOAB wrappers
# ------------------------------------------------------------------

function baoab_BA_2d!(rx, ry, vx, vy, f0x, f0y, params::BAOABParams{T}, dt::T, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA2!(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2])
    k(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function baoab_BA_3d!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params::BAOABParams{T}, dt::T, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA3!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

function baoab_OU_2d!(vx, vy, beta_x, beta_y, params::BAOABParams{T}, dt::T, dq) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU2!(vx, vy, beta_x, beta_y, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, beta_x, beta_y, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

function baoab_OU_3d!(vx, vy, vz, beta_x, beta_y, beta_z, params::BAOABParams{T}, dt::T, dq) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU3!(vx, vy, vz, beta_x, beta_y, beta_z, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, vz, beta_x, beta_y, beta_z, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

function baoab_A_2d!(rx, ry, vx, vy, dt::T, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A2!(rx, ry, vx, vy, dt, box[1], box[2])
    k(rx, ry, vx, vy, dt, box[1], box[2]; threads, blocks)
    return nothing
end

function baoab_A_3d!(rx, ry, rz, vx, vy, vz, dt::T, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A3!(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

function baoab_B_2d!(vx, vy, fx, fy, params::BAOABParams{T}, dt::T, Ekin) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B2!(vx, vy, fx, fy, params.mass, dt, Ekin)
    k(vx, vy, fx, fy, params.mass, dt, Ekin; threads, blocks)
    return nothing
end

function baoab_B_3d!(vx, vy, vz, fx, fy, fz, params::BAOABParams{T}, dt::T, Ekin) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B3!(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin)
    k(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin; threads, blocks)
    return nothing
end

end # module LangevinIntegrators
