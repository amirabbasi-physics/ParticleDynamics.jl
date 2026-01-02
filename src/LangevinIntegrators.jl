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

"""
    VVParams(gamma, mass, noise_scale; corr_time=nothing)

Parameters for the Grønbech-Jensen/Farago (velocity-Verlet) Langevin scheme.
`gamma` and `noise_scale` are CuArrays so `Filters.set_temperature!` can update
them in-place (see `examples/TwoT_2D_LD_VV.jl`). `corr_time` stores optional
per-particle Ornstein–Uhlenbeck correlation times.
"""
struct VVParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
end
VVParams(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1}; corr_time::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat} = begin
    @assert corr_time === nothing || length(corr_time) == length(gamma)
    VVParams{T}(gamma, mass, noise_scale, corr_time)
end

"""
    BAOABParams(gamma, mass, noise_scale; corr_time=nothing)

Parameter container for BAOAB/BAOA/GSM splitting schemes. Matches the API
expected by `baoab_BA_*`, `baoab_OU_*`, `baoab_A_*`, and `baoab_B_*`.
"""
struct BAOABParams{T<:AbstractFloat}
    gamma::CuArray{T,1}
    mass::T
    noise_scale::CuArray{T,1}
    corr_time::Union{Nothing,CuArray{T,1}}
end
BAOABParams(gamma::CuArray{T,1}, mass::T, noise_scale::CuArray{T,1}; corr_time::Union{Nothing,CuArray{T,1}}=nothing) where {T<:AbstractFloat} = begin
    @assert corr_time === nothing || length(corr_time) == length(gamma)
    BAOABParams{T}(gamma, mass, noise_scale, corr_time)
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

function _vv_noise2_ou_kernel!(beta_x::CuDeviceVector{T},
                               beta_y::CuDeviceVector{T},
                               noise_scale::CuDeviceVector{T},
                               corr_time::CuDeviceVector{T},
                               state_x::CuDeviceVector{T},
                               state_y::CuDeviceVector{T},
                               dt::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        τ = corr_time[i]
        if τ <= zero(T)
            # Fallback to white noise if correlation time is zero or negative
            valx = s * randn(T)
            valy = s * randn(T)
            beta_x[i] = valx; beta_y[i] = valy
            state_x[i] = valx; state_y[i] = valy
        else
            a = exp(-dt / τ)
            b = sqrt(max(one(T) - a*a, zero(T)))
            nx = a * state_x[i] + b * s * randn(T)
            ny = a * state_y[i] + b * s * randn(T)
            beta_x[i] = nx; beta_y[i] = ny
            state_x[i] = nx; state_y[i] = ny
        end
    end
    return
end

function _vv_noise3_ou_kernel!(beta_x::CuDeviceVector{T},
                               beta_y::CuDeviceVector{T},
                               beta_z::CuDeviceVector{T},
                               noise_scale::CuDeviceVector{T},
                               corr_time::CuDeviceVector{T},
                               state_x::CuDeviceVector{T},
                               state_y::CuDeviceVector{T},
                               state_z::CuDeviceVector{T},
                               dt::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(beta_x); if i > N; return; end
    @inbounds begin
        s = noise_scale[i]
        τ = corr_time[i]
        if τ <= zero(T)
            valx = s * randn(T); valy = s * randn(T); valz = s * randn(T)
            beta_x[i] = valx; beta_y[i] = valy; beta_z[i] = valz
            state_x[i] = valx; state_y[i] = valy; state_z[i] = valz
        else
            a = exp(-dt / τ)
            b = sqrt(max(one(T) - a*a, zero(T)))
            nx = a * state_x[i] + b * s * randn(T)
            ny = a * state_y[i] + b * s * randn(T)
            nz = a * state_z[i] + b * s * randn(T)
            beta_x[i] = nx; beta_y[i] = ny; beta_z[i] = nz
            state_x[i] = nx; state_y[i] = ny; state_z[i] = nz
        end
    end
    return
end

"""
    vv_prepare_noise!(βx, βy[, βz], noise_scale; corr_time, state_x, state_y[, state_z], dt)

Draw the stochastic impulses used by the velocity-Verlet Langevin solver.
Supports both white noise and correlated (OU) draws. The correlated path is
used in `test/runtests.jl` by supplying `noise_corr_time=0.05f0`.
"""
function vv_prepare_noise!(beta_x::CuArray{T,1},
                           beta_y::CuArray{T,1},
                           noise_scale::CuArray{T,1};
                           beta_z::Union{Nothing,CuArray{T,1}}=nothing,
                           corr_time::Union{Nothing,CuArray{T,1}}=nothing,
                           state_x::Union{Nothing,CuArray{T,1}}=nothing,
                           state_y::Union{Nothing,CuArray{T,1}}=nothing,
                           state_z::Union{Nothing,CuArray{T,1}}=nothing,
                           dt::Union{Nothing,T}=nothing) where {T<:AbstractFloat}
    @assert length(beta_x) == length(noise_scale) == length(beta_y)
    N = length(beta_x)
    threads = (N < 100_000) ? 128 : 256
    blocks  = cld(N, threads)
    use_correlated = (corr_time !== nothing)
    if beta_z === nothing
        if use_correlated
            @assert state_x !== nothing && state_y !== nothing "OU state buffers required for correlated noise"
            @assert dt !== nothing "dt required for correlated noise"
            k = CUDA.@cuda launch=false _vv_noise2_ou_kernel!(beta_x, beta_y, noise_scale, corr_time, state_x, state_y, dt::T)
            k(beta_x, beta_y, noise_scale, corr_time, state_x, state_y, dt::T; threads, blocks)
        else
            k = CUDA.@cuda launch=false _vv_noise2_kernel!(beta_x, beta_y, noise_scale)
            k(beta_x, beta_y, noise_scale; threads, blocks)
        end
    else
        @assert length(beta_z) == length(noise_scale)
        if use_correlated
            @assert state_x !== nothing && state_y !== nothing && state_z !== nothing "OU state buffers required for correlated noise"
            @assert dt !== nothing "dt required for correlated noise"
            k = CUDA.@cuda launch=false _vv_noise3_ou_kernel!(beta_x, beta_y, beta_z, noise_scale, corr_time, state_x, state_y, state_z, dt::T)
            k(beta_x, beta_y, beta_z, noise_scale, corr_time, state_x, state_y, state_z, dt::T; threads, blocks)
        else
            k = CUDA.@cuda launch=false _vv_noise3_kernel!(beta_x, beta_y, beta_z, noise_scale)
            k(beta_x, beta_y, beta_z, noise_scale; threads, blocks)
        end
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

"""
    vv_positions_soa!(rx, ry[, rz], vx, vy[, vz], fx, fy[, fz], βx, βy[, βz], params, dt, box)

Velocity-Verlet B step: update positions using half-step velocities,
deterministic forces, and stochastic impulses. Called internally by `step!` and
mirrors the implementation in `examples/TwoT_2D_LD_VV.jl`.
"""
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
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        # Parameters in base precision
        g = gamma[i]
        q = g * dt / (T(2) * mass)
        a = (one(T) - q) / (one(T) + q)
        b = one(T) / (one(T) + q)
        # Promote to Float64 for sensitive math
        vpx = Float64(vx[i]); vpy = Float64(vy[i])
        f0x_i = Float64(f0x[i]); f0y_i = Float64(f0y[i])
        fx_i  = Float64(fx[i]);  fy_i  = Float64(fy[i])
        bx    = Float64(beta_x[i]); by = Float64(beta_y[i])
        aA = Float64(a); bA = Float64(b)
        dtA = Float64(dt); gA = Float64(g); mA = Float64(mass)

        # Velocity update
        vnx = aA*vpx + (dtA/(2*mA))*(aA*f0x_i + fx_i) + (bA/mA)*bx
        vny = aA*vpy + (dtA/(2*mA))*(aA*f0y_i + fy_i) + (bA/mA)*by

        v2 = vnx*vnx + vny*vny
        vbarx = 0.5*(vpx + vnx); vbary = 0.5*(vpy + vny)

        # Power terms (dq is power)
        P_diss = - gA * v2
        P_sto  = (vbarx*(bx/bA) + vbary*(by/bA)) / dtA
        P_cons = vnx*fx_i + vny*fy_i

        Ekin[i] = T(0.5) * mass * T(v2)
        dq[i]   = dq[i] - T(P_diss + P_sto)
        dU[i]   = dU[i] + T(P_cons)
        vx[i] = T(vnx); vy[i] = T(vny)
    end
    return
end

function _vv_vel3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                   f0x::CuDeviceVector{T}, f0y::CuDeviceVector{T}, f0z::CuDeviceVector{T},
                   fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                   beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T}, beta_z::CuDeviceVector{T},
                   dq::CuDeviceVector{T}, dU::CuDeviceVector{T}, Ekin::CuDeviceVector{T},
                   noise_scale::CuDeviceVector{T},
                   gamma::CuDeviceVector{T}, mass::T, dt::T) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        g = gamma[i]
        q = g * dt / (T(2) * mass)
        a = (one(T) - q) / (one(T) + q)
        b = one(T) / (one(T) + q)

        vpx = Float64(vx[i]); vpy = Float64(vy[i]); vpz = Float64(vz[i])
        f0x_i = Float64(f0x[i]); f0y_i = Float64(f0y[i]); f0z_i = Float64(f0z[i])
        fx_i  = Float64(fx[i]);  fy_i  = Float64(fy[i]);  fz_i  = Float64(fz[i])
        bx    = Float64(beta_x[i]); by = Float64(beta_y[i]); bz = Float64(beta_z[i])
        aA = Float64(a); bA = Float64(b)
        dtA = Float64(dt); gA = Float64(g); mA = Float64(mass)

        vnx = aA*vpx + (dtA/(2*mA))*(aA*f0x_i + fx_i) + (bA/mA)*bx
        vny = aA*vpy + (dtA/(2*mA))*(aA*f0y_i + fy_i) + (bA/mA)*by
        vnz = aA*vpz + (dtA/(2*mA))*(aA*f0z_i + fz_i) + (bA/mA)*bz

        v2 = vnx*vnx + vny*vny + vnz*vnz
        vbarx = 0.5*(vpx + vnx); vbary = 0.5*(vpy + vny); vbarz = 0.5*(vpz + vnz)

        P_diss = - gA * v2
        P_sto  = (vbarx*(bx/bA) + vbary*(by/bA) + vbarz*(bz/bA)) / dtA
        P_cons = vnx*fx_i + vny*fy_i + vnz*fz_i

        Ekin[i] = T(0.5) * mass * T(v2)
        dq[i]   = dq[i] - T(P_diss + P_sto)
        dU[i]   = dU[i] + T(P_cons)
        vx[i] = T(vnx); vy[i] = T(vny); vz[i] = T(vnz)
    end
    return
end

"""
    vv_velocities_soa!(vx, vy[, vz], f0x, f0y[, f0z], fx, fy[, fz], βx, βy[, βz], dq, dU, Ekin, params, dt)

Velocity-Verlet O step: advance velocities with previous and current forces,
then update the heat and kinetic-energy buffers. The power accounting matches
the entropy production diagnostics in `examples/TwoT_2D_LD_VV.jl`.
"""
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
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        # Mixed precision computations in Float64
        gA = Float64(gamma[i])
        dtA = Float64(dt)
        mA = Float64(mass)
        c  = exp(-gA*dtA/mA)
        r = sqrt((one(T) - c*c) / (T(2)*gA*dtA*mA))
        vpx = Float64(vx[i]); vpy = Float64(vy[i])
        bx  = Float64(beta_x[i]); by = Float64(beta_y[i])
        vnx = c*vpx + r*bx
        vny = c*vpy + r*by
        vbarx = 0.5*(vpx + vnx); vbary = 0.5*(vpy + vny)
        # Same power formulas as VV
        P_diss = - gA * (vbarx*vbarx + vbary*vbary)
        P_sto  = (vbarx*(bx) + vbary*(by)) / dtA
        dq[i] = dq[i] + T(P_diss + P_sto)
        vx[i] = T(vnx); vy[i] = T(vny)
    end
    return
end

function _baoab_OU3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                     beta_x::CuDeviceVector{T}, beta_y::CuDeviceVector{T}, beta_z::CuDeviceVector{T},
                     noise_scale::CuDeviceVector{T},
                     gamma::CuDeviceVector{T}, mass::T, dt::T,
                     dq::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        gA = Float64(gamma[i])
        dtA = Float64(dt)
        mA = Float64(mass)
        c  = exp(-gA*dtA/mA)
        r = sqrt((one(T) - c*c) / (T(2)*gA*dtA*mA))

        vpx = Float64(vx[i]); vpy = Float64(vy[i]); vpz = Float64(vz[i])
        bx  = Float64(beta_x[i]); by = Float64(beta_y[i]); bz = Float64(beta_z[i])
        vnx = c*vpx + r*bx
        vny = c*vpy + r*by
        vnz = c*vpz + r*bz
        vbarx = 0.5*(vpx + vnx); vbary = 0.5*(vpy + vny); vbarz = 0.5*(vpz + vnz)
        P_diss = - gA * (vbarx*vbarx + vbary*vbary + vbarz*vbarz)
        P_sto  = (vbarx*(bx) + vbary*(by) + vbarz*(bz)) / dtA
        dq[i] = dq[i] + T(P_diss + P_sto)
        vx[i] = T(vnx); vy[i] = T(vny); vz[i] = T(vnz)
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
                    Ekin::CuDeviceVector{T}, dU::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        # Mixed precision for robustness
        dtA = Float64(dt); mA = Float64(mass)
        vpx = Float64(vx[i]); vpy = Float64(vy[i])
        fx_i = Float64(fx[i]); fy_i = Float64(fy[i])
        # Half-kick update
        vxi = vpx + (dtA/(2*mA)) * fx_i
        vyi = vpy + (dtA/(2*mA)) * fy_i
        # Conservative power like VV: v · f at end step
        if dtA != 0.0
            P_cons = vxi*fx_i + vyi*fy_i
            dU[i] += T(P_cons)
        end
        vx[i] = T(vxi); vy[i] = T(vyi)
        v2 = vxi*vxi + vyi*vyi
        Ekin[i] = T(0.5) * mass * T(v2)
    end
    return
end

function _baoab_B3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                    mass::T, dt::T,
                    Ekin::CuDeviceVector{T}, dU::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        dtA = Float64(dt); mA = Float64(mass)
        vpx = Float64(vx[i]); vpy = Float64(vy[i]); vpz = Float64(vz[i])
        fx_i = Float64(fx[i]); fy_i = Float64(fy[i]); fz_i = Float64(fz[i])
        vxi = vpx + (dtA/(2*mA)) * fx_i
        vyi = vpy + (dtA/(2*mA)) * fy_i
        vzi = vpz + (dtA/(2*mA)) * fz_i
        if dtA != 0.0
            P_cons = vxi*fx_i + vyi*fy_i + vzi*fz_i
            dU[i] += T(P_cons)
        end
        vx[i] = T(vxi); vy[i] = T(vyi); vz[i] = T(vzi)
        v2 = vxi*vxi + vyi*vyi + vzi*vzi
        Ekin[i] = T(0.5) * mass * T(v2)
    end
    return
end

# ------------------------------------------------------------------
# Public BAOAB wrappers
# ------------------------------------------------------------------

"""
    baoab_BA_2d!(rx, ry, vx, vy, f0x, f0y, params, dt, box)

Combined B/A half-step for BAOAB in 2D: kick velocities by half a force step
and drift positions by `½ Δt`. Mirrors the integrator sequencing in
`examples/TwoT_2D_LD_BAOAB.jl`.
"""
function baoab_BA_2d!(rx, ry, vx, vy, f0x, f0y, params::BAOABParams{T}, dt::T, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA2!(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2])
    k(rx, ry, vx, vy, f0x, f0y, params.mass, dt, box[1], box[2]; threads, blocks)
    return nothing
end

# Conservative power accumulators (match VV formula: P_cons = v · f at end step)
function _cons_power2!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T},
                       fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
                       dU::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        dU[i] += vx[i]*fx[i] + vy[i]*fy[i]
    end
    return
end

function _cons_power3!(vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                       fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                       dU::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(vx); if i > N; return; end
    @inbounds begin
        dU[i] += vx[i]*fx[i] + vy[i]*fy[i] + vz[i]*fz[i]
    end
    return
end

function cons_power_2d!(vx::CuArray{T,1}, vy::CuArray{T,1}, fx::CuArray{T,1}, fy::CuArray{T,1}, dU::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _cons_power2!(vx, vy, fx, fy, dU)
    k(vx, vy, fx, fy, dU; threads, blocks)
    return nothing
end

function cons_power_3d!(vx::CuArray{T,1}, vy::CuArray{T,1}, vz::CuArray{T,1}, fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, dU::CuArray{T,1}) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _cons_power3!(vx, vy, vz, fx, fy, fz, dU)
    k(vx, vy, vz, fx, fy, fz, dU; threads, blocks)
    return nothing
end

"""
3D variant of [`baoab_BA_2d!`](@ref).
"""
function baoab_BA_3d!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params::BAOABParams{T}, dt::T, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_BA3!(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, f0x, f0y, f0z, params.mass, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

"""
    baoab_OU_2d!(vx, vy, βx, βy, params, dt, dq)

Ornstein–Uhlenbeck velocity update (O step) for the BAOAB scheme in 2D. Uses
the same per-particle `gamma` and noise scales as `VVParams`.
"""
function baoab_OU_2d!(vx, vy, beta_x, beta_y, params::BAOABParams{T}, dt::T, dq) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU2!(vx, vy, beta_x, beta_y, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, beta_x, beta_y, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

"""
3D OU step for BAOAB (see [`baoab_OU_2d!`](@ref)).
"""
function baoab_OU_3d!(vx, vy, vz, beta_x, beta_y, beta_z, params::BAOABParams{T}, dt::T, dq) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_OU3!(vx, vy, vz, beta_x, beta_y, beta_z, params.noise_scale, params.gamma, params.mass, dt, dq)
    k(vx, vy, vz, beta_x, beta_y, beta_z, params.noise_scale, params.gamma, params.mass, dt, dq; threads, blocks)
    return nothing
end

"""
    baoab_A_2d!(rx, ry, vx, vy, dt, box)

Pure drift (A step) that advances coordinates by `½ Δt` using the already
updated velocities.
"""
function baoab_A_2d!(rx, ry, vx, vy, dt::T, box::Definitions.Box2{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A2!(rx, ry, vx, vy, dt, box[1], box[2])
    k(rx, ry, vx, vy, dt, box[1], box[2]; threads, blocks)
    return nothing
end

"""
3D drift step (see [`baoab_A_2d!`](@ref)).
"""
function baoab_A_3d!(rx, ry, rz, vx, vy, vz, dt::T, box::Definitions.Box3{T}) where {T<:AbstractFloat}
    N = length(rx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_A3!(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3])
    k(rx, ry, rz, vx, vy, vz, dt, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

"""
    baoab_B_2d!(vx, vy, fx, fy, params, dt, Ekin, dU)

Final B step of BAOAB: kick velocities with the fresh forces and update the
conservative power/kinetic energy accumulators.
"""
function baoab_B_2d!(vx, vy, fx, fy, params::BAOABParams{T}, dt::T, Ekin, dU) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B2!(vx, vy, fx, fy, params.mass, dt, Ekin, dU)
    k(vx, vy, fx, fy, params.mass, dt, Ekin, dU; threads, blocks)
    return nothing
end

"""
3D version of [`baoab_B_2d!`](@ref).
"""
function baoab_B_3d!(vx, vy, vz, fx, fy, fz, params::BAOABParams{T}, dt::T, Ekin, dU) where {T<:AbstractFloat}
    N = length(vx); threads = min(256, N); blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _baoab_B3!(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin, dU)
    k(vx, vy, vz, fx, fy, fz, params.mass, dt, Ekin, dU; threads, blocks)
    return nothing
end

end # module LangevinIntegrators
