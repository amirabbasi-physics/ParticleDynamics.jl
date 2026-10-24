# Fuse the first kick and drift where no constraint or diagnostic stage needs
# the intermediate state. Keep the arithmetic identical to the staged kernels.
function _nve_kick_drift3_kernel!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
                                  vx::CuDeviceVector{T}, vy::CuDeviceVector{T}, vz::CuDeviceVector{T},
                                  fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
                                  c::T, dt::T, box) where {T}
    i = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        ux = muladd(c, fx[i], vx[i])
        uy = muladd(c, fy[i], vy[i])
        uz = muladd(c, fz[i], vz[i])
        vx[i] = ux; vy[i] = uy; vz[i] = uz
        x = muladd(dt, ux, rx[i])
        y = muladd(dt, uy, ry[i])
        z = muladd(dt, uz, rz[i])
        Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
        x -= (x >= T(0.5)*Lx)*Lx; x += (x < -T(0.5)*Lx)*Lx
        y -= (y >= T(0.5)*Ly)*Ly; y += (y < -T(0.5)*Ly)*Ly
        z -= (z >= T(0.5)*Lz)*Lz; z += (z < -T(0.5)*Lz)*Lz
        rx[i] = x; ry[i] = y; rz[i] = z
    end
    return
end



