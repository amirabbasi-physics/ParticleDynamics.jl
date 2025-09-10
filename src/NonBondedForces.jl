module NonBondedForces

using CUDA
using ..Definitions

export lj_forces_soa!

@inline function lj_pair(dx::Float32, dy::Float32, r2::Float32, ϵ::Float32, σ::Float32)
    invr2 = 1f0 / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
    fx = f_over_r*dx  # FIXED: removed negative sign
    fy = f_over_r*dy  # FIXED: removed negative sign
    ep = 4f0*ϵ*(s12 - s6)
    return (fx, fy, ep)
end

@inline function lj_pair(dx::Float32, dy::Float32, dz::Float32, r2::Float32, ϵ::Float32, σ::Float32)
    invr2 = 1f0 / r2
    s2    = (σ*σ) * invr2
    s6    = s2*s2*s2
    s12   = s6*s6
    f_over_r = 24f0*ϵ*(2f0*s12 - s6)*invr2
    fx = f_over_r*dx  # FIXED: removed negative sign
    fy = f_over_r*dy  # FIXED: removed negative sign
    fz = f_over_r*dz  # FIXED: removed negative sign
    ep = 4f0*ϵ*(s12 - s6)
    return (fx, fy, fz, ep)
end

function _lj2_soa_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Float32, Ly::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    @inbounds for k in 1:cap
        j = nbr[i,k]; if j == 0; break; end
        dx = xi - rx[j]; dy = yi - ry[j]
        dx = (2abs(dx) > Lx) ? dx - sign(dx)*Lx : dx
        dy = (2abs(dy) > Ly) ? dy - sign(dy)*Ly : dy
        r2 = dx*dx + dy*dy
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj3_soa_kernel!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    Epot::CuDeviceVector{Float32},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Float32, Ly::Float32, Lz::Float32,
    ϵ::Float32, σ::Float32, cutoff2::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    @inbounds for k in 1:cap
        j = nbr[i,k]; if j == 0; break; end
        dx = xi - rx[j]; dy = yi - ry[j]; dz = zi - rz[j]
        dx = (2abs(dx) > Lx) ? dx - sign(dx)*Lx : dx
        dy = (2abs(dy) > Ly) ? dy - sign(dy)*Ly : dy
        dz = (2abs(dz) > Lz) ? dz - sign(dz)*Lz : dz
        r2 = dx*dx + dy*dy + dz*dz
        if (r2 > 0f0) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += 0.5f0 * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh, box::Definitions.Box2, params::Definitions.LJParams{Float32})
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    cap = nbh.cap
    cutoff2 = params.rcut*params.rcut
    k = CUDA.@cuda launch=false _lj2_soa_kernel!(rx, ry, fx, fy, Epot, nbh.neighbors, cap,
                                                 box[1], box[2], params.ϵ, params.σ, cutoff2)
    CUDA.@sync k(rx, ry, fx, fy, Epot, nbh.neighbors, cap, box[1], box[2], params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                        fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                        Epot::CuArray{Float32,1},
                        nbh, box::Definitions.Box3, params::Definitions.LJParams{Float32})
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    cap = nbh.cap
    cutoff2 = params.rcut*params.rcut
    k = CUDA.@cuda launch=false _lj3_soa_kernel!(rx, ry, rz, fx, fy, fz, Epot, nbh.neighbors, cap,
                                                 box[1], box[2], box[3], params.ϵ, params.σ, cutoff2)
    CUDA.@sync k(rx, ry, rz, fx, fy, fz, Epot, nbh.neighbors, cap, box[1], box[2], box[3], params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

end # module