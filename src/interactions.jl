export forces!

function forces!(r::CuVector{SVector{N,T}}, f₀::CuVector{SVector{N,T}}, Eₚ₀::CuVector{T},periodicity::SVector{N,T}, ϵ::T, cut_off::T;nthreads=128) where {N,T}
    Npart = length(r)
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads forces_kernel!(r, f₀, Eₚ₀, cut_off, periodicity, ϵ)
    return f₀
end

export harm_rep2D
export harm_rep3D

@inline function harm_rep2D(dx::T, dy::T, dist::T, ϵ::T, σ::T) where {T}
    inv_dist = 1.0f0/dist
    f_int = ϵ*(inv_dist - 1.0f0/σ)
    e_int = 0.25f0*ϵ*(1.0f0-dist/σ)^2.0f0
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y) , e_int
end

@inline function harm_rep3D(dx::T, dy::T, dz::T, dist::T, ϵ::T, σ::T) where {T}
    inv_dist = 1.0f0/dist
    f_int = ϵ*(inv_dist - 1.0f0/σ)
    e_int = 0.25f0*ϵ*(1.0f0-dist/σ)^2.0f0
    f_x = f_int*dx
    f_y = f_int*dy
    f_z = f_int*dz
    return SVector{3,T}(f_x,f_y,f_z) , e_int
end

export forces_kernel!

function forces_kernel!(r::CuDeviceVector{T}, f::CuDeviceVector{T},Eₚ₀::CuDeviceVector{Float32},
     cut_off::Float32, periodicity::T, ϵ::Float32) where {T}
    Npart = length(r)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id
    tid = threadIdx().x
    shared = CuStaticSharedArray(T, 128)
    dim = length(periodicity)
    tile = 0
    pos = r[gtid]
    acc = zero(T)
    epot= 0.0f0

    if dim == 2
        for i in 1:blockDim().x:Npart
            idx = tile * blockDim().x + tid
            shared[tid] = r[idx]
            sync_threads()

            @inbounds for j in 1:blockDim().x
                dx  = pos[1] - shared[j][1]
                dy  = pos[2] - shared[j][2]
                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dr² = dx*dx + dy*dy
                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
                    frc, ep = harm_rep2D(dx,dy,dist, ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            tile += 1
        end
        f[gtid] = acc
        Eₚ₀[gtid] = epot
        return nothing
    elseif dim == 3
        for i in 1:blockDim().x:Npart
            idx = tile * blockDim().x + tid
            shared[tid] = r[idx]
            sync_threads()

            @inbounds for j in 1:blockDim().x
                dx  = pos[1] - shared[j][1]
                dy  = pos[2] - shared[j][2]
                dz  = pos[3] - shared[j][3]
                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dz = ifelse(abs(dz) > periodicity[3] / 2, dz - sign(dz) * periodicity[3] ,dz)
                dr² = dx*dx + dy*dy + dz*dz
                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
                    frc, ep = harm_rep3D(dx,dy,dz,dist, ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            tile += 1
        end
        f[gtid] = acc
        Eₚ₀[gtid] = epot
        return nothing
    end
    return nothing
end
