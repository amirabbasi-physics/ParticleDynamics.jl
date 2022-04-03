export forces_fun!

function forces_fun!(r::CuVector{SVector{N,T}}, f₀::CuVector{SVector{N,T}},periodicity::SVector{N,T}, cut_off::T; nthreads=128) where {N,T}
    Npart = length(r)
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads calculate_forces!(r, f₀, cut_off, periodicity)
    return nothing
end
export harm_rep

@inline function harm_rep(dx::T, dy::T, dist::T, k::T, σ::T) where T
    inv_dist = 1.0f0/dist
    f_int = k*(σ*inv_dist - 1.0f0)
    e_int = (1.0f0/2.0f0)*(k/2.0f0)*(σ*inv_dist - 1.0f0)^2.0f0
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y) , f_int
end

export calculate_forces!

function calculate_forces!(r::CuDeviceVector{T}, f::CuDeviceVector{T},
     cut_off::Float32, periodicity::T) where T
    Npart = length(r)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id

    shared = CuStaticSharedArray(T, 128)
    dim = length(periodicity)
    tile = 0
    pos = r[gtid]
    acc = zero(T)
    epot= 0.0f0

    for i in 1:blockDim().x:Npart
        idx = tile * blockDim().x + threadIdx().x
        shared[threadIdx().x] = r[idx]
        sync_threads()

        @inbounds @simd for j in 1:blockDim().x
            dx  = pos[1] - shared[j][1]
            dy  = pos[2] - shared[j][2]
            dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
            dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
            dr² = dx*dx + dy*dy
            dist  = sqrt(dr²)
            if 0.0f0 < dist < cut_off
                k = Float32(1.0e8)
                acc, epot = harm_rep(dx,dy,dist, k, cut_off)
            end
        end
        sync_threads()
        tile += 1
    end
    f[gtid] = acc
    return nothing
end
