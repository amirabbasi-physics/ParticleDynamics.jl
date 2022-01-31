export forces_fun
function forces_fun(coords::CuVector{SVector{N,T}}, periodicity::SVector{N,T}, cut_off::T; nthreads=128) where {N,T}
    Npart = UInt32(length(coords))

    f_d = similar(coords)
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads calculate_forces!(coords, f_d, cut_off, periodicity)
    return f_d
end



export calculate_forces!

function calculate_forces!(r::CuDeviceVector{T}, f::CuDeviceVector{T}, cut_off::Float32, periodicity::T) where T
    Npart = UInt32(length(r))
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id
    #shared = CuStaticSharedArray(T, 128)
    shared = @cuStaticSharedMem(T,128)
    tile = 0
    pos = r[gtid]
    acc = zero(T)
    f[gtid] = acc
    for i in 1:blockDim().x:Npart
        idx = tile * blockDim().x + threadIdx().x
        shared[threadIdx().x] = r[idx]
        sync_threads()

        @inbounds @simd for j in 1:blockDim().x
            dr = shared[j]-pos
            dr = mod.(dr,periodicity)
            dist = sqrt(sum(abs2, dr))
            if dist < cut_off
                k = Float32(2.5e3)
                inv_dist = 1.0f0/(dist+Float32(1e-7))
                f_int = k*(cut_off*inv_dist-1.0f0)
                acc -= f_int * dr
            end
        end
        sync_threads()
        tile += 1
    end
    f[gtid] = acc
    return nothing
end
