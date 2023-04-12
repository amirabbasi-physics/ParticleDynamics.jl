export PBC_kernel!
export PBC!


function PBC_kernel!(r::CuDeviceVector{SVector{N,T}}, box::SVector{N,T}) where {N,T}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2
        end
        if gtid <= Npart
            r[gtid] = pos
        end
    end
    return nothing
end

function PBC!(r::CuVector{SVector{N,T}}, box::SVector{N,T}) where {N,T}
    Npart = length(r)    
    kernel = @cuda launch = false PBC_kernel!(r, box)
    config = launch_configuration(kernel.fun)
    threads = min(Npart, config.threads)
    blocks = cld(Npart, threads)
    CUDA.@sync kernel(r, box; threads, blocks)
    return nothing
end
