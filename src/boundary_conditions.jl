export PBC_kernel!
export PBC!


function PBC_kernel!(r::CuDeviceVector{T}, box::T,::Val{TH}) where {T,TH}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2
            #pos = mod.(pos, box)
#        else
#            pos = zero(T)
        end
        sync_threads()
        if gtid <= Npart

            r[gtid] = pos
        end
        sync_threads()
    end
    return nothing
end

function PBC!(coords::CuVector{T}, box::T; nthreads=128) where T
    Npart = length(coords)
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads PBC_kernel!(coords, box,Val(nthreads))
    return nothing
end
