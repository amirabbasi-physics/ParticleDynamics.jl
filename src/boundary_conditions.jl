export PBC_kernel!
export PBC
function PBC_kernel!(r::CuDeviceVector{T}, r_d::CuDeviceVector{T}, periodicity::T,::Val{TH}) where {T,TH}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            pos = mod.(pos, periodicity)
#        else
#            pos = zero(T)
        end
        sync_threads()
        if gtid <= Npart
            r_d[gtid] = pos
        end
    end
    return nothing
end

function PBC!(coords::CuVector{T}, periodicity::T; nthreads=128) where T
    Npart = length(coords)
    r_d = similar(coords)
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads PBC_kernel!(coords, r_d, periodicity,Val(nthreads))
    return r_d
end
