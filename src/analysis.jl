export kinetic!
export kinetic_kernel!

function kinetic!(Ekin0::CuVector{T}, v::CuVector{SVector{N,T}}, a²::T; nthreads=128) where {N,T}
    Npart = length(v)
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads kinetic_kernel!(Ekin0, v ,a²,Val(nthreads))
    return nothing
end


function kinetic_kernel!(Ekin0::CuDeviceVector{T}, v::CuDeviceVector{SVector{N,T}}, a²::T,::Val{TH}) where {N,T,TH}
    Npart = length(v)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            vel = v[gtid]
			kin = Ekin0[gtid]
            kin+= (1.0f0/(2.0f0*a²))*dot(vel,vel)
        end
        sync_threads()

        if gtid <= Npart
            Ekin0[gtid] = kin
        end
        sync_threads()
    end
    return nothing
end
