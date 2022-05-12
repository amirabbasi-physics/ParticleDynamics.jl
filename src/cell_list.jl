"""

export entropy_prod!
export entropy_kernel!

function entropy_prod!(S₀::CuVector{T},Epot₀::CuVector{T},Epot_tmp::CuVector{T},v₀::CuVector{SVector{N,T}},
    v_tmp::CuVector{SVector{N,T}},c2s::CuVector{T},αs::CuVector{T},a²::T; nthreads=128) where {N,T}
    Npart = length(S₀)
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads entropy_kernel!(S₀, Epot₀,Epot_tmp,v₀,v_tmp,c2s,αs,a²,Val(nthreads))
    return nothing
end

function entropy_kernel!(S₀::CuDeviceVector{T},Epot₀::CuDeviceVector{T},Epot_tmp::CuDeviceVector{T},v₀::CuDeviceVector{SVector{N,T}},
    v_tmp::CuDeviceVector{SVector{N,T}},c2s::CuDeviceVector{T},αs::CuDeviceVector{T}, a²::T,::Val{TH}) where {N,T,TH}
    Npart = length(v_tmp)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            vel_tmp = v_tmp[gtid]
            epot_tmp = Epot_tmp[gtid]
            vel = v₀[gtid]
            epot = Epot₀[gtid]
            c2  = c2s[gtid]
            α  = αs[gtid]
            sdt = S₀[gtid]
            sdt += ((0.50f0/a²)*(dot(vel,vel) - dot(vel_tmp,vel_tmp))+(epot - epot_tmp))/(c2*α)
        end

        sync_threads()

        if gtid <= Npart
            S₀[gtid] = sdt
        end
        sync_threads()
    end
    return nothing
end



function entropy_prod!(S₀₂::CuVector{T},v₀::CuVector{SVector{N,T}},f_noise::CuVector{SVector{N,T}},c2s::CuVector{T},
    c3s::CuVector{T}, αs::CuVector{T},a²::T; nthreads=128) where {N,T}
    Npart = length(S₀₂)
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads entropy_kernel!(S₀₂,v₀,f_noise,c2s,c3s,αs,a²,Val(nthreads))
    return nothing
end

function entropy_kernel!(S₀₂::CuDeviceVector{T},v₀::CuDeviceVector{SVector{N,T}},f_noise::CuDeviceVector{SVector{N,T}},
    c2s::CuDeviceVector{T},c3s::CuDeviceVector{T},αs::CuDeviceVector{T}, a²::T,::Val{TH}) where {N,T,TH}
    Npart = length(v₀)
    tid = threadIdx().x
    gtid = (blockIdx().x - UInt32(1)) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            vel = v₀[gtid]
            rnd = f_noise[gtid]
            c2  = c2s[gtid]
            c3  = c3s[gtid]
            α  = αs[gtid]
            sdt = S₀₂[gtid]
            sdt +=  (a²/α) * (-dot(vel,vel)+(c3/c2)*dot(vel,rnd))
        end
        sync_threads()

        if gtid <= Npart
            S₀₂[gtid] = sdt
        end
        sync_threads()
    end
    return nothing
end
"""
