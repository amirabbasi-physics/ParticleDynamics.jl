export update_parts_LD!
export update_parts_BD!

function update_parts_LD!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    noise::CuVector{SVector{N,T}},sdot::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},
    c3s::CuVector{T},αs::CuVector{T},a::T; nthreads=128) where {N,T}
    Npart = UInt32(length(r))



    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads Langevin!(r, v, f, noise, sdot,c1s, c2s, c3s, αs, a)
    return nothing
end

export Langevin!

function Langevin!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},sdot::CuDeviceVector{T},c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T},
     αs::CuDeviceVector{T}, a::T) where {N,T}
     Npart = length(r)

     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             c3  = c3s[gtid]
             α   = αs[gtid]
             sdt = sdot[gtid]
             pos = pos .+ c2 .* vel
             vel = c1 .* vel .+ c3 .* rnd .+ c2 .* frc
             sdt = (a*dot(vel,vel)- dot(c3 .* rnd,vel))/α
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = vel
             sdot[gtid] = sdt
         end
        sync_threads()
     end
     return nothing
end


function update_parts_BD!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}},
     f::CuVector{SVector{N,T}},noises::CuVector{SVector{N,T}}, sdot::CuVector{T},c1s::CuVector{T},
     c2s::CuVector{T},αs::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))

    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads Brownian!(r, v, f, noises, sdot, c1s, c2s, αs)
    return nothing
end


export Brownian!

function Brownian!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},sdot::CuDeviceVector{T},c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T},αs::CuDeviceVector{T}) where {N,T}
     Npart = length(r)

     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             α   = αs[gtid]
             sdt = sdot[gtid]
             pos = pos .+ c1 .* vel
             vel = frc .+ c2 .* rnd
             sdt = (dot(vel,vel)- dot(c2 .* rnd,vel))/α
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = vel
             sdot[gtid] = sdt
         end
        sync_threads()
     end
     return nothing
end
