export update_parts_LD
export update_parts_BD

function update_parts_LD(coords::CuVector{SVector{N,T}}, vels::CuVector{SVector{N,T}}, frcs::CuVector{SVector{N,T}},
    noises::CuVector{SVector{N,T}},c1s::CuVector{T},c2s::CuVector{T},
    c3s::CuVector{T},αs::CuVector{T},a::T; nthreads=128) where {N,T}
    Npart = UInt32(length(coords))

    r_d = similar(coords)
    v_d = similar(coords)
    sdot_d = similar(αs)

    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads calculate_state!(coords, vels, frcs, noises, c1s, c2s, c3s, αs, a, r_d, v_d, sdot_d)
    return r_d, v_d, sdot_d
end

function update_parts_BD(coords::CuVector{SVector{N,T}}, vels::CuVector{SVector{N,T}}, frcs::CuVector{SVector{N,T}},
    noises::CuVector{SVector{N,T}},c1s::CuVector{T},c2s::CuVector{T},
    c3s::CuVector{T},αs::CuVector{T},a::T; nthreads=128) where {N,T}
    Npart = UInt32(length(coords))

    r_d = similar(coords)
    v_d = similar(coords)
    sdot_d = similar(αs)

    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads Brownian!(coords, vels, frcs, noises, c1s, c2s, αs, r_d, v_d, sdot_d)
    return r_d, v_d, sdot_d
end





export calculate_state!

function calculate_state!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T},
     αs::CuDeviceVector{T}, a::T,rr::CuDeviceVector{SVector{N,T}}, vv::CuDeviceVector{SVector{N,T}}, ssdot::CuDeviceVector{T}) where {N,T}
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
             sdt = ssdot[gtid]
             vel = c1 .* vel .+ c3 .* rnd .+ c2 .* frc
             pos = pos .+ c2 .* vel
             sdt = (a*dot(vel,vel)- dot(c3 .* rnd,vel))/α
         end
         sync_threads()
         if gtid <= Npart
             rr[gtid] = pos
             vv[gtid] = vel
             ssdot[gtid] = sdt
         end
        sync_threads()
     end
     return nothing
end


export Brownian!

function Brownian!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T},αs::CuDeviceVector{T},
     rr::CuDeviceVector{SVector{N,T}}, vv::CuDeviceVector{SVector{N,T}}, ssdot::CuDeviceVector{T}) where {N,T}
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
             sdt = ssdot[gtid]
             vel = frc .+ c2 .* rnd
             pos = pos .+ c1 .* vel
             sdt = (dot(vel,vel)- dot(c2 .* rnd,vel))/α
         end
         sync_threads()
         if gtid <= Npart
             rr[gtid] = pos
             vv[gtid] = vel
             ssdot[gtid] = sdt
         end
        sync_threads()
     end
     return nothing
end
