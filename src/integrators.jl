export update_parts_LD!
export update_parts_BD!


function update_parts_LD!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},dq::CuVector{T},eₖ::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T},a²::T; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads Langevin!(r, v, f, fR, dq, eₖ, c1s, c2s, c3s, a²)
    return nothing
end

export Langevin!

function Langevin!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},dq::CuDeviceVector{T},eₖ::CuDeviceVector{T}, c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T},a²::T) where {N,T}
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
             dQ  = dq[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = c3 * rnd
             vel_prev = vel

             #Euler-Heaun Method
             #vel_tmp = vel_prev .- (a²*c2) .* vel_prev - (a²*c2) .* frc + (a²*c2) .* rnd_force
             #d_vel =  - (0.5*a²*c2) .* (vel_prev .+ vel_tmp) + (a²*c2) .* rnd_force
             #vel = vel_prev + d_vel
             #pos = pos .+ c2 .* vel

             # Euler-Maruyama method
             vel = c1 .* vel_prev + c2 .* frc + (a²*c2) .* rnd_force
             d_vel =  vel .- vel_prev
             pos = pos .+ c2 .* vel

             dQ = - c2 .* dot(vel_prev,vel_prev) .+ 0.5 * dot((2 .* vel_prev .+ d_vel), c2 .* rnd_force)
             Eₖ = 0.5f0*dot(vel,vel)/a²
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = vel
             noise[gtid] = rnd_force
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end

"""
function update_parts_BD!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}},
     f::CuVector{SVector{N,T}},noises::CuVector{SVector{N,T}},c1s::CuVector{T},
     c2s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads Brownian!(r, v, f, noises, sdot, c1s, c2s, αs)
    return nothing
end

export Brownian!

function Brownian!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T}) where {N,T}
     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos_tmp = r[gtid]
             vel_tmp = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             α   = αs[gtid]
             pos = pos_tmp .+ c1 .* frc + c2 .* rnd
             vel = (pos .- pos_tmp) ./ c1
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = vel
         end
        sync_threads()
     end
     return nothing
end
"""
