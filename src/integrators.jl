export update_parts_LD!
export update_parts_BD!


function update_parts_LD!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},dq::CuVector{T},eₖ::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads Langevin_kernel!(r, v, f, fR, dq, eₖ, c1s, c2s, c3s)
    return nothing
end

export update_positions!
export update_velocities!

function update_positions!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(r, v, f, fR, c1s, c2s, c3s)
    return nothing
end

function update_velocities!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},fR::CuVector{SVector{N,T}},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T};
     nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_kernel!(r, v, f₀, f, fR, c1s, c2s, c3s)
    return nothing
end

export update_positions_kernel!
export update_velocities_kernel!

function update_positions_kernel!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}}, c1s::CuDeviceVector{T},c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T}) where {N,T}
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

             rnd_force = (c2*c3) .* rnd

             b = 1.0f0 / (1.0f0 + 0.50f0*c1*c2)
             bdt = b*c2
             pos = pos .+ bdt .* vel .+ (0.50f0*bdt*c1*c2) .* frc .+ (0.50f0*bdt*c1) .* rnd_force
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
         end
        sync_threads()
     end
     return nothing
end

function update_velocities_kernel!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}}, noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T}, eₖ::CuDeviceVector{T}, c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T},
     c3s::CuDeviceVector{T}) where {N,T}
     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel_prev = v[gtid]
             frc_prev = f₀[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             c3  = c3s[gtid]
             dQ  = dq[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = (c2*c3) .* rnd
             a = (1.0f0 - 0.50f0*c1*c2) / (1.0f0 + 0.50f0*c1*c2)
             b = 1.0f0 / (1.0f0 + 0.50f0*c1*c2)
             vel_next = a .* vel_prev .+ (0.5f0*c1*c2) .* (a .* frc_prev .+ frc) .+ (b*c1) .* rnd_force

             dQ = - c2 .* dot(vel_prev,vel_prev) .+ 0.5f0 * dot((vel_prev .+ vel_next), c2 .* rnd_force)
             Eₖ = 0.5f0*dot(vel_next,vel_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             v[gtid]  = vel_next
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end


export Langevin_kernel!

function Langevin_kernel!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}},dq::CuDeviceVector{T},eₖ::CuDeviceVector{T}, c1s::CuDeviceVector{T},
     c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T}) where {N,T}
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

             rnd_force = c3 .* rnd
             vel_prev = vel

             # Euler-Maruyama method
             a = c1*c2
             vel_next = (1.0f0-a).* vel_prev .+ a .* frc .+ a .* rnd_force
             d_vel =  vel_next .- vel_prev
             pos = pos .+ c2 .* vel_prev

             dQ = - c2 .* dot(vel_prev,vel_prev) .+ 0.5f0 * dot((2.0f0 .* vel_prev .+ d_vel), c2 .* rnd_force)
             Eₖ = 0.5f0*dot(vel_next,vel_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = vel_next
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
