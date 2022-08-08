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

function update_positions!(dr::CuVector{SVector{N,T}},r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(dr, r, v, f, fR, c1s, c2s, c3s)
    return nothing
end

function update_velocities!(dr::CuVector{SVector{N,T}},v::CuVector{SVector{N,T}}, f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},fR::CuVector{SVector{N,T}},dq::CuVector{T},eₖ::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T};
     nthreads=128) where {N,T}
    Npart = UInt32(length(v))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_kernel!(dr, v, f₀, f, fR, dq, eₖ, c1s, c2s, c3s)
    return nothing
end


export update_positions_kernel!
export update_velocities_kernel!



#####################################################################################
#               Positions and velocities update for Verlet-type algorithm           #
#####################################################################################
function update_positions_kernel!(dr::CuDeviceVector{SVector{N,T}},r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
     noise::CuDeviceVector{SVector{N,T}}, c1s::CuDeviceVector{T},c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T}) where {N,T}
     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             d_pos = dr[gtid]
             pos = r[gtid]
             vel = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             c3  = c3s[gtid]

             rnd_force = c3 .* rnd
             a = c1*c2
             bb = 1.0f0 / (1.0f0 + 0.50f0*a)
             bbdt = bb*c2
             d_pos = bbdt .* vel + (0.50f0*bbdt*a) .* frc + (0.50f0*bbdt*a) .* rnd_force
             pos = pos + d_pos
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             dr[gtid] = d_pos
         end
        sync_threads()
     end
     return nothing
end

function update_velocities_kernel!(dr::CuDeviceVector{SVector{N,T}},v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}}, noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T}, eₖ::CuDeviceVector{T}, c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T},
     c3s::CuDeviceVector{T}) where {N,T}
     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             d_pos = dr[gtid]
             v_prev = v[gtid]
             frc_prev = f₀[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             c3  = c3s[gtid]
             dQ  = dq[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = c3 .* rnd
             a = c1*c2
             aa = (1.0f0 - 0.50f0*a) / (1.0f0 + 0.50f0*a)
             bb = 1.0f0 / (1.0f0 + 0.50f0*a)
             v_next = aa .* v_prev + (0.0f0*0.5f0*a*aa) .* frc_prev + (0.0f0*0.5f0*a) .* frc + (bb*a) .* rnd_force

             injected_energy = 0.50f0 * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.5f0 *dot((v_prev .+ v_next) ,v_prev) #works nicely!

             dQ = injected_energy + dissipated_energy
             Eₖ = 0.5f0*dot(v_next,v_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             v[gtid]  = v_next
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end

#####################################################################################
#               Positions and velocities update for Euler-Maruyama algorithm        #
#####################################################################################
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
             v_prev = vel

             # Euler-Maruyama method
             a = c1*c2
             v_next = (1.0f0-a).* v_prev .+ (0.0f0*a) .* frc .+ a .* rnd_force
             pos = pos .+ c2 .* v_next

             injected_energy   = 0.50f0  * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.50f0 * dot((v_prev .+ v_next) ,v_prev) #works nicely!

             dQ = injected_energy + dissipated_energy
             Eₖ = 0.50f0*dot(v_next,v_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             v[gtid] = v_next
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


#####################################################################################
#               Positions and velocities update for leap-frog algorithm             #
#####################################################################################
function update_positions!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}},c2s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(r, v, c2s)
    return nothing
end

function update_velocities!(v::CuVector{SVector{N,T}}, f₀::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},dq::CuVector{T},eₖ::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T};
     nthreads=128) where {N,T}
    Npart = UInt32(length(v))
    nblocks = Npart ÷ nthreads
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_kernel!(v, f₀, fR, dq, eₖ, c1s, c2s, c3s)
    return nothing
end
function update_positions_kernel!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}},c2s::CuDeviceVector{T} ) where {N,T}
     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel = v[gtid]
             c2  = c2s[gtid]

             pos = pos + (0.50f0*c2) .* vel
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
         end
        sync_threads()
     end
     return nothing
end

function update_velocities_kernel!(v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}}, noise::CuDeviceVector{SVector{N,T}},dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T}, c1s::CuDeviceVector{T}, c2s::CuDeviceVector{T}, c3s::CuDeviceVector{T}) where {N,T}
     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c1s[gtid]
             c2  = c2s[gtid]
             c3  = c3s[gtid]
             dQ  = dq[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = c3 .* rnd
             a = c1*c2

             v_next = (1.0f0-a) .* v_prev + a .* frc + a .* rnd_force

             injected_energy   = 0.50f0 * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.50f0*dot((v_prev .+ v_next) ,v_prev) #works nicely!

             dQ = injected_energy + dissipated_energy
             Eₖ = 0.5f0*dot(v_next,v_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             v[gtid]  = v_next
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end
