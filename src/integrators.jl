"""
#####################################################################################
#               Positions and velocities update for Verlet-type algorithm           #
#####################################################################################
export update_positions!

function update_positions!(
    r::CuVector{SVector{N,T}},
    v::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    c1s::CuVector{T},
    c2s::CuVector{T},
    c3s::CuVector{T};
    nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(r, v, f, fR, c1s, c2s, c3s)
    return nothing
end


export update_positions_kernel!

function update_positions_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    c1s::CuDeviceVector{T},
    c2s::CuDeviceVector{T},
    c3s::CuDeviceVector{T}) where {N,T}

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
         end
        sync_threads()
     end
     return nothing
end



export update_velocities!

function update_velocities!(
    v::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T},
    c1s::CuVector{T},
    c2s::CuVector{T},
    c3s::CuVector{T};
     nthreads=128) where {N,T}
    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_kernel!(v, f₀, f, fR, dq, eₖ, c1s, c2s, c3s)
    return nothing
end


export update_velocities_kernel!

function update_velocities_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T},
    c₁::CuDeviceVector{T},
    c₂::CuDeviceVector{T},
    c₃::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             frc_prev = f₀[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c2  = c₂[gtid]
             c3  = c₃[gtid]
             dQ  = dq[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = c3 .* rnd
             a = c1*c2
             aa = (1.f0 - 0.5f0*a) / (1.f0 + 0.5f0*a)
             bb = 1.f0 / (1.f0 + 0.5f0*a)
             v_next = aa .* v_prev + (0.5f0*a*aa) .* frc_prev + (0.5f0*a) .* frc + (bb*a) .* rnd_force

             injected_energy = 0.5f0 * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.5f0 *dot((v_prev .+ v_next) ,v_next) #works nicely!

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
"""

export update_positions!

function update_positions!(
    r::CuVector{SVector{N,T}},
    v::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    dq::CuVector{T},
    c1s::CuVector{T},
    c2s::CuVector{T},
    c3s::CuVector{T};
    nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(r, v, f, fR,dq, c1s, c2s, c3s)
    return nothing
end


export update_positions_kernel!

function update_positions_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    c1s::CuDeviceVector{T},
    c2s::CuDeviceVector{T},
    c3s::CuDeviceVector{T}) where {N,T}

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

             rnd_force = c3 .* rnd
             a = c1*c2
             bb = 1.0f0 / (1.0f0 + 0.50f0*a)
             bbdt = bb*c2
             d_pos = bbdt .* vel + (0.50f0*bbdt*a) .* frc + (0.50f0*bbdt*a) .* rnd_force
             pos = pos + d_pos
             dQ = (-dot(d_pos,vel) + dot(d_pos, rnd_force)) /c2
         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
             dq[gtid] = dQ
         end
        sync_threads()
     end
     return nothing
end



export update_velocities!

function update_velocities!(
    v::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    eₖ::CuVector{T},
    c1s::CuVector{T},
    c2s::CuVector{T},
    c3s::CuVector{T};
     nthreads=128) where {N,T}
    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_kernel!(v, f₀, f, fR, eₖ, c1s, c2s, c3s)
    return nothing
end


export update_velocities_kernel!

function update_velocities_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    eₖ::CuDeviceVector{T},
    c₁::CuDeviceVector{T},
    c₂::CuDeviceVector{T},
    c₃::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             frc_prev = f₀[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c2  = c₂[gtid]
             c3  = c₃[gtid]
             Eₖ   = eₖ[gtid]

             rnd_force = c3 .* rnd
             a = c1*c2
             aa = (1.f0 - 0.5f0*a) / (1.f0 + 0.5f0*a)
             bb = 1.f0 / (1.f0 + 0.5f0*a)
             v_next = aa .* v_prev + (0.5f0*a*aa) .* frc_prev + (0.5f0*a) .* frc + (bb*a) .* rnd_force

             Eₖ = 0.5f0*dot(v_next,v_next)/c1
         end
         sync_threads()
         if gtid <= Npart
             v[gtid]  = v_next
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end


#####################################################################################
#####################################################################################
#               Positions and velocities update for leap-frog algorithm             #
#####################################################################################
#####################################################################################

function update_positions!(
    r  ::CuVector{SVector{N,T}},
    v  ::CuVector{SVector{N,T}},
    c₂ ::CuVector{T};
    nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    nblocks=ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_kernel!(r, v, c₂)
    return nothing
end

function update_positions_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    c₂::CuDeviceVector{T} ) where {N,T}

     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel = v[gtid]
             c2  = c₂[gtid]

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


function update_velocities!(
    v::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T},
    c₁::CuVector{T},
    c₂::CuVector{T},
    c₃::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(v))
    nblocks=ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks= nblocks threads=nthreads update_velocities_kernel!(v, f₀, fR, dq, eₖ, c₁, c₂, c₃)
    return nothing
end


function update_velocities_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T},
    c₁::CuDeviceVector{T},
    c₂::CuDeviceVector{T},
    c₃::CuDeviceVector{T}) where {N,T}

    Npart = length(v)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c2  = c₂[gtid]
             c3  = c₃[gtid]
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

#####################################################################################
#####################################################################################
#              Positions and velocities update for Melchionna algorithm             #
#####################################################################################
#####################################################################################

function update_positions_ml!(
    r  ::CuVector{SVector{N,T}},
    v  ::CuVector{SVector{N,T}},
    c₂ ::CuVector{T};
    nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    nblocks=ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_ml_kernel!(r, v, c₂)
    return nothing
end

function update_positions_ml_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    c₂::CuDeviceVector{T} ) where {N,T}

     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos = r[gtid]
             vel = v[gtid]
             c2  = c₂[gtid]

             pos = pos + c2 .* vel
         end
         sync_threads()

         if gtid <= Npart
             r[gtid] = pos
         end
        sync_threads()
     end
     return nothing
end


export update_velocities_ml₁!

function update_velocities_ml₁!(
    v::CuVector{SVector{N,T}},
    vˢ::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    c1s::CuVector{T},
    c3s::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_ml₁_kernel!(v, vˢ, f, fR, c1s, c3s)
    return nothing
end


export update_velocities_ml₁_kernel!

function update_velocities_ml₁_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    vˢ::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    c₁::CuDeviceVector{T},
    c₃::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             vs     = vˢ[gtid]
             frc_prev = f₀[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c3  = c₃[gtid]

             rnd_force = c3 .* rnd

             v_next = c1 .* v_prev .+ (1.0f0 - c1) .* frc_prev .+ rnd_force

         end
         sync_threads()
         if gtid <= Npart
             vˢ[gtid] = v_next
         end
        sync_threads()
     end
     return nothing
end



export update_velocities_ml₂!

function update_velocities_ml₂!(
    v::CuVector{SVector{N,T}},
    vˢ::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T},
    c₁::CuVector{T},
    c₂::CuVector{T},
    c₃::CuVector{T},
    a::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_ml₂_kernel!(v, vˢ, f, fR, dq, eₖ, c₁, c₂, c₃, a)
    return nothing
end


export update_velocities_ml₂_kernel!

function update_velocities_ml₂_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    vˢ::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T},
    c₁::CuDeviceVector{T},
    c₂::CuDeviceVector{T},
    c₃::CuDeviceVector{T},
    a::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             vs     = vˢ[gtid]
             frc_prev = f₀[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c2  = c₂[gtid]
             c3  = c₃[gtid]
             aa  = a[gtid]

             rnd_force = c3 .* rnd

             v_next = c1 .* vs .+ (1.0f0 - c1) .* frc_prev .+ rnd_force

             injected_energy   = 0.50f0 * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.50f0* dot((v_prev .+ v_next) ,v_next)

             dQ = injected_energy + dissipated_energy

             Eₖ = 0.5f0*dot(v_next,v_next)/aa

         end
         sync_threads()
         if gtid <= Npart
             v[gtid] = v_next
             vˢ[gtid] = vs
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end




#####################################################################################
#####################################################################################
##              Positions and velocities update for Euler-Maruyama algorithm       ##
#####################################################################################
#####################################################################################

export update_parts_EM!


function update_parts_EM!(r::CuVector{SVector{N,T}}, v::CuVector{SVector{N,T}}, f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},dq::CuVector{T},eₖ::CuVector{T},c1s::CuVector{T},c2s::CuVector{T},c3s::CuVector{T}; nthreads=128) where {N,T}
    Npart = UInt32(length(r))
    blocks=ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_parts_EM_kernel!(r, v, f, fR, dq, eₖ, c1s, c2s, c3s)
    return nothing
end

export update_parts_EM_kernel!

function update_parts_EM_kernel!(r::CuDeviceVector{SVector{N,T}}, v::CuDeviceVector{SVector{N,T}}, f::CuDeviceVector{SVector{N,T}},
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


             a = c1*c2
             v_next = (1.0f0-a).* v_prev .+ a .* frc .+ a .* rnd_force
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
function update_parts_BD!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    noise::CuVector{SVector{N,T}},
    c1s::CuVector{T},
    c2s::CuVector{T};
    nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    CUDA.@sync @cuda blocks=ceil(Int, Npart/nthreads) threads=nthreads Brownian!(r, f, noises, sdot, c1s, c2s, αs)
    return nothing
end

export Brownian!

function Brownian!(
    r::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    c1s::CuDeviceVector{T},
    c2s::CuDeviceVector{T}) where {N,T}

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
#####################################################################################
#        Positions and velocities update for Bussi-Parrinello algorithm             #
#####################################################################################
#####################################################################################


export update_velocities_bp₁!

function update_velocities_bp₁!(
    v::CuVector{SVector{N,T}},
    vˢ::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    c₁::CuVector{T},
    c₃::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_bp₁_kernel!(v, vˢ, fR, c₁, c₃)
    return nothing
end


export update_velocities_bp₁_kernel!

function update_velocities_bp₁_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    vˢ::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    c₁::CuDeviceVector{T},
    c₃::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             rnd    = noise[gtid]
             c1     = c₁[gtid]
             c3     = c₃[gtid]

             vs = c1 .* v_prev .+ c3 .* rnd
         end
         sync_threads()

         if gtid <= Npart
             vˢ[gtid] = vs
         end
        sync_threads()
     end
     return nothing
end


export update_positions_bp!

function update_positions_bp!(
    r::CuVector{SVector{N,T}},
    vˢ::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    c₂::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(r))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_positions_bp_kernel!(r, vˢ, f, c₂)
    return nothing
end


export update_positions_bp_kernel!

function update_positions_bp_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    vˢ::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    c₂::CuDeviceVector{T}) where {N,T}

     Npart = length(r)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             pos        = r[gtid]
             vs         = vˢ[gtid]
             frc_prev   = f[gtid]
             c2         = c₂[gtid]

             pos = pos + c2 .* vs + (0.5f0*c2*c2) .* frc_prev

         end
         sync_threads()
         if gtid <= Npart
             r[gtid] = pos
         end
        sync_threads()
     end
     return nothing
end


export update_velocities_bp₂!

function update_velocities_bp₂!(
    v::CuVector{SVector{N,T}},
    vˢ::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    fR::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T},
    c₁::CuVector{T},
    c₂::CuVector{T},
    c₃::CuVector{T},
    a::CuVector{T};
     nthreads=128) where {N,T}

    Npart = UInt32(length(v))
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads update_velocities_bp₂_kernel!(v, vˢ, f₀, f, fR, dq, eₖ, c₁, c₂, c₃, a)
    return nothing
end


export update_velocities_bp₂_kernel!

function update_velocities_bp₂_kernel!(
    v::CuDeviceVector{SVector{N,T}},
    vˢ::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    noise::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T},
    c₁::CuDeviceVector{T},
    c₂::CuDeviceVector{T},
    c₃::CuDeviceVector{T},
    a::CuDeviceVector{T}) where {N,T}

     Npart = length(v)
     tid = threadIdx().x
     gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

     @inbounds begin
         if gtid <= Npart
             v_prev = v[gtid]
             vs     = vˢ[gtid]
             frc_prev = f₀[gtid]
             frc = f[gtid]
             rnd = noise[gtid]
             c1  = c₁[gtid]
             c2  = c₂[gtid]
             c3  = c₃[gtid]
             aa  = a[gtid]

             rnd_force = c3 .* rnd
             v_m = vs .+ (0.5f0*c2) .* (frc .+ frc_prev)
             v_next = c1 .* v_m .+ rnd_force

             injected_energy   = 0.50f0 * dot((v_prev .+ v_next), rnd_force)
             dissipated_energy = -0.50f0* dot((v_prev .+ v_next) ,v_next)

             dQ = injected_energy + dissipated_energy

             Eₖ = 0.5f0*dot(v_next,v_next)/aa

         end
         sync_threads()
         if gtid <= Npart
             v[gtid] = v_next
             dq[gtid] = dQ
             eₖ[gtid] = Eₖ
         end
        sync_threads()
     end
     return nothing
end
