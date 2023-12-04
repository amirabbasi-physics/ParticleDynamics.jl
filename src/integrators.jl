#####################################################################################
#####################################################################################
##              Positions and velocities update for Euler-Maruyama algorithm       ##
#####################################################################################
#####################################################################################

"""
export update_particles_em!

function update_particles_em!(
    r::CuVector{SVector{N,T}}, 
    v::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false em_kernel!(r, v, f, dq, eₖ, c1, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, f, dq, eₖ, c1, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end

export em_kernel!

function em_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}}, 
    f::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T1}, 
    c1::T1,
    dt::T1,
    c3s::CuDeviceVector{T}, 
    box::SVector{N,T}) where {N,T,T1}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            v_prev = v[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            Eₖ   = eₖ[gtid]
            rnd = @SVector randn(T1,N)

            rnd_force = T1(c3) .* T1.(rnd)
            a = T1(c1*dt)
            v_next = (1-a) .* v_prev .+ dt .* (frc .+ rnd_force)
            pos = pos .+ dt .* v_next

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!
            r[gtid] = T.(pos)
            v[gtid] = T.(v_next)

            injected_energy = dot((T1.(v_prev) .+ T1.(v_next)), rnd_force)/2
            dissipated_energy = - c1*dot(T1.(v_prev) ,T1.(v_prev))*(1-a/2)

            dQ += -(injected_energy + dissipated_energy)               # Minus sign indicates the dQ of the heat bath
            Eₖ += dot(T1.(v_next),T1.(v_next))/2
            
            dq[gtid] = T(dQ)
            eₖ[gtid] = Eₖ
        end
    end
    return nothing
end
"""
#####################################################################################
#               Positions and velocities update for Verlet-type algorithm           #
#####################################################################################
export update_positions_vv!

function update_positions_vv!(
    r::CuVector{SVector{N,T}},
    v::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    f_r::CuVector{SVector{N,T}},
    c1::T1,
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false update_positions_kernel_vv!(r, v, f,f_r, c1, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, f, f_r, c1, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end

export update_positions_kernel_vv!


function update_positions_kernel_vv!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    f_r::CuDeviceVector{SVector{N,T}},
    c1::T1,
    dt::T1,
    c3s::CuDeviceVector{T},
    box::SVector{N,T}) where {N,T,T1}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            vel = v[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            rnd = @SVector randn(T1,N)

            rnd_force = T1(c3) .* rnd
            a = c1*dt
            bb = (1 / (1 + a/2))
            bbdt = bb*dt
            d_pos = bbdt .* vel .+ (bbdt*dt/2) .* (frc .+ rnd_force)
            pos = pos .+ d_pos
            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!

            r[gtid] = T.(pos)
            f_r[gtid] = T.(rnd)
        end

     end
     return nothing
end



export update_velocities_vv!

function update_velocities_vv!(
    v::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    f_r::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuVector{T}) where {N,T,T1}

    kernel = @cuda launch=false update_velocities_kernel_vv!(v, f₀, f, f_r, dq, eₖ, c1, dt, c3s)

    Npart = length(v)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(v, f₀, f, f_r, dq, eₖ, c1, dt, c3s; threads=nthreads, blocks=nblocks)
    return nothing
end


export update_velocities_kernel_vv!

function update_velocities_kernel_vv!(
    v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    f_r::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuDeviceVector{T}) where {N,T,T1}

    Npart = length(v)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            v_prev = v[gtid]
            frc_prev = f₀[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            Eₖ   = eₖ[gtid]
            rnd = f_r[gtid]

            rnd_force = T1(c3) .* T1.(rnd)
            a = c1*dt
            bb = 1 / (1 + a/2)
            aa = (1 - a/2) * bb
            
            v_next = aa .* T1.(v_prev) + (dt*aa/2) .* frc_prev + (dt/2) .* frc + (bb*dt) .* rnd_force
            v[gtid]  = T.(v_next)
            injected_energy = dot((T1.(v_prev) .+ v_next), rnd_force)/(2bb)
            dissipated_energy = - c1*dot(T1.(v_prev) ,T1.(v_prev))

            dQ += -(injected_energy + dissipated_energy)                 # Minus sign indicates the dQ of the heat bath
            Eₖ += dot(T1.(v_next),T1.(v_next))/2
            
            dq[gtid] = T(dQ)
            eₖ[gtid] = Eₖ
        end

     end
     return nothing
end



function update_velocities_vv!(
    v::CuVector{SVector{N,T}},
    f₀::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    f_r::CuVector{SVector{N,T}},
    dq::CuVector{T},
    du::CuVector{T},
    eₖ::CuVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuVector{T}) where {N,T,T1}

    kernel = @cuda launch=false update_velocities_kernel_vv!(v, f₀, f, f_r, dq, du, eₖ, c1, dt, c3s)

    Npart = length(v)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(v, f₀, f, f_r, dq, du, eₖ, c1, dt, c3s; threads=nthreads, blocks=nblocks)
    return nothing
end

function update_velocities_kernel_vv!(
    v::CuDeviceVector{SVector{N,T}},
    f₀::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    f_r::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    du::CuDeviceVector{T},
    eₖ::CuDeviceVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuDeviceVector{T}) where {N,T,T1}

    Npart = length(v)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id

    @inbounds begin
        if gtid <= Npart
            v_prev = v[gtid]
            frc_prev = f₀[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            dU  = du[gtid]
            Eₖ   = eₖ[gtid]
            rnd = f_r[gtid]

            rnd_force = T1(c3) .* T1.(rnd)
            a = c1*dt
            bb = 1 / (1 + a/2)
            aa = (1 - a/2) * bb
            
            v_next = aa .* T1.(v_prev) + (dt*aa/2) .* frc_prev + (dt/2) .* frc + (bb*dt) .* rnd_force
            v[gtid]  = T.(v_next)
            injected_energy = dot((T1.(v_prev) .+ v_next), rnd_force)/(2bb)
            dissipated_energy = - c1*dot(T1.(v_next) ,T1.(v_next))


            dQ += -(injected_energy + dissipated_energy)                 # Minus sign indicates the dQ of the heat bath
            dU += dot(v_next, T1.(frc))
            Eₖ += dot(T1.(v_next),T1.(v_next))/2
            
            dq[gtid] = T(dQ)
            du[gtid] = T(dU)
            eₖ[gtid] = Eₖ
        end

     end
     return nothing
end




#####################################################################################
#####################################################################################
#               Positions and velocities update for leap-frog algorithm             #
#####################################################################################
#####################################################################################
#This integrator has problem: The problem is that it does not update postions of the particles
#on the upper right of the box in a 2D simulation.

export update_positions_lf!
function update_positions_lf!(
    r::CuVector{SVector{N,T}},
    v::CuVector{SVector{N,T}},
    dt::T1,
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false update_positions_kernel_lf!(r, v, dt, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, dt, box; threads=nthreads, blocks=nblocks)
    return nothing
end

export update_positions_kernel_lf!
function update_positions_kernel_lf!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}},
    dt::T1,
    box::SVector{N,T}) where {N,T,T1}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            vel = v[gtid]
            pos = T1.(pos) .+ (dt/2) .* T1.(vel)
            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2 
            r[gtid] = T.(pos)
        end
    end
    return nothing
end

export update_velocities_lf!
function update_velocities_lf!(
    v::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    dq::CuVector{T},
    eₖ::CuVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuVector{T}) where {N,T,T1}

    kernel = @cuda launch=false update_velocities_kernel_lf!(v, f, dq, eₖ, c1, dt, c3s)

    Npart = length(v)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(v, f, dq, eₖ, c1, dt, c3s; threads=nthreads, blocks=nblocks)
    return nothing
end

export update_velocities_kernel_lf!
function update_velocities_kernel_lf!(
    v::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    eₖ::CuDeviceVector{T1},
    c1::T1,
    dt::T1,
    c3s::CuDeviceVector{T}) where {N,T,T1}

    Npart = length(v)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            v_prev = v[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            Eₖ   = eₖ[gtid]
            rnd = @SVector randn(T1,N)
            rnd_force = T1(c3) .* rnd
            a = c1*dt
            v_next = (1-a) .* T1.(v_prev) .+ dt .* (frc .+ rnd_force)
            v[gtid]  = T.(v_next)
            injected_energy = dot((T1.(v_prev) .+ T1.(v_next)), rnd_force)/2
            dissipated_energy = - c1*dot(T1.(v_prev) ,T1.(v_prev))*(1-a/2)

            dQ += -(injected_energy + dissipated_energy)                 # Minus sign indicates the dQ of the heat bath
            Eₖ += dot(T1.(v_next),T1.(v_next))/2
            
            dq[gtid] = T(dQ)
            eₖ[gtid] = Eₖ
        end
    end
    return nothing
end




function update_particles_em!(
    r::CuVector{SVector{N,T}}, 
    v::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    dq::CuVector{T},
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false em_kernel!(r, v, f, dq, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, f, dq, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end

function em_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}}, 
    f::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    dt::T1,
    c3s::CuDeviceVector{T},
    box::SVector{N,T}) where {N,T,T1}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            rnd = @SVector randn(T1,N)

            rnd_force = T1(c3) .* rnd
            v_next = (frc .+ rnd_force)
            
            pos = pos .+ dt .* v_next

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!
            r[gtid] = T.(pos)
            v[gtid] = T.(v_next)

            
            dQ += dot(T1.(frc), v_next) 
            dq[gtid] = T(dQ)
        end
    end
    return nothing
end



function update_particles_mem!(
    r::CuVector{SVector{N,T}}, 
    v::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    dq::CuVector{T},
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false mem_kernel!(r, v, f, dq, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, f, dq, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end

function mem_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}}, 
    f::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    dt::T1,
    c3s::CuDeviceVector{T}, 
    box::SVector{N,T}) where {N,T,T1}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            v_prev = v[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]
            rnd1 = @SVector randn(T1,N)
            rnd2 = @SVector randn(T1,N)
            rnd = 0.5 .* (rnd1 .+ rnd2)
            rnd_force = T1(c3) .* rnd
            v_next = (frc .+ rnd_force)
            
            pos = pos .+ dt .* v_next

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!
            r[gtid] = T.(pos)
            v[gtid] = T.(v_next)

            """
            injected_energy = dot(T1.(v_next), rnd_force)
            
            dissipated_energy = -dot(T1.(v_next) ,T1.(v_next))

            dQ += -(injected_energy + dissipated_energy)
            

            if frc !== zero(frc)
                dQ += dot(T1.(frc),T1.(frc))- 2T1(5.0e7) + (T1(sqrt(200)) * dot(T1.(frc), rnd)) / dt
            end
            """
            #dQ += (T1(sqrt(2)) * dot(T1.(frc), rnd)) / dt 
            dQ += dot(T1.(frc), rnd_force)
            dq[gtid] = T(dQ)
        end
    end
    return nothing
end



function predictor_Heun!(
    r::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    f_r::CuVector{SVector{N,T}},
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false predictor_Heun_kernel!(r, f, f_r, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, f, f_r, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end


function predictor_Heun_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    f::CuDeviceVector{SVector{N,T}},
    f_r::CuDeviceVector{SVector{N,T}},
    dt::T1,
    c3s::CuDeviceVector{T}, 
    box::SVector{N,T}) where {N,T,T1}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            frc = f[gtid]
            c3  = c3s[gtid]

            rnd = @SVector randn(T1,N)

            rnd_force = T1(c3) .* T1.(rnd)
            v_next = (frc .+ rnd_force)
            pos = pos .+ dt .* v_next

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!

            r[gtid] = T.(pos)
            f_r[gtid] = T1.(rnd)
        end
    end
    return nothing
end


function corrector_Heun!(
    r₀::CuVector{SVector{N,T}},
    r::CuVector{SVector{N,T}}, 
    v::CuVector{SVector{N,T}}, 
    f₀::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    f_r::CuVector{SVector{N,T}}, 
    dq::CuVector{T},
    dt::T1,
    c3s::CuVector{T},
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false corrector_Heun_kernel!(r₀, r, v, f₀, f, f_r, dq, dt, c3s, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r₀, r, v, f₀, f, f_r, dq, dt, c3s, box; threads=nthreads, blocks=nblocks)
    return nothing
end

function corrector_Heun_kernel!(
    r₀::CuDeviceVector{SVector{N,T}},
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}}, 
    f₀::CuDeviceVector{SVector{N,T}}, 
    f::CuDeviceVector{SVector{N,T}},
    f_r::CuDeviceVector{SVector{N,T}},
    dq::CuDeviceVector{T},
    dt::T1,
    c3s::CuDeviceVector{T}, 
    box::SVector{N,T}) where {N,T,T1}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    @inbounds begin
        if gtid <= Npart
            pos = r₀[gtid]
            frc = f[gtid]
            v_prev = v[gtid]
            frc_prev = f₀[gtid]
            rnd = f_r[gtid]
            c3  = c3s[gtid]
            dQ  = dq[gtid]

            rnd_force = T1(c3) .* T1.(rnd)
            v_next = T1.(frc) .+ rnd_force
            pos = pos .+ dt .* v_next

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!

            r[gtid] = T.(pos)
            v[gtid] = T.(v_next)

            #dQ += dot(T1.(frc), T1.(frc))

            dQ += dot(T1.(frc), rnd_force)
            dq[gtid] = T(dQ)
        end
    end
    return nothing
end




#####################################################################################
#####################################################################################
#                APMO Positions and velocities update for EM algorithm              #
#                                                                                   #
#####################################################################################
#####################################################################################


"""

function update_particles_em!(
    r::CuVector{SVector{N,T}}, 
    v::CuVector{SVector{N,T}}, 
    f::CuVector{SVector{N,T}},
    r_pseu::CuVector{SVector{N,T}}, 
    v_pseu::CuVector{SVector{N,T}}, 
    dQ::CuVector{T},
    dt::T1,
    c2::T1,
    c3::T1,
    box::SVector{N,T}) where {N,T,T1}

    kernel = @cuda launch=false em_kernel!(r, v, f, r_pseu, v_pseu, dQ, dt, c2, c3, box)

    Npart = length(r)
    config = launch_configuration(kernel.fun)
    nthreads = Base.min(Npart, config.threads)
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, v, f, r_pseu, v_pseu, dQ, dt, c2, c3, box; threads=nthreads, blocks=nblocks)
    return nothing
end

function em_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    v::CuDeviceVector{SVector{N,T}}, 
    f::CuDeviceVector{SVector{N,T}},
    r_pseu::CuDeviceVector{SVector{N,T}},
    v_pseu::CuDeviceVector{SVector{N,T}}, 
    dq::CuDeviceVector{T},
    dt::T1,
    c2::T1,
    c3::T1, 
    box::SVector{N,T}) where {N,T,T1}
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    sqrt_2 = T1(1.41421356237)
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
            v_prev = v[gtid]
            frc = f[gtid]
            pos_pseu = r_pseu[gtid]
            v_pseu_prev = v_pseu[gtid]

            dQ  = dq[gtid]

            rnd = @SVector randn(T1,N)
            rnd_pseu = @SVector randn(T1,N)

            rnd_force = sqrt_2 .* rnd
            rnd_force_pseu = T1(c3) .* rnd_pseu
            v_next = (frc .- c2 .* (pos .- pos_pseu) .+ rnd_force)
            v_next_pseu = (c2 .* (pos .- pos_pseu) .+ rnd_force_pseu)
            
            pos = pos .+ dt .* v_next
            pos_pseu = pos_pseu .+ dt .* v_next_pseu

            rel_dist = pos .- pos_pseu

            pos = mod.(pos .+ box ./ 2, box) .- box ./ 2                    # Applying PBC!
            rel_dist = mod.(rel_dist .+ box ./ 2, box) .- box ./ 2 

            pos_pseu = pos .- rel_dist

            r[gtid] = T.(pos)
            v[gtid] = T.(v_next)

            r_pseu[gtid] = T.(pos_pseu)
            v_pseu[gtid] = T.(v_next_pseu)

            #injected_energy = dot(T1.(v_next), rnd_force)
            
            #dissipated_energy = -dot(T1.(v_next) ,T1.(v_next))

            #dQ += -(injected_energy + dissipated_energy)

            
            #if frc !== zero(frc)
            #    dQ += dot(T1.(frc),T1.(frc))- 2T1(5.0e7) + (T1(sqrt(2)) * dot(T1.(frc), rnd)) / dt
            #end
            
            #dQ += (T1(sqrt(2)) * dot(T1.(frc), rnd)) / dt 
            dq[gtid] = T(dQ)
        end
    end
    return nothing
end

"""