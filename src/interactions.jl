export harm_rep


@inline function harm_rep(dr::SVector{N,T}, dist::T, ϵ::T, σ::T) where {N,T}
    inv_dist = T(1.0/dist)
    f_int = T(ϵ*(inv_dist - 1.0/σ))
    e_int = T(0.25*ϵ*(1.0-dist/σ)^2.0)
    return SVector(f_int .* dr) , e_int
end

@inline function wca(dr::SVector{N,T}, dist::T, ϵ::T, σ::T) where {N,T}
    inv_dist = T(1.0/dist)
    r_div_sig = dist/σ
    r_div_sig6 = r_div_sig^6
    r_div_sig12 = r_div_sig6^2
    e_int = 4 * ϵ * (r_div_sig12 - r_div_sig6 + 0.25)
    f_int = -24 * ϵ * r_div_sig * (2 * r_div_sig12 - r_div_sig6) * inv_dist
    return f_int .* dr, e_int
end


export forces!

function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    box::SVector{N,T},
    ϵ::T,
    cut_off::T; nthreads=128) where {N,T}

    kernel = @cuda launch=false forces_kernel!(r, f, Epot, box, ϵ, cut_off, Val(nthreads))
    Npart = size(r,1)
    config = launch_configuration(kernel.fun)

    nthreads = Base.min(Npart, ceil(Int,sqrt(config.threads)))
    nblocks = cld(Npart, nthreads)
    Npart = length(r)
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync kernel(r, f, Epot, box, ϵ, cut_off; threads= nthreads, blocks=nblocks)
    return f, Epot
end

function min_image(dx::T, L_x::T) where T
    return ifelse(abs(dx) > L_x / 2, dx - sign(dx) * L_x ,dx)
end
export forces_kernel!

function forces_kernel!(
    r::CuDeviceVector{T1},
    f::CuDeviceVector{T1},
    Epot::CuDeviceVector{T2},
    box::T1,
    ϵ::T2,
    cut_off::T2,::Val{TH}) where {T1 <: SVector,T2 <: AbstractFloat,TH}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id

    shared_pos = CuStaticSharedArray(T1, TH)
    full_blocks = Npart ÷ blockDim().x
    rest = Npart % blockDim().x

    dim = length(box)
    tile = 0
    acc = zero(T1)
    epot= T2(0.0)
    
    @inbounds begin
        if gtid <= Npart
            pos = r[gtid]
        else
            pos = zero(T1)
        end
        acc = zero(T1)
        epot= T2(0.0)
        for i in 1:full_blocks
            idx = tile * blockDim().x + tid
            shared_pos[tid] = r[idx]
            sync_threads()
            @inbounds for j in 1:blockDim().x
                dr  = pos - shared_pos[j]
                dr = min_image.(dr, box)
                dr² = dot(dr,dr)
                dist  = sqrt(dr²)
                frc, ep = harm_rep(dr,dist, ϵ, cut_off)
                if dist > cut_off || dist == T2(0.0)
                    frc =  zero(T1)
                    ep = T2(0.0)
                end
                acc = acc .+ frc
                epot = epot + ep
            end
            sync_threads()
            tile += 1
        end
        if tid <= rest
            idx = tile * blockDim().x + tid
            shared_pos[tid] = r[idx]
        end
        sync_threads()
        @inbounds for j in 1:rest
            dr  = pos - shared_pos[j]
            dr = min_image.(dr, box)
            dr² = dot(dr,dr)
            dist  = sqrt(dr²)
            frc, ep = harm_rep(dr,dist, ϵ, cut_off)
            if dist >= cut_off || dist == T2(0.0)
                frc =  zero(T1)
                ep = T2(0.0)
            end
            acc = acc .+ frc
            epot = epot + ep
        end
        sync_threads()
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] = epot
        end
    end
    return nothing
end



################################################################################
#                                                                              #
#                   Calculating forces and collision events                    #
#                                                                              #
################################################################################
export collisions!

function collisions!(
    r::CuVector{SVector{N,T}},
    coll::CuMatrix{T},
    coll_switch::CuMatrix{I},
    cut_off::Float32,
    box::SVector{N,T}) where {N,I,T}

    kernel = @cuda launch=false collisions_kernel!(r, coll, coll_switch, cut_off, box)
    Npart = size(r,1)
    config = launch_configuration(kernel.fun)

    nthreads = Base.min(Npart, ceil(Int,sqrt(config.threads)))
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, coll, coll_switch, cut_off, box; threads=(nthreads,nthreads), blocks=(nblocks,nblocks))
    return coll, coll_switch
end


export collisions_kernel!

function collisions_kernel!(
    r::CuDeviceVector{T},
    coll::CuDeviceMatrix{Float32},
    coll_switch::CuDeviceMatrix{I},
    cut_off::Float32,
    box::T) where {T,I}

    Npart = length(r)
    dim   = length(box)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if dim == 2
        @inbounds begin
            if i <= Npart && j <= Npart
                pos₁  = r[i]
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                
                dr² = dx*dx + dy*dy

                dist  = sqrt(dr²)
                if T(0.0) < dist < cut_off
                    if coll_switch[i,j] == 0
                        coll[i,j] = 1
                        coll_switch[i,j] = 1
                    else
                        coll[i,j] = 0
                    end
                elseif dist > cut_off
                    coll_switch[i,j] = 0
                end
            end
        end
    elseif dim == 3
        @inbounds begin
            if i <= Npart && j <= Npart
                pos₁  = r[i]
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]
                dz  = pos₁[3] - pos₂[3]

                dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                dz = ifelse(abs(dz) > box[3] / 2, dz - sign(dz) * box[3] ,dz)
                dr² = dx*dx + dy*dy + dz*dz

                dist  = sqrt(dr²)
                if T(0.0) < dist < cut_off
                    if coll_switch[i,j] == 0
                        coll[i,j] = 1
                        coll_switch[i,j] = 1
                    else
                        coll[i,j] = 0
                    end
                elseif dist > cut_off
                    coll_switch[i,j] = 0
                end
            end
        end
    end
    return nothing
end
