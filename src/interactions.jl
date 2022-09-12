export harm_rep2D
export harm_rep3D

@inline function harm_rep2D(dx::T, dy::T, dist::T, ϵ::T, σ::T) where {T}
    inv_dist = 1.0f0/dist
    f_int = ϵ*(inv_dist - 1.0f0/σ)
    e_int = 0.25f0*ϵ*(1.0f0-dist/σ)^2.0f0
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y) , e_int
end


@inline function harm_rep3D(dx::T, dy::T, dz::T, dist::T, ϵ::T, σ::T) where {T}
    inv_dist = 1.0f0/dist
    f_int = ϵ*(inv_dist - 1.0f0/σ)
    e_int = 0.25f0*ϵ*(1.0f0-dist/σ)^2.0f0
    f_x = f_int*dx
    f_y = f_int*dy
    f_z = f_int*dz
    return SVector{3,T}(f_x,f_y,f_z) , e_int
end


export forces!

function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    periodicity::SVector{N,T},
    ϵ::T,
    cut_off::T; nthreads=128) where {N,T}

    Npart = length(r)
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads forces_kernel!(r, f, Epot, periodicity, ϵ, cut_off, Val(nthreads))
    return f, Epot
end

export forces_kernel!

function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{Float32},
    periodicity::T,
    ϵ::Float32,
    cut_off::Float32,::Val{TH}) where {T,TH}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id

    shared_pos = CuStaticSharedArray(T, TH)
    full_blocks = Npart ÷ blockDim().x
    rest = Npart % blockDim().x

    dim = length(periodicity)
    tile = 0
    acc = zero(T)
    epot= 0.0f0

    if dim == 2
        @inbounds begin
            if gtid <= Npart
                pos = r[gtid]
            else
                pos = zero(T)
            end
            acc = zero(T)
            epot= 0.0f0
            for i in 1:full_blocks
                idx = tile * blockDim().x + tid
                shared_pos[tid] = r[idx]
                sync_threads()
                @inbounds for j in 1:blockDim().x
                    dx  = pos[1] - shared_pos[j][1]
                    dy  = pos[2] - shared_pos[j][2]
                    dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                    dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                    dr² = dx*dx + dy*dy
                    dist  = sqrt(dr²)
                    if 0.0f0 < dist < cut_off
                        frc, ep = harm_rep2D(dx,dy,dist, ϵ, cut_off)
                        acc = acc .+ frc
                        epot = epot + ep
                    end
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
                dx  = pos[1] - shared_pos[j][1]
                dy  = pos[2] - shared_pos[j][2]
                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dr² = dx*dx + dy*dy
                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
                    frc, ep = harm_rep2D(dx,dy,dist, ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            if gtid <= Npart
                f[gtid] = acc
                Epot[gtid] = epot
            end
        end
        return nothing
    elseif dim == 3
        @inbounds begin
            if gtid <= Npart
                pos = r[gtid]
            else
                pos = zero(T)
            end
            acc = zero(T)
            epot= 0.0f0
            for i in 1:full_blocks
                idx = tile * blockDim().x + tid
                shared_pos[tid] = r[idx]
                sync_threads()
                @inbounds for j in 1:blockDim().x
                    dx  = pos[1] - shared_pos[j][1]
                    dy  = pos[2] - shared_pos[j][2]
                    dz  = pos[3] - shared_pos[j][3]
                    dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                    dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                    dz = ifelse(abs(dz) > periodicity[3] / 2, dz - sign(dz) * periodicity[3] ,dz)
                    dr² = dx*dx + dy*dy + dz*dz
                    dist  = sqrt(dr²)
                    if 0.0f0 < dist < cut_off
                        frc, ep = harm_rep3D(dx,dy,dz,dist, ϵ, cut_off)
                        acc = acc .+ frc
                        epot = epot + ep
                    end
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
                dx  = pos[1] - shared_pos[j][1]
                dy  = pos[2] - shared_pos[j][2]
                dz  = pos[3] - shared_pos[j][3]
                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dz = ifelse(abs(dz) > periodicity[3] / 2, dz - sign(dz) * periodicity[3] ,dz)
                dr² = dx*dx + dy*dy + dz*dz
                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
                    frc, ep = harm_rep3D(dx,dy,dz,dist, ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            if gtid <= Npart
                f[gtid] = acc
                Epot[gtid] = epot
            end
        end
        return nothing
    end
end



################################################################################
#                                                                              #
#               Calculating forces and collision events                        #
#                                                                              #
################################################################################

export collisions!

function collision!(
    r::CuVector{SVector{N,T}},
    coll::CuMatrix{I},
    coll_switch::CuMatrix{I},
    cut_off::Float32,
    periodicity::SVector{N,T}) where {N,I,T}

    kernel = @cuda launch=false collision_kernel!(r, coll, coll_switch, cut_off, periodicity)
    Npart = size(r,1)
    config = launch_configuration(kernel.fun)

    nthreads = Base.min(Npart, ceil(Int,sqrt(config.threads)))
    nblocks = cld(Npart, nthreads)
    CUDA.@sync kernel(r, coll, coll_switch, cut_off, periodicity; threads=(nthreads,nthreads), blocks=(nblocks,nblocks))
    return coll, coll_switch
end

export collisions_kernel!

function collision_kernel!(
    r::CuDeviceVector{T},
    coll::CuDeviceMatrix{I},
    coll_switch::CuDeviceMatrix{I},
    cut_off::Float32,
    periodicity::T) where {T,I}

    Npart = length(r)
    dim   = length(periodicity)
    coll_event = 1
    coll_event_switch = 1
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if dim == 2
        @inbounds begin
            if i <= Npart && j <= Npart
                pos₁  = r[i]
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dr² = dx*dx + dy*dy

                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
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

                dx = ifelse(abs(dx) > periodicity[1] / 2, dx - sign(dx) * periodicity[1] ,dx)
                dy = ifelse(abs(dy) > periodicity[2] / 2, dy - sign(dy) * periodicity[2] ,dy)
                dz = ifelse(abs(dz) > periodicity[3] / 2, dz - sign(dz) * periodicity[3] ,dz)
                dr² = dx*dx + dy*dy + dz*dz

                dist  = sqrt(dr²)
                if 0.0f0 < dist < cut_off
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
