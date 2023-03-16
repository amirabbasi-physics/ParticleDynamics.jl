export harm_rep2D
export harm_rep3D

@inline function harm_rep2D(dx::T, dy::T, dr²::T, ϵ::T, σ::T) where T
    dist = sqrt(dr²)
    inv_dist = dist^(-1)
    f_int = ϵ*(inv_dist - σ^(-1))
    e_int = f_int*dist*f_int*dist/(4ϵ)
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y), e_int
end


@inline function harm_rep3D(dx::T, dy::T, dz::T, dr²::T, ϵ::T, σ::T) where {T}
    dist = sqrt(dr²)
    inv_dist = dist^(-1)
    f_int = ϵ*(inv_dist - σ^(-1))
    e_int = e_int = f_int*dist*f_int*dist/(4ϵ)
    f_x = f_int*dx
    f_y = f_int*dy
    f_z = f_int*dz
    return SVector{3,T}(f_x,f_y,f_z) , e_int
end




export neighbor_list!

function neighbor_list!(
    r::CuVector{SVector{N,T}},
    Neighbors::CuMatrix{I},
    neigh_cut_off::T,
    box::SVector{N,T}) where {N,I,T}

    Npart = length(r)
    
    neighbor_matrix = CuArray(cu(zeros(I,Npart,Npart)))
    block_dim = (16, 16)
    grid_dim = (div(Npart + block_dim[1] - 1, block_dim[1]), div(Npart + block_dim[2] - 1, block_dim[2]), 1)
    CUDA.@sync @cuda threads=block_dim blocks=grid_dim neighbor_matrix_kernel!(r, neighbor_matrix, neigh_cut_off, box)
    
    kernel = @cuda launch = false neighbor_list_kernel!(neighbor_matrix,Neighbors)
    config = launch_configuration(kernel.fun)
    threads = min(Npart, config.threads)
    blocks = cld(Npart, threads)
    CUDA.@sync kernel(neighbor_matrix,Neighbors; threads, blocks)
    return nothing
end

export neighbor_list_kernel!

"""
function neighbor_list_kernel!(
    neighbor_matrix::CuDeviceMatrix{I},
    Neighbors::CuDeviceMatrix{I}) where I

    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid
    Npart = size(neighbor_matrix, 2)
    @inbounds begin
        if gtid <= Npart
            col = view(neighbor_matrix, :, gtid)
            k = 1 
            for j = 1:length(col)
                if col[j] == 1 && gtid != j 
                    Neighbors[k, gtid] = j
                    k += 1
                end
            end
        end
    end
    return nothing            
end

"""

function neighbor_list_kernel!(
    neighbor_matrix::CuDeviceMatrix{I},
    Neighbors::CuDeviceMatrix{I}) where I

    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid
    Npart = size(neighbor_matrix, 1)
    @inbounds begin
        if gtid <= Npart
            row = view(neighbor_matrix, gtid, :)
            k = 1 
            for j = 1:length(row)
                if row[j] == 1 && gtid != j 
                    Neighbors[gtid,k] = j
                    k += 1
                end
            end
        end
    end
    return nothing            
end


export neighbor_matrix_kernel!

function neighbor_matrix_kernel!(
    r::CuDeviceVector{T1},
    neighbor_matrix::CuDeviceMatrix{I},
    neigh_cut_off::T,
    box::T1) where {T,T1,I}

    Npart = length(r)
    ncut_off² = neigh_cut_off^2

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        if i <= Npart && j <= Npart
            pos₁  = r[i]
            pos₂  = r[j]
            dx  = pos₁[1] - pos₂[1]
            dy  = pos₁[2] - pos₂[2]

            dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
            dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
            
            dr² = dx*dx + dy*dy

            if 0 < dr² < ncut_off²
                neighbor_matrix[i,j] = 1
            end
        end
    end
    return nothing
end

export forces!

function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    box::SVector{N,T},
    ϵ::T,
    cut_off::T; nthreads=128) where {N,T}

    Npart = length(r)
    nblocks = ceil(Int, Npart/nthreads)
    CUDA.@sync @cuda blocks=nblocks threads=nthreads forces_kernel!(r, f, Epot, box, ϵ, cut_off, Val(nthreads))
    return nothing
end

function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    box::T,
    ϵ::T1,
    cut_off::T1,::Val{TH}) where {T,T1,TH}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # global thread id

    shared_pos = CuStaticSharedArray(T, TH)
    full_blocks = Npart ÷ blockDim().x
    rest = Npart % blockDim().x
    cut_off² = cut_off^2
    dim = length(box)
    tile = 0
    acc = zero(T)
    epot= zero(T1)

    if dim == 2
        @inbounds begin
            if gtid <= Npart
                pos = r[gtid]
            else
                pos = zero(T)
            end
            acc = zero(T)
            epot= zero(T1)
            for i in 1:full_blocks
                idx = tile * blockDim().x + tid
                shared_pos[tid] = r[idx]
                sync_threads()
                @inbounds for j in 1:blockDim().x
                    dx  = pos[1] - shared_pos[j][1]
                    dy  = pos[2] - shared_pos[j][2]
                    dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                    dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                    dr² = dx*dx + dy*dy
                    if 0 < dr² < cut_off²
                        frc, ep = harm_rep2D(dx,dy,dr², ϵ, cut_off)
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
                dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                dr² = dx*dx + dy*dy 
                if 0 < dr² < cut_off²
                    frc, ep = harm_rep2D(dx,dy,dr², ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            if gtid <= Npart
                f[gtid] = acc
                Epot[gtid] += epot
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
            epot= zero(T1)
            for i in 1:full_blocks
                idx = tile * blockDim().x + tid
                shared_pos[tid] = r[idx]
                sync_threads()
                @inbounds for j in 1:blockDim().x
                    dx  = pos[1] - shared_pos[j][1]
                    dy  = pos[2] - shared_pos[j][2]
                    dz  = pos[3] - shared_pos[j][3]
                    dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                    dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                    dz = ifelse(abs(dz) > box[3] / 2, dz - sign(dz) * box[3] ,dz)
                    dr² = dx*dx + dy*dy + dz*dz
                    if 0 < dr² < cut_off²
                        frc, ep = harm_rep3D(dx,dy,dz,dr², ϵ, cut_off)
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
                dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
                dz = ifelse(abs(dz) > box[3] / 2, dz - sign(dz) * box[3] ,dz)
                dr² = dx*dx + dy*dy + dz*dz
                if 0 < dr² < cut_off²
                    frc, ep = harm_rep3D(dx,dy,dz,dr², ϵ, cut_off)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
            sync_threads()
            if gtid <= Npart
                f[gtid] = acc
                Epot[gtid] += epot
            end
        end
        return nothing
    end
end

function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    box::SVector{N,T},
    ϵ::T,
    cut_off::T) where {N,T,I}

    #Npart = length(r)


    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, box, ϵ, cut_off)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, box, ϵ, cut_off; threads, blocks)

    return nothing
end

"""
function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    cut_off::T1) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,1)

    cut_off² = cut_off^2
    acc = zero(T)
    epot= zero(T1)

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
        else
            pos₁ = zero(T)
        end
        acc = zero(T)
        epot= zero(T1)


        @inbounds for j = 1:NNeigh
            idx = Neighbors[j,gtid]
            if (idx != 0 && idx <= Npart)
                pos₂  = r[idx]
            else
                break
            end
            dx  = pos₁[1] - pos₂[1]
            dy  = pos₁[2] - pos₂[2]

            dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
            dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
            dr² = dx*dx + dy*dy

            if 0 < dr² < cut_off²
                frc, ep = harm_rep2D(dx, dy, dr², ϵ, cut_off)
                acc = acc .+ frc
                epot = epot + ep
            end
        end
        sync_threads()
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
        end
    end
    return nothing
end


function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    cut_off::T1) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    shared_pos = CuStaticSharedArray(T, 40)

    cut_off² = cut_off^2
    acc = zero(T)
    epot= zero(T1)

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
        else
            pos₁ = zero(T)
        end
        acc = zero(T)
        epot= zero(T1)


        @inbounds for j = 1:NNeigh
            idx = Neighbors[gtid,j]
            if (idx != 0 && idx <= Npart)
                shared_pos[j]  = r[idx]
            else
                break
            end

            dx  = pos₁[1] - shared_pos[j][1]
            dy  = pos₁[2] - shared_pos[j][2]

            dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
            dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)
            dr² = dx*dx + dy*dy

            if 0 < dr² < cut_off²
                frc, ep = harm_rep2D(dx, dy, dr², ϵ, cut_off)
                acc = acc .+ frc
                epot = epot + ep
            end
        end
        sync_threads()
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
        end
    end
    return nothing
end


"""

function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    cut_off::T1) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    
    cut_off² = cut_off^2
    acc = zero(T)
    epot= zero(T1)

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
        else
            pos₁ = zero(T)
        end
        acc = zero(T)
        epot= zero(T1)


        @inbounds for j = 1:NNeigh
            idx = Neighbors[gtid,j]
            if idx != 0 
                pos₂  = r[idx]
            else
                break
            end
            dx  = pos₁[1] - pos₂[1]
            dy  = pos₁[2] - pos₂[2]

            dx = ifelse(2abs(dx) > box[1] , dx - sign(dx) * box[1] ,dx)
            dy = ifelse(2abs(dy) > box[2] , dy - sign(dy) * box[2] ,dy)

            #dx = rem(dx + box[1]/2, box[1]) - box[1]/2
            #dy = rem(dy + box[2]/2, box[2]) - box[2]/2

            dr² = dx*dx + dy*dy

            if 0 < dr² < cut_off²
                frc, ep = harm_rep2D(dx, dy, dr², ϵ, cut_off)
                acc = acc .+ frc
                epot = epot + ep
            end
        end
        #sync_threads()
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
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

function collisions_backup!(
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

function collisions!(
    r::CuVector{SVector{N,T}},
    coll::CuMatrix{T},
    coll_switch::CuMatrix{I},
    cut_off::T,
    box::SVector{N,T}) where {N,I,T}

    Npart = length(r)
    block_dim = (32, 32)
    grid_dim = (div(Npart + block_dim[1] - 1, block_dim[1]), div(Npart + block_dim[2] - 1, block_dim[2]), 1)
    CUDA.@sync @cuda threads=block_dim blocks=grid_dim collisions_kernel!(r, coll, coll_switch, cut_off, box)
    return nothing
end

export collisions_kernel!

function collisions_kernel!(
    r::CuDeviceVector{T},
    coll::CuDeviceMatrix{T1},
    coll_switch::CuDeviceMatrix{I},
    cut_off::T1,
    box::T) where {T,T1,I}
    Npart = length(r)
    dim   = length(box)
    cut_off²=cut_off^2
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
    
                if zero(T1) < dr² < cut_off²
                    if coll_switch[i,j] == 0
                        coll[i,j] += 1
                        coll_switch[i,j] = 1
                    else
                        coll[i,j] += 0
                    end
                elseif dr² > cut_off²
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

                if zero(T1) < dr² < cut_off²
                    if coll_switch[i,j] == 0
                        coll[i,j] = 1
                        coll_switch[i,j] = 1
                    else
                        coll[i,j] = 0
                    end
                elseif dr² > cut_off²
                    coll_switch[i,j] = 0
                end
            end
        end
    end
    return nothing
end
