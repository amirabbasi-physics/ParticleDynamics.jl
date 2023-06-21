export harm_rep
export WCA

#checked!
@inline function harm_rep(dx::T, dy::T, dr²::T, ϵ::T, σ::T) where T
    dist = dr²^(1/2)
    f_int = ϵ*(1/dist - 1/σ)
    e_int = (ϵ/2)*(1 - dist/σ)^2
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y), e_int
end


#checked!
@inline function harm_rep(dx::T, dy::T, dz::T, dr²::T, ϵ::T, σ::T) where T
    dist = dr²^(1/2)
    f_int = ϵ*(1/dist - 1/σ)
    e_int = (ϵ/2)*(1 - dist/σ)^2
    f_x = f_int*dx
    f_y = f_int*dy
    f_z = f_int*dz
    return SVector{3,T}(f_x,f_y, f_z), e_int
end

# checked!
@inline function WCA(dx::T, dy::T, dr²::T, ϵ::T, σ::T) where T
    inv_dr² = 1/dr²
    σ² = σ^2
    σ²_inv_dr² = σ²*inv_dr²
    σ6_inv_dr6 = σ²_inv_dr²^3
    σ12_inv_dr12 = σ6_inv_dr6^2
    f_int = 24ϵ*(2σ12_inv_dr12 - σ6_inv_dr6)*inv_dr²
    e_int = 4ϵ*(2σ12_inv_dr12 - σ6_inv_dr6) + ϵ
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y), e_int
end

#checked!
@inline function WCA(dx::T, dy::T, dz::T, dr²::T, ϵ::T, σ::T) where T
    inv_dr² = 1/dr²
    σ² = σ^2
    σ²_inv_dr² = σ²*inv_dr²
    σ6_inv_dr6 = σ²_inv_dr²^3
    σ12_inv_dr12 = σ6_inv_dr6^2
    f_int = 24ϵ*(2σ12_inv_dr12 - σ6_inv_dr6)*inv_dr²
    e_int = 4ϵ*(2σ12_inv_dr12 - σ6_inv_dr6) + ϵ
    f_x = f_int*dx
    f_y = f_int*dy
    f_y = f_int*dz
    return SVector{3,T}(f_x,f_y, f_z), e_int
end

export forces!
export forces_kernel!




################################################################################
#                                                                              #
#                           Calculating forces                                 #
#                           USING NEIGHBORLIST                                 #
#                                                                              #
################################################################################

"""
# backupcode
#checked!
function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    box::SVector{N,T},
    ϵ::T,
    σ::T) where {N,T,I}


    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, box, ϵ, σ)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, box, ϵ, σ; threads, blocks)

    return nothing
end

#checked!

function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    σ::T1) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    
    cut_off = T1(2^(1/6))*σ
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

        for j = 1:NNeigh
            idx = Neighbors[gtid,j]
            if idx != 0 
                pos₂  = r[idx]
            else
                break
            end
            dx  = pos₁[1] - pos₂[1]
            dy  = pos₁[2] - pos₂[2]
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
            
            dr² = dx*dx + dy*dy

            if  0 < dr² < cut_off²
                frc, ep = WCA(dx, dy, dr², ϵ, σ)
                acc += frc
                epot = epot + ep
            end
        end
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
        end
    end
    return nothing
end
"""

# Modify forces! function to accept force calculation function as argument
function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    box::SVector{N,T},
    ϵ::T,
    σ::T,
    force_func::Function) where {N,T,I}

    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, box, ϵ, σ, force_func)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, box, ϵ, σ, force_func; threads, blocks)

    return nothing
end

# Modify forces_kernel! to use force_func instead of WCA
function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    σ::T1,
    force_func::Function) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    
    if force_func == WCA
        cut_off = T1(2^(1/6))*σ
        cut_off² = cut_off^2
    elseif force_func == harm_rep
        cut_off = T1(σ)
        cut_off² = cut_off^2
    end
    

    """
    cut_off = T1(2^(1/6))*σ
    cut_off² = cut_off^2
    """

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

        for j = 1:NNeigh
            idx = Neighbors[gtid,j]
            if idx != 0 
                pos₂  = r[idx]
            else
                break
            end
            dx  = pos₁[1] - pos₂[1]
            dy  = pos₁[2] - pos₂[2]
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
            
            dr² = dx*dx + dy*dy

            if  0 < dr² < cut_off²
                frc, ep = force_func(dx, dy, dr², ϵ, σ) # Call the passed function here
                acc += frc
                epot = epot + ep
            end
        end
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
        end
    end
    return nothing
end

################################################################################
#                                                                              #
#                   Calculating forces and collisions                          #
#                           USING NEIGHBORLIST                                 #
#                                                                              #
################################################################################

"""
#checked!
function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    cold_num::I,
    colls::CuVector{T},
    coll_switch::CuMatrix{Bool},
    box::SVector{N,T},
    ϵ::T,
    σ::T) where {N,T,I}


    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, cold_num, colls, coll_switch, box, ϵ, σ)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, cold_num, colls, coll_switch, box, ϵ, σ; threads, blocks)

    return nothing
end


#checked!
function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    num_cold::Int,
    colls::CuDeviceVector{T1},
    coll_switch::CuDeviceMatrix{Bool},
    box::T,
    ϵ::T1,
    σ::T1) where {T,T1,I}

    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    cut_off = T1(2^(1/6))*σ
    cut_off² = cut_off^2

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
            acc = zero(T)
            epot= zero(T1)
            coll = zero(T1)

            @inbounds for j = 1:NNeigh
                idx = Neighbors[gtid,j]
                if idx != 0 
                    pos₂  = r[idx]
                else
                    break
                end

                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy

                dr² = dx*dx + dy*dy

                if dr² > cut_off²
                    coll_switch[gtid,j] = false
                else
                    if !coll_switch[gtid,j]                    
                        if gtid <= num_cold
                            if idx > num_cold
                                coll += one(T1)
                                coll_switch[gtid,j] = true
                            end
                        else
                            if idx <= num_cold
                                coll += one(T1)
                                coll_switch[gtid,j] = true
                            end
                        end
                    end
                    frc, ep = WCA(dx, dy, dr², ϵ, σ)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
              
            f[gtid] = acc
            Epot[gtid] += epot
            colls[gtid] += coll 
        end
    end
    return nothing
end
"""


# Modify launcher to include force calculation function as an argument
function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    cold_num::I,
    colls::CuVector{T},
    coll_switch::CuMatrix{Bool},
    box::SVector{N,T},
    ϵ::T,
    σ::T,
    force_func::Function) where {N,T,I}

    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, cold_num, colls, coll_switch, box, ϵ, σ, force_func)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, cold_num, colls, coll_switch, box, ϵ, σ, force_func; threads, blocks)

    return nothing
end

# Modify kernel to include force calculation function as an argument
function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    num_cold::Int,
    colls::CuDeviceVector{T1},
    coll_switch::CuDeviceMatrix{Bool},
    box::T,
    ϵ::T1,
    σ::T1,
    force_func::Function) where {T,T1,I}

    
    Npart = length(r)
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    NNeigh = size(Neighbors,2)

    
    if force_func == WCA
        cut_off = T1(2^(1/6))*σ
        cut_off² = cut_off^2
    elseif force_func == harm_rep
        cut_off = T1(σ)
        cut_off² = cut_off^2
    end
    
    """
    cut_off = T1(2^(1/6))*σ
    cut_off² = cut_off^2
    """

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
            acc = zero(T)
            epot= zero(T1)
            coll = zero(T1)

            @inbounds for j = 1:NNeigh
                idx = Neighbors[gtid,j]
                if idx != 0 
                    pos₂  = r[idx]
                else
                    break
                end

                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy

                dr² = dx*dx + dy*dy

                if dr² > cut_off²
                    coll_switch[gtid,j] = false
                else
                    if !coll_switch[gtid,j]                    
                        if gtid <= num_cold
                            if idx > num_cold
                                coll += one(T1)
                                coll_switch[gtid,j] = true
                            end
                        else
                            if idx <= num_cold
                                coll += one(T1)
                                coll_switch[gtid,j] = true
                            end
                        end
                    end
                    frc, ep = force_func(dx, dy, dr², ϵ, σ)
                    acc = acc .+ frc
                    epot = epot + ep
                end
            end
              
            f[gtid] = acc
            Epot[gtid] += epot
            colls[gtid] += coll 
        end
    end
    return nothing
end




# NOT CHECKED!!!!!!!

################################################################################
#                                                                              #
#                   Calculating forces and collisions                          #
#                           WITHOUT NEIGHBORLIST                               #
#                                                                              #
################################################################################
export collisions!

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
    
                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
                
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
                        frc, ep = harm_rep(dx,dy,dr², ϵ, cut_off)
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
                    frc, ep = harm_rep(dx,dy,dr², ϵ, cut_off)
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