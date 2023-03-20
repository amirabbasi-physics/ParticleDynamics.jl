export neighbor_list!
function neighbor_list!(
    r::CuVector{SVector{N,T}},
    Neighbors::CuMatrix{I},
    neigh_cut_off::T,
    box::SVector{N,T}) where {N,I,T}

    Npart = length(r)
    
    neighbor_matrix = CuArray(zeros(Int32,Npart,Npart))    
    block_dim = (32, 32)
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

function neighbor_list_kernel!(
    neighbor_matrix::CuDeviceMatrix{I},
    Neighbors::CuDeviceMatrix{I}) where I

    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid
    Npart = size(neighbor_matrix, 1)
    @inbounds begin
        if gtid <= Npart
            row = view(neighbor_matrix, gtid:gtid, :)
            tmp_neighbor = view(Neighbors, gtid:gtid, :)
            k = 1 
            for j = 1:length(row)
                if row[j] == 1 && gtid != j 
                    tmp_neighbor[k] = j
                    k += 1
                end
            end
        end
        #sync_threads()
    end
    sync_threads()
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

    # Allocate index array for particle i
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
    sync_threads()
    return nothing
end

export neighbor_list_new!

function neighbor_list_new!(
    r::CuVector{SVector{N,T}},
    Neighbors::CuMatrix{I},
    neigh_cut_off::T,
    box::SVector{N,T}) where {N,I,T}

    Npart = length(r)    
    kernel = @cuda launch = false neighbor_list_kernel!(r, Neighbors, neigh_cut_off, box)
    config = launch_configuration(kernel.fun)
    threads = min(Npart, config.threads)
    blocks = cld(Npart, threads)
    CUDA.@sync kernel(r, Neighbors, neigh_cut_off, box; threads, blocks)
    return nothing
end

export neighbor_list_kernel!
function neighbor_list_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    Neighbors::CuDeviceMatrix{I},
    neigh_cut_off::T,
    box::SVector{N,T}) where {N,I,T}
    # Get global index of current thread
    gtid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    Npart = length(r)
    NNeigh = size(Neighbors,2)
    ncut_off² = neigh_cut_off^2
    # Only operate on valid particle indices
    if gtid <= Npart
        # Initialize counter for this particle
        cnt = 0
        # Loop over all other particles
        for j = 1:Npart
            if cnt > NNeigh
                break
            end
            # Skip self-particle
            if i != j
                pos₁  = r[i]
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]
    
                dx = ifelse(abs(dx) > box[1] / 2, dx - sign(dx) * box[1] ,dx)
                dy = ifelse(abs(dy) > box[2] / 2, dy - sign(dy) * box[2] ,dy)

                # Check if j is within the cutoff distance from i
                dr² = dx*dx + dy*dy

                if 0 < dr² < ncut_off²
                    # Use atomicAdd to increment counter for particle i
                    idx = atomicAdd(Neighbor[gtid, 1:end], 0)
                    Neighbors[gtid, idx] = j
                    cnt += 1
                end
            end
        end
    end
    return nothing
end


