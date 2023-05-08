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
        pos₁  = r[gtid]
        # Initialize counter for this particle
        idx = 0
        # Loop over all other particles
        @inbounds for j = 1:Npart
            if idx > NNeigh
                break
            end
            # Skip self-particle
            if gtid != j
                
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]
    
                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy

                # Check if j is within the cutoff distance from i
                dr² = dx*dx + dy*dy

                if 0 < dr² < ncut_off²
                    idx += 1
                    Neighbors[gtid, idx] = j    
                end
            end
        end
    end
    return nothing
end