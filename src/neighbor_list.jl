export neighbor_list!
export neighbor_list_kernel!


function neighbor_list!(
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



function neighbor_list_kernel!(
    r::CuDeviceVector{SVector{N,T}},
    Neighbors::CuDeviceMatrix{I},
    neigh_cut_off::T,
    box::SVector{N,T}) where {N,I,T}

    gtid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    Npart = length(r)
    NNeigh = size(Neighbors,2)
    ncut_off² = neigh_cut_off^2
    dim = length(box)
    if gtid <= Npart
        pos₁  = r[gtid]
        idx = 0
        # Loop over all other particles
        if dim == 2
            @inbounds for j = 1:gtid-1
                if idx > NNeigh
                    break
                end
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy

                dr² = dx*dx + dy*dy

                if 0 < dr² < ncut_off²
                    idx += 1
                    Neighbors[gtid, idx] = j    
                end
            end

            for j = gtid+1:Npart
                if idx > NNeigh
                    break
                end
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]

                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy

                dr² = dx*dx + dy*dy

                if 0 < dr² < ncut_off²
                    idx += 1
                    Neighbors[gtid, idx] = j    
                end
            end
        elseif dim == 3
            @inbounds for j = 1:gtid-1
                if idx > NNeigh
                    break
                end
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]
                dz  = pos₁[3] - pos₂[3]
    
                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
                dz = (2abs(dz) > box[3] ) ? dz - sign(dz) * box[3] : dz
    
                dr² = dx*dx + dy*dy + dz*dz
    
                if 0 < dr² < ncut_off²
                    idx += 1
                    Neighbors[gtid, idx] = j    
                end
            end
    
            for j = gtid+1:Npart
                if idx > NNeigh
                    break
                end
                pos₂  = r[j]
                dx  = pos₁[1] - pos₂[1]
                dy  = pos₁[2] - pos₂[2]
                dz  = pos₁[3] - pos₂[3]
    
                dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
                dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
                dz = (2abs(dz) > box[3] ) ? dz - sign(dz) * box[3] : dz
    
                dr² = dx*dx + dy*dy + dz*dz
    
                if 0 < dr² < ncut_off²
                    idx += 1
                    Neighbors[gtid, idx] = j    
                end
            end
        end
    end
    return nothing
end