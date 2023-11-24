export harm_rep
export WCA


@inline function harm_rep(dx::T, dy::T, dr²::T, ϵ::T, σ::T) where T
    dist = dr²^(1/2)
    f_int = ϵ*(1/dist - 1/σ)
    e_int = (ϵ/2)*(1 - dist/σ)^2
    f_x = f_int*dx
    f_y = f_int*dy
    return SVector{2,T}(f_x,f_y), e_int
end


#Newly added!
@inline function harm_rep(dx::T, dy::T, dr²::T, ϵ::T, σ::T, alpha::T1) where {T, T1}
    dist = dr²^(1/2)
    f_int = ϵ*(1/dist - 1/σ)
    e_int = (ϵ/2)*(1 - dist/σ)^2
    f_x = f_int*dx
    f_y = f_int*dy
    dqt = T1(f_x^2 + f_y^2 - 2ϵ * alpha)
    return SVector{2,T}(f_x,f_y), e_int, dqt
end
#Newly added finished



# uncomment for 3D case

"""
@inline function harm_rep(dx::T, dy::T, dz::T, dr²::T, ϵ::T, σ::T) where T
    dist = dr²^(1/2)
    f_int = ϵ*(1/dist - 1/σ)
    e_int = (ϵ/2)*(1 - dist/σ)^2
    f_x = f_int*dx
    f_y = f_int*dy
    f_z = f_int*dz
    return SVector{3,T}(f_x,f_y, f_z), e_int
end
"""

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






################################################################################
#                                                                              #
#                           Calculating forces                                 #
#                           USING NEIGHBORLIST                                 #
#                                                                              #
################################################################################
export forces!
export forces_kernel!

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




#Newly added!
function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    alpha_list::CuVector{T1},
    dQ::CuVector{T},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    box::SVector{N,T},
    ϵ::T,
    σ::T,
    force_func::Function) where {N,T,T1,I}

    kernel = @cuda launch = false forces_kernel!(r, f, alpha_list, dQ, Epot, Neighbors, box, ϵ, σ, force_func)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, alpha_list, dQ, Epot, Neighbors, box, ϵ, σ, force_func; threads, blocks)

    return nothing
end


function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    alpha_list::CuDeviceVector{T2},
    dQ::CuDeviceVector{T1},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    box::T,
    ϵ::T1,
    σ::T1,
    force_func::Function) where {T,T1, T2, I}

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
    

    acc  = zero(T)
    epot = zero(T1)
    dqt  = zero(T1) 

    @inbounds begin
        if gtid <= Npart
            pos₁ = r[gtid]
            alpha = alpha_list[gtid]
        else
            pos₁ = zero(T)
            alpha = zero(T1)
        end

        acc = zero(T)
        epot= zero(T1)
        dqt  = zero(T1)

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
                frc, ep, dq = force_func(dx, dy, dr², ϵ, σ, alpha) # Call the passed function here
                acc += frc
                epot = epot + ep
                dqt += dq
            end
        end
              
        if gtid <= Npart
            f[gtid] = acc
            Epot[gtid] += epot
            dQ[gtid] += dqt
        end
    end
    return nothing
end
#Newly added finished!

################################################################################
#                                                                              #
#                   Calculating forces and collisions                          #
#                           USING NEIGHBORLIST                                 #
#                                                                              #
################################################################################



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



function forces!(
    r::CuVector{SVector{N,T}},
    f::CuVector{SVector{N,T}},
    Epot::CuVector{T},
    Neighbors::CuMatrix{I},
    cold_num::I,
    colls::CuVector{T},
    box::SVector{N,T},
    ϵ::T,
    σ::T,
    force_func::Function) where {N,T,I}

    kernel = @cuda launch = false forces_kernel!(r, f, Epot, Neighbors, cold_num, colls, box, ϵ, σ, force_func)
    config = launch_configuration(kernel.fun)
    threads = min(length(r), config.threads)
    blocks = cld(length(r), threads)
    CUDA.@sync kernel(r, f, Epot, Neighbors, cold_num, colls, box, ϵ, σ, force_func; threads, blocks)

    return nothing
end


"""

function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    num_cold::Int,
    colls::CuDeviceVector{T1},
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

                if dr² < cut_off²                  
                    if gtid <= num_cold
                        if idx > num_cold
                            coll += one(T1)
                        end
                    else
                        if idx <= num_cold
                            coll += one(T1)
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


"""


function forces_kernel!(
    r::CuDeviceVector{T},
    f::CuDeviceVector{T},
    Epot::CuDeviceVector{T1},
    Neighbors::CuDeviceMatrix{I},
    num_cold::Int,
    colls::CuDeviceVector{T1},
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

                if dr² < cut_off²                  
                    coll += one(T1)
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