export Box
 # This should be changed for polydisperse particles!
function Box(; dim::Int, Npart::Int, ϕ::T, sigma::T) where T <: AbstractFloat   
    if dim == 2
        L = sqrt(π*sigma^2*Npart/(4*ϕ))
        return SVector{2,T}([L,L])
    elseif dim == 3
        L = (π*sigma^3*Npart/(6*ϕ))^(1/3)
        return SVector{3,T}([L,L,L])
    end
end

export sort_pos_by_dist
function sort_pos_by_dist(positions::Array{SVector{2,T},1}, x0::T, y0::T) where T
    distances = [(pos[1]-x0)^2 + (pos[2]-y0)^2 for pos in positions]
    sorted_indices = sortperm(distances)
    return positions[sorted_indices]
end


export shuffle_pos!
function shuffle_pos!(simulation)
    r_tmp = [simulation.particles[i].r for i = 1:length(simulation.particles)]
    shuffle!(r_tmp)
    [simulation.particles[i].r = r_tmp[i] for i = 1:length(simulation.particles)]
    return nothing
end


export hexagonal_neighbors
function hexagonal_neighbors(; sigma::T, circ_R::T) where T
    n_max = floor(Int,circ_R / sigma)
    num_circles = 10 + 6 * sum(1:n_max)
    return num_circles
end 

export rectangular_lattice

function rectangular_lattice(Npart::Int, box::SVector{2, T}) where T
    n_side = ceil(Int,sqrt(Npart))
    delta = box ./ n_side
    x = range(delta[1]/2, stop=box[1]-delta[1]/2, length=n_side)
    y = range(delta[2]/2, stop=box[2]-delta[2]/2, length=n_side)
    positions = [SVector{2, T}(xi, yj) .- box/2 for xi in x, yj in y]
    return positions[1:Npart]
end

export triangular_lattice


function triangular_lattice(Npart::Int, σ::T) where T
    lattice_const = σ
    positions = Array{SVector{2,T}, 1}()
    M_x = ceil(Int,sqrt(Npart)) 
    M_y = ceil(Int,sqrt(Npart))
    for i = 1:M_x
        for j = 1:M_y
            pos = SVector{2,T}([(i-1)*lattice_const + (j%2)*lattice_const/2, (j-1)*lattice_const*sqrt(3)/2]) 
            push!(positions, pos)
        end
    end
    max_x = maximum([pos[1] for pos in positions])
    min_x = minimum([pos[1] for pos in positions])

    max_y = maximum([pos[2] for pos in positions])
    min_y = minimum([pos[2] for pos in positions])

    L_x = max_x - min_x
    L_y = max_y - min_y

    size = @SVector [L_x, L_y]
    positions = [pos = pos .- size/2 for pos in positions]
    return positions[1:Npart]
end

function triangular_lattice(Npart::Int, box::SVector{N,T}, σ::T) where {N,T}
    positions = Array{SVector{2,T}, 1}()
    M_x = ceil(Int,box[1]/σ)
    M_y = ceil(Int,2box[2]/(sqrt(3)*σ))
    for i = 1:M_x
        for j = 1:M_y
            pos = SVector{2,T}([(i-1)*σ + (j%2)*σ/2, (j-1)*σ*sqrt(3)/2]) 
            push!(positions, pos)
        end
    end
    positions = [pos = pos .- box/2 for pos in positions]
    return positions[1:Npart]
end

export simplecubic_lattice
function simplecubic_lattice(Npart::Int, box::SVector{3,T}) where T
    positions = Array{SVector{3,T}, 1}()
    L_x = box[1]
    L_y = L_x
    L_z = L_x

    M_x = ceil(Int,Npart^(1/3))
    M_y = M_x
    M_z = M_y

    s_x, s_y, s_z = L_x/(Npart)^(1/3), L_y/(Npart)^(1/3), L_z/(Npart)^(1/3)

    for i = 0 : M_x - 1, j = 0 : M_y - 1, k = 0 : M_z - 1
        push!(positions, SVector{3,T}([(i + 1/2) * s_x, (j + 1/2) * s_y, (k + 1/2) * s_z] .- [L_x/2, L_y/2, L_z/2]))
    end
    return positions
end

export fcc_lattice
function fcc_lattice(box::SVector{N,T},σ::T,M_x::Int64, M_y::Int64, M_z::Int64) where {N,T}
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,T}, 1}()
    for i = 0:M_x-1
        for j = 0:M_y-1
            for k = 0:M_z-1
                pos = [pos_fcc(σ)[n] .+ @SVector [i * σ, j * σ, k * σ] for n = 1:4]
                pos = [pos[i] .- box ./2 for i=1:4]
                for nn = 1:4
                    push!(positions, pos[nn])
                end
            end
        end
    end
    return positions
end
export pos_fcc
function pos_fcc(a::T) where T
    """returns the positions (x,y,z) of the 4 atoms in a fcc unit cell with the lattice constant a."""
    p₁ = @SVector [0.f0, 0.f0, 0.f0]
    p₂ = @SVector [0.f0, 0.5f0*a, 0.5f0*a]
    p₃ = @SVector [0.5f0*a, 0.f0, 0.5f0*a]
    p₄ = @SVector [0.5f0*a, 0.5f0*a, 0.f0]
    return p₁, p₂, p₃, p₄
end

"""
export fcc_lattice
function fcc_lattice(box::SVector{N,T},a::T,M_x::Int, M_y::Int, M_z::Int) where {N,T}
    positions = Array{SVector{3,T}, 1}()
    for i = 0:M_x-1, j = 0:M_y-1, k = 0:M_z-1
        x = i*a
        y = j*a
        z = k*a
        push!(positions, SVector{3,T}([x, y, z]) .- box ./ 2)
        push!(positions, SVector{3,T}([x+a/2, y+a/2, z]) .- box ./ 2)
        push!(positions, SVector{3,T}([x+a/2, y, z+a/2]) .- box ./ 2)
        push!(positions, SVector{3,T}([x, y+a/2, z+a/2]) .- box ./ 2)
    end
    return positions
end
"""

export isinsphere
function isinsphere(pos::SVector{3,T}, rad::T, r_margin::T) where T
    if norm(pos) < rad + r_margin
        return true
    else
        return false
    end
end

export isincircle
function isincircle(pos::SVector{2,T}, rad::T, r_margin::T) where T
    #mid_point = @SVector T[0.50*L, 0.50*L]                
    if norm(pos) < rad + r_margin
        return true
    else
        return false
    end
end

export check_overlap
function check_overlap(pos::SVector{N,T}, positions::Vector{SVector{N,T}}, box::SVector{N,T}, sigma::T) where {N,T}
    dim = length(box)

    if dim == 2
        for p in positions

            dx  = pos[1] - p[1]
            dy  = pos[2] - p[2]
    
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
    
            dr² = dx*dx + dy*dy
            if dr² <= sigma^2
                return true
            end
        end
        return false
    elseif dim == 3
        for p in positions

            dx  = pos[1] - p[1]
            dy  = pos[2] - p[2]
            dz  = pos[3] - p[3]
    
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
            dz = (2abs(dz) > box[2] ) ? dz - sign(dz) * box[2] : dz
    
            dr² = dx*dx + dy*dy + dz*dz
            if dr² <= sigma^2
                return true
            end
        end
        return false
    end
end

export cut_circle_sphere

function cut_circle_sphere!(box::SVector{N,T}, σ::T, Npart::Int, fraction::T,cold_frac::T) where {N,T}
    dim = length(box)
    R = T(σ/2)
    if dim == 2
        N_circle = Npart * fraction * cold_frac

        rad = T(sqrt(N_circle)/(2π/(1.0675*2sqrt(3))))
        #rad = sqrt(N_circle)/(π/(2sqrt(3)))
        N_lattice = ceil(Int,2N_circle)
        r = triangular_lattice(N_lattice, σ)
        r_init  = circle_cut(r, rad, true)
    elseif dim == 3

        N_sphere = Int(ceil(Npart .* fraction))
        rad = T(cbrt(N_sphere))
        nn = Int(ceil(rad))
        lattice_const = T(sqrt(2)*σ)
        r = fcc_lattice(box,lattice_const, nn, nn, nn)
        
        r1  = sphere_cut(r, rad)

        num_pl = length(r1)
        n_remain = Int(ceil(length(r1) *(1/fraction -1)))
        Npart_new = n_remain + num_pl 
        r2  = sphere_cut(r, rad*cold_frac)
        n_remain = Npart_new-length(r2)

        r_init = Array{SVector{3,T}}(undef, Npart_new)
        [r_init[i]=r2[i] for i = 1:length(r2)]
        for ii in 1:n_remain
            # Generate a random position
            pos = random_pos(box)    
            # Check for overlap with other particles
            while check_overlap(pos, r_init, R)
                pos = random_pos(box)
            end        
            # Append the position to the list of positions
            r_init[length(r2)+ ii] = pos
        end
    end
    return r_init
end

export circle_cut
function circle_cut(r0::Array{SVector{N,T}}, rad::T, in::Bool) where {N,T}
    new_pos = Array{SVector{2,T}, 1}()
    if in
        for i = 1:length(r0)
            if norm(r0[i])/rad < T(1.0)
                push!(new_pos , r0[i])
            end
        end
    else
        for i = 1:length(r0)
            if norm(r0[i])/rad > 1.01
                push!(new_pos , r0[i])
            end
        end
    end

    return new_pos
end

export sphere_cut
function sphere_cut(r0::Array{SVector{N,T}}, rad::T) where {N,T}
    r0_new = Array{SVector{3,T}, 1}()
    #r_center = @SVector T[0.5*L_box,0.5*L_box,0.5*L_box]
    for i = 1:length(r0)
        if norm(r0[i])/rad <= 1
            push!(r0_new , r0[i])
        end
    end
    return r0_new
end

export volume
@inline function volume(R::T)::T where T
    return T((4.0/3.0)*π*R*R*R)
end

export friction

@inline function friction(η::T, R::T)::T where T
    return T(6.0*π*η*R)
end

export random_pos
function random_pos(box::SVector{N,T}) where {N,T}
    dim = length(box)
	return SVector{dim,T}((rand(dim)) .-T(0.5) ) .* box
end


################################################################################
################################################################################
#           			APMs / Passive Brownian Particles
################################################################################
################################################################################

export Particle
abstract type Particle end


export PassiveP
mutable struct PassiveP{T <: AbstractFloat} <: Particle
	part_type::String
    part_id::Int
	rad::T
	α::T
	τm::T
	τD::T
    r::SVector
	v::SVector
	f::SVector
end


function PassiveP(; part_type::String = "Cold",
    part_id::Int = 0,
    r::SVector=SVector{3,T}(ones(T,3)), 
    v::SVector=SVector{3,T}(ones(T,3)), 
    f::SVector=SVector{3,T}(ones(T,3)),
    density::T, η::T, Radii::T, α::T) where {T<:AbstractFloat}
    kB = T(1.380649*10^(-23))
    Temp = T(300.0)
    rad = T(1.0)               # Needs to be modified for polydispersed systems
    m = density*volume(Radii)
    γ = friction(η,Radii)
    τm = m/γ
    τD = γ*(2Radii)^2/(kB*Temp)

    PassiveP{T}(part_type,part_id, rad,α,τm, τD,r,v,f)
end

export PassiveOP
mutable struct PassiveOP{T <: AbstractFloat} <: Particle
	part_type::String
    part_id::Int
	rad::T
	α::T
	τD::T
    r::SVector
	v::SVector
	f::SVector
end


function PassiveOP(; part_type::String = "Cold",
    part_id::Int = 0,
    r::SVector=SVector{3,T}(ones(T,3)), 
    v::SVector=SVector{3,T}(ones(T,3)),  
    f::SVector=SVector{3,T}(ones(T,3)), η::T, Radii::T, α::T) where {T<:AbstractFloat}
    kB = T(1.380649*10^(-23))
    Temp = T(300.0)
    rad = T(1.0)               # Needs to be modified for polydispersed systems
    γ = friction(η,Radii)
    τD = γ*(2Radii)^2/(kB*Temp)

    PassiveOP{T}(part_type,part_id, rad, α, τD, r, v, f)
end

"""
export APM
mutable struct APM{T <: AbstractFloat, N <: Int} <: Particle
	part_type::String
	rad::T
	α::T
	τm::T
	τD::T
    τΓ::T
    r::SVector{N,T}
	v::SVector{N,T}
	f::SVector{N,T}
	r_pseu::SVector{N,T}
	v_pseu::SVector{N,T}
end

function APM(; part_type::String = "APM",r::SVector{N,T}, v::SVector{N,T}, f::SVector{N,T},
    density::T, η::T, Radii::T, α::T, Temp::T, k::T, r_pseu::SVector{N,T}, v_pseu::SVector{N,T}) where {T <: AbstractFloat, N <: Int}
    kB = T(1.380649*10^(-23))
    rad = Radii/1.0e-6
    m = density*volume(Radii)
    γ = friction(η,Radii)
    τm = m/γ
    τD = γ*(Radii)^2/(kB*Temp)
    τΓ = γ/k
    APM{T,N}(part_type, rad, α, τm, τD, τΓ, r, v, f, r_pseu, v_pseu)
end
"""
################################################################################
#
#           			Interactions definition
#
################################################################################

"""
export WCA
export Harmonic_Repulsive

abstract type Interaction end

struct WCA{T <: AbstractFloat} <: Interaction
    ϵ::T
    σ::T
    r_cut::T
end

function WCA(; ϵ::T, σ::T, r_cut::T) where T<:AbstractFloat
    WCA{T}(ϵ, σ, r_cut, particles)
end


struct Harmonic_Repulsive{T <: AbstractFloat} <: Interaction
    k::T
    r_cut::T
end

function Harmonic_Repulsive(;k::T, r_cut::T) where T<:AbstractFloat
    Harmonic_Repulsive{T}(k, r_cut, particles)
end
"""

################################################################################
################################################################################
#           			Simulation definition                                  #
################################################################################                                                                              
################################################################################

export Simulation

mutable struct Simulation
    descriptor::String
    box::SVector
    particles::Array{Particle, 1}
    part_types::Vector{String}
    ϵ::Union{Float32,Float64}
    σ::Union{Float32,Float64}
    neigh_cut_off::Union{Float32,Float64}
    neigh_update::Int
    num_cold::Int
    dt::Union{Float32,Float64}
    integrator::String
    num_steps::Int
    save_interval::Int
    particles_to_save::Array{Particle, 1}
	output_file::String
end

function Simulation(; descriptor::String = "No description given...",
    box::SVector=SVector{3,Union{Float32,Float64}}(ones(Float32,3)),
    particles::Array{Particle, 1} = Particle[],
    part_types::Vector{String}=["A","B"],
    ϵ::Union{Float32,Float64} = 100.0f0,
    σ::Union{Float32,Float64} = 1.0f0,
    neigh_cut_off::Union{Float32,Float64} = 5.0f0,
    neigh_update::Int = 100000, 
    num_cold::Int = 1,
    dt::Union{Float32,Float64} = 0.00001f0,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output")
    Simulation(descriptor,box, particles, part_types, ϵ, σ, neigh_cut_off, neigh_update, num_cold, dt,integrator, num_steps, save_interval, particles_to_save,output_file)
end


"""

mutable struct Simulation
    descriptor::String
    box::SVector
    particles::Array{Particle, 1}
    part_types::Vector{String}
    ϵ::Union{Float32,Float64}
    σ::Union{Float32,Float64}
    neigh_cut_off::Union{Float32,Float64}
    neigh_update::Int
    num_cold::Int
    dt::Union{Float32,Float64}
    integrator::String
    num_steps::Int
    save_interval::Int
    particles_to_save::Array{Particle, 1}
    output_file::String
    num_runs::Int
    homogeneous::Bool
    collision_calc::Bool
    ϕ::Union{Float32,Float64}
    fraction::Union{Float32,Float64}
    cold_frac::Union{Float32,Float64}
    R::Union{Float32,Float64}
    α₁::Union{Float32,Float64}
    α₂::Union{Float32,Float64}
    random_positions::Bool
end

function Simulation(; descriptor::String = "No description given...",
    box::SVector=SVector{3,Union{Float32,Float64}}(ones(Float32,3)),
    particles::Array{Particle, 1} = Particle[],
    part_types::Vector{String}=["A","B"],
    ϵ::Union{Float32,Float64} = 100.0f0,
    σ::Union{Float32,Float64} = 1.0f0,
    neigh_cut_off::Union{Float32,Float64} = 5.0f0,
    neigh_update::Int = 100000, 
    num_cold::Int = 1,
    dt::Union{Float32,Float64} = 0.00001f0,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output",
    num_runs::Int = 1,
    homogeneous::Bool = true,
    collision_calc::Bool = true,
    Npart::Int = 1,
    p_ids::Vector{Int} = [1],
    dim::Int = 3,
    ϕ::Union{Float32,Float64} = 1.0f0,
    fraction::Union{Float32,Float64} = 0.5f0,
    cold_frac::Union{Float32,Float64} = 0.5f0,
    R::Union{Float32,Float64} = 1.0f0,
    α₁::Union{Float32,Float64} = 1.0f0,
    α₂::Union{Float32,Float64} = 1.0f0,
    Δt_prod::Union{Float32,Float64} = 0.00001f0,
    random_positions::Bool = true)
    Simulation(descriptor,box, particles, part_types, ϵ, σ, neigh_cut_off, neigh_update, num_cold, dt,integrator, num_steps, save_interval, particles_to_save,output_file)
end   

descriptor, box,particles,part_types,ϵ,σ,neigh_cut_off,neigh_update, num_cold,dt,integrator,num_steps,save_interval,particles_to_save,output_file,num_runs,homogeneous, collision_calc,Npart,
    p_ids::Vector{Int} = [1],
    dim::Int = 3,
    ϕ::Union{Float32,Float64} = 1.0f0,
    fraction::Union{Float32,Float64} = 0.5f0,
    cold_frac::Union{Float32,Float64} = 0.5f0,
    R::Union{Float32,Float64} = 1.0f0,
    α₁::Union{Float32,Float64} = 1.0f0,
    α₂::Union{Float32,Float64} = 1.0f0,
    Δt_prod::Union{Float32,Float64} = 0.00001f0,
    random_positions::Bool = true

"""