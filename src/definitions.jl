
"""
export noise2D
export noise3D
@inline function noise2D(Npart::Int)
    return SVector{2,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart))
end


@inline function noise2D(Npart::Int)
    return SVector{2,Float64}.(CUDA.randn(Float64,Npart) ,CUDA.randn(Float64,Npart))
end



function noise3D(Npart::Int) 
    return SVector{3,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart),CUDA.randn(Float32,Npart))
end
"""




export Box

function Box(; dim::Int, Npart::Int, ϕ::T, σ::T) where T <: AbstractFloat    # This should be changed for polydisperse particles!
    if dim == 2
        L = sqrt(π*σ^2*Npart/(4*ϕ))
        return SVector{2,T}([L,L])
    elseif dim == 3
        L = (π*σ^3*Npart/(6*ϕ))^(1/3)
        return SVector{3,T}([L,L,L])
    end
end


function shuffle_pos!(simulation)
    r_tmp = [simulation.particles[i].r for i = 1:length(simulation.particles)]
    shuffle!(r_tmp)
    [simulation.particles[i].r = r_tmp[i] for i = 1:length(simulation.particles)]
    return nothing
end

export rectangular_lattice

function rectangular_lattice(Npart::Int, box::SVector{2,T}) where T
    positions = Array{SVector{2,T}, 1}()
    L_x = box[1]
    L_y = L_x
    s_x, s_y = L_x/sqrt(Npart), L_y/sqrt(Npart)
    M_x = ceil(Int,Npart^(1/2))
    M_y = M_x
    for i = 0 : M_x - 1, j = 0 : M_y - 1
        push!(positions, SVector{2,T}([(i + 1/2) * s_x, (j + 1/2) * s_y] .- [L_x/2, L_y/2]))
    end
    return positions
end

export triangular_lattice
function triangular_lattice(box::SVector{N,T},lattice_const::T,M_x::Int64, M_y::Int64) where {N,T}
    positions = Array{SVector{2,T}, 1}()
    for i = 1:M_x
        for j = 1:M_y
            pos = SVector{2,T}([(i-1)*lattice_const + (j%2)*lattice_const/2, (j-1)*lattice_const*sqrt(3)/2]) 
            pos = pos .- box ./ 2
            push!(positions, pos)
        end
    end
    return positions
end

export circle_cut
function circle_cut(r0::Array{SVector{N,T}}, rad::T) where {N,T}
    new_pos = Array{SVector{2,T}, 1}()
    #r_center = @SVector T[0.5*L_box,0.5*L_box]
    for i = 1:length(r0)
        if norm(r0[i])/rad <= 1
            push!(new_pos , r0[i])
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
function fcc_lattice(box::SVector{N,T},lattice_const::T,M_x::Int64, M_y::Int64, M_z::Int64) where {N,T}
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,Float32}, 1}()
    for i = 0:M_x-1
        for j = 0:M_y-1
            for k = 0:M_z-1
                pos = [pos_fcc(lattice_const)[n] .+ @SVector [i * lattice_const, j * lattice_const, k * lattice_const] for n = 1:4]
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
function check_overlap(pos, positions, R)
    for p in positions
        if norm(pos - p) < 2R
            return true
        end
    end
    return false
end

export cut_circle_sphere

function cut_circle_sphere!(box::SVector{N,T}, σ::T, Npart::Int, fraction::T,cold_frac::T) where {N,T}
    dim = length(box)
    R = T(σ/2)
    if dim == 2
        N_circle = Int(ceil(Npart .* fraction))
        lattice_const = σ
        r = triangular_lattice(box,lattice_const, N_circle, N_circle)
        rad = T(sqrt(N_circle))

        r1  = circle_cut(r, rad)
        num_pl = length(r1)
        n_remain = Int(ceil(length(r1) *(1/fraction -1)))
        Npart_new = n_remain + num_pl 
        r2  = circle_cut(r, rad*cold_frac)
        n_remain = Npart_new-length(r2)

        r_init = Array{SVector{2,T}}(undef, Npart_new)
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
    elseif dim == 3

        N_sphere = Int(ceil(Npart .* fraction))
        rad = T(cbrt(N_sphere))
        nn = Int(ceil(rad))
        lattice_const = sqrt(2.0f0)*σ
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
    return r_init, num_pl
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
mutable struct PassiveP{T<:AbstractFloat} <: Particle
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
    r::SVector=SVector{3,Float32}(ones(Float32,3)), 
    v::SVector=SVector{3,Float32}(ones(Float32,3)), 
    f::SVector=SVector{3,Float32}(ones(Float32,3)),
    density::T, η::T, Radii::T, α::T) where {T<:AbstractFloat}
    kB = T(1.380649*10^(-23))
    Temp = T(300.0)
    rad = T(1.0)               # Needs to be modified for polydispersed systems
    m = density*volume(Radii)
    γ = friction(η,Radii)
    τm = m/γ
    τD = γ*(Radii)^2/(kB*Temp)

    PassiveP{T}(part_type,part_id, rad,α,τm, τD,r,v,f)
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
    ϵ::Float64
    σ::Float64
    dt::Float64
    integrator::String
    num_steps::Int
    save_interval::Int
    particles_to_save::Array{Particle, 1}
	output_file::String
end

function Simulation(; descriptor::String = "No description given...",
    box::SVector=SVector{3,Float32}(ones(Float32,3)),
    particles::Array{Particle, 1} = Particle[],
    part_types::Vector{String}=["A","B"],
    ϵ::Float64 = 5.0e6,
    σ::Float64 = 2.0,

    dt::Float64=0.0001,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output")
    Simulation(descriptor,box, particles, part_types, ϵ, σ, dt,integrator, num_steps, save_interval, particles_to_save,output_file)
end