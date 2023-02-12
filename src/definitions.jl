
export noise2D
export noise3D
@inline function noise2D(Npart::Int)
    return SVector{2,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart))
end

function noise3D(Npart::Int) 
    return SVector{3,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart),CUDA.randn(Float32,Npart))
end

export rectangular_lattice

function rectangular_lattice(s_x::T, s_y::T, M_x::Int, M_y::Int) where T
    positions = Array{SVector{2,T}, 1}()
    for i = 0 : M_x - 1, j = 0 : M_y - 1
        push!(positions, SVector{2,T}([(i + 0.50) * s_x, (j + 0.50) * s_y]))
    end
    return positions
end


export triangular_lattice

function triangular_lattice(L_box::T,lattice_const::T,M_x::Int64, M_y::Int64) where T
    """Calculates the positions of an hexagonal lattice with the lattice constant a
    in a square box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{2,T}, 1}()
    r_x = lattice_const*M_x/2
    r_y = lattice_const*M_y/2
    for i = 0:M_x-1
        for j = 0:M_y-1
            pos = [pos_triangular(lattice_const)[n] .+ @SVector T[i * lattice_const - r_x, j * lattice_const - r_y] for n = 1:4]
            for nn = 1:4
                #if pos_num < Npart
                    push!(positions, pos[nn])
                    #pos_num += 1
                #end
            end
        end
    end
    return positions
end


export pos_triangular
function pos_triangular(a::T) where T
    """returns the positions (x,y) of the 4 atoms in a hexagonal unit cell with the lattice constant a."""
    p₁ = @SVector T[0.0, 0.0]
    p₂ = @SVector T[0.0, a]
    p₃ = @SVector T[-0.50*a, 0.50 *sqrt(3)*a]
    p₄ = @SVector T[0.50*a, 0.50 *sqrt(3)*a]
    return p₁, p₂, p₃, p₄
end


export triangular_circle
function triangular_circle(L_box::T, r0::Array{SVector{N,T}}, rad::T) where {N,T}
    new_pos = Array{SVector{2,T}, 1}()
    r_center = @SVector T[0.5*L_box,0.5*L_box]
    for i = 1:length(r0)
        dist = r0[i] .- r_center
        if norm(dist)/rad <= 1
            push!(new_pos , r0[i])
        end
    end
    return new_pos
end


export simplecubic_lattice

function simplecubic_lattice(s_x::T, s_y::T, s_z::T, M_x::Int64, M_y::Int64, M_z::Int64) where T
    positions = Array{SVector{3,T}, 1}()
    for i = 0 : M_x - 1, j = 0 : M_y - 1, k = 0 : M_z - 1
        push!(positions, @SVector T[(i + 0.50) * s_x, (j + 0.50) * s_y, (k + 0.50) * s_z])
    end
    return positions
end


export fcc_sphere
function fcc_sphere(L_box::T, r0::Array{SVector{N,T}}, rad::T) where {N,T}
    new_pos = Array{SVector{3,T}, 1}()
    r_center = @SVector T[0.5*L_box,0.5*L_box,0.5*L_box]
    for i = 1:length(r0)
        dist = r0[i] .- r_center
        if norm(dist)/rad <= 1
            push!(new_pos , r0[i])
        end
    end
    return new_pos
end

export fcc_lattice
function fcc_lattice(L_box::T,lattice_const::T,M_x::Int, M_y::Int, M_z::Int) where T
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,T}, 1}()
    r_x = lattice_const*M_x/2
    r_y = lattice_const*M_y/2
    r_z = lattice_const*M_z/2
    for i = 0:M_x-1
        for j = 0:M_y-1
            for k = 0:M_z-1
                pos = [pos_fcc(lattice_const)[n] .+ @SVector T[i * lattice_const - r_x, j * lattice_const - r_y, k * lattice_const - r_z] for n = 1:4]
                for nn = 1:4
                    #if pos_num < Npart
                        push!(positions, pos[nn])
                        #pos_num += 1
                    #end
                end
            end
        end
    end
    return positions
end
export pos_fcc
function pos_fcc(a::T) where T
    """returns the positions (x,y,z) of the 4 atoms in a fcc unit cell with the lattice constant a."""
    p₁ = @SVector T[0.0, 0.0, 0.0]
    p₂ = @SVector T[0.0, 0.50*a, 0.50*a]
    p₃ = @SVector T[0.50*a, 0.0, 0.50*a]
    p₄ = @SVector T[0.50*a, 0.50*a, 0.0]
    return p₁, p₂, p₃, p₄
end

export isinsphere
function isinsphere(L::T, N::Int32, σ::T, pos::SVector{3,T}) where T
    mid_point = @SVector T[0.50*L, 0.50*L, 0.50*L]
    sphere_rad = T((N/8)^(1/3)*σ)
    if norm(pos-mid_point) < sphere_rad
        return true
    else
        return false
    end
end

export isincircle
function isincircle(L::T, N::Int32, σ::T, pos::SVector{2,T}) where T
    mid_point = @SVector T[0.50*L, 0.50*L]
    circle_rad = T((N/8)^(1/3)*σ)                   # Correct the formula!!!!!!!!!!!!
    if norm(pos-mid_point) < circle_rad
        return true
    else
        return false
    end
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

function random_pos(dim::Int64, L::T) where T
	position = SVector{dim,T}(rand(dim)) .* L
	return position
end


################################################################################
#
#           			APMs / Passive Brownian Particles
#
################################################################################

export Particle
abstract type Particle end


export PassiveP
mutable struct PassiveP{T<:AbstractFloat, N<:Int} <: Particle
	part_type::String
	rad::T
	α::T
	τm::T
	τD::T
    r::SVector{N,T}
	v::SVector{N,T}
	f::SVector{N,T}
    dQ::T
end


function PassiveP(; part_type::String = "Cold",r::SVector{N,T}, v::SVector{N,T}, f::SVector{N,T},
    density::T, η::T, Radii::T, α::T, Temp::T) where {T<:AbstractFloat, N<:Int}
    kB = T(1.380649*10^(-23))
    rad = Radii/1.0e-6
    m = density*volume(Radii)
    γ = friction(η,Radii)
    τm = m/γ
    τD = γ*(Radii)^2/(kB*Temp)

    PassiveP{T,N}(part_type,rad,α,τm, τD,r,v,f)
end



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
################################################################################
#
#           			Interactions definition
#
################################################################################
export WCA
export Harmonic_Repulsive
export AbstractInteraction
abstract type AbstractInteraction end



struct WCA{T <: AbstractFloat} <: AbstractInteraction
    ϵ::T
    σ::T
    r_cut::T
    particles::Array{Particle, 1}
end
function WCA(ϵ::T, σ::T, r_cut::T, particles::Array{T, 1}) where T<:AbstractFloat
    WCA{T}(ϵ, σ, r_cut, particles)
end

struct Harmonic_Repulsive{T<:AbstractFloat} <: AbstractInteraction
    k::T
    r_cut::T
    particles::Array{T, 1}
end

function Harmonic_Repulsive(k::T, r_cut::T, particles::Array{T, 1}) where T<:AbstractFloat
    Harmonic_Repulsive{T}(k, r_cut, particles)
end



################################################################################
#
#           			Simulation definition                                   
#                                                                              
################################################################################


export Simulation

mutable struct Simulation{T <: AbstractFloat, N<: Int}
    descriptor::String

    periodicity::SVector{N,T}

    particles::Array{Particle, 1}
    interactions::Array{AbstractInteraction, 1}

    dt::T
    integrators::Array{AbstractIntegrator, 1}
    num_steps::N

    save_interval::N
    particles_to_save::Array{Particle, 1}
	output_file::String
end

function Simulation(; descriptor::String = "No description given...",
    periodicity::SVector{N,T},
    particles::Array{Particle, 1} = Particle[],
    interactions::Array{AbstractInteraction, 1} = AbstractInteraction[],
    dt::T,
    integrators::Array{AbstractIntegrator, 1} = AbstractIntegrator[],
    num_steps::N = 0,
    save_interval::N = 0,
    particles_to_save::Array{Particle, 1}=  Particle[],
    output_file::String = "output") where {T <: AbstractFloat, N<: Int}
new(descriptor, periodicity, particles, interactions, dt,integrators, num_steps, save_interval, particles_to_save,output_file)
end