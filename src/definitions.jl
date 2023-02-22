
export noise2D
export noise3D
@inline function noise2D(Npart::Int)
    return SVector{2,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart))
end

function noise3D(Npart::Int) 
    return SVector{3,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart),CUDA.randn(Float32,Npart))
end

export Box

function Box(; dim::Int, Npart::Int, ϕ::T, σ::T) where T <: AbstractFloat    # This should be changed for polydisperse particles!
    if dim == 2
        L = T(sqrt(π*σ^2.0*Npart/(4.0*ϕ)))
        return SVector{2,T}([L,L])
    elseif dim == 3
        L = T((π*σ^3.0*Npart/(6.0*ϕ))^(1.0/3.0))
        return SVector{3,T}([L,L,L])
    end
end


export rectangular_lattice

function rectangular_lattice(Npart::Int, box::SVector{2,T}) where T
    positions = Array{SVector{2,T}, 1}()
    L_x = box[1]
    L_y = L_x
    s_x, s_y = T(L_x/sqrt(Npart)), T(L_y/sqrt(Npart))
    M_x = ceil(Int,Npart^(1/2))
    M_y = M_x
    for i = 0 : M_x - 1, j = 0 : M_y - 1
        push!(positions, SVector{2,T}([(i + 0.50) * s_x, (j + 0.50) * s_y] .- [L_x/2, L_y/2]))
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

    s_x, s_y, s_z = T(L_x/(Npart)^(1.0/3.0)), T(L_y/(Npart)^(1.0/3.0)), T(L_z/(Npart)^(1.0/3.0))

    for i = 0 : M_x - 1, j = 0 : M_y - 1, k = 0 : M_z - 1
        push!(positions, SVector{3,T}(T[(i + 0.50) * s_x, (j + 0.50) * s_y, (k + 0.50) * s_z] .- [L_x/2, L_y/2, L_z/2]))
    end
    return positions
end

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

export cut_circle_sphere

function cut_circle_sphere!(box::SVector{N,T}, σ::T, Npart::Int, fraction::T) where {N,T}
    dim = length(box)
    R = T(σ/2)
    if dim == 2
        N_circle = ceil(Npart .* fraction)
        nn = Int(50*ceil(0.5*sqrt(N_circle)))
        L_x, L_y = box[1], box[2]
        lattice_const = σ
        r = triangular_lattice(box,lattice_const, nn, nn)
        rad = T(sqrt(N_circle*3*sqrt(3)*(σ/2)^2 /2π))
        #println(norm.(r),rad)
        r  = circle_cut(r, rad)
        n_remain = Int(ceil(length(r) *(1/fraction -1)))
        Npart_new = n_remain + length(r)
        
        r_init = Array{SVector{2,T}}(undef, Npart_new)
        [r_init[i]=r[i] for i = 1:length(r)]
        #println(r_init)
        ii = 0
        while ii < n_remain
            pos = random_pos(box)
            if !isincircle(pos,rad,R)
                ii += 1
                r_init[length(r)+ ii] = pos                
            end
        end
    elseif dim == 3
        N_shape = ceil(Npart .* fraction)
        nn = Int(ceil(cbrt(N_shape)))
        L_x, L_y, L_z = box[1], box[2], box[3]
        lattice_const = sqrt(2.0f0)*σ
        r = fcc_lattice(box,lattice_const, nn, nn, nn)
        rad = T((N_shape*4*sqrt(2)*(σ/2)^3 / 3π)^(1/3))
        r  = sphere_cut(r, rad)

        n_remain = Int(ceil(length(r) *(1/fraction -1)))
        n_remain = Npart - length(r)
        Npart_new = n_remain + length(r)
        r_init = Array{SVector{3,T}}(undef, Npart_new)
        [r_init[i]=r[i] for i = 1:length(r)]
        ii = 0
        while ii < n_remain
            pos = random_pos(box)
            if !isinsphere(pos,rad,R)
                ii += 1
                r_init[length(r)+ ii] = pos                
            end
        end
    end
    return r_init, length(r)
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
    rad = Radii/1.0e-6
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
    ϵ::Float32
    σ::Float32
    dt::Float32
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
    ϵ::Float32 = 0.001f0,
    σ::Float32 = 2.0f0,

    #interaction_type::Interaction,
    dt::Float32=0.0001f0,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output")
    Simulation(descriptor,box, particles, part_types, ϵ, σ, dt,integrator, num_steps, save_interval, particles_to_save,output_file)
end