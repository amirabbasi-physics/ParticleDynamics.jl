export volume, friction, hexagonal_neighbors, max_neighbors
    
export zero_velocities_kernel!

function zero_velocities_kernel!(
    r::CuDeviceVector{T},
    rr::CuDeviceVector{T},
    num_cold::I ) where {I, T}
    tid = threadIdx().x
    gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    
    @inbounds begin
        if gtid <= num_cold
            r[gtid] = rr[gtid]
        end
    end
    return
end

    
function hexagonal_neighbors(; sigma::T, circ_R::T) where T
    n_max = ceil(Int,circ_R / sigma)
    num_circles = 7 * sum(1:n_max)
    return num_circles
end 

function max_neighbors(; sigma::T, R::T, box::SVector{N,T}) where {N,T}
    dim = length(box)
    return ceil(Int, 5*(2 * R/sigma)^dim)
end

@inline function volume(R::T)::T where T
    return T((4.0/3.0)*π*R*R*R)
end

@inline function friction(η::T, R::T)::T where T
    return T(6.0*π*η*R)
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


function PassiveOP(; part_type::String = "C",
    part_id::Int = 0,
    r::SVector=SVector{3,T}(ones(T,3)), 
    v::SVector=SVector{3,T}(ones(T,3)),  
    f::SVector=SVector{3,T}(ones(T,3)), 
    η::T, 
    Radii::T,
    α::T) where {T<:AbstractFloat}
    kB = T(1.380649*10^(-23))
    Temp = T(300.0)
    rad = T(1.0)               # Needs to be modified for polydispersed systems
    γ = friction(η,Radii)
    τD = γ*(2Radii)^2/(kB*Temp)

    PassiveOP{T}(part_type,part_id, rad, α, τD, r, v, f)
end


export APMO
mutable struct APMO{T <: AbstractFloat} <: Particle
	part_type::String
    part_id::Int
	rad::T
	α::T
	τD::T
    r::SVector
	v::SVector
	f::SVector
    τΓ::T
	r_pseu::SVector
	v_pseu::SVector
end

function APMO(; part_type::String = "H",
    part_id::Int = 1,
    r::SVector=SVector{3,T}(ones(T,3)), 
    v::SVector=SVector{3,T}(ones(T,3)),  
    f::SVector=SVector{3,T}(ones(T,3)),
    η::T, 
    Radii::T, 
    α::T, 
    r_pseu::SVector=SVector{3,T}(ones(T,3)),
    v_pseu::SVector=SVector{3,T}(ones(T,3)),
    τΓ::T) where {T <: AbstractFloat}
    kB = T(1.380649*10^(-23))
    Temp = T(300.0)
    rad = T(1.0)               # Needs to be modified for polydispersed systems
    γ = friction(η,Radii)
    τD = γ*(2Radii)^2/(kB*Temp)
    τΓ = τΓ
    APMO{T}(part_type, part_id, rad, α, τD, r, v, f, τΓ, r_pseu, v_pseu)
end



################################################################################
################################################################################
#           			Simulation definition                                  #
################################################################################                                                                              
################################################################################

export Simulation

# Add force_func field to Simulation struct
mutable struct Simulation
    descriptor::String
    type::String
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
    force_func::Function 
end

function Simulation(; descriptor::String = "No description given...",
    type::String = "Langevin",
    box::SVector=SVector{3,Union{Float32,Float64}}(ones(Float32,3)),
    particles::Array{Particle, 1} = Particle[],
    part_types::Vector{String}=["H"],
    ϵ::Union{Float32,Float64} = 100.0f0,
    σ::Union{Float32,Float64} = 1.0f0,
    neigh_cut_off::Union{Float32,Float64} = 5.0f0,
    neigh_update::Int = 100000, 
    num_cold::Int = 0,
    dt::Union{Float32,Float64} = 0.00001f0,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output",
    force_func::Function = WCA)  
    Simulation(descriptor, type, box, particles, part_types, ϵ, σ, neigh_cut_off, neigh_update, num_cold, dt,integrator, num_steps, save_interval, particles_to_save,output_file, force_func)  # Add force_func argument here
end

export SimulationActive

# Add force_func field to Simulation struct
mutable struct SimulationActive
    descriptor::String
    box::SVector
    particles::Array{Particle, 1}
    part_types::Vector{String}
    ϵ::Union{Float32,Float64}
    σ::Union{Float32,Float64}
    neigh_cut_off::Union{Float32,Float64}
    neigh_update::Int
    dt::Union{Float32,Float64}
    integrator::String
    num_steps::Int
    save_interval::Int
    particles_to_save::Array{Particle, 1}
    output_file::String
    force_func::Function 
end

function SimulationActive(; descriptor::String = "No description given...",
    box::SVector=SVector{3,Union{Float32,Float64}}(ones(Float32,3)),
    particles::Array{Particle, 1} = Particle[],
    part_types::Vector{String}=["A","B"],
    ϵ::Union{Float32,Float64} = 100.0f0,
    σ::Union{Float32,Float64} = 1.0f0,
    neigh_cut_off::Union{Float32,Float64} = 5.0f0,
    neigh_update::Int = 100000, 
    dt::Union{Float32,Float64} = 0.00001f0,
    integrator::String = "vv",
    num_steps::Int = 0,
    save_interval::Int = 0,
    particles_to_save::Array{Particle, 1} =  Particle[],
    output_file::String = "output",
    force_func::Function = WCA)  
    SimulationActive(descriptor,box, particles, part_types, ϵ, σ, neigh_cut_off, neigh_update, dt,integrator, num_steps, save_interval, particles_to_save,output_file, force_func)  # Add force_func argument here
end
