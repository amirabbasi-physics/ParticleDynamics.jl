"""
    `@use_threads multithreaded ...`

Applies `Threads.@threads` if `multithreaded == true`.  This is mostly used to
shorten code.
"""


################################################################################
#
#           			Active Tandem/ Passive Brownian Particles
#
################################################################################
export Particle

struct Particle{T<:AbstractFloat}
	part_type::String
	σ::T
	α::T
	τm::T
	τD::T
	Q̇ ::T

    r::SVector
	v::SVector
	f::SVector


    function Particle(; part_type::String = "PassiveBP",r::SVector, v::SVector, f::SVector,
		density::T, η::T, Radii::T, α::T, β::T, Q̇::T=0.0, free::Int64 = 1, τΓ::Union{T, Nothing} = nothing, r_pseu::Union{SVector, Nothing} = nothing, v_pseu::Union{SVector, Nothing} = nothing) where T<:AbstractFloat
		σ = 2.0*Radii/2.0e-6
		m = density*volume(Radii)
		γ = friction(η,Radii)
		τm = m/γ
		τD = γ*β*(2*Radii)^2
		if !isnothing(τΓ)
			τΓ = τΓ
		end
        new{T}(part_type,σ,α,τm, τD,Q̇,r,v,f, free,τΓ,r_pseu,v_pseu)
    end
end


export AbstractIntegrator
export Brownian
export Langevin


abstract type AbstractIntegrator end

"""
Stores properties of a Brownian integrator

    Brownian(; Particles, dt, rotations, multithreaded)

Initialize a Brownian integrator for `Particles` with timestep `dt`.  If `rotations == true`, integrate the orientational degree of freedom.
If `multithreaded == true`, split Particles between threads.
"""
struct Brownian <: AbstractIntegrator
    dt::Float64
    particles::Array{Particle, 1}


    function Brownian(; particles::Array{Particle, 1}, dt::Float64)
        new(dt, particles)
    end
end

struct Langevin <: AbstractIntegrator
    dt::Float64
    particles::Array{Particle, 1}

    function Langevin(; particles::Array{Particle, 1}, dt::Float64)
        new(dt, particles)
    end
end






export group_by_type

function group_by_type(particles::Array{Particle, 1}; part_type::Union{String, Array{String, 1}})
    if !(part_type isa Array)
        part_type = [part_type]
    end

    pgroup = Array{Particle, 1}()
    for particle in particles
        if particle.part_type in part_type
            push!(pgroup, particle)
        end
    end
    return pgroup
end

export rectangular_lattice
export random_positions
export remove_overlaps!

"""
    rectangular_lattice(; s_x, s_y, M_x, M_y)

Generates a rectangular lattice with lattice constants `s_x, s_y` in the x, y
directions.  Each cell is duplicated `M_x, M_y` times in the x, y directions.
"""
function rectangular_lattice(; lat_const::SVector, dup_vec::SVector)
	positions = []
    if size(lat_const,1) == 1
		for i = 1 : dup_vec[1]
			push!(positions, SVector{1,Float64}([(i-1) * lat_const[1]]))
		end
		return positions
	elseif size(lat_const,1) == 2
		for i = 1 : dup_vec[1], j = 1 : dup_vec[2]
			push!(positions, SVector{2,Float64}([(i-1) * lat_const[1], (j-1) * lat_const[2]]))
		end
		return positions
	elseif size(lat_const,1) == 3
	    for i = 1 : dup_vec[1], j = 1 : dup_vec[2], k = 1: dup_vec[3]
	        push!(positions, SVector{3,Float64}([(i-1) * lat_const[1], (j-1) * lat_const[2], (k-1) * lat_const[3]]))
	    end
	    return positions
	end
end

"""
    rectangular_lattice(; s_x, s_y, M_x, M_y)

Generates a rectangular lattice with lattice constants `s_x, s_y` in the x, y
directions.  Each cell is duplicated `M_x, M_y` times in the x, y directions.
"""
function random_positions(nPart::Int64, dim::Int64, L::Float64)
	positions = [SZeros(dim) for i = 1:nPart]
	for i = 1:nPart
		positions[i] = L .* SRands(dim)
	end
	return positions
end

function remove_overlaps!(; positions, fixed_positions, minimum_distance::Float64, periodicity::SVector)
	dim = size(periodicity,1)
	indices_to_remove = Array{Int64, 1}()
	if positions == fixed_positions
		if dim == 2
			for i = 1: size(positions,1) - 1
		        @inbounds for j = i + 1 : size(positions,1)
					Δx = wrap_displacement(positions[i][1] - positions[j][1]; period = periodicity[1])
				    Δy = wrap_displacement(positions[i][2] - positions[j][2]; period = periodicity[2])

		            if Δx^2 + Δy^2 < minimum_distance^2
		                push!(indices_to_remove, i)
		                break
		            end
		        end
		    end
		    deleteat!(positions, indices_to_remove)
		elseif dim == 3
			for (i, position) in enumerate(positions)
		        @inbounds for j = i + 1 : size(positions,1)
					Δx = wrap_displacement(positions[i][1] - positions[j][1]; period = periodicity[1])
				    Δy = wrap_displacement(positions[i][2] - positions[j][2]; period = periodicity[2])
					Δz = wrap_displacement(positions[i][3] - positions[j][3]; period = periodicity[3])

		            if Δx^2 + Δy^2 + Δz^2 < minimum_distance^2
		                push!(indices_to_remove, i)
		                break
		            end
		        end
		    end
		    deleteat!(positions, indices_to_remove)

		end
	end


end

export CellList

################################################################################
#                       Structure for creating cell list
#
################################################################################
struct CellList
    start_pid::Array
    next_pid::Array{Int64, 1}

    particles::Array{Particle, 1}

    num_cells::SVector
    cell_dr::SVector

    function CellList(; particles::Array{Particle,1},dim::Int64,L::Float64, rc::Float64)
		if dim == 2
			num_cells_x = floor(Int64, L / rc)
        	num_cells_y = floor(Int64, L / rc)
        	cell_drx = L / num_cells_x
        	cell_dry = L / num_cells_y

        	start_pid = -ones(Int64, num_cells_x, num_cells_y)
        	next_pid = -ones(Int64, length(particles))

        	for (n, particle) in enumerate(particles)
            	i = floor(Int64, particle.r[1] / cell_drx) + 1
            	j = floor(Int64, particle.r[2] / cell_dry) + 1

            	if start_pid[i, j] > 0
                	next_pid[n] = start_pid[i, j]
            	end
            	start_pid[i, j] = n
        	end
			num_cells = SVector{2,Int64}([num_cells_x, num_cells_y])
			cell_dr   = SVector{2,Float64}([cell_drx, cell_dry])
		elseif dim == 3
			num_cells_x = floor(Int64, L / rc)
        	num_cells_y = floor(Int64, L / rc)
			num_cells_z = floor(Int64, L / rc)

        	cell_drx = L / num_cells_x
        	cell_dry = L / num_cells_y
			cell_drz = L / num_cells_z

        	start_pid = -ones(Int64, num_cells_x, num_cells_y, num_cells_z)
        	next_pid = -ones(Int64, length(particles))

        	for (n, particle) in enumerate(particles)
            	i = floor(Int64, particle.r[1] / cell_drx) + 1
            	j = floor(Int64, particle.r[2] / cell_dry) + 1
				k = floor(Int64, particle.r[3] / cell_drz) + 1

            	if start_pid[i, j, k] > 0
                	next_pid[n] = start_pid[i, j, k]
            	end
            	start_pid[i, j, k] = n
        	end
			num_cells = SVector{3,Int64}([num_cells_x, num_cells_y, num_cells_z])
			cell_dr   = SVector{3,Float64}([cell_drx, cell_dry, cell_drz])
		end

        new(start_pid, next_pid, particles, num_cells, cell_dr)
    end
end


export WCA
export Harmonic_Repulsive
export AbstractInteraction
abstract type AbstractInteraction end



struct WCA <: AbstractInteraction
    ϵ::Float64
    σ::Float64
    rc::Float64

    particles::Array{Particle, 1}
    cell_list::CellList

	multithreaded::Bool
    use_newton_3rd::Bool

    function WCA(; particles::Array{Particle, 1}, cell_list::CellList,
                            ϵ::Float64, σ::Float64 = 1.0, rc::Float64,
            				multithreaded::Bool = true, use_newton_3rd::Bool = false)
        new(ϵ, σ, rc, particles, cell_list,multithreaded, use_newton_3rd)
    end
end

struct Harmonic_Repulsive <: AbstractInteraction

    particles::Array{Particle, 1}
    cell_list::CellList

	k::Float64
	rc::Float64

	multithreaded::Bool
    use_newton_3rd::Bool

    function Harmonic_Repulsive(; particles::Array{Particle, 1}, cell_list::CellList,
                            k::Float64=1.0e8, rc::Float64,
            				multithreaded::Bool = true, use_newton_3rd::Bool = false)
        new(particles, cell_list,k, rc, multithreaded, use_newton_3rd)
    end
end


export Simulation

mutable struct Simulation
    descriptor::String

    L::Float64
    periodicity::SVector

    particles::Array{Particle, 1}

    cell_lists::Array{CellList, 1}
    interactions::Array{AbstractInteraction, 1}

    dt::Float64
    integrators::Array{AbstractIntegrator, 1}
    num_steps::Int64

    save_interval::Int64
    particles_to_save::Array{Particle, 1}
	output_file::String
    function Simulation(; descriptor::String = "No description given...",
                          L::Float64 = 0.0,
                          periodicity::SVector = SVector{3,Float64}([-1.0,-1.0,-1.0]),
                          particles::Array{Particle, 1} = Particle[],
                          cell_lists::Array{CellList, 1} = CellList[],
                          interactions::Array{AbstractInteraction, 1} = AbstractInteraction[],
                          dt::Float64 = 0.0,
                          integrators::Array{AbstractIntegrator, 1} = AbstractIntegrator[],
                          num_steps::Int64 = 0,
                          save_interval::Int64 = 0,
                          particles_to_save::Array{Particle, 1}=  Particle[],
						  output_file::String = "output")
        new(descriptor, L, periodicity, particles, cell_lists, interactions, dt,integrators, num_steps, save_interval, particles_to_save,output_file)
    end
end

"""
###############################################################################
"""


export wrap_displacement
export volume
export friction
export SOnes
export SZeros
export SRands
export SRandns



"""
    wrap_displacement(displacement; period)

Returns a new displacement after applying periodic boundary conditions.  The
periodicity is given by `period`.  If period < 0, then no periodicity is
applied.
"""
@inline function wrap_displacement(displacement::Float64; period::Float64)
    if period > 0.0 && abs(displacement) > period / 2
        return displacement - sign(displacement) * period
    end
    return displacement
end



@inline function volume(R::T)::T where T <: Float64
	return (4.0/3.0)*π*R*R*R
end

@inline function friction(η::T, R::T)::T where T<: Float64
	return 6*π*η*R
end



@inline function SZeros(dim::Int64)
	return @SVector zeros(dim)
end

@inline function SOnes(dim::Int64)
	return @SVector ones(dim)
end

@inline function SRands(dim::Int64)
	return @SVector rand(dim)
end

@inline function SRandns(dim::Int64)
	return @SVector randn(dim)
end
