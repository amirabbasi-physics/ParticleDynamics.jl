export sim_run

function sim_run(;
    num_runs::N,
    homogeneous::Bool,
	collision_calc::Bool,
    num_steps::N,
	save_interval::N,
    Npart::N,
    ptypes::Vector{String},
	p_ids::Vector{Int},
    dim::N,
    ϕ::T,
    fraction::T,
	cold_frac::T,
    R::T,
	neigh_cut_off::T,
	neigh_update::I,
	ϵ::T,
    α₁::T,
    α₂::T,
    Δt_prod::T,
    integ::String) where {N,I,T}

    η		= T(8.9e-4)
    density = T(1.0e3) # mass density of particles (kg/m³)
	σ = T(2.0)


    box = Box(dim = dim, Npart = Npart, ϕ = ϕ, σ = σ )
    num_pl = ceil(Int, Npart*fraction)
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################

    if homogeneous 
        r_init = [random_pos(box)]
		for i in 1:Npart-1
			pos = random_pos(box)    
			# Check for overlap with other particles
			while check_overlap(pos, r_init, σ/2)
				pos = random_pos(box)
			end        
			# Append the position to the list of positions
			push!(r_init,pos)
		end
		homog = "homogeneous"
    else
        r_init, num_pl = cut_circle_sphere!(box, σ, Npart, fraction,cold_frac)
		homog = "inhomogeneous"
    end

    Npart = length(r_init)

    for run = 1:num_runs
		simulation = Simulation()
		simulation.neigh_update = neigh_update
		simulation.neigh_cut_off = neigh_cut_off
        output_file = "Npart,$Npart,deltat-$Δt_prod,alpha_1-$α₁,alpha_2-$α₂,fraction-$ϕ,integ-$integ,run_num-$run,$homog,$neigh_update,$neigh_cut_off"  # check this!
		simulation.part_types = ptypes
        simulation.output_file = output_file
        
		simulation.integrator = integ
		simulation.ϵ = ϵ 
		simulation.σ = σ 
        simulation.box = box
        r0 = r_init
        for i = 1:num_pl
            push!(simulation.particles, PassiveP(part_type = ptypes[1], part_id = p_ids[1],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁))
        end
        for i = num_pl+1:Npart
            push!(simulation.particles, PassiveP(part_type = ptypes[2], part_id = p_ids[2],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂))
        end

		for i = 1:num_pl
			simulation.particles[i].α = α₁
			simulation.particles[i].v = sqrt(simulation.particles[i].α*(simulation.particles[i].τD/simulation.particles[i].τm)) .* @SVector randn(T,dim)
		end
		for i = num_pl+1:Npart
			simulation.particles[i].α = α₂
			simulation.particles[i].v = sqrt(simulation.particles[i].α*(simulation.particles[i].τD/simulation.particles[i].τm)) .* @SVector randn(T,dim)
		end

		simulation.dt = Δt_prod
		simulation.num_steps = num_steps
		simulation.save_interval = save_interval
		simulate!(simulation, collision_calc, num_pl)
		yield()
    end
    return nothing
end


#####################################################################################
#####################################################################################
#        Simulation scheme for Verlet-type and Euler-Maruyama algorithms            #
#####################################################################################
#####################################################################################
export simulate!

function simulate!(
	simulation::Simulation,
	collision_calc::Bool,
	num_pl::Int)



	Npart = length(simulation.particles)

	dQ = CuVector(zeros(Float64,Npart))
	Eₖ = similar(dQ)
	Eₚ = similar(dQ)

	c1 = [(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:Npart]
	c1 = CuVector(c1)
	c2 = [simulation.dt for i=1:Npart]
	c2 = CuVector(c2)
	c3 = [sqrt(2*simulation.particles[i].α/simulation.dt) for i=1:Npart]
	c3 = CuVector(c3)

	part_id = [simulation.particles[i].part_id for i=1:Npart]

	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	v_c = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v_c)
	f_c = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f_c)
	if collision_calc
		coll = CuArray(zeros(Float64,Npart,Npart))
		#coll₀ = CuArray(cu(zeros(Npart,Npart)))
    	coll_switch = CuArray(cu(Matrix{Int32}(I,Npart,Npart)))
	end

	dQ₀ = zero(dQ)
	Ekin = zero(Eₖ)
	Epot = zero(Eₚ)

	if simulation.integrator == "em_fast"
		for step = 0:simulation.num_steps
			forces!(r, f, Eₚ, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				collisions!(r, coll, coll_switch, simulation.σ, simulation.box)
			end
			EM_integrate!(r, v, f, dQ, Eₖ, c1, c2, c3,simulation.box)
			#yield()
			if step % simulation.save_interval == 0
				dQ₀, Epot, Ekin = dQ, Eₚ, Eₖ
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
				if collision_calc
					coll₀ = coll
					coll = zero(coll)
				end
				@async begin
					if collision_calc
						coll₀ ./= simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀), Array(coll₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
					else
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
					end
				end		
			end
		end
	elseif simulation.integrator == "em_fast_neigh"
		NN = hexagonal_neighbors(1.0, simulation.neigh_cut_off)
		Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)

		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end
			forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ)
			
			if collision_calc
				collisions!(r, coll, coll_switch, simulation.σ, simulation.box)
			end
			EM_integrate!(r, v, f, dQ, Eₖ, c1, c2, c3,simulation.box)
			#yield()
			if step % simulation.save_interval == 0
				dQ₀, Epot, Ekin = dQ, Eₚ, Eₖ
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
				if collision_calc
					coll₀ = coll
					coll = zero(coll)
				end
				@async begin
					if collision_calc
						coll₀ ./= simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀), Array(coll₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
					else
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
					end
				end		
			end
		end	
	elseif simulation.integrator == "em_new"
		for step = 0:simulation.num_steps
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			#f, Epot = forces!(r, f, Epot, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			em_new!(r, v, f, dQ₀, Ekin, c1, c2, c3, noisefun)
			PBC!(r,simulation.box)
			dQ .+= dQ₀
			Eₖ .+= Ekin
			Eₚ .+= Epot
			yield()
			if step % simulation.save_interval == 0
				if collision_calc
					coll ./= simulation.save_interval
				end
				dQ ./= simulation.save_interval
				Eₖ ./= simulation.save_interval
				Eₚ ./= simulation.save_interval
				if collision_calc
					r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Eₖ), Vector(Eₚ), Vector(dQ), Array(coll)
					dQ = zero(dQ)
					Eₖ = zero(Eₖ)
					Eₚ = zero(Eₚ)
					coll = zero(coll)
					write_gsd(step,simulation, part_id, r_c, v_c)
					write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
				else
					r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Eₖ), Vector(Eₚ), Vector(dQ)
					dQ = zero(dQ)
					Eₖ = zero(Eₖ)
					Eₚ = zero(Eₚ)
					write_gsd(step,simulation, part_id, r_c, v_c)
					write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
				end		
			end
		end
	
	elseif simulation.integrator == "vv"
		f₀ = zero(f)
		for step = 0:simulation.num_steps
			f = f₀
			update_positions_vv!(r, v, f₀, c1, c2, c3,simulation.box)
			forces!(r, f, Eₚ, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			update_velocities_vv!(v, f₀, f, dQ, Eₖ, c1, c2, c3, simulation.box)
			f₀ = f

			if step % simulation.save_interval == 0
				dQ₀, Epot, Ekin = dQ, Eₚ, Eₖ
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
				if collision_calc
					coll₀ = coll
					coll = zero(coll)
				end
				@async begin
					if collision_calc
						coll₀ ./= simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀), Array(coll₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
					else
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
					end
				end		
			end
		end
	elseif simulation.integrator == "vv_neigh"
		f₀ = zero(f)
		NN = hexagonal_neighbors(1.0, simulation.neigh_cut_off)
		Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
		#Neighbors = CuArray(Matrix(zeros(Int,NN,Npart)))
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)

		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			f = f₀
			update_positions_vv!(r, v, f₀, c1, c2, c3,simulation.box)
			forces!(r, f, Eₚ,Neighbors, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			update_velocities_vv!(v, f₀, f, dQ, Eₖ, c1, c2, c3, simulation.box)
			f₀ = f

			if step % simulation.save_interval == 0
				dQ₀, Epot, Ekin = dQ, Eₚ, Eₖ
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
				if collision_calc
					coll₀ = coll
					coll = zero(coll)
				end
				@async begin
					if collision_calc
						coll₀ ./= simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀), Array(coll₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
					else
						r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Ekin), Vector(Epot), Vector(dQ₀)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
					end
				end		
			end
		end
		# The leapfrog integrator should be revised carefully!!!!
	"""
	elseif simulation.integrator == "lf"
		for step = 0:simulation.num_steps
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			fR = noisefun(Npart)
			update_positions_lf!(r, v, c2)
			PBC!(r,simulation.box)
			#f, Epot = forces!(r, f, Epot, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			update_velocities_lf!(v, f, fR, dQ₀, Ekin, c1, c2, c3)
			update_positions_lf!(r, v, c2)
			PBC!(r,simulation.box)
			dQ .+= dQ₀ 
			Eₖ .+= Ekin
			Eₚ .+= Epot
			yield()
			if step % simulation.save_interval == 0
				if collision_calc
					coll ./= simulation.save_interval
				end
				dQ ./= simulation.save_interval
				Eₖ ./= simulation.save_interval
				Eₚ ./= simulation.save_interval
				if collision_calc
					r_c, v_c, Eₖ_c, Eₚ_c, dQ_c, coll_c = Vector(r), Vector(v), Vector(Eₖ), Vector(Eₚ), Vector(dQ), Array(coll)
					dQ = zero(dQ)
					Eₖ = zero(Eₖ)
					Eₚ = zero(Eₚ)
					coll = zero(coll)
					write_gsd(step,simulation, part_id, r_c, v_c)
					write_log(step, simulation, num_pl, Eₖ_c, Eₚ_c, dQ_c,coll_c)
				else
					r_c, v_c, Eₖ_c, Eₚ_c, dQ_c = Vector(r), Vector(v), Vector(Eₖ), Vector(Eₚ), Vector(dQ)
					dQ = zero(dQ)
					Eₖ = zero(Eₖ)
					Eₚ = zero(Eₚ)
					write_gsd(step,simulation, part_id, r_c, v_c)
					write_log(step, simulation, Eₖ_c, Eₚ_c, dQ_c)
				end		
			end
		end
	"""
	end
	
	r_c, v_c, f_c = Vector(r), Vector(v), Vector(f)
	# After finishing the simulation it saves the positions, velocities and forces back into the simulation structure! 
	[simulation.particles[i].r = r_c[i] for i=1:Npart]
	[simulation.particles[i].v = v_c[i] for i=1:Npart]
	[simulation.particles[i].f = f_c[i] for i=1:Npart]
    return nothing
end





















#####################################################################################
#####################################################################################
#                      Simulation scheme for leap-frog algorithm                    #
#####################################################################################
#####################################################################################
function simulation_lf!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r::CuVector{SVector{N,T}},
	v::CuVector{SVector{N,T}},
	f::CuVector{SVector{N,T}},
	fR::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁::CuVector{T},
	c₂::CuVector{T},
	c₃::CuVector{T},
	ϵ::T,
	cut_off::T,
	box::SVector{N,T},
	forces!::Function,
	update_positions_lf!::Function,
	update_velocities_lf!::Function,
	noisefun::Function) where {N,T}
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)

    for _ in 1:freq
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR = noisefun(Npart)
        update_positions_lf!(r, v, c₂)
		PBC!(r,box)
		f, Epot = forces!(r, f, Epot, box, ϵ, cut_off)
		update_velocities_lf!(v, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
        update_positions_lf!(r, v, c₂)
		PBC!(r,box)
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end
