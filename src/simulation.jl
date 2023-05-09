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
    integ::String,
	random_positions::Bool) where {N,I,T}

    η		= T(8.9e-4)
    density = T(1.0e3) # mass density of particles (kg/m³)
	σ = T(1.0)
	

		 
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################
	sigma = T(2^(1/6))*σ
	neigh_cut_off *= sigma
    if homogeneous
		if random_positions
			box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
    		num_cold = ceil(Int, Npart*fraction)
			r_init = [random_pos(box)]
			for _ in 1:Npart-1
				pos = random_pos(box)    
				# Check for overlap with other particles
				while check_overlap(pos, r_init, sigma + (1e-5))
					pos = random_pos(box)
				end        
				# Append the position to the list of positions
				push!(r_init,pos)
			end
		else
			box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
    		num_cold = ceil(Int, Npart*fraction)
			r_init = rectangular_lattice(Npart,box)
			#r_init = triangular_lattice(Npart, box, σ)
			r_init = sort_pos_by_dist(r_init, zero(T), zero(T))
		end
		shuffle!(r_init)
		homog = "homogeneous"
    else
		box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
		num_cold = ceil(Int, Npart*fraction)
        r_init = cut_circle_sphere!(box, sigma, Npart, fraction, cold_frac)
		if length(r_init) <= num_cold
			rr_remain = rectangular_lattice(2Npart,box)
			rad = T(sqrt(ceil(Npart .* fraction))/(2π/(1.0675*2sqrt(3))))
			r_remain = circle_cut(rr_remain, rad, false)
			shuffle!(r_remain)
		else
			error("r_droplet size is more than cold particles size!")
		end
		n_remain = Npart - length(r_init)
		r_init = append!(r_init, r_remain[1:n_remain])
		r_init = r_init[1:Npart]
		homog = "inhomogeneous"
    end

    Npart = length(r_init)

    for run = 1:num_runs
		simulation = Simulation()
		simulation.neigh_update = neigh_update
		simulation.neigh_cut_off = neigh_cut_off
		simulation.num_cold = num_cold
        output_file = "Npart,$Npart,deltat-$Δt_prod,epsilon-$ϵ,alpha_1-$α₁,alpha_2-$α₂,fraction-$ϕ,integ-$integ,run_num-$run,$homog"  # check this!
		simulation.part_types = ptypes
        simulation.output_file = output_file
        
		simulation.integrator = integ
		simulation.ϵ = ϵ 
		simulation.σ = σ 
        simulation.box = box
        r0 = r_init
        for i = 1:num_cold
            push!(simulation.particles, PassiveP(part_type = ptypes[1], part_id = p_ids[1],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁))
        end
        for i = num_cold+1:Npart
            push!(simulation.particles, PassiveP(part_type = ptypes[2], part_id = p_ids[2],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂))
        end
		println("System initialized!")
		for i = 1:num_cold
			simulation.particles[i].α = α₁
			simulation.particles[i].v = sqrt(simulation.particles[i].α) .* @SVector randn(T,dim)
		end
		for i = num_cold+1:Npart
			simulation.particles[i].α = α₂
			simulation.particles[i].v = sqrt(simulation.particles[i].α) .* @SVector randn(T,dim)
		end


		simulation.dt = Δt_prod
		simulation.num_steps = num_steps
		simulation.save_interval = save_interval
		println("Simulation starts!")
		simulate!(simulation, collision_calc,box)
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
	box::SVector{N,T}) where {N,T}


	Npart = length(simulation.particles)

	dQ = CuVector(zeros(T,Npart))
	Eₖ = similar(dQ)
	Eₚ = similar(dQ)
	
	NN = hexagonal_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
	Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
	#println("check 1")
	if collision_calc
		colls = CuVector(zeros(T,Npart))
		coll_switch = CuArray(falses(Npart,NN))
	end
	
	#println("check 2")
	c1 = [T(sqrt(simulation.particles[i].τD/simulation.particles[i].τm)) for i=1:Npart]
	scale = similar(c1)
	#println("check 3")
	if simulation.integrator == "lf" || simulation.integrator == "em"
		scale = T(1.0) .- c1 .* simulation.dt ./2
	else
		scale = ones(Npart)
	end
	#println("check 4")
	c3 = [T(sqrt(2*c1[i]*simulation.particles[i].α * scale[i] /simulation.dt)) for i=1:Npart]
	c1 = CuVector(c1)
	c3 = CuVector(c3)

	part_id = [simulation.particles[i].part_id for i=1:Npart]

	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	v_c = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v_c)
	f_c = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f_c)
	#println("check 5")
	dQ₀ = zeros(T,Npart)
	Ekin = similar(dQ₀)
	Epot = similar(dQ₀)
	coll₀ = similar(dQ₀)

	if simulation.integrator == "vv"
		f_r_c = [simulation.particles[i].f for i=1:Npart]
		f_r_c = zero(f_r_c)
		f_r = CuVector(f_r_c)
		f₀ = zero(f)
		#println("check 6")
		neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		#println("check 7")
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end
			copyto!(f , f₀)
			update_positions_vv!(r, v, f₀, f_r, c1, simulation.dt, c3,simulation.box)

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ)
			end


			update_velocities_vv!(v, f₀, f, f_r, dQ, Eₖ, c1, simulation.dt, c3)
			copyto!(f₀ , f)
			if step % simulation.save_interval == 0
				if collision_calc
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
					copyto!(coll₀,colls)
				else
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
				end

				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end

				@async begin
					if collision_calc
						coll₀ ./= 2simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						#copyto!(r_c, Vector(r))
						#copyto!(v_c, Vector(v))
						r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀, coll₀)
					else
						#copyto!(r_c, Vector(r))
						#copyto!(v_c, Vector(v))
						r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
				end		
			end
		end
	elseif simulation.integrator == "em"
		neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ)
			end

			update_particles_em!(r, v, f, dQ, Eₖ, c1, simulation.dt, c3,simulation.box)
			if step % simulation.save_interval == 0
				if collision_calc
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
					copyto!(coll₀,colls)
					#dQ₀, Epot, Ekin, coll₀ = Vector(dQ), Vector(Eₚ), Vector(Eₖ), Vector{Float64}(colls)
				else
					#dQ₀, Epot, Ekin = Vector(dQ), Vector(Eₚ), Vector(Eₖ)
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
				end

				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end

				@spawn begin
					if collision_calc
						coll₀ ./= 2simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						copyto!(r_c, Vector(r))
						copyto!(v_c, Vector(v))
						#r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀, coll₀)
					else
						copyto!(r_c, Vector(r))
						copyto!(v_c, Vector(v))
						#r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
				end		
			end
		end
	elseif simulation.integrator == "lf"
		neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list_new!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end
			update_positions_lf!(r, v, simulation.dt, simulation.box)


			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ)
			end


			update_velocities_lf!(v, f, dQ, Eₖ, c1, simulation.dt, c3, simulation.box)
			update_positions_lf!(r, v, simulation.dt, simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
					copyto!(coll₀,colls)
				else
					copyto!(dQ₀,dQ)
					copyto!(Epot,Eₚ)
					copyto!(Ekin,Eₖ)
				end

				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end

				@spawn begin
					if collision_calc
						coll₀ ./= 2simulation.save_interval
					end
					dQ₀ ./= simulation.save_interval
					Ekin ./= simulation.save_interval
					Epot ./= simulation.save_interval
					if collision_calc
						copyto!(r_c, Vector(r))
						copyto!(v_c, Vector(v))
						#r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀, coll₀)
					else
						copyto!(r_c, Vector(r))
						copyto!(v_c, Vector(v))
						#r_c, v_c = Vector(r), Vector(v)
						write_gsd(step,simulation, part_id, r_c, v_c)
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
				end		
			end
		end
	end
	

	copyto!(r_c,r)
	copyto!(v_c,v)
	copyto!(f_c,f)
	# After finishing the simulation it saves the positions, velocities and forces back into the simulation structure! 
	[simulation.particles[i].r = r_c[i] for i=1:Npart]
	[simulation.particles[i].v = v_c[i] for i=1:Npart]
	[simulation.particles[i].f = f_c[i] for i=1:Npart]
    return nothing
end

















