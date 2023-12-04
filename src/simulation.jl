export sim_run

function sim_run(;
	type::String,
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
	density::T,
	η::T,
    Δt_prod::T,
    integ::String,
	random_positions::Bool,
	force_func::Function) where {N,I,T}

    #η		= T(8.9e-4)
    #density = T(1.0e3) # mass density of particles (kg/m³)
	σ = T(1.0)
	

		 
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################
	if force_func == WCA
		sigma = T(2^(1/6))*σ
	elseif force_func == harm_rep
		sigma = σ
	end

	neigh_cut_off *= sigma
	box, r_init, num_cold = initialization(homogeneous = homogeneous , dim = dim, Npart = Npart, ϕ = ϕ, fraction = fraction, sigma = sigma, random_positions = random_positions, cold_frac = cold_frac)

    Npart = length(r_init)

    for run = 1:num_runs
		simulation = Simulation()
		simulation.type = type
		simulation.neigh_update = neigh_update
		simulation.neigh_cut_off = neigh_cut_off
		simulation.num_cold = num_cold
		alpha = α₂/α₁
        output_file = "$type,Npart,$Npart,$dim-D,deltat-$Δt_prod,epsilon-$ϵ,alpha-$alpha,fraction-$ϕ,integ-$integ,run_num-$run,homo_$homogeneous"  # check this!
		simulation.part_types = ptypes
        simulation.output_file = output_file
        
		simulation.integrator = integ
		simulation.force_func = force_func
		simulation.ϵ = ϵ 
		simulation.σ = σ 
        simulation.box = box
        r0 = r_init

		if simulation.type == "Brownian"
			for i = 1:num_cold
				push!(simulation.particles, PassiveOP(part_type = ptypes[1], part_id = p_ids[1],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₁))
			end
			for i = num_cold+1:Npart
				push!(simulation.particles, PassiveOP(part_type = ptypes[2], part_id = p_ids[2],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₂))
			end
		elseif simulation.type == "Langevin"	
			for i = 1:num_cold
				push!(simulation.particles, PassiveP(part_type = ptypes[1], part_id = p_ids[1],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁))
			end
			for i = num_cold+1:Npart
				push!(simulation.particles, PassiveP(part_type = ptypes[2], part_id = p_ids[2],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂))
			end
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

		if simulation.type == "Brownian"
			simulateO!(simulation, collision_calc,box)
		elseif simulation.type == "Langevin"
			simulate!(simulation, collision_calc,box)
		end
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

	dQ = CUDA.zeros(T,Npart)
	dU = CUDA.zeros(T,Npart)
	Eₖ = CUDA.zeros(Float64,Npart)
	virial = CUDA.zeros(Float64,Npart)
	Eₚ = similar(dQ)
	
	if simulation.force_func == WCA
		NN = max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
	elseif simulation.force_func == harm_rep
		NN = max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
	end
	
	Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
	if collision_calc
		colls = CUDA.zeros(T,Npart)
		coll_switch = CuArray(falses(Npart,NN))
	end

	c1 = Float64(sqrt(simulation.particles[1].τD/simulation.particles[1].τm))
	alpha_list = [Float64(simulation.particles[i].α) for i=1:Npart]
	if simulation.integrator == "lf" || simulation.integrator == "em"
		scale = T(1.0 - c1 * simulation.dt /2)
	else
		scale = T(1.0)
	end
	c3 = [T(sqrt(2*c1*simulation.particles[i].α * scale /simulation.dt)) for i=1:Npart]
	c3 = CuVector(c3)
	alpha_d = CuVector(alpha_list)

	part_id = [simulation.particles[i].part_id for i=1:Npart]

	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	v_c = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v_c)
	f_c = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f_c)

	dQ₀ = zeros(Float64)
	dU₀	= zeros(Float64)
	Ekin = similar(dQ₀)
	virial_sum = similar(dQ₀)
	Ekin_alpha = similar(dQ₀)
	Epot = similar(dQ₀)
	coll₀ = similar(dQ₀)
	if simulation.integrator == "vv"
		f_r = CUDA.zeros(eltype(f), size(f))
		f₀ = CUDA.zeros(eltype(f), size(f))
		neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
	
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
			end
			
			update_positions_vv!(r, v, f₀, f_r, c1, Float64(simulation.dt), c3, simulation.box)
	
			
			if collision_calc
				#forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				#forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end
			
			virial!(r,f,virial)
			#update_velocities_vv!(v, f₀, f, f_r, dQ, Eₖ, c1, Float64(simulation.dt), c3)
			update_velocities_vv!(v, f₀, f, f_r, dQ, dU, Eₖ, c1, Float64(simulation.dt), c3)
			copyto!(f₀,f)
			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				# Pre-allocate
				c1_d = Float64(c1)
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval

				@. virial = virial ./ alpha_d
				virial_sum = Float64.(sum(virial)) / simulation.save_interval

				# Naive way to define dU/T
				@. dU = dU ./ alpha_d
				dU₀ = Float64.(sum(dU)) / simulation.save_interval


				Ekin = sum(Eₖ) / simulation.save_interval
				
				@. Eₖ = 2 * c1_d * Eₖ ./ alpha_d
				Ekin_alpha_numerator = sum(Eₖ)
				Ekin_alpha = (Ekin_alpha_numerator / simulation.save_interval) - length(simulation.particles)*Float64.(sum(c1_d .* length(simulation.box)))
				
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(virial, zero(eltype(virial)))
				fill!(dU, zero(eltype(dU)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						#write_log(step, simulation, Ekin, Epot, dQ₀, Ekin_alpha, coll₀)
						write_log(step, simulation, Ekin, Epot, dQ₀, virial_sum, dU₀, Ekin_alpha, coll₀)
					else
						println("The correct write_log function to write virial is not implemented yet!")
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
				end
			end
		end	
	elseif simulation.integrator == "em"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end

			update_particles_em!(r, v, f, dQ, Eₖ, c1, Float64(simulation.dt), c3,simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				# Pre-allocate
				c1_d = Float64(c1)
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				
				Ekin = sum(Eₖ) / simulation.save_interval
				
				@. Eₖ = 2 * c1_d * Eₖ ./ alpha_d
				Ekin_alpha_numerator = sum(Eₖ)
				Ekin_alpha = (Ekin_alpha_numerator / simulation.save_interval) - length(simulation.particles)*Float64.(sum(c1_d * length(simulation.box)))
				
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Ekin, Epot, dQ₀, Ekin_alpha, coll₀)
					else
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
				end
			end
		end
	elseif simulation.integrator == "lf"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end
			update_positions_lf!(r, v, Float64(simulation.dt), simulation.box)


			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end

			update_velocities_lf!(v, f, dQ, Eₖ, c1, Float64(simulation.dt), c3)
			
			update_positions_lf!(r, v, Float64(simulation.dt), simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				# Pre-allocate
				c1_d = Float64(c1)
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				
				Ekin = sum(Eₖ) / simulation.save_interval
				
				@. Eₖ = 2 * c1_d * Eₖ ./ alpha_d
				Ekin_alpha_numerator = sum(Eₖ)
				Ekin_alpha = (Ekin_alpha_numerator / simulation.save_interval) - length(simulation.particles)*Float64.(sum(c1_d * length(simulation.box)))
				
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₖ, zero(eltype(Eₖ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					write_log(step, simulation, Ekin, Epot, dQ₀, Ekin_alpha, coll₀)
					
					if collision_calc
						write_log(step, simulation, Ekin, Epot, dQ₀, Ekin_alpha, coll₀)
					else
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


#####################################################################################
#####################################################################################
#        Simulation scheme for Verlet-type and Euler-Maruyama algorithms            #
#####################################################################################
#####################################################################################
export simulateO!

function simulateO!(
	simulation::Simulation,
	collision_calc::Bool,
	box::SVector{N,T}) where {N,T}


	Npart = length(simulation.particles)

	dQ = CUDA.zeros(T,Npart)
	Eₚ = similar(dQ)
	

	if simulation.force_func == WCA
		NN = max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
	elseif simulation.force_func == harm_rep
		NN = max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
	end

	"""
	if length(box) == 2
		if simulation.force_func == WCA
			NN = hexagonal_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
		elseif simulation.force_func == harm_rep
			NN = hexagonal_neighbors(sigma = T(1.0), circ_R = simulation.neigh_cut_off)
		end
	elseif length(box) == 3
		if simulation.force_func == WCA
			NN = hcp_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
		elseif simulation.force_func == harm_rep
			NN = hcp_neighbors(sigma = T(1.0), circ_R = simulation.neigh_cut_off)
		end
	end
	"""

	Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
	if collision_calc
		colls = CUDA.zeros(T,Npart)
		coll_switch = CuArray(falses(Npart,NN))
	end

	alpha_list = [Float64(simulation.particles[i].α) for i=1:Npart]
	sqrt_2alpha = Float64.(sqrt.(2 .* alpha_list))

	if simulation.integrator == "em" || simulation.integrator == "mem" || simulation.integrator == "Heun" || simulation.integrator == "em-nocollswitch"
		scale = T(1.0)
	end

	c3 = [T(sqrt(2*simulation.particles[i].α * scale /simulation.dt)) for i=1:Npart]
	c3 = CuVector(c3)
	alpha_d = CuVector(alpha_list)
	sqrt_2alpha_d = CuVector(sqrt_2alpha)
	part_id = [simulation.particles[i].part_id for i=1:Npart]

	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	v_c = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v_c)
	f_c = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f_c)

	

	tt = simulation.particles[1].τD
	dQ₀ = zeros(Float64)
	Epot = similar(dQ₀)
	coll₀ = similar(dQ₀)
	if simulation.integrator == "em"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				#forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				#forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			
			end

			update_particles_em!(r, v, f, dQ, Float64(simulation.dt), c3, simulation.box)
			#update_particles_em!(r, v, f, dQ, Float64(simulation.dt), c3, sqrt_2alpha_d, simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
					end
				end
			end
		end
	elseif simulation.integrator == "em-nocollswitch"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				#forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				#forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			
			end

			update_particles_em!(r, v, f, dQ, Float64(simulation.dt), c3, simulation.box)
			#update_particles_em!(r, v, f, dQ, Float64(simulation.dt), c3, sqrt_2alpha_d, simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
					end
				end
			end
		end
	elseif simulation.integrator == "mem"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				#forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end

			update_particles_mem!(r, v, f, dQ, Float64(simulation.dt), c3, simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
					end
				end
			end
		end
	elseif simulation.integrator == "Heun"
		f_r = CUDA.zeros(eltype(f), size(f))
		f₀ = CUDA.zeros(eltype(f), size(f))
		r₀ = CUDA.zeros(eltype(f), size(f))
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps
			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			copyto!(r₀,r)
			if collision_calc
				forces!(r, f₀, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f₀, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end
			
			predictor_Heun!(r, f₀, f_r, Float64(simulation.dt), c3, simulation.box)


			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				#forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, alpha_d, dQ, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end
			
			corrector_Heun!(r₀, r, v, f₀, f, f_r, dQ, Float64(simulation.dt), c3, simulation.box)


			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
				
				
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
					end
				end
			end
		end
	end
	

	copyto!(r_c,r)
	copyto!(v_c,v)
	copyto!(f_c,f)
	
	[simulation.particles[i].r = r_c[i] for i=1:Npart]
	[simulation.particles[i].v = v_c[i] for i=1:Npart]
	[simulation.particles[i].f = f_c[i] for i=1:Npart]
    return nothing
end

export sim_runActiveO

function sim_runActiveO(;
    num_runs::N,
    #homogeneous::Bool,
	collision_calc::Bool,
    num_steps::N,
	save_interval::N,
    Npart::N,
    ptypes::Vector{String},
	p_ids::Vector{Int},
    dim::N,
    ϕ::T,
    #fraction::T,
	#cold_frac::T,
    R::T,
	neigh_cut_off::T,
	neigh_update::I,
	ϵ::T,
    α::T,
    τΓ::T,
    Δt_prod::T,
    integ::String,
	random_positions::Bool,
	force_func::Function) where {N,I,T}

    η		= T(8.9e-4)
	σ = T(1.0)
	

		 
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################
	if force_func == WCA
		sigma = T(2^(1/6))*σ
	elseif force_func == harm_rep
		sigma = σ
	end

	neigh_cut_off *= sigma

	if random_positions && (Npart <= 100000)
		box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
		
		r_init = [random_pos(box)]
		for _ in 1:Npart-1
			pos = random_pos(box)    
			# Check for overlap with other particles
			while check_overlap(pos, r_init, box, sigma * T(1.05))
				pos = random_pos(box)
			end        
			# Append the position to the list of positions
			push!(r_init,pos)
		end
	else
		box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
		
		r_init = rectangular_lattice(Npart,box)
		#r_init = triangular_lattice(Npart, box, σ)
		r_init = sort_pos_by_dist(r_init, zero(T), zero(T))
	end
	shuffle!(r_init)


    Npart = length(r_init)

    for run = 1:num_runs
		simulation = Simulation()
		simulation.neigh_update = neigh_update
		simulation.neigh_cut_off = neigh_cut_off
		#simulation.num_cold = num_cold
        output_file = "Brownian_APM_Npart,$Npart,deltat-$Δt_prod,epsilon-$ϵ,alpha-$α,memory-$τΓ,fraction-$ϕ,integ-$integ,run_num-$run"  # check this!
		simulation.part_types = ptypes
        simulation.output_file = output_file
        
		simulation.integrator = integ
		simulation.force_func = force_func
		simulation.ϵ = ϵ 
		simulation.σ = σ 
        simulation.box = box
        r0 = r_init
        for i = 1:Npart
            push!(simulation.particles, APMO(part_type = ptypes[1], part_id = p_ids[1],r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), r_pseu = r0[i], v_pseu = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α, τΓ = τΓ))
        end
		
		println("System initialized!")
		for i = 1:Npart
			simulation.particles[i].α = α
			simulation.particles[i].v = T(sqrt(2)) .* @SVector randn(T,dim)
			simulation.particles[i].v_pseu = T(sqrt(2*(simulation.particles[i].α+1))) .* @SVector randn(T,dim)
		end



		simulation.dt = Float64(Δt_prod)
		simulation.num_steps = num_steps
		simulation.save_interval = save_interval
		println("Simulation starts!")
		simulateAPMO!(simulation, collision_calc,box)
		yield()
    end
    return nothing
end

export simulateAPMO!

function simulateAPMO!(
	simulation::Simulation,
	collision_calc::Bool,
	box::SVector{N,T}) where {N,T}

	Npart = length(simulation.particles)

	dQ = CUDA.zeros(T,Npart)
	Eₚ = similar(dQ)
	
	if simulation.force_func == WCA
		NN = hexagonal_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
	elseif simulation.force_func == harm_rep
		NN = hexagonal_neighbors(sigma = T(1.0), circ_R = simulation.neigh_cut_off)
	end

	Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
	if collision_calc
		colls = CUDA.zeros(T,Npart)
		coll_switch = CuArray(falses(Npart,NN))
	end

	alpha_list = [Float64(simulation.particles[i].α) for i=1:Npart]

	if simulation.integrator == "em" || simulation.integrator == "Heun"
		scale = T(1.0)
	end

	"""
	c2 = [T(simulation.particles[i].τD /simulation.particles[i].τΓ) for i=1:Npart]
	c2_d = CuVector(c2)
	c3 = [T(sqrt(2*(simulation.particles[i].α+1) * scale /simulation.dt)) for i=1:Npart]
	c3_d = CuVector(c3)
	"""

	c2 = Float64(simulation.particles[1].τD /simulation.particles[1].τΓ) 
	c3 = Float64(sqrt(2*(simulation.particles[1].α + 1) * scale /simulation.dt)) 
	alpha_d = CuVector(alpha_list)

	part_id = [simulation.particles[i].part_id for i=1:Npart]

	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	v_c = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v_c)
	f_c = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f_c)

	r_pseu_c = [simulation.particles[i].r_pseu for i=1:Npart]
	r_pseu = CuVector(r_pseu_c)
	v_pseu_c = [simulation.particles[i].v_pseu for i=1:Npart]
	v_pseu = CuVector(v_pseu_c)

	dQ₀ = zeros(Float64)
	Epot = similar(dQ₀)
	coll₀ = similar(dQ₀)
	if simulation.integrator == "em"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
			end

			if collision_calc
				println("Collision is not implemented for active particles yet!")
				#forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end

			update_particles_em!(r, v, f, r_pseu, v_pseu, dQ, Float64(simulation.dt), c2, c3, simulation.box)

			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
					end
				end
			end
		end
	elseif simulation.integrator == "Heun"
		f_r = CUDA.zeros(eltype(f), size(f))
		f₀ = CUDA.zeros(eltype(f), size(f))
		r₀ = CUDA.zeros(eltype(f), size(f))
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = 0:simulation.num_steps

			if step % simulation.neigh_update == 0
				neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
			end

			copyto!(r₀,r)
			if collision_calc
				forces!(r, f₀, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f₀, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end
			
			predictor_Heun!(r, v, f₀, f_r, Float64(simulation.dt), c3, simulation.box)


			if collision_calc
				forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end
			
			corrector_Heun!(r₀, r, v, f₀, f, f_r, dQ, Float64(simulation.dt), c3, simulation.box)


			if step % simulation.save_interval == 0
				if collision_calc
					coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
				end
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64(sum(dQ)) / simulation.save_interval
				Epot = Float64(sum(Eₚ)) / simulation.save_interval
	
				fill!(dQ, zero(eltype(dQ)))
				fill!(Eₚ, zero(eltype(Eₚ)))
				
				if collision_calc
					fill!(colls, zero(eltype(colls)))
				end
	
				@async begin
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
					if collision_calc
						write_log(step, simulation, Epot, dQ₀, coll₀)
					else
						write_log(step, simulation, Epot, dQ₀)
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














