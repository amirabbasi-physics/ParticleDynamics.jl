export sim_run, relax_system!


function sim_run(;
    type::String,
    restart::Union{String,Nothing},
    num_runs::N,
    homogeneous::Union{String,Bool},
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
    force_func::Function,
    relax_steps::N,
    relax_temp::T) where {N,I,T}

    σ = T(1.0)
    
    prev_step = 0

    if force_func == WCA
        sigma = T(2^(1/6))*σ
    elseif force_func == Harmonic
        sigma = σ
    end

    neigh_cut_off *= sigma
    if restart != nothing
        prev_step, r_init, velocities, p_ids, part_types, box, num_cold = read_last_gsd(restart, dim)
    else
        box, r_init, num_cold = initialization(homogeneous = homogeneous, dim = dim, Npart = Npart, ϕ = ϕ, fraction = fraction, sigma = sigma, random_positions = random_positions, cold_frac = cold_frac)
    end

    Npart = length(r_init)

    for run = 1:num_runs
        simulation = Simulation()
        simulation.type = type
        simulation.neigh_update = neigh_update
        simulation.neigh_cut_off = neigh_cut_off
        simulation.num_cold = num_cold
        alpha = α₂/α₁
        output_file = "$type,Npart,$Npart,$dim-D,deltat-$Δt_prod,epsilon-$ϵ,alpha-$alpha,fraction-$ϕ,integ-$integ,run_num-$run,homo_$homogeneous"
        simulation.part_types = ptypes
        simulation.output_file = output_file
        
        simulation.integrator = integ
        simulation.force_func = force_func
        simulation.ϵ = ϵ
        simulation.σ = σ
        simulation.box = box
        r0 = r_init

        if restart !== nothing
            if simulation.type == "Brownian"
                for i = 1:num_cold
                    push!(simulation.particles, PassiveOP(part_type = ptypes[1], part_id = p_ids[1], r = r0[i], v = velocities[i], f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₁))
                end
                for i = num_cold+1:Npart
                    push!(simulation.particles, PassiveOP(part_type = ptypes[2], part_id = p_ids[2], r = r0[i], v = velocities[i], f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₂))
                end
            elseif simulation.type == "Langevin"
                for i = 1:num_cold
                    push!(simulation.particles, PassiveP(part_type = ptypes[1], part_id = p_ids[1], r = r0[i], v = velocities[i], f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁))
                end
                for i = num_cold+1:Npart
                    push!(simulation.particles, PassiveP(part_type = ptypes[2], part_id = p_ids[2], r = r0[i], v = velocities[i], f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂))
                end
            end
        else
            if simulation.type == "Brownian"
                for i = 1:num_cold
                    push!(simulation.particles, PassiveOP(part_type = ptypes[1], part_id = p_ids[1], r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₁))
                end
                for i = num_cold+1:Npart
                    push!(simulation.particles, PassiveOP(part_type = ptypes[2], part_id = p_ids[2], r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), η = η, Radii = R, α = α₂))
                end
            elseif simulation.type == "Langevin"
                for i = 1:num_cold
                    push!(simulation.particles, PassiveP(part_type = ptypes[1], part_id = p_ids[1], r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁))
                end
                for i = num_cold+1:Npart
                    push!(simulation.particles, PassiveP(part_type = ptypes[2], part_id = p_ids[2], r = r0[i], v = SVector{dim,T}(zeros(T,dim)), f = SVector{dim,T}(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂))
                end
            end
        end

		"""
        println("System initialized!")

        # Relax the system with a single temperature
        relax_system!(simulation, box, relax_steps, relax_temp)

		println("Relaxation is done!")

		"""

		
        for i = 1:num_cold
            simulation.particles[i].α = α₁
            if restart !== nothing
                simulation.particles[i].v = sqrt(simulation.particles[i].α) .* @SVector randn(T,dim)
            end
        end
        for i = num_cold+1:Npart
            simulation.particles[i].α = α₂
            if restart !== nothing
                simulation.particles[i].v = sqrt(simulation.particles[i].α) .* @SVector randn(T,dim)
            end
        end

        simulation.dt = Δt_prod
        simulation.num_steps = num_steps
        simulation.save_interval = save_interval
        println("Simulation starts!")

        if simulation.type == "Brownian"
            simulateO!(simulation, collision_calc, box, prev_step, homogeneous)
        elseif simulation.type == "Langevin"
            simulate!(simulation, collision_calc, box, prev_step, homogeneous)
        end
        yield()
    end
    return nothing
end

function relax_system!(
    simulation::Simulation,
    box::SVector{N,T}, 
    num_steps::I,
    single_temp::T) where {N,I,T}
    
    Npart = length(simulation.particles)

    dQ = CUDA.zeros(T,Npart)
    dU = CUDA.zeros(T,Npart)
    Eₖ = CUDA.zeros(Float64,Npart)
    virial = CUDA.zeros(Float64,Npart)
    Eₚ = similar(dQ)
    
    if simulation.force_func == WCA
        NN = max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
    elseif simulation.force_func == Harmonic
        NN = max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
    end
    
    Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))

    c1 = Float64(sqrt(simulation.particles[1].τD/simulation.particles[1].τm))
    c3 = [T(sqrt(2*c1*single_temp/simulation.dt)) for i=1:Npart]
    c3 = CuVector(c3)

    part_id = [simulation.particles[i].part_id for i=1:Npart]
    part_id_d = CuVector(part_id)
    r_c = [simulation.particles[i].r for i=1:Npart]
    r = CuVector(r_c)
    v_c = [simulation.particles[i].v for i=1:Npart]
    v = CuVector(v_c)
    f_c = [simulation.particles[i].f for i=1:Npart]
    f = CuVector(f_c)

    neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
    f_r = CUDA.zeros(eltype(f), size(f))
	f₀ = CUDA.zeros(eltype(f), size(f))

    for step = 1:num_steps
        if step % simulation.neigh_update == 0
            neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
        end

		if step <= 1e2
			if simulation.force_func == Harmonic
				ϵ = 10000.0f0
			elseif simulation.force_func == WCA
				ϵ = 0.1f0
			end
		else
			ϵ = simulation.ϵ
		end

		update_positions_vv!(r, v, f₀, f_r, c1, Float64(simulation.dt), c3, simulation.box)
        forces!(r, f, Eₚ, Neighbors, simulation.box, ϵ, simulation.σ, simulation.force_func)
        update_velocities_vv!(v, f₀, f, f_r, dQ, dU, Eₖ, c1, Float64(simulation.dt), c3)
		copyto!(f₀,f)
    end

    # Update particle states
    copyto!(r_c, r)
    copyto!(v_c, v)
    copyto!(f_c, f)
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
export simulate!


function simulate!(
	simulation::Simulation,
	collision_calc::Bool,
	box::SVector{N,T}, 
	prev_step::I,
	homogeneous::Union{Bool, String}) where {N, I,T}


	Npart = length(simulation.particles)

	dQ = CUDA.zeros(T,Npart)
	dU = CUDA.zeros(T,Npart)
	Eₖ = CUDA.zeros(Float64,Npart)
	virial = CUDA.zeros(Float64,Npart)
	Eₚ = similar(dQ)
	
	if simulation.force_func == WCA
		NN = max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
	elseif simulation.force_func == Harmonic
		NN = max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
	end
	
	Neighbors = CuArray(Matrix(zeros(Int,Npart,NN)))
	if collision_calc
		colls_c = [SVector{2,I}(zeros(I,2)) for i = 1:Npart]
		colls = CuVector(colls_c)
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
	#println(c3)
	c3 = CuVector(c3)
	alpha_d = CuVector(alpha_list)

	part_id = [simulation.particles[i].part_id for i=1:Npart]
	part_id_d = CuVector(part_id)
	r_c = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r_c)
	#rr = r
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
	coll₀ = zeros(Float64,3)
	if prev_step !== 0
		prev_step += 1
	end
	
	if simulation.integrator == "vv"
		f_r = CUDA.zeros(eltype(f), size(f))
		f₀ = CUDA.zeros(eltype(f), size(f))
		neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
	
		for step = prev_step:simulation.num_steps + prev_step
			if step % simulation.neigh_update == 0
				fill!(Neighbors, zero(eltype(Neighbors)))
				neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
			end	

			update_positions_vv!(r, v, f₀, f_r, c1, Float64(simulation.dt), c3, simulation.box)
			
			if collision_calc
				#forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, coll_switch, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, Eₚ, Neighbors, part_id_d, colls, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			else
				#forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
				forces!(r, f, Eₚ, Neighbors, simulation.box, simulation.ϵ, simulation.σ, simulation.force_func)
			end

		
			#virial!(r,f,virial)
			update_velocities_vv!(v, f₀, f, f_r, dQ, dU, Eₖ, c1, Float64(simulation.dt), c3)
			copyto!(f₀,f)
			if step % simulation.save_interval == 0
				if collision_calc
					copyto!(colls_c,Array(colls))
					copyto!(coll₀,[sum([colls_c[i][1] for i = 1:simulation.num_cold])/2, sum([colls_c[i][2] for i in 1:length(colls_c)])/2, sum([colls_c[i][1] for i = simulation.num_cold+1:length(colls_c)])/2] ./ (simulation.save_interval*simulation.dt))
				end

				c1_d = Float64(c1)
				
				@. dQ = dQ ./ alpha_d
				dQ₀ = Float64.(sum(dQ)) / simulation.save_interval

				@. virial = virial ./ alpha_d
				virial_sum = Float64.(sum(virial)) / simulation.save_interval

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
				
				
				begin
					if collision_calc
						#write_log(step, simulation, Ekin, Epot, dQ₀, Ekin_alpha, coll₀)
						write_log(step, simulation, Ekin, Epot, dQ₀, virial_sum, dU₀, Ekin_alpha, coll₀)
					else
						println("The correct write_log function to write virial is not implemented yet!")
						write_log(step, simulation, Ekin, Epot, dQ₀)
					end
					r_c, v_c = Vector(r), Vector(v)
					write_gsd(step, simulation, part_id, r_c, v_c)
				end
				
			end
		end	
	elseif simulation.integrator == "em"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = prev_step:simulation.num_steps + prev_step

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
		for step = prev_step:simulation.num_steps + prev_step
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
	box::SVector{N,T},
	prev_step::I,
	homogeneous::Union{Bool, String}) where {N,I,T}


	Npart = length(simulation.particles)

	dQ = CUDA.zeros(T,Npart)
	Eₚ = similar(dQ)
	

	if simulation.force_func == WCA
		NN = max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
	elseif simulation.force_func == Harmonic
		NN = max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
	end

	"""
	if length(box) == 2
		if simulation.force_func == WCA
			NN = hexagonal_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
		elseif simulation.force_func == Harmonic
			NN = hexagonal_neighbors(sigma = T(1.0), circ_R = simulation.neigh_cut_off)
		end
	elseif length(box) == 3
		if simulation.force_func == WCA
			NN = hcp_neighbors(sigma = T(2^(1/6)), circ_R = simulation.neigh_cut_off)
		elseif simulation.force_func == Harmonic
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

	if prev_step !== 0
		prev_step += 1
	end

	if simulation.integrator == "em"
		neighbor_list!(r,Neighbors,simulation.neigh_cut_off,simulation.box)
		for step = prev_step:simulation.num_steps + prev_step

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
		for step = prev_step:simulation.num_steps + prev_step

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
		for step = prev_step:simulation.num_steps + prev_step

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
		for step = prev_step:simulation.num_steps + prev_step
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
	elseif force_func == Harmonic
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
	elseif simulation.force_func == Harmonic
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
















"""
function initialize_simulation_arrays(simulation::Simulation, Npart::Int, T::DataType, box::SVector{N,T}) where {N, T}
    dQ = CUDA.zeros(T, Npart)
    dU = CUDA.zeros(T, Npart)
    Eₖ = CUDA.zeros(Float64, Npart)
    virial = CUDA.zeros(Float64, Npart)
    Eₚ = similar(dQ)

    NN = determine_max_neighbors(simulation, T, box)

    Neighbors = CuArray(Matrix(zeros(Int, Npart, NN)))
    colls, coll_switch = initialize_collision_arrays(Npart, NN)

    c1, c3, alpha_d, part_id, r, rr, v, f = initialize_particle_properties(simulation, T, Npart, box)

    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀ = initialize_save_interval_variables()

    return dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, coll_switch, c1, c3, alpha_d, part_id, r, rr, v, f, dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
end

function determine_max_neighbors(simulation::Simulation, T::DataType, box::SVector{N,T}) where {N, T}
    if simulation.force_func == WCA
        return max_neighbors(sigma = T(2^(1/6)), R = simulation.neigh_cut_off, box = box)
    elseif simulation.force_func == Harmonic
        return max_neighbors(sigma = T(1.0), R = simulation.neigh_cut_off, box = box)
    end
end

function initialize_collision_arrays(Npart::Int, NN::Int)
    colls = CUDA.zeros(T, Npart)
    coll_switch = CuArray(falses(Npart, NN))
    return colls, coll_switch
end

function initialize_particle_properties(simulation::Simulation, T::DataType, Npart::Int, box::SVector{N,T}) where {N, T}
    c1 = Float64(sqrt(simulation.particles[1].τD / simulation.particles[1].τm))
    alpha_list = [Float64(simulation.particles[i].α) for i = 1:Npart]

    scale = if simulation.integrator in ["lf", "em"]
                T(1.0 - c1 * simulation.dt / 2)
             else
                T(1.0)
             end

    c3 = [T(sqrt(2 * c1 * simulation.particles[i].α * scale / simulation.dt)) for i = 1:Npart]
    c3 = CuVector(c3)
    alpha_d = CuVector(alpha_list)
    part_id = [simulation.particles[i].part_id for i = 1:Npart]

    r_c = [simulation.particles[i].r for i = 1:Npart]
    r = CuVector(r_c)
    rr = r
    v_c = [simulation.particles[i].v for i = 1:Npart]
    v = CuVector(v_c)
    f_c = [simulation.particles[i].f for i = 1:Npart]
    f = CuVector(f_c)

    return c1, c3, alpha_d, part_id, r, rr, v, f
end

function initialize_save_interval_variables()
    dQ₀ = zeros(Float64)
    dU₀ = zeros(Float64)
    Ekin = similar(dQ₀)
    virial_sum = similar(dQ₀)
    Ekin_alpha = similar(dQ₀)
    Epot = similar(dQ₀)
    coll₀ = similar(dQ₀)

    return dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
end


function simulate_vv!(
    simulation::Simulation,
    collision_calc::Bool,
    homogeneous::Union{Bool, String},
    prev_step::I,
    dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
) where {I}

f_r = CUDA.zeros(eltype(f), size(f))
f₀ = CUDA.zeros(eltype(f), size(f))
neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)

for step = prev_step:simulation.num_steps + prev_step
    neighbor_list_update!(simulation, step, r, Neighbors)

    ϵ = determine_epsilon(simulation, step, homogeneous)
    update_positions_vv!(r, v, f₀, f_r, c1, Float64(simulation.dt), c3, simulation.box)

    apply_forces!(simulation, collision_calc, r, f, Eₚ, Neighbors, colls, ϵ)
    virial!(r, f, virial)
    update_velocities_vv!(v, f₀, f, f_r, dQ, dU, Eₖ, c1, Float64(simulation.dt), c3)
    copyto!(f₀, f)

    if step % simulation.save_interval == 0
        perform_save_interval_operations!(simulation, collision_calc, step, colls, c1, alpha_d, dQ, dU, Eₖ, virial, Eₚ, dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀, part_id, r, v)
    end
end
end

function neighbor_list_update!(simulation::Simulation, step::Int, r, Neighbors)
if step % simulation.neigh_update == 0
    neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
end
end

function determine_epsilon(simulation::Simulation, step::Int, homogeneous::Union{Bool, String})
if homogeneous !== true && step <= 1e5
    if simulation.force_func == Harmonic
        return 10000.0f0
    elseif simulation.force_func == WCA
        return 0.1f0
    end
else
    return simulation.ϵ
end
end

function apply_forces!(
    simulation::Simulation,
    collision_calc::Bool,
    r, f, Eₚ, Neighbors, colls, ϵ
)
if collision_calc
    forces!(r, f, Eₚ, Neighbors, simulation.num_cold, colls, simulation.box, ϵ, simulation.σ, simulation.force_func)
else
    forces!(r, f, Eₚ, Neighbors, simulation.box, ϵ, simulation.σ, simulation.force_func)
end
end

function perform_save_interval_operations!(
    simulation::Simulation,
    collision_calc::Bool,
    step::Int,
    colls, c1, alpha_d, dQ, dU, Eₖ, virial, Eₚ, 
    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀, part_id, r, v
)

    if collision_calc
        coll₀ = Float64.(sum(colls)) / (2 * simulation.save_interval)
    end

    c1_d = Float64(c1)

    @. dQ = dQ ./ alpha_d
    dQ₀ = Float64.(sum(dQ)) / simulation.save_interval

    @. virial = virial ./ alpha_d
    virial_sum = Float64.(sum(virial)) / simulation.save_interval

    @. dU = dU ./ alpha_d
    dU₀ = Float64.(sum(dU)) / simulation.save_interval

    Ekin = sum(Eₖ) / simulation.save_interval

    @. Eₖ = 2 * c1_d * Eₖ ./ alpha_d
    Ekin_alpha_numerator = sum(Eₖ)
    Ekin_alpha = (Ekin_alpha_numerator / simulation.save_interval) - length(simulation.particles) * Float64.(sum(c1_d .* length(simulation.box)))

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
            write_log(step, simulation, Ekin, Epot, dQ₀, virial_sum, dU₀, Ekin_alpha, coll₀)
        else
            println("The correct write_log function to write virial is not implemented yet!")
            write_log(step, simulation, Ekin, Epot, dQ₀)
        end
    end
end



function simulate_em!(
    simulation::Simulation,
    collision_calc::Bool,
    prev_step::I,
    dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
) where {I}

    neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
    for step = prev_step:simulation.num_steps + prev_step
        neighbor_list_update!(simulation, step, r, Neighbors)

        apply_forces!(simulation, collision_calc, r, f, Eₚ, Neighbors, colls, simulation.ϵ)
        update_particles_em!(r, v, f, dQ, Eₖ, c1, Float64(simulation.dt), c3, simulation.box)

        if step % simulation.save_interval == 0
            perform_save_interval_operations!(simulation, collision_calc, step, colls, c1, alpha_d, dQ, dU, Eₖ, virial, Eₚ, dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀, part_id, r, v)
        end
    end
end

function simulate_lf!(
    simulation::Simulation,
    collision_calc::Bool,
    prev_step::I,
    dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
) where {I}

    neighbor_list!(r, Neighbors, simulation.neigh_cut_off, simulation.box)
    for step = prev_step:simulation.num_steps + prev_step
        neighbor_list_update!(simulation, step, r, Neighbors)

        update_positions_lf!(r, v, Float64(simulation.dt), simulation.box)
        apply_forces!(simulation, collision_calc, r, f, Eₚ, Neighbors, colls, simulation.ϵ)
        update_velocities_lf!(v, f, dQ, Eₖ, c1, Float64(simulation.dt), c3)
        update_positions_lf!(r, v, Float64(simulation.dt), simulation.box)

        if step % simulation.save_interval == 0
            perform_save_interval_operations!(simulation, collision_calc, step, colls, c1, alpha_d, dQ, dU, Eₖ, virial, Eₚ, dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀, part_id, r, v)
        end
    end
end


function simulate!(
    simulation::Simulation,
    collision_calc::Bool,
    box::SVector{N,T}, 
    prev_step::I,
    homogeneous::Union{Bool, String}
) where {N, I, T}

    Npart = length(simulation.particles)
    dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, rr, v, f, 
    dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀ = initialize_simulation_arrays(simulation, Npart, T, box)

    if prev_step !== 0
        prev_step += 1
    end

    if simulation.integrator == "vv"
        simulate_vv!(
            simulation, collision_calc, homogeneous, prev_step,
            dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
            dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
        )
    elseif simulation.integrator == "em"
        simulate_em!(
            simulation, collision_calc, prev_step,
            dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
            dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
        )
    elseif simulation.integrator == "lf"
        simulate_lf!(
            simulation, collision_calc, prev_step,
            dQ, dU, Eₖ, virial, Eₚ, Neighbors, colls, c1, c3, alpha_d, part_id, r, v, f, 
            dQ₀, dU₀, Ekin, virial_sum, Ekin_alpha, Epot, coll₀
        )
    end

    finalize_simulation(simulation, r, v, f, Npart)
    return nothing
end

function finalize_simulation(simulation::Simulation, r, v, f, Npart::Int)
    r_c, v_c, f_c = Vector(r), Vector(v), Vector(f)
    copyto!(r_c, r)
    copyto!(v_c, v)
    copyto!(f_c, f)
    # After finishing the simulation it saves the positions, velocities and forces back into the simulation structure! 
    [simulation.particles[i].r = r_c[i] for i = 1:Npart]
    [simulation.particles[i].v = v_c[i] for i = 1:Npart]
    [simulation.particles[i].f = f_c[i] for i = 1:Npart]
end
"""










