export hr_min_sec
@inline function hr_min_sec(time::Float64)
    hours = trunc(Int64, time / 3600.0)
    minutes = trunc(Int64, mod(time, 3600.0) / 60.0)
    seconds = trunc(Int64, mod(time, 60.0))

    return string(hours < 10 ? "0" : "", hours,
                  minutes < 10 ? ":0" : ":", minutes,
                  seconds < 10 ? ":0" : ":", seconds)
end


export sim_run
function sim_run(;
    num_runs::N,
    homogeneous::Union{Nothing,Vector},
	collision_calc::Bool,
    num_steps::N,
	dump_freq::N,
    Npart::N,
    ptypes::Vector{String},
	p_ids::Vector{Int},
    dim::N,
    ϕ::T,
    fraction::T,
    R::T,
	ϵ::T,
    α₁::T,
    α₂::T,
    Δt_prod::T,
    integ::String) where {N,T}

    η		= T(8.9e-4)
    density = T(1.0e3) # mass density of particles (kg/m³)
	σ = T(2R/(1.0e-6))
    box = Box(dim = dim, Npart = Npart, ϕ = ϕ, σ = σ )
    num_pl = ceil(Int, Npart*fraction)

	if dim == 2
		noisefun = noise2D
	elseif dim == 3
		noisefun = noise3D
	end
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################

    if homogeneous != nothing
        if dim == 2
            r_init = rectangular_lattice(Npart, box)
        elseif dim == 3
            r_init = simplecubic_lattice(Npart, box)
        end
    else
        r_init, num_pl = cut_circle_sphere!(box, σ, Npart, fraction)
    end

    Npart = length(r_init)

    for run = 1:num_runs
		simulation = Simulation()
        output_file = "$dim,dimension_Npart,$Npart,run_num-$run"  # check this!
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

		# Check for homogeneous simulation to see wether it performs simulation up to the correct number of steps
        if homogeneous != nothing
			idx = randperm(Npart)
            for i = 1:num_pl
                simulation.particles[i].α = α₁
                simulation.particles[i].v = @SVector zeros(T,dim)
            end
            for i = num_pl+1:Npart
                simulation.particles[i].α = α₂
                simulation.particles[i].v = @SVector zeros(T,dim)
            end
			for i = 1:Npart
				simulation.particles[i].r, simulation.particles[idx[i]].r = simulation.particles[idx[i]].r, simulation.particles[i].r
			end

            Δt_relax,α_init,num_steps_relax,freq_relax = homogeneous[1],homogeneous[2],homogeneous[3],homogeneous[4]  
            batchs  = num_steps_relax ÷ freq_relax

			simulation.num_steps = num_steps_relax
        	simulation.save_interval = freq_relax
            for i = 0:batchs
                α_relax  =   α_init - i*(α_init - max(α₁,α₂))/batchs

                for particle in simulation.particles
                    particle.α = α_relax
                end
                simulation.dt = T(Δt_relax)
                simulate!(simulation, collision_calc, noisefun) 
            end
			simulation.num_steps = num_steps
        	simulation.save_interval = dump_freq
			simulate!(simulation, collision_calc, noisefun)
        elseif homogeneous == nothing
			simulation.num_steps = num_steps
        	simulation.save_interval = dump_freq
			# Check for simulation to see wether it performs simulation up to the correct number of steps 
			simulation.dt = T(Δt_prod)	
			simulate!(simulation, collision_calc, noisefun)
		end
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
	noisefun::Function)


	Npart = length(simulation.particles)
	dQ = CuVector(zeros(Float32,Npart))
	Eₖ = similar(dQ)
	Eₚ = similar(dQ)


	c1 = [(simulation.particles[i].τD/simulation.particles[i].τm) for i=1:Npart]
	c1 = CuVector(c1)
	c2 = simulation.dt
	c3 = [Float32(sqrt(2.0*simulation.particles[i].α/simulation.dt)) for i=1:Npart]
	c3 = CuVector(c3)

	part_id = [simulation.particles[i].part_id for i=1:Npart]
	r = [simulation.particles[i].r for i=1:Npart]
	r = CuVector(r)
	v = [simulation.particles[i].v for i=1:Npart]
	v = CuVector(v)
	f = [simulation.particles[i].f for i=1:Npart]
	f = CuVector(f)
	if collision_calc
		coll = CuArray(cu(zeros(Npart,Npart)))
		coll₀ = CuArray(cu(zeros(Npart,Npart)))
    	coll_switch₀ = CuArray(cu(Matrix{Int32}(I,Npart,Npart)))
	end

	dQ₀ = zero(dQ)
	Ekin = zero(Eₖ)
	Epot = zero(Eₚ)

	
	if simulation.integrator == "vv"
		f₀ = similar(f)
		for step = 0:simulation.num_steps
			f = f₀
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			fR = noisefun(Npart)
			r = update_positions_vv!(r, v, f₀, fR, c1, c2, c3)
			PBC!(r,simulation.box)
			f, Epot = forces!(r, f, Epot, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			v = update_velocities_vv!(v, f₀, f, fR, dQ₀, Ekin, c1, c2, c3)
			f₀ = f
			dQ .+= dQ₀
			Eₖ .+= Ekin
			Eₚ .+= Epot
			if step % simulation.save_interval == 0
				@async write_gsd(step,simulation, part_id, r, v)
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
			end
		end
	elseif simulation.integrator == "em"
		for step = 1:simulation.num_steps
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			fR = noisefun(Npart)
			f, Epot = forces!(r, f, Epot, simulation.box, simulation.ϵ, simulation.σ)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, simulation.σ, simulation.box)
				coll .+= coll₀
			end
			update_parts_em!(r, v, f, fR, dQ₀, Ekin, c1, c2, c3)
			PBC!(r,simulation.box)
			dQ .+= dQ₀
			Eₖ .+= Ekin
			Eₚ .+= Epot
			if step % simulation.save_interval == 0
				@async write_gsd(step,simulation, part_id, r, v)
				dQ = zero(dQ)
				Eₖ = zero(Eₖ)
				Eₚ = zero(Eₚ)
			end
		end
		# The leapfrog integrator should be revised carefully!!!!
	elseif simulation.integrator == "lf"
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
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, cut_off, box)
				coll .+= coll₀
			end
			PBC!(r,box)
			dQ .+= dQ₀ ./freq
			Eₖ .+= Ekin ./freq
			Eₚ .+= Epot ./freq
		end
	end

	if collision_calc
		coll ./= simulation.save_interval
	end
    dQ ./= simulation.save_interval
    Eₖ ./= simulation.save_interval
    Eₚ ./= simulation.save_interval
	
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
