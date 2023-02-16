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
    homogeneous::Union{Nothing,Array{4,1}},
    num_steps::N,
    Npart::N,
    ptypes::Vector{String},
    interaction_pot::Interaction,
    dim::N,
    ϕ::T,
    fraction::T,
    Temp::T,
    R::T,
    α₁::T,
    α₂::T,
    Δt_prod::T,
    dump_freq::N,
    integ::String) where {N,T}

    η		= T(8.9e-4)
    density = T(1.0e3) # mass density of particles (kg/m³)

    box = Box(dim = dim, Npart = Npart, ϕ = ϕ, σ = (2R/10^(-6)))
    num_pl = ceil(Int, Npart*fraction)
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
        r_init, num_pl = cut_circle_sphere!(box, R, Npart, fraction)
    end

    Npart = length(r_init)

    for run = 1:num_runs
        output_file = "GPU_$Npart,dim_$dim,phi_$ϕ,alpha1_$α₁,alpha2_$α₂,epsilon$ϵ,dt$Δt₂,ns,$run,$integ"   # check this!
        simulation = Simulation()
        simulation.output_file = output_file
        simulation.num_steps = num_steps
        simulation.save_interval = dump_freq

        simulation.interaction_type = interaction_pot
        simulation.box = box
        r0 = r_init
        for i = 1:num_pl
            push!(simulation.particles, PassiveP(part_type = ptypes[1], r = r0[i], v = SVector(randn(T,dim)), f = SVector(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁, Temp = Temp))
        end
        for i = num_pl+1:Npart
            push!(simulation.particles, PassiveP(part_type = ptypes[2], r = r0[i], v = SVector(randn(T,dim)), f = SVector(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂, Temp = Temp))
        end

        if homogeneous != nothing
            Δt,α_init,num_steps_relax,freq_relax = homogeneous[1],homogeneous[2],homogeneous[3],homogeneous[4]  
            batchs  = num_steps_relax ÷ freq_relax
            for i = 0:batchs
                α_relax  =   α_init - i*(α_init - max(α₁,α₂))/batchs

                for particle in simulation.particles
                    particle.α = α_relax
                end
                Δt = T(Δt₁)
                steps = freq_relax
                simulate!(simulation, collision_calc, noisefun) 
            end

            idx = randperm(Npart)
            for i = 1:num_pl
                simulation.particles[i].α = α₁
                simulation.particles[i].v = @SVector zeros(T,dim)
                simulation.particles[i].r, simulation.particles[idx[i]].r = simulation.particles[idx[i]].r, simulation.particles[i].r
            end
            for i = num_pl+1:Npart
                simulation.particles[i].α = α₂
                simulation.particles[i].v = @SVector zeros(T,dim)
                simulation.particles[i].r, simulation.particles[idx[i]].r = simulation.particles[idx[i]].r, simulation.particles[i].r
            end
        end

        ###############################################################################
        #                           Production run
        ###############################################################################

        Δt = Δt_prod 
        freq = dump_freq
        steps = num_steps ÷ dump_freq

        simulate!(simulation, collision_calc, noisefun) 
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
	noisefun::Function) where {N,T}

	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)

	if collision_calc
		coll = CuArray(cu(zeros(Npart,Npart)))
		coll₀ = CuArray(cu(zeros(Npart,Npart)))
    	coll_switch₀ = CuArray(cu(Matrix{Int32}(I,Npart,Npart)))
	end

	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)

	if simulation.integrator == "vv"
		for _ in 1:freq
			f = f₀
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			fR = noisefun(Npart)
			update_positions_vv!(r, v, f₀, fR, c₁, c₂, c₃)
			PBC!(r,box)
			f, Epot = forces!(r, f, Epot, box, ϵ, cut_off)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, cut_off, box)
				coll .+= coll₀
			end
			update_velocities_vv!(v, f₀, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
			f₀ = f
			dQ .+= dQ₀
			Eₖ .+= Ekin
			Eₚ .+= Epot
		end
	elseif simulation.integrator == "em"
		for _ in 1:freq
			dQ₀ = zero(dQ₀)
			Ekin = zero(Ekin)
			Epot = zero(Epot)
			fR = noisefun(Npart)
			f, Epot = forces!(r, f, Epot, box, ϵ, cut_off)
			if collision_calc
				coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, cut_off, box)
				coll .+= coll₀
			end
			update_parts_em!(r, v, f, fR, dQ₀,Ekin, c₁, c₂, c₃)
			PBC!(r,box)
			dQ .+= dQ₀ ./freq
			Eₖ .+= Ekin ./freq
			Eₚ .+= Epot ./freq
		end
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
		coll ./= freq
	end
    dQ ./= freq
    Eₖ ./= freq
    Eₚ ./= freq
    return r, v, f, dQ, Eₖ, Eₚ, coll
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
"""