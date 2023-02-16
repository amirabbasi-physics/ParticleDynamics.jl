export sim_run
function sim_run(;
    num_runs::N,
    homogeneous::Union{Nothing,Array{4,1}},
    nsteps::N,
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
    simulation = Simulation()
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
        out_file = "GPU_$Npart,dim_$dim,phi_$ϕ,alpha1_$α₁,alpha2_$α₂,epsilon$ϵ,dt$Δt₂,ns,$run,$integ"   # check this!
        r0 = r_init
        for i = 1:num_pl
            push!(simulation.particles, PassiveP(part_type = ptypes[1], r = r0[i], v = SVector(randn(T,dim)), f = SVector(zeros(T,dim)), density = density, η = η, Radii = R, α = α₁, Temp = Temp))
        end
        for i = num_pl+1:Npart
            push!(simulation.particles, PassiveP(part_type = ptypes[2], r = r0[i], v = SVector(randn(T,dim)), f = SVector(zeros(T,dim)), density = density, η = η, Radii = R, α = α₂, Temp = Temp))
        end

        if homogeneous != nothing
            Δt,α_init,nsteps_relax,freq_relax = homogeneous[1],homogeneous[2],homogeneous[3],homogeneous[4]  
            batchs  = nsteps_relax ÷ freq_relax
            for i = 0:batchs
                α_relax  =   α_init - i*(α_init - max(α₁,α₂))/batchs

                for particle in simulation.particles
                    particle.α = α_relax
                end
                Δt = T(Δt₁)
                steps = freq_relax
                simulate!(out_file, integ, collision_calc, dim, steps, freq, simulation.particles, interaction_pot, box) 
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
        steps = nsteps ÷ dump_freq

        simulate!(out_file, integ, collision_calc, dim, steps, freq, simulation.particles, interaction_pot, box) 
    end
    return nothing
end


function simulate!(
    out_file::String,
    integ::String,
    collision_calc::Bool, 
    dim::Int,
    steps::Int,
    freq::Int,
    particles::Array{Particle,1},
    interaction_pot::Interaction,
    box::SVector{N,T}) where {N,T}

    if dim == 2
        noise_fun = noise2D(args...)
    elseif dim == 3
        noise_fun = noise3D(args...)
    end
    
    if integ == "em"
        if collision_calc
            simulation_em!(dim, steps, freq, out_file, particles, box, interaction_pot, forces!, collisions!,update_parts_em!, noise_fun)
        else
            simulation_em!(dim, steps, freq, out_file, particles, box, interaction_pot, forces!,update_parts_em!, noise_fun)
        end
    elseif integ == "vv"
        if collision_calc
            simulation_vv!(dim, steps, freq, out_file, particles, box, interaction_pot, forces!, collisions!,update_positions_vv!,update_velocities_vv!, noise_fun)
        else
            simulation_vv!(dim, steps, freq, out_file, particles, box, interaction_pot, forces!, update_positions_vv!,update_velocities_vv!, noise_fun)
        end
    end
    
    return nothing
end
