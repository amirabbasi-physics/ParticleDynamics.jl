export sim_run
function sim_run(;
    num_runs::N,
    homogeneous::Union{Nothing,Array{4,1}},
    nsteps::N,
    Npart::N,
    ptypes::Vector{String},
    dim::N,
    ϕ::T,
    fraction::T,
    Temp::T,
    R::T,
    α₁::T,
    α₂::T,
    Δt_prod::T,
    dump_freq::N,
    integ::String
    ) where {N,T}

    η		= T(8.9e-4)
    density = T(1.0e3) # mass density of particles (kg/m³)

    box = Box(dim = dim, Npart = Npart, ϕ = ϕ, σ = (2R/10^6))
    num_pl = ceil(Int, Npart*fraction)
    ###############################################################################
    #   Initializing the system to get randomly distributed positions
    ###############################################################################
    if dim == 2
        r_init = rectangular_lattice(Npart, box)
    elseif dim == 3
        r_init = simplecubic_lattice(Npart, box)
    end

    Npart = length(r_init)

    for run = 1:num_runs
        out_file = "GPU_$Npart,dim_$dim,rho_$ρ,alpha1_$α₁,alpha2_$α₂,epsilon$ϵ,dt$Δt₂,ns,$run,$integ"
        r0 = r_init
        for i = 1:num_pl
            push!(simulation.particles, PassiveP(part_type = ptypes[1], r = r0[i], v = @SVector randn(T,dim), f = @SVector zeros(T,dim), density = density, η = η, Radii = R, α = α₁, Temp = Temp))
        end
        for i = num_pl+1:Npart
            push!(simulation.particles, PassiveP(part_type = ptypes[2], r = r0[i], v = @SVector randn(T,dim), f = @SVector zeros(T,dim), density = density, η = η, Radii = R, α = α₂, Temp = Temp))
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
                simulate!(out_file, integ, collision_calc, dim, steps, freq, particles, interaction_pot, box) 
            end

            idx = randperm(Npart)
            for i = 1:num_pl
                simulation.particles[i].α = α₁
                simulation.particles[i].v = @SVector zeros(T,dim)
                simulation.particles[i].position, simulation.particles[idx[i]].position = simulation.particles[idx[i]].position, simulation.particles[i].position
            end
            for i = num_pl+1:Npart
                simulation.particles[i].α = α₂
                simulation.particles[i].v = @SVector zeros(T,dim)
                simulation.particles[i].position, simulation.particles[idx[i]].position = simulation.particles[idx[i]].position, simulation.particles[i].position
            end
        end

        ###############################################################################
        #                           Production run
        ###############################################################################

        Δt = Δt_prod 
        freq = dump_freq
        steps = nsteps ÷ dump_freq

        simulate!(out_file, integ, collision_calc, dim, steps, freq, particles, interaction_pot, box) 
    end
    return nothing
end


export fully_phase_separated

function fully_phase_separated(;
    num_runs::Int,
    nsteps::Int,
    Npart::Int,
    ptypes::Vector{String},
    dim::Int,
    nn::Int,
    ρ::Float32,
    fraction::Float32,
    T::Float32,
    R::Float32,
    α₁::Float32,
    α₂::Float32,
    ϵ::Float32,
    σ::Float32,
    Δt₁::Float32,
    α_init::Float32,
    nsteps_relax::Int,
    freq_relax::Int,
    Δt₂::Float32,
    dump_freq::Int,
    integ::String
    )

    kB      = Float32(1.38064e-23)
    η		= Float32(8.9e-4)
    density = Float32(1.0e3) # mass density of particles (kg/m³)
    

    V		= volume(R)
    m 		= density*V
    γ		= friction(R,η)

    α       = [α₁, α₂]

    #interaction parameters
    cut_off = σ
    # Time-scales
    τD      = γ*R^2/(kB*T)
    τm      = m/γ
    a²      = (τD/τm)

    if dim == 2
        L = Float32(sqrt(π*σ^2.0f0*Npart/(4.0f0*ρ)))
        s_x, s_y = Float32(L/sqrt(Npart)), Float32(L/sqrt(Npart))
        box = SVector{2,Float32}([L,L])
    elseif dim == 3
        L = Float32((π*σ^3.0f0*Npart/(6.0f0*ρ))^(1.0f0/3.0f0))
        s_x, s_y, s_z = Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0))
        box = SVector{3,Float32}([L,L,L])
    end
    r0 = Array{SVector{dim,Float32}, 1}()
    ###############################################################################
    #   Initializing the system to get a randomly distributed positions
    ###############################################################################
    for run = 1:num_runs

        if dim == 2
            a_x = 0.5f0*L
            a_y = 0.5f0*L
            r_mean = @SVector [a_x, a_y]
            lattice_const = σ
            r = triangular_lattice(L,lattice_const, nn, nn)
            r = [r[i] .+ r_mean for i in 1:length(r)]
            rad = 0.3f0*nn*lattice_const
            r  = triangular_circle(L, r, rad)
            [push!(r0,r[i]) for i = 1:length(r)]
            n_remain = Npart - length(r)
            ii = 0
            while ii < n_remain
                pos = random_pos(dim,L)
                if (pos[1] > L - rad + 2.0f0*σ || pos[1] < rad - 2.0f0*σ) || (pos[2] > L - rad + 2.0f0*σ || pos[2] < rad - 2.0f0*σ)
                    push!(r0,pos)
                    ii += 1
                end
            end
        elseif dim == 3
            a_x = 0.5f0*L
            a_y = 0.5f0*L
            a_z = 0.5f0*L
            r_mean = @SVector [a_x, a_y, a_z]

            lattice_const = sqrt(2.0f0)*σ
            r = fcc_lattice(L,lattice_const, nn, nn, nn)
            r = [r[i] .+ r_mean for i in 1:length(r)]
            rad = 0.412f0*nn*lattice_const
            r  = fcc_sphere(L, r, rad)
            [push!(r0,r[i]) for i = 1:length(r)]
            n_remain = Npart - length(r)
            ii = 0
            while ii < n_remain
                pos = random_pos(dim,L)
                if (pos[1] > L - rad + 2.0f0*σ || pos[1] < rad - 2.0f0*σ) || (pos[2] > L - rad + 2.0f0*σ || pos[2] < rad - 2.0f0*σ) || (pos[3] > L - rad + 2.0f0*σ || pos[3] < rad - 2.0f0*σ)
                    push!(r0,pos)
                    ii += 1
                end
            end
        end
        num_pl = ceil(Int, Npart * fraction)
        output_file = "GPU_$Npart,dim_$dim,rho_$ρ,alpha1_$α₁,alpha2_$α₂,epsilon$ϵ,dt$Δt₂,ns,$run,$integ"
        """
        batchs  = nsteps_relax ÷ freq_relax
        for i = 0:batchs
            α_relax  =   α_init - i*(α_init - max(α₁,α₂))/batchs
            part_type = Vector{String}(undef,Npart)
            alpha_lst = Vector{Float32}(undef,Npart)
            v0 = Array{SVector{dim,Float32}, 1}()
            f0 = Array{SVector{dim,Float32}, 1}()
            fR0 = Array{SVector{dim,Float32}, 1}()

            [push!(v0, @SVector ones(Float32,dim)) for _ in 1:Npart]
            [push!(f0, @SVector zeros(Float32,dim)) for _ in 1:Npart]
            [push!(fR0, @SVector zeros(Float32,dim)) for _ in 1:Npart]

            dQ   = zeros(Float32,Npart)
            Eₖ   = zeros(Float32,Npart)
            Eₚ   = zeros(Float32,Npart)
            c₁   = zeros(Float32,Npart)
            c₂   = zeros(Float32,Npart)
            c₃   = zeros(Float32,Npart)

            Δt      = Float32(Δt₁*τm)

            if integ == "em"
                c1      = a²
                c2      = Δt
                [c₁[i]  = c1 for i in 1:Npart]
                [c₂[i]  = c2 for i in 1:Npart]
                for i = 1:num_pl
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[1]
                    alpha_lst[i] = Float32(α_relax)
                end
                for i = num_pl+1:Npart
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[2]
                    alpha_lst[i] = Float32(α_relax)
                end
            elseif integ == "vv"
                c1      = a²
                c2      = Δt
                [c₁[i]  = c1 for i in 1:Npart]
                [c₂[i]  = c2 for i in 1:Npart]
                for i = 1:num_pl
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[1]
                    alpha_lst[i] = Float32(α_relax)
                end
                for i = num_pl+1:Npart
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[2]
                    alpha_lst[i] = Float32(α_relax)
                end
            elseif integ == "lf"
                c1      = a²
                c2      = Δt
                [c₁[i]  = c1 for i in 1:Npart]
                [c₂[i]  = c2 for i in 1:Npart]
                for i = 1:num_pl
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[1]
                    alpha_lst[i] = Float32(α_relax)
                end
                for i = num_pl+1:Npart
                    c₃[i] = Float32(sqrt(2.0f0*α_relax/Δt))
                    v0[i] = Float32(sqrt(a²*α_relax)) .* v0[i]
                    part_type[i] = ptypes[2]
                    alpha_lst[i] = Float32(α_relax)
                end
            end


            c₁_d     = CuVector(c₁)
            c₂_d     = CuVector(c₂)
            c₃_d     = CuVector(c₃)
            α_d  = CuVector(alpha_lst)
            r0_d     = CuVector(r0)
            v0_d     = CuVector(v0)
            f0_d     = CuVector(f0)
            fR_d     = CuVector(fR0)

            dQ_d     = CuVector(dQ)
            Eₖ_d     = CuVector(Eₖ)
            Eₚ_d     = CuVector(Eₚ)


            r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,α_d
            steps = nsteps ÷ dump_freq

            r0 , v0, f0, dQ, Eₖ, Eₚ = simul!(integ, dim, Npart, steps, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,box)
        end
        """

        #r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
        #write_xyz(output_file, Npart, c2, alpha_lst, σ,L, 0, dim, part_type, r, v, dQ)
        #write_log(output_file, 0,c2,alpha_lst, Eₖ, Eₚ, dQ)

    ###############################################################################
    #                           Production run
    ###############################################################################
        part_type = Vector{String}(undef,Npart)
        alpha_lst = Vector{Float32}(undef,Npart)
        v0 = Array{SVector{dim,Float32}, 1}()
        f0 = Array{SVector{dim,Float32}, 1}()
        fR0 = Array{SVector{dim,Float32}, 1}()

        [push!(v0, @SVector ones(Float32,dim)) for _ in 1:Npart]
        [push!(f0, @SVector zeros(Float32,dim)) for _ in 1:Npart]
        [push!(fR0, @SVector zeros(Float32,dim)) for _ in 1:Npart]

        dQ   = zeros(Float32,Npart)
        Eₖ   = zeros(Float32,Npart)
        Eₚ   = zeros(Float32,Npart)
        c₁   = zeros(Float32,Npart)
        c₂   = zeros(Float32,Npart)
        c₃   = zeros(Float32,Npart)

        Δt      = Float32(Δt₂*τm)
        if integ == "em"
            c1      = a²
            c2      = Δt
            [c₁[i]  = c1 for i in 1:Npart]
            [c₂[i]  = c2 for i in 1:Npart]
            for i = 1:num_pl
                c₃[i] = Float32(sqrt(2.0f0*α[1]/Δt))
                v0[i] = Float32(sqrt(a²*α[1])) .* v0[i]
                part_type[i] = ptypes[1]
                alpha_lst[i] = Float32(α[1])
            end
            for i = num_pl+1:Npart
                c₃[i] = Float32(sqrt(2.0f0*α[2]/Δt))
                v0[i] = Float32(sqrt(a²*α[2])) .* v0[i]
                part_type[i] = ptypes[2]
                alpha_lst[i] = Float32(α[2])
            end
        elseif integ == "vv"
            c1      = a²
            c2      = Δt
            [c₁[i]  = c1 for i in 1:Npart]
            [c₂[i]  = c2 for i in 1:Npart]
            for i = 1:num_pl
                c₃[i] = Float32(sqrt(2.0f0*α[1]/Δt))
                v0[i] = Float32(sqrt(a²*α[1])) .* v0[i]
                part_type[i] = ptypes[1]
                alpha_lst[i] = Float32(α[1])
            end
            for i = num_pl+1:Npart
                c₃[i] = Float32(sqrt(2.0f0*α[2]/Δt))
                v0[i] = Float32(sqrt(a²*α[2])) .* v0[i]
                part_type[i] = ptypes[2]
                alpha_lst[i] = Float32(α[2])
            end
        elseif integ == "lf"
            c1      = a²
            c2      = Δt
            [c₁[i]  = c1 for i in 1:Npart]
            [c₂[i]  = c2 for i in 1:Npart]
            for i = 1:num_pl
                c₃[i] = Float32(sqrt(2.0f0*α[1]/Δt))
                v0[i] = Float32(sqrt(a²*α[1])) .* v0[i]
                part_type[i] = ptypes[1]
                alpha_lst[i] = Float32(α[1])
            end
            for i = num_pl+1:Npart
                c₃[i] = Float32(sqrt(2.0f0*α[2]/Δt))
                v0[i] = Float32(sqrt(a²*α[2])) .* v0[i]
                part_type[i] = ptypes[2]
                alpha_lst[i] = Float32(α[2])
            end
        end

        c₁_d     = CuVector(c₁)
        c₂_d     = CuVector(c₂)
        c₃_d     = CuVector(c₃)
        α_d     = CuVector(alpha_lst)
        r0_d     = CuVector(r0)
        v0_d     = CuVector(v0)
        f0_d     = CuVector(f0)
        fR_d     = CuVector(fR0)
        dQ_d     = CuVector(dQ)
        Eₖ_d     = CuVector(Eₖ)
        Eₚ_d     = CuVector(Eₚ)
        r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d
        freq = dump_freq
        steps = nsteps ÷ dump_freq
        r0 , v0, f0, dQ, Eₖ, Eₚ = simul!(output_file,integ, dim, Npart, steps, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,box)
    end
end

function simulate!(
    out_file::String,
    integ::String,
    collision_calc::Bool, 
    dim::Int,
    steps::Int,
    freq::Int,
    particles::Vector{Particle,1},
    interaction_pot::AbstractInteraction,
    box::SVector{N,T}) where {N,T}

    if integ == "em"
        if dim == 2
            if collision_calc
                simulation_em!(dim, steps, freq, out_file, particles,box,forces!, collisions!,update_parts_em!, noise2D)
            else
                simulation_em!(dim, steps, freq, out_file, particles, ϵ,box,forces!,update_parts_em!, noise2D)
            end
        elseif dim == 3
            if collision_calc
                simulation_em!(dim, steps, freq, particles, ϵ,box,forces!, collisions!,update_parts_em!, noise3D)
            else
                simulation_em!(dim, steps, freq, particles, ϵ,box,forces!,update_parts_em!, noise3D)
            end
        end
    elseif integ == "vv"
        if dim == 2
            if collision_calc
                simulation_vv!(dim, steps, freq, particles, ϵ,box,forces!, collisions!,update_positions_vv!,update_velocities_vv!, noise2D)
            else
                simulation_vv!(dim, steps, freq, particles, ϵ,box,forces!,update_positions_vv!,update_velocities_vv!, noise2D)
            end
        elseif dim == 3
            if collision_calc
                simulation_vv!(dim, steps, freq, particles, ϵ,box,forces!, collisions!,update_positions_vv!,update_velocities_vv!, noise3D)
            else
                simulation_vv!(dim, steps, freq, particles, ϵ,box,forces!,update_positions_vv!,update_velocities_vv!, noise3D)
            end
        end
    end
    return r0 , v0, f0, dQ, Eₖ, Eₚ
end