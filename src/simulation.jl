export hr_min_sec
export simulation!


@inline function hr_min_sec(time::Float64)
    hours = trunc(Int64, time / 3600.0)
    minutes = trunc(Int64, mod(time, 3600.0) / 60.0)
    seconds = trunc(Int64, mod(time, 60.0))

    return string(hours < 10 ? "0" : "", hours,
                  minutes < 10 ? ":0" : ":", minutes,
                  seconds < 10 ? ":0" : ":", seconds)
end



export run_sim

function run_sim(;
    num_runs::Int,
    nsteps::Int,
    Npart::Int,
    ptypes::Vector{String},
    dim::Int,
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
        periodicity = SVector{2,Float32}([L,L])
    elseif dim == 3
        L = Float32((π*σ^3.0f0*Npart/(6.0f0*ρ))^(1.0f0/3.0f0))
        s_x, s_y, s_z = Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0))
        periodicity = SVector{3,Float32}([L,L,L])
    end
    num_pl = ceil(Int, Npart*fraction)
    ###############################################################################
    #   Initializing the system to get a randomly distributed positions
    ###############################################################################
    for run = 1:num_runs

        if dim == 2
            NN = ceil(Int,Npart^(1/2))
            r0 = rectangular_lattice(s_x,s_y,NN,NN)
        elseif dim == 3
            NN = ceil(Int,Npart^(1/3))
            r0 = simplecubic_lattice(s_x,s_y,s_z,NN,NN,NN)
        end
        output_file = "GPU_$Npart,dim_$dim,rho_$ρ,alpha1_$α₁,alpha2_$α₂,epsilon$ϵ,dt$Δt₂,ns,$run,$integ"

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
            elseif (integ == "ml" || integ == "bp")
                a   = zeros(Float32,Npart)
                c1      = exp(-0.5f0*a²*Δt)
                c2      = Δt
                [c₁[i]  = c1 for i in 1:Npart]
                [c₂[i]  = c2 for i in 1:Npart]
                [a[i]   = a² for i in 1:Npart]
                for i = 1:num_pl
                    c₃[i] = Float32(sqrt(a²*α_relax*(1-c1^2)))
                    v0[i] = c₃[i] .* v0[i]
                    part_type[i] = ptypes[1]
                    alpha_lst[i] = Float32(α_relax)
                end
                for i = num_pl+1:Npart
                    c₃[i] = Float32(sqrt(a²*α_relax*(1-c1^2)))
                    v0[i] = c₃[i] .* v0[i]
                    part_type[i] = ptypes[2]
                    alpha_lst[i] = Float32(α_relax)
                end
            end

            idx = randperm(Npart)
            c₃  = c₃[idx]
            v0  = v0[idx]
            part_type = part_type[idx]
            alpha_lst = alpha_lst[idx]


            c₁_d     = CuVector(c₁)
            c₂_d     = CuVector(c₂)
            c₃_d     = CuVector(c₃)
            alpha_d  = CuVector(alpha_lst)
            r0_d     = CuVector(r0)
            v0_d     = CuVector(v0)
            f0_d     = CuVector(f0)
            fR_d     = CuVector(fR0)

            dQ_d     = CuVector(dQ)
            Eₖ_d     = CuVector(Eₖ)
            Eₚ_d     = CuVector(Eₚ)

            if (integ == "ml" || integ == "bp")
              a_d     = CuVector(a)
              r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀,a₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d, a_d
            else
              r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d
            end



            if integ == "em"
              if dim == 2
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise2D)
              elseif dim == 3
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise3D)
              end
            elseif integ == "vv"
              if dim == 2
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise2D)
              elseif dim == 3
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise3D)
              end
            elseif integ == "lf"
              if dim == 2
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise2D)
              elseif dim == 3
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise3D)
              end
            elseif integ == "ml"
              if dim == 2
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise2D)
              elseif dim == 3
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise3D)
              end
            elseif integ == "bp"
              if dim == 2
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise2D)
              elseif dim == 3
                  r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise3D)
              end
            end
        end
        r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
        write_xyz(output_file, Npart, c2, alpha_lst, σ,L, 0, dim, part_type, r, v, dQ)
        write_log(output_file, 0,c2,alpha_lst, Eₖ, Eₚ, dQ)

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
        elseif (integ == "ml" || integ == "bp")
            a   = zeros(Float32,Npart)
            c1      = Float32(exp(-0.5f0*a²*Δt))
            c2      = Δt
            [c₁[i]  = c1 for i in 1:Npart]
            [c₂[i]  = c2 for i in 1:Npart]
            [a[i]   = a² for i in 1:Npart]
            for i = 1:num_pl
                c₃[i] = Float32(sqrt(a²*α[1]*(1-c1^2)))
                v0[i] = c₃[i] .* v0[i]
                part_type[i] = ptypes[1]
                alpha_lst[i] = Float32(α[1])
            end
            for i = num_pl+1:Npart
                c₃[i] = Float32(sqrt(a²*α[2]*(1-c1^2)))
                v0[i] = c₃[i] .* v0[i]
                part_type[i] = ptypes[2]
                alpha_lst[i] = Float32(α[2])
            end
        end

        idx = randperm(Npart)
        c₃  = c₃[idx]
        v0  = v0[idx]
        part_type = part_type[idx]
        alpha_lst = alpha_lst[idx]

        c₁_d     = CuVector(c₁)
        c₂_d     = CuVector(c₂)
        c₃_d     = CuVector(c₃)
        alpha_d  = CuVector(alpha_lst)
        r0_d     = CuVector(r)
        v0_d     = CuVector(v0)
        f0_d     = CuVector(f0)
        fR_d     = CuVector(fR0)
        dQ_d     = CuVector(dQ)
        Eₖ_d     = CuVector(Eₖ)
        Eₚ_d     = CuVector(Eₚ)

        if (integ == "ml" || integ == "bp")
            a_d     = CuVector(a)
            r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀,a₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d, a_d
        else
            r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d
        end

        freq = dump_freq
        steps = nsteps ÷ dump_freq
        if integ =="em"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "vv"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "lf"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "ml"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "bp"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        end
    end
end


function run_sim(;
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
    lattice_const = sqrt(2.0f0)*σ

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
        periodicity = SVector{2,Float32}([L,L])
    elseif dim == 3
        L = Float32((π*σ^3.0f0*Npart/(6.0f0*ρ))^(1.0f0/3.0f0))
        s_x, s_y, s_z = Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0)), Float32(L/(Npart)^(1.0f0/3.0f0))
        periodicity = SVector{3,Float32}([L,L,L])
    end
    r0 = Array{SVector{dim,Float32}, 1}()
    ###############################################################################
    #   Initializing the system to get a randomly distributed positions
    ###############################################################################
    for run = 1:num_runs

        if dim == 2
            NN = ceil(Int,Npart^(1/2))
            r0 = rectangular_lattice(s_x,s_y,NN,NN)
        elseif dim == 3
            a_x = 0.5f0*L
            a_y = 0.5f0*L
            a_z = 0.5f0*L
            r_mean = @SVector [a_x, a_y, a_z]
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
              elseif (integ == "ml" || integ == "bp")
                  a   = zeros(Float32,Npart)
                  c1      = exp(-0.5f0*a²*Δt)
                  c2      = Δt
                  [c₁[i]  = c1 for i in 1:Npart]
                  [c₂[i]  = c2 for i in 1:Npart]
                  [a[i]   = a² for i in 1:Npart]
                  for i = 1:num_pl
                      c₃[i] = Float32(sqrt(a²*α_relax*(1-c1^2)))
                      v0[i] = c₃[i] .* v0[i]
                      part_type[i] = ptypes[1]
                      alpha_lst[i] = Float32(α_relax)
                  end
                  for i = num_pl+1:Npart
                      c₃[i] = Float32(sqrt(a²*α_relax*(1-c1^2)))
                      v0[i] = c₃[i] .* v0[i]
                      part_type[i] = ptypes[2]
                      alpha_lst[i] = Float32(α_relax)
                  end
              end


              c₁_d     = CuVector(c₁)
              c₂_d     = CuVector(c₂)
              c₃_d     = CuVector(c₃)
              alpha_d  = CuVector(alpha_lst)
              r0_d     = CuVector(r0)
              v0_d     = CuVector(v0)
              f0_d     = CuVector(f0)
              fR_d     = CuVector(fR0)

              dQ_d     = CuVector(dQ)
              Eₖ_d     = CuVector(Eₖ)
              Eₚ_d     = CuVector(Eₚ)

              if (integ == "ml" || integ == "bp")
                  a_d     = CuVector(a)
                  r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀,a₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d, a_d
              else
                  r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d
              end



              if integ == "em"
                  if dim == 2
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise2D)
                  elseif dim == 3
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise3D)
                  end
              elseif integ == "vv"
                  if dim == 2
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise2D)
                  elseif dim == 3
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise3D)
                  end
              elseif integ == "lf"
                  if dim == 2
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise2D)
                  elseif dim == 3
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise3D)
                  end
              elseif integ == "ml"
                  if dim == 2
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise2D)
                  elseif dim == 3
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise3D)
                  end
              elseif integ == "bp"
                  if dim == 2
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise2D)
                  elseif dim == 3
                      r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq_relax, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise3D)
                  end
              end
        end

        r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
        write_xyz(output_file, Npart, c2, alpha_lst, σ,L, 0, dim, part_type, r, v, dQ)
        write_log(output_file, 0,c2,alpha_lst, Eₖ, Eₚ, dQ)

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
        elseif (integ == "ml" || integ == "bp")
            a   = zeros(Float32,Npart)
            c1      = Float32(exp(-0.5f0*a²*Δt))
            c2      = Δt
            [c₁[i]  = c1 for i in 1:Npart]
            [c₂[i]  = c2 for i in 1:Npart]
            [a[i]   = a² for i in 1:Npart]
            for i = 1:num_pl
                c₃[i] = Float32(sqrt(a²*α[1]*(1-c1^2)))
                v0[i] = c₃[i] .* v0[i]
                part_type[i] = ptypes[1]
                alpha_lst[i] = Float32(α[1])
            end
            for i = num_pl+1:Npart
                c₃[i] = Float32(sqrt(a²*α[2]*(1-c1^2)))
                v0[i] = c₃[i] .* v0[i]
                part_type[i] = ptypes[2]
                alpha_lst[i] = Float32(α[2])
            end
        end

        c₁_d     = CuVector(c₁)
        c₂_d     = CuVector(c₂)
        c₃_d     = CuVector(c₃)
        alpha_d  = CuVector(alpha_lst)
        r0_d     = CuVector(r)
        v0_d     = CuVector(v0)
        f0_d     = CuVector(f0)
        fR_d     = CuVector(fR0)
        dQ_d     = CuVector(dQ)
        Eₖ_d     = CuVector(Eₖ)
        Eₚ_d     = CuVector(Eₚ)

        if (integ == "ml" || integ == "bp")
            a_d     = CuVector(a)
            r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀,a₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d, a_d
        else
            r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d
        end

        freq = dump_freq
        steps = nsteps ÷ dump_freq
        if integ =="em"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_em!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_parts_em!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "vv"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_vv!,update_velocities_vv!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "lf"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀, ϵ, cut_off,periodicity,forces!,update_positions_lf!,update_velocities_lf!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "ml"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_ml!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_ml!,update_velocities_ml₁!,update_velocities_ml₂!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "bp"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_bp!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,a₀, ϵ, cut_off,periodicity,forces!,update_positions_bp!,update_velocities_bp₁!,update_velocities_bp₂!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        end
    end
end
#####################################################################################
#####################################################################################
#                 Simulation scheme for Euler-Maruyama algorithm                    #
#####################################################################################
#####################################################################################
function simulation_em!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r::CuVector{SVector{N,T}},
	v::CuVector{SVector{N,T}},
	f₀::CuVector{SVector{N,T}},
	fR::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁::CuVector{T},
	c₂::CuVector{T},
	c₃::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_parts_em!::Function,
	noisefun::Function) where {N,T}
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)
    for _ in 1:freq
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR = noisefun(Npart)
        f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
        update_parts_em!(r, v, f, fR, dQ₀,Ekin, c₁, c₂, c₃)
		PBC!(r,periodicity)
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end



#####################################################################################
#####################################################################################
#                    Simulation scheme for Verlet-type algorithm                    #
#####################################################################################
#####################################################################################
function simulation_vv!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r::CuVector{SVector{N,T}},
	v::CuVector{SVector{N,T}},
	f₀::CuVector{SVector{N,T}},
	fR::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁::CuVector{T},
	c₂::CuVector{T},
	c₃::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions_vv!::Function,
	update_velocities_vv!::Function,
	noisefun::Function) where {N,T}
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)
    for _ in 1:freq
		f = f₀
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR = noisefun(Npart)
        update_positions_vv!(r, v, f₀, fR, c₁, c₂, c₃)
		PBC!(r,periodicity)
		f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
		update_velocities_vv!(v, f₀, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
		f₀ = f
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
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
	periodicity::SVector{N,T},
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
		PBC!(r,periodicity)
		f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
		update_velocities_lf!(v, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
        update_positions_lf!(r, v, c₂)
		PBC!(r,periodicity)
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end


"""
#####################################################################################
#####################################################################################
#                    Simulation step for Melchionna algorithm                       #
#####################################################################################
#####################################################################################

function simulation_ml!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r::CuVector{SVector{N,T}},
	v::CuVector{SVector{N,T}},
	f₀::CuVector{SVector{N,T}},
	fR::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁::CuVector{T},
	c₂::CuVector{T},
	c₃::CuVector{T},
	a::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions_ml!::Function,
	update_velocities_ml₁!::Function,
    update_velocities_ml₂!::Function,
	noisefun::Function) where {N,T}
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)
    vˢ = zero(v)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)
    for _ in 1:freq
		f = f₀
        vˢ = zero(v)
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR = noisefun(Npart)
        update_velocities_ml₁!(v, vˢ, f₀, fR, c₁, c₃)
        update_positions_ml!(r, vˢ, c₂)
		PBC!(r,periodicity)
		#f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
        fR = noisefun(Npart)
		update_velocities_ml₂!(v, vˢ, f, fR, dQ₀, Ekin, c₁, c₂, c₃, a)
		f₀ = f
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end



#####################################################################################
#####################################################################################
#                  Simulation step for Bussi-Parrinello algorithm                   #
#####################################################################################
#####################################################################################

function simulation_bp!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r::CuVector{SVector{N,T}},
	v::CuVector{SVector{N,T}},
	f₀::CuVector{SVector{N,T}},
	fR::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁::CuVector{T},
	c₂::CuVector{T},
	c₃::CuVector{T},
	a::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions_bp!::Function,
	update_velocities_bp₁!::Function,
    update_velocities_bp₂!::Function,
	noisefun::Function) where {N,T}
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)
    vˢ = zero(v)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)
    for _ in 1:freq
		f = f₀
        vˢ = zero(v)
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR = noisefun(Npart)
        update_velocities_bp₁!(v, vˢ, fR, c₁, c₃)
        update_positions_bp!(r, vˢ,f₀, c₂)
		PBC!(r,periodicity)
		#f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
        fR = noisefun(Npart)
		update_velocities_bp₂!(v, vˢ, f₀, f, fR, dQ₀, Ekin, c₁, c₂, c₃, a)
		f₀ = f
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end
"""
