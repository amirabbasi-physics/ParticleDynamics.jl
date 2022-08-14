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
    relax_freq::Int,
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

    for run = 1:num_runs
    ###############################################################################
    #   Initializing the system to get a randomly distributed positions
    ###############################################################################
        Δt      = Float32(1.0e-9*Δt₁)
        c1      = a²
        c2      = Δt
        c₁      = zeros(Float32,Npart)
        c₂      = zeros(Float32,Npart)
        c₃      = zeros(Float32,Npart)

        [c₁[i]  = c1 for i in 1:Npart]
        [c₂[i]  = c2 for i in 1:Npart]

        part_type = Vector{String}(undef,Npart)
        alpha_lst = Vector{Float32}(undef,Npart)

        if dim == 2
            NN = ceil(Int,Npart^(1/2))
            r0 = rectangular_lattice(s_x,s_y,NN,NN)
        elseif dim == 3
            NN = ceil(Int,Npart^(1/3))
            r0 = simplecubic_lattice(s_x,s_y,s_z,NN,NN,NN)
        end
        v0 = Array{SVector{dim,Float32}, 1}()
        f0 = Array{SVector{dim,Float32}, 1}()
        fR0 = Array{SVector{dim,Float32}, 1}()

        [push!(v0, @SVector ones(Float32,dim)) for _ in 1:Npart]
        [push!(f0, @SVector zeros(Float32,dim)) for _ in 1:Npart]
        [push!(fR0, @SVector zeros(Float32,dim)) for _ in 1:Npart]

        dQ   = zeros(Float32,Npart)
        Eₖ   = zeros(Float32,Npart)
        Eₚ   = zeros(Float32,Npart)

        num_pl = floor(Int32, Npart * fraction)
        for i = 1:num_pl
            c₃[i] = Float32(sqrt(2.0f0*α_init/Δt))
            v0[i] = Float32(sqrt(a²*α_init)) .* v0[i]
            part_type[i] = ptypes[1]
            alpha_lst[i] = Float32(α_init)
        end

        for i = num_pl+1:Npart
            c₃[i] = Float32(sqrt(2.0f0*α_init/Δt))
            v0[i] = Float32(sqrt(a²*α_init)) .* v0[i]
            part_type[i] = ptypes[2]
            alpha_lst[i] = Float32(α_init)
        end

        idx = randperm(Npart)
        c₃  = c₃[idx]
        v0  = v0[idx]
        part_type = part_type[idx]
        alpha_lst = alpha_lst[idx]
        output_file = "GPU_$Npart,dim_$dim,rho_$ρ,alpha1_$α₁,alpha2_$α₂,dt$Δt₂,ns,$run,$integ"


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

        r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d

        freq = relax_freq
        if integ == "vv"
            if dim == 2
                r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise2D)
            elseif dim == 3
                r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise3D)
            end
        elseif integ == "lf"
            if dim == 2
                r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise2D)
            elseif dim == 3
                r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise3D)
            end
        end
        r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
        write_xyz(output_file, Npart, c2, alpha_lst, σ,L, 0, dim, part_type, r, v, dQ)
        write_log(output_file, 0,c2,alpha_lst, Eₖ, Eₚ, dQ)

    ###############################################################################
    #                           Production run
    ###############################################################################
        Δt      = Float32(1.0e-9*Δt₂)
        c1      = a²
        c2      = Δt
        c₁      = zeros(Float32,Npart)
        c₂      = zeros(Float32,Npart)
        c₃      = zeros(Float32,Npart)

        [c₁[i]  = c1 for i in 1:Npart]
        [c₂[i]  = c2 for i in 1:Npart]

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

        r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, Eₚ₀, c₁₀, c₂₀, c₃₀, α₀ = r0_d, v0_d, f0_d, fR_d, dQ_d, Eₖ_d, Eₚ_d, c₁_d, c₂_d, c₃_d ,alpha_d
        freq = dump_freq
        steps = nsteps ÷ dump_freq

        if integ == "vv"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_vv!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise3D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            end
        elseif integ == "lf"
            if dim == 2
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise2D)
                    r , v, f, dQ, Eₖ, Eₚ = Vector(r₀), Vector(v₀), Vector(f₀), Vector(dQ₀), Vector(Eₖ₀), Vector(Eₚ₀)
                    write_xyz(output_file, Npart, c2, alpha_lst, σ,L, t, dim, part_type, r, v, dQ)
                    write_log(output_file, t,c2,alpha_lst, Eₖ, Eₚ, dQ)
                    yield()
                end
            elseif dim == 3
                for t = 0:steps
                    r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀ = simulation_lf!(dim, Npart, freq, r₀,v₀,f₀,fR₀, dQ₀, Eₖ₀,Eₚ₀,c₁₀, c₂₀, c₃₀,α₀, ϵ, cut_off,periodicity,forces!,update_positions!,update_velocities!, noise3D)
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
#             Positions and velocities update for Verlet-type algorithm             #
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
	α₀::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions!::Function,
	update_velocities!::Function,
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
        update_positions!(r, v, f₀, fR, c₁, c₂, c₃)
		PBC!(r,periodicity)
		f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
		update_velocities!(v, f₀, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
		f₀ = f
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end


#####################################################################################
#####################################################################################
#             Positions and velocities update for leap-frog algorithm               #
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
	α₀::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions!::Function,
	update_velocities!::Function,
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
        update_positions!(r, v, c₂)
		PBC!(r,periodicity)
		f, Epot = forces!(r, f, Epot, periodicity, ϵ, cut_off)
		update_velocities!(v, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
        update_positions!(r, v, c₂)
		PBC!(r,periodicity)
		dQ .+= dQ₀ ./freq
		Eₖ .+= Ekin ./freq
		Eₚ .+= Epot ./freq
    end
    return r, v, f, dQ, Eₖ, Eₚ
end
