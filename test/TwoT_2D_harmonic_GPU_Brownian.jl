using NonEqSimGPU

using StaticArrays
using CUDA
import Random: randperm

function main(steps)
    kB      = Float32(1.38064e-23)
    T       = 300.f0
    η		= Float32(8.9e-4)
    density = Float32(1.0e3) # mass density of particles (kg/m³)
    dim     = 2

    ϕ       = 0.75f0

    fraction = 0.5f0

    R 		= Float32(1.0e-6)
    V		= volume(R)
    m 		= density*V
    γ		= friction(R,η)

    T₁      = T
    T₂      = T

    α₁      = T₁/T
    α₂      = T₂/T
    α       = [α₁, α₂]
    ptypes  = ["H","He"]
    # Time-scales
    τD      = γ*(R)^2/(kB*T)
    τm      = m/γ
    τ       = τD
    σ       = 2.0f0
    cut_off = σ
    ε       = 100.0

    Δt      = 0.1f0*τm
    c1      = Δt
    Npart   = 1024*4        # Its square root must be an Integer
    c₁      = zeros(Float32,Npart)
    c₂      = zeros(Float32,Npart)

    [c₁[i]=c1 for i in 1:Npart]

    part_type = Vector{String}(undef,Npart)
    alpha_lst = Vector{Float32}(undef,Npart)

    if mod(Npart,128) != 0
        println("Number of particles must be ")
    end

    if dim == 2
        L = Float32(sqrt(π*σ^2*Npart/(4*ϕ)))
        s_x, s_y = Float32(L/sqrt(Npart)), Float32(L/sqrt(Npart))
        periodicity = SVector{2,Float32}([L,L])
    elseif dim == 3
        L = Float32((π*σ^3*Npart/(6*ϕ))^(1/3))
        periodicity = SVector{3,Float32}([L,L,L])
    end

    r0 = rectangular_lattice(s_x,s_y,Int(sqrt(Npart)),Int(sqrt(Npart)))

    v0 = Array{SVector{dim,Float32}, 1}()
    f0 = Array{SVector{dim,Float32}, 1}()

    [push!(v0, @SVector rand(Float32,dim)) for _ in 1:Npart]
    [push!(f0, @SVector rand(Float32,dim)) for _ in 1:Npart]

    #Epot0      = zeros(Float32,Npart)
    #Ekin0      = zeros(Float32,Npart)
    Sdot0   = zeros(Float32,Npart)

    num_pl = floor(Int32, Npart * fraction)

    for i = 1:num_pl
        c₂[i] = Float32(sqrt(2.0f0*α[1]/Δt))
        part_type[i] = ptypes[1]
        alpha_lst[i] = Float32(α[1])
    end

    for i = num_pl+1:Npart
        c₂[i] = Float32(sqrt(2.0f0*α[2]/Δt))
        part_type[i] = ptypes[2]
        alpha_lst[i] = Float32(α[2])
    end

    idx = randperm(Npart)
    c₂  = c₂[idx]
    part_type = part_type[idx]
    alpha_lst = alpha_lst[idx]
    output_file = "nPart_$Npart,dim_$dim,rho_$ϕ"
    write_xyz(output_file, Npart, σ,L, 0, dim, part_type, r0, v0,f0)
    write_log(output_file, Npart, 0, Sdot0)
    c₁_d     = CuVector(c₁)
    c₂_d     = CuVector(c₂)
    alpha_d  = CuVector(alpha_lst)
    r0_d     = CuVector(r0)
    v0_d     = CuVector(v0)
    f0_d    = CuVector(f0)

    Sdot0_d = CuVector(Sdot0)

    r₀, v₀, f₀,  Sdot₀, c₁₀, c₂₀, α₀ = r0_d, v0_d, f0_d, Sdot0_d, c₁_d, c₂_d, alpha_d
    freq = 10000
    for t in 1:steps
        r₀, v₀, f₀, Sdot₀ = run_simulation!(dim, Npart, freq, r₀, v₀, Sdot₀, c₁₀, c₂₀, α₀, cut_off, periodicity, forces_fun,update_parts_BD,noise2D)
        r , v, f, Sdot_ave = Vector(r₀), Vector(v₀), Vector(f₀),Vector(Sdot₀)
        write_xyz(output_file, Npart, σ,L, t, dim, part_type, r, v,f)
        write_log(output_file, Npart, t, Sdot_ave)
        yield()
    end
end
#@device_code_warntype @cuda threads = 1024 main(10)
@time main(100)
