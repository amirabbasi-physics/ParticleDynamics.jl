export hr_min_sec
export run_simulation!
export Force
export kinetic



@inline function hr_min_sec(time::Float64)
    hours = trunc(Int64, time / 3600.0)
    minutes = trunc(Int64, mod(time, 3600.0) / 60.0)
    seconds = trunc(Int64, mod(time, 60.0))

    return string(hours < 10 ? "0" : "", hours,
                  minutes < 10 ? ":0" : ":", minutes,
                  seconds < 10 ? ":0" : ":", seconds)
end


function run_simulation!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},f₀::CuVector{SVector{N,T}},S₀::CuVector{T},Epot₀::CuVector{T}, c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},
	α₀::CuVector{T}, a²::T, cut_off::T, periodicity::SVector{N,T},forces_fun::Function,
	update_parts_LD!::Function, noise2D::Function) where {N,T}
	S₀ = deepcopy(zero(S₀))
    for _ in 1:freq
		Epot_tmp = deepcopy(Epot₀)
		v_tmp = deepcopy(v₀)
		#@cuprintln(sum(Epot_tmp))
        forces_fun!(r₀,f₀,Epot₀,periodicity,cut_off)
		#@cuprintln(sum(Epot₀)-sum(Epot_tmp))
        f_noise = noise2D(Npart)
		#@cuprintln(sum(dot.(v₀,v₀)))
        update_parts_LD!(r₀, v₀, f₀, f_noise, c₁₀, c₂₀, c₃₀,a²)
		#@cuprintln(sum(dot.(v₀,v₀)))
		entropy_prod!(S₀, Epot₀,Epot_tmp,v₀,v_tmp,c₂₀,α₀,a²)
		#@cuprintln(sum(S₀))
		PBC!(r₀,periodicity)
		#@cuprintln(sum(dot.(v_tmp,v_tmp) .- dot.(v₀,v₀)))
    end
	#S₀ .= S₀./freq
    return S₀
end



function run_simulation!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},S₀::CuVector{T}, c₁₀::CuVector{T}, c₂₀::CuVector{T},
	α₀::CuVector{T},cut_off::T, periodicity::SVector{N,T},forces_fun::Function,
	update_parts_BD!::Function, noise2D::Function) where {N,T}
    for _ in 1:freq
		sdot	= zero(similar(S₀))
		f_int = zero(similar(v₀))
	    f_noise = zero(similar(v₀))
        f_int   = forces_fun(r₀,periodicity,cut_off)
        f_noise = noise2D(Npart)
        update_parts_BD!(r₀, v₀, f_int, f_noise,S₀, c₁₀, c₂₀,α₀)

		PBC!(r₀,periodicity)
    end
    return nothing
end

@inline function kinetic(v::SVector,τm::Float64,τD::Float64)
	return (τm/(2.0*τD))*dot(v,v)
end
