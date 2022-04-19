export hr_min_sec
export run_simulation2D!
export run_simulation3D!
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


function run_simulation2D!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},f₀::CuVector{SVector{N,T}},fR₀::CuVector{SVector{N,T}}, dQ₀::CuVector{T}, Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, a²::T, ϵ::T, cut_off::T,
	periodicity::SVector{N,T},forces!::Function,update_parts_LD!::Function, noise2D::Function) where {N,T}
	dQ = similar(dQ₀)
	Eₖ = similar(Eₖ₀)
	Eₚ = similar(Eₚ₀)
	dQ = zero(dQ)
	Eₖ = zero(Eₖ)
	Eₚ = zero(Eₚ)
	dQ₀ = zero(dQ₀)
	Eₖ₀ = zero(Eₖ₀)
	Eₚ₀ = zero(Eₚ₀)
    for _ in 1:freq
		f₀ = zero(f₀)
        f₀ = forces!(r₀,f₀,Eₚ₀, periodicity, ϵ, cut_off)
        fR₀ = noise2D(Npart)
        update_parts_LD!(r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀,a²)
		PBC!(r₀,periodicity)
		dQ = dQ .+ dQ₀ ./freq
		Eₖ = Eₖ .+ Eₖ₀ ./freq
		Eₚ = Eₚ .+ Eₚ₀ ./freq
    end
	dQ₀ = dQ
	Eₖ₀ = Eₖ
	Eₚ₀ = Eₚ
    return r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀
end

function run_simulation3D!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},f₀::CuVector{SVector{N,T}},fR₀::CuVector{SVector{N,T}}, dQ₀::CuVector{T}, Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, a²::T, ϵ::T, cut_off::T,
	periodicity::SVector{N,T},forces!::Function,update_parts_LD!::Function, noise3D::Function) where {N,T}
	dQ = similar(dQ₀)
	Eₖ = similar(Eₖ₀)
	Eₚ = similar(Eₚ₀)
	dQ = zero(dQ)
	Eₖ = zero(Eₖ)
	Eₚ = zero(Eₚ)
	dQ₀ = zero(dQ₀)
	Eₖ₀ = zero(Eₖ₀)
	Eₚ₀ = zero(Eₚ₀)
    for _ in 1:freq
		f₀ = zero(f₀)
        f₀ = forces!(r₀,f₀,Eₚ₀, periodicity, ϵ, cut_off)
        fR₀ = noise3D(Npart)
        update_parts_LD!(r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀,a²)
		PBC!(r₀,periodicity)
		dQ = dQ .+ dQ₀ ./freq
		Eₖ = Eₖ .+ Eₖ₀ ./freq
		Eₚ = Eₚ .+ Eₚ₀ ./freq
    end
	dQ₀ = dQ
	Eₖ₀ = Eₖ
	Eₚ₀ = Eₚ
    return r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀
end



function run_simulation3D_new!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},f₀::CuVector{SVector{N,T}},fR₀::CuVector{SVector{N,T}}, dQ₀::CuVector{T}, Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, a²::T, ϵ::T, cut_off::T,
	periodicity::SVector{N,T},forces!::Function,update_parts_LD!::Function, noise3D::Function) where {N,T}
	dQ = similar(dQ₀)
	Eₖ = similar(Eₖ₀)
	Eₚ = similar(Eₚ₀)
	dQ = zero(dQ)
	Eₖ = zero(Eₖ)
	Eₚ = zero(Eₚ)
	dQ₀ = zero(dQ₀)
	Eₖ₀ = zero(Eₖ₀)
	Eₚ₀ = zero(Eₚ₀)
    for _ in 1:freq
		
        f₀ = forces!(r₀,f₀,Eₚ₀, periodicity, ϵ, cut_off)
        fR₀ = noise3D(Npart)
        update_parts_LD!(r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀,a²)
		PBC!(r₀,periodicity)
		dQ = dQ .+ dQ₀ ./freq
		Eₖ = Eₖ .+ Eₖ₀ ./freq
		Eₚ = Eₚ .+ Eₚ₀ ./freq
    end
	dQ₀ = dQ
	Eₖ₀ = Eₖ
	Eₚ₀ = Eₚ
    return r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀
end
"""
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
"""
