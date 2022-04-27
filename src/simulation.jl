export hr_min_sec
export run_simulation2D!
export run_simulation3D!


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
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, ϵ::T, cut_off::T,
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
        update_parts_LD!(r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀)
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
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, ϵ::T, cut_off::T,
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
        update_parts_LD!(r₀, v₀, f₀, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀)
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
	Eₚ₀::CuVector{T},c₁₀::CuVector{T}, c₂₀::CuVector{T}, c₃₀::CuVector{T},α₀::CuVector{T}, ϵ::T, cut_off::T,
	periodicity::SVector{N,T},forces!::Function,update_positions!::Function,update_velocities!::Function, noise3D::Function) where {N,T}
	dQ = similar(dQ₀)
	Eₖ = similar(Eₖ₀)
	Eₚ = similar(Eₚ₀)
	f = similar(f₀)

	dQ = zero(dQ)
	Eₖ = zero(Eₖ)
	Eₚ = zero(Eₚ)
	f = zero(f)
	dQ₀ = zero(dQ₀)
	Eₖ₀ = zero(Eₖ₀)
	Eₚ₀ = zero(Eₚ₀)
    for _ in 1:freq
		f = f₀
        fR₀ = noise3D(Npart)
        update_positions!(r₀, v₀, f₀, fR₀, c₁₀, c₂₀, c₃₀)
		PBC!(r₀,periodicity)
		f = forces!(r₀, f, Eₚ₀, periodicity, ϵ, cut_off)
		update_velocities!(v₀, f₀, f, fR₀, dQ₀, Eₖ₀, c₁₀, c₂₀, c₃₀)
		f₀ = f
		dQ = dQ .+ dQ₀ ./freq
		Eₖ = Eₖ .+ Eₖ₀ ./freq
		Eₚ = Eₚ .+ Eₚ₀ ./freq
    end
	dQ₀ = dQ
	Eₖ₀ = Eₖ
	Eₚ₀ = Eₚ
    return r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀
end
