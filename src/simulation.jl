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

#####################################################################################
#####################################################################################
#             Positions and velocities update for Verlet-type algorithm             #
#####################################################################################
#####################################################################################
function simulation!(
	dim::Int,
	Npart::Int,
	freq::Int,
	r₀::CuVector{SVector{N,T}},
	v₀::CuVector{SVector{N,T}},
	f₀::CuVector{SVector{N,T}},
	fR₀::CuVector{SVector{N,T}},
	dQ₀::CuVector{T},
	Eₖ₀::CuVector{T},
	Eₚ₀::CuVector{T},
	c₁₀::CuVector{T},
	c₂₀::CuVector{T},
	c₃₀::CuVector{T},
	α₀::CuVector{T},
	ϵ::T,
	cut_off::T,
	periodicity::SVector{N,T},
	forces!::Function,
	update_positions!::Function,
	update_velocities!::Function,
	noise3D::Function) where {N,T}
	dQ = similar(dQ₀)
	Eₖ = similar(Eₖ₀)
	Eₚ = similar(Eₚ₀)
	f = zero(f₀)
	dQ = zero(dQ)
	Eₖ = zero(Eₖ)
	Eₚ = zero(Eₚ)
	dQ₀ = zero(dQ₀)
	Ekin = zero(Eₖ₀)
	Epot = zero(Eₚ₀)
    for _ in 1:freq
		f = zero(f₀)
		dQ₀ = zero(dQ₀)
		Ekin = zero(Ekin)
		Epot = zero(Epot)
        fR₀ = noise3D(Npart)
        update_positions!(r₀, v₀, f₀, fR₀, c₁₀, c₂₀, c₃₀)
		PBC!(r₀,periodicity)
		f, Epot = forces!(r₀, f, Epot, periodicity, ϵ, cut_off)
		update_velocities!(v₀, f₀, f, fR₀, dQ₀, Ekin, c₁₀, c₂₀, c₃₀)
		f₀ = f
		dQ = dQ .+ dQ₀ ./freq
		Eₖ = Eₖ .+ Ekin ./freq
		Eₚ = Eₚ .+ Epot ./freq
    end
	dQ₀ = dQ
	Eₖ₀ = Eₖ
	Eₚ₀ = Eₚ
    return r₀, v₀, f₀, dQ₀, Eₖ₀, Eₚ₀
end
