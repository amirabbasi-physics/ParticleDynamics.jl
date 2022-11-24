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
export simulation_vv!


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
    collisions!::Function,
	update_positions_vv!::Function,
	update_velocities_vv!::Function,
	noisefun::Function) where {N,T}

    coll = CuArray(cu(zeros(Npart,Npart)))
	dQ = zero(dQ₀)
	Eₖ = zero(Eₖ₀)
	Eₚ = zero(Eₚ₀)
	f = zero(f₀)
    coll₀ = CuArray(cu(zeros(Npart,Npart)))
    coll_switch₀ = CuArray(cu(Matrix{Int32}(I,Npart,Npart)))
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
        coll₀, coll_switch₀ = collisions!(r, coll₀, coll_switch₀, cut_off, periodicity)
		update_velocities_vv!(v, f₀, f, fR, dQ₀, Ekin, c₁, c₂, c₃)
		f₀ = f
        coll .+= coll₀
		dQ .+= dQ₀
		Eₖ .+= Ekin
		Eₚ .+= Epot
    end

    coll ./= freq
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