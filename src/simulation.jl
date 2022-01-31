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


function run_simulation!(dim::Int, Npart::Int, freq::Int, r₀::CuVector{SVector{N,T}}, v₀::CuVector{SVector{N,T}}, 
        S₀::CuVector{T}, c₁₀::CuVector{T}, c₂₀::CuVector{T},c₃₀::CuVector{T}, α₀::CuVector{T}, a::T,
        cut_off::T, periodicity::SVector{N,T}, forces_fun::Function,update_parts::Function, noise2D!::Function) where {N,T}
    r  = copy(r₀)
    v  = copy(v₀)


    c1 = copy(c₁₀)
    c2 = copy(c₂₀)
    c3 = copy(c₃₀)
    α  = copy(α₀)

    f_int = similar(v₀)
    f_noise = similar(v₀)
    sdot  = similar(S₀)


    for _ in 1:freq
        f_int = forces_fun(r,periodicity,cut_off)
        f_noise = noise2D!(Npart)
        r, v, sdot = update_parts(r, v, f_int, f_noise, c1, c2, c3, α, a)
        r  = PBC!(r,periodicity)
    end
    return r, v, f_int, sdot
end

@inline function kinetic(v::SVector,τm::Float64,τD::Float64)
	return (τm/(2.0*τD))*dot(v,v)
end
