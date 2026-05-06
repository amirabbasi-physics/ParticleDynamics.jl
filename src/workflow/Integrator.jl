abstract type Method end

@kwdef struct ConstantVolume <: Method
    group
    thermostat = nothing
end

@kwdef struct Langevin <: Method
    group
    gamma
    kT
end

@kwdef struct Brownian <: Method
    group
    gamma
    kT
end

@kwdef struct ActiveOrnsteinUhlenbeck <: Method
    group
    gamma
    kT
    tau = nothing
    noise_scale = nothing
    spectrum = nothing
end

@kwdef struct Integrator
    dt
    scheme = nothing
    forces::Vector{Force} = Force[]
    methods::Vector{Method} = Method[]
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end
