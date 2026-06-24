using ..Filters
using ..SimulationCore

abstract type AbstractWorkflowScheme end

"""`VelocityVerlet()` selects the workflow velocity-Verlet stepping scheme."""
struct VelocityVerlet <: AbstractWorkflowScheme end
"""`BAOAB()` selects the workflow BAOAB Langevin splitting."""
struct BAOAB <: AbstractWorkflowScheme end
"""`BAOA()` selects the workflow BAOA Langevin splitting."""
struct BAOA <: AbstractWorkflowScheme end
"""`GSM()` selects the workflow Grønbech-Jensen/Farago style scheme."""
struct GSM <: AbstractWorkflowScheme end
"""`EulerHeun()` selects the workflow Brownian midpoint scheme."""
struct EulerHeun <: AbstractWorkflowScheme end
"""`EulerMaruyama()` selects the workflow Euler-Maruyama scheme."""
struct EulerMaruyama <: AbstractWorkflowScheme end

"""
    OUSpectrum(taus, scales)

Multi-mode Ornstein-Uhlenbeck spectrum used by
[`ActiveOrnsteinUhlenbeck`](@ref).
"""
struct OUSpectrum
    taus::Vector{Float64}
    scales::Vector{Float64}
    function OUSpectrum(taus::AbstractVector{<:Real}, scales::AbstractVector{<:Real})
        tau_host = Float64.(collect(taus))
        scale_host = Float64.(collect(scales))
        length(tau_host) == length(scale_host) ||
            throw(ArgumentError("OUSpectrum taus and scales must have the same length."))
        isempty(tau_host) && throw(ArgumentError("OUSpectrum must contain at least one mode."))
        new(tau_host, scale_host)
    end
end

"""
    Method

Abstract workflow method applied to a [`Group`](@ref).
"""
abstract type Method end

"""
    ConstantVolume(group; thermostat=nothing)

Molecular-dynamics method for a particle group, optionally coupled to a
workflow thermostat. When `thermostat === nothing`, the group evolves in the
microcanonical NVE limit.
"""
@kwdef struct ConstantVolume <: Method
    group
    thermostat = nothing
end
ConstantVolume(group; thermostat=nothing) = ConstantVolume(group=group, thermostat=thermostat)

"""
    Langevin(group; gamma, kT)

Langevin method applied to a workflow group.
"""
@kwdef struct Langevin <: Method
    group
    gamma
    kT
end
Langevin(group; gamma, kT) = Langevin(group=group, gamma=gamma, kT=kT)

"""
    Brownian(group; gamma, kT)

Brownian dynamics method applied to a workflow group.
"""
@kwdef struct Brownian <: Method
    group
    gamma
    kT
end
Brownian(group; gamma, kT) = Brownian(group=group, gamma=gamma, kT=kT)

"""
    ActiveOrnsteinUhlenbeck(group; gamma, kT, tau=nothing, noise_scale=nothing, spectrum=nothing)

Active Ornstein-Uhlenbeck method applied to a workflow group.
"""
@kwdef struct ActiveOrnsteinUhlenbeck <: Method
    group
    gamma
    kT
    tau = nothing
    noise_scale = nothing
    spectrum = nothing
end
function ActiveOrnsteinUhlenbeck(group; gamma, kT, tau=nothing, noise_scale=nothing, spectrum=nothing)
    return ActiveOrnsteinUhlenbeck(group=group,
                                   gamma=gamma,
                                   kT=kT,
                                   tau=tau,
                                   noise_scale=noise_scale,
                                   spectrum=spectrum)
end

"""
    Integrator(; dt, scheme=nothing, forces=Force[], methods=Method[], metadata=Dict())

High-level workflow integrator object. It owns the timestep, force objects,
methods, and thermostat-bearing methods that define the simulation dynamics.
"""
@kwdef struct Integrator
    dt
    scheme = nothing
    forces::Vector{Force} = Force[]
    methods::Vector{Method} = Method[]
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

struct CompiledIntegrator
    dt::Float64
    scheme::AbstractWorkflowScheme
    build_spec!::Function
    metadata::Dict{Symbol,Any}
end

function build_lowlevel_integrator(compiled::CompiledIntegrator,
                                   st::SimulationState;
                                   system=nothing,
                                   materialized_groups::Dict{Symbol,Any}=Dict{Symbol,Any}())
    return compiled.build_spec!(st, system, materialized_groups)
end

function _method_list(methods)
    methods === nothing && return Method[]
    host = methods isa Method ? Method[methods] : Method[collect(methods)...]
    all(method -> method isa Method, host) ||
        throw(ArgumentError("methods must be a Method or a collection of Method objects."))
    return host
end

function _normalize_scheme(explicit_scheme, methods::Vector{Method})
    if explicit_scheme === nothing
        isempty(methods) && throw(ArgumentError("Integrator requires at least one method to infer a scheme."))
        if all(method -> method isa ConstantVolume, methods)
            return VelocityVerlet()
        elseif all(method -> method isa Langevin, methods)
            return VelocityVerlet()
        elseif all(method -> method isa Union{Brownian,ActiveOrnsteinUhlenbeck}, methods)
            return EulerMaruyama()
        else
            throw(ArgumentError("Unable to infer an integrator scheme for the supplied workflow methods."))
        end
    end

    explicit_scheme isa AbstractWorkflowScheme ||
        throw(ArgumentError("Unsupported workflow scheme $(typeof(explicit_scheme))."))
    return explicit_scheme
end

_scheme_family(::Union{VelocityVerlet,BAOAB,BAOA,GSM}) = :langevin
_scheme_family(::Union{EulerHeun,EulerMaruyama}) = :brownian

function _materialized_group_filter(system::ParticleSystem,
                                    materialized_groups::Dict{Symbol,Any},
                                    ref)
    if ref isa Symbol
        haskey(materialized_groups, ref) ||
            throw(ArgumentError("Unknown workflow group $(ref). Add it to `Simulation(groups=...)` or pass the Group object directly."))
        return materialized_groups[ref]
    elseif ref isa Group
        return get(materialized_groups, ref.name, materialize_group(system, ref))
    else
        throw(ArgumentError("Workflow methods must target a Group or Symbol group name; got $(typeof(ref))."))
    end
end

function _apply_stochastic_method!(spec,
                                   st::SimulationState,
                                   filter,
                                   method::Union{Langevin,Brownian,ActiveOrnsteinUhlenbeck},
                                   dtT)
    Filters.set_friction!(spec, st, method.gamma; filter=filter)
    Filters.set_temperature!(spec, st, dtT, method.kT; filter=filter)
    return spec
end

function _apply_ou_controls!(spec,
                             st::SimulationState,
                             filter,
                             method::ActiveOrnsteinUhlenbeck,
                             dtT)
    if method isa ActiveOrnsteinUhlenbeck
        if method.spectrum !== nothing
            (method.tau === nothing && method.noise_scale === nothing) ||
                throw(ArgumentError("ActiveOrnsteinUhlenbeck accepts either `spectrum` or (`tau`, `noise_scale`), not both."))
            spec_in = method.spectrum
            if spec_in isa OUSpectrum
                Filters.set_ou_spectrum!(spec, st, spec_in.taus, spec_in.scales; filter=filter, dt=dtT)
            elseif spec_in isa Tuple && length(spec_in) == 2
                Filters.set_ou_spectrum!(spec, st, spec_in[1], spec_in[2]; filter=filter, dt=dtT)
            else
                throw(ArgumentError("Unsupported OU spectrum container $(typeof(spec_in)). Use OUSpectrum(taus, scales) or a two-tuple."))
            end
        else
            (method.tau !== nothing && method.noise_scale !== nothing) ||
                throw(ArgumentError("ActiveOrnsteinUhlenbeck requires either `spectrum` or both `tau` and `noise_scale`."))
            Filters.set_ou_spectrum!(spec, st, method.tau, method.noise_scale; filter=filter, dt=dtT)
        end
    end
    return spec
end

function _stochastic_builder(scheme::AbstractWorkflowScheme,
                             methods::Vector{Method},
                             dtT)
    family = _scheme_family(scheme)
    if family === :langevin
        all(method -> method isa Union{Langevin,ActiveOrnsteinUhlenbeck}, methods) ||
            throw(ArgumentError("VelocityVerlet/BAOAB/BAOA/GSM only support Langevin and ActiveOrnsteinUhlenbeck workflow methods right now."))
        seed = methods[1]
        constructor = if scheme isa VelocityVerlet
            SimulationCore.velocityverlet
        elseif scheme isa BAOAB
            SimulationCore.baoab
        elseif scheme isa BAOA
            SimulationCore.baoa
        else
            SimulationCore.gsm
        end
        return function (st::SimulationState, system::ParticleSystem, materialized_groups::Dict{Symbol,Any})
            spec = constructor(st; gamma=seed.gamma, temperature=seed.kT, dt=dtT)
            for method in methods
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                _apply_stochastic_method!(spec, st, filter, method, dtT)
            end
            for method in methods
                method isa ActiveOrnsteinUhlenbeck || continue
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                _apply_ou_controls!(spec, st, filter, method, dtT)
            end
            return spec
        end
    else
        all(method -> method isa Union{Brownian,ActiveOrnsteinUhlenbeck}, methods) ||
            throw(ArgumentError("EulerHeun/EulerMaruyama only support Brownian and ActiveOrnsteinUhlenbeck methods right now."))
        seed = methods[1]
        constructor = scheme isa EulerHeun ? SimulationCore.eulerheun : SimulationCore.eulermaruyama
        return function (st::SimulationState, system::ParticleSystem, materialized_groups::Dict{Symbol,Any})
            spec = constructor(st; gamma=seed.gamma, temperature=seed.kT, dt=dtT)
            for method in methods
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                _apply_stochastic_method!(spec, st, filter, method, dtT)
            end
            for method in methods
                method isa ActiveOrnsteinUhlenbeck || continue
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                _apply_ou_controls!(spec, st, filter, method, dtT)
            end
            return spec
        end
    end
end

function _constant_volume_builder(methods::Vector{Method}, dtT)
    thermostats = Any[method.thermostat for method in methods]
    all(thermo -> thermo === nothing || thermo isa Thermostat, thermostats) ||
        throw(ArgumentError("ConstantVolume methods require either `thermostat=nothing` for NVE or a workflow Thermostat."))

    active_idx = findall(thermo -> thermo !== nothing, thermostats)
    isempty(active_idx) && return function (st::SimulationState, system::ParticleSystem, materialized_groups::Dict{Symbol,Any})
        return SimulationCore.nve(st; dt=dtT)
    end

    bath_ids = Int32[]
    nbaths = 0
    for thermo in thermostats
        if thermo === nothing
            push!(bath_ids, Int32(0))
        else
            nbaths += 1
            push!(bath_ids, Int32(nbaths))
        end
    end

    active_thermostats = Thermostat[thermostats[i] for i in active_idx]

    if all(thermo -> thermo isa CSVR, active_thermostats)
        temperatures = Float64[(thermostats[i]::CSVR).kT for i in active_idx]
        taus = Float64[(thermostats[i]::CSVR).tau for i in active_idx]
        return function (st::SimulationState, system::ParticleSystem, materialized_groups::Dict{Symbol,Any})
            spec = SimulationCore.csvr(st; temperatures=temperatures, taus=taus, mass=st.mass)
            fill!(spec.workspace.particle_bath_id, Int32(0))
            for (method, bath_id) in zip(methods, bath_ids)
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                idx = Filters.resolve_gpu(filter, st)
                Filters.assign_scalar!(spec.workspace.particle_bath_id, idx, bath_id)
            end
            fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(eltype(spec.workspace.cumulative_energy_exchange_per_bath)))
            fill!(spec.workspace.last_velocity_scale_per_bath, one(eltype(spec.workspace.last_velocity_scale_per_bath)))
            spec.workspace.kinetic_initialized = false
            spec.workspace.dof_dirty = true
            return spec
        end
    elseif all(thermo -> thermo isa NoseHooverChain, active_thermostats)
        chain_lengths = unique((thermo::NoseHooverChain).chain_length for thermo in active_thermostats)
        substeps = unique((thermo::NoseHooverChain).substeps for thermo in active_thermostats)
        length(chain_lengths) == 1 ||
            throw(ArgumentError("All NoseHooverChain workflow methods must use the same chain_length."))
        length(substeps) == 1 ||
            throw(ArgumentError("All NoseHooverChain workflow methods must use the same substeps."))
        temperatures = Float64[(thermostats[i]::NoseHooverChain).kT for i in active_idx]
        taus = Float64[(thermostats[i]::NoseHooverChain).tau for i in active_idx]
        return function (st::SimulationState, system::ParticleSystem, materialized_groups::Dict{Symbol,Any})
            spec = SimulationCore.nosehooverchain(st;
                                                  temperatures=temperatures,
                                                  taus=taus,
                                                  chain_length=chain_lengths[1],
                                                  substeps=substeps[1],
                                                  mass=st.mass)
            fill!(spec.workspace.particle_bath_id, Int32(0))
            for (method, bath_id) in zip(methods, bath_ids)
                filter = _materialized_group_filter(system, materialized_groups, method.group)
                idx = Filters.resolve_gpu(filter, st)
                Filters.assign_scalar!(spec.workspace.particle_bath_id, idx, bath_id)
            end
            fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(eltype(spec.workspace.cumulative_energy_exchange_per_bath)))
            fill!(spec.workspace.last_velocity_scale_per_bath, one(eltype(spec.workspace.last_velocity_scale_per_bath)))
            spec.workspace.kinetic_initialized = false
            spec.workspace.dof_dirty = true
            return spec
        end
    else
        throw(ArgumentError("ConstantVolume methods cannot mix CSVR and NoseHooverChain thermostats in one workflow integrator."))
    end
end

function compile_integrator(system::ParticleSystem, integrator::Integrator; precision=:f64)
    methods = _method_list(integrator.methods)
    isempty(methods) && throw(ArgumentError("Integrator requires at least one workflow method."))
    scheme = _normalize_scheme(integrator.scheme, methods)
    dtT = _precision_type(precision)(integrator.dt)

    builder = if all(method -> method isa ConstantVolume, methods)
        scheme isa VelocityVerlet ||
            throw(ArgumentError("ConstantVolume workflow methods currently support only the default/VelocityVerlet MD stepping path."))
        _constant_volume_builder(methods, dtT)
    else
        all(method -> method isa Union{Langevin,Brownian,ActiveOrnsteinUhlenbeck}, methods) ||
            throw(ArgumentError("Workflow integrator methods may not mix ConstantVolume with stochastic methods."))
        _stochastic_builder(scheme, methods, dtT)
    end

    metadata = Dict{Symbol,Any}(
        :scheme => nameof(typeof(scheme)),
        :method_types => Symbol[nameof(typeof(method)) for method in methods],
        :dt => Float64(dtT),
    )
    return CompiledIntegrator(Float64(dtT), scheme, builder, metadata)
end
