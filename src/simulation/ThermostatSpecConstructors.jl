const NHC_PROPAGATOR_LEGACY  = UInt8(1)
const NHC_PROPAGATOR_GROMACS = UInt8(2)
const NHC_PROPAGATOR_LAMMPS  = UInt8(3)

@inline function _nhc_propagator_id(propagator::Symbol)
    if propagator === :legacy
        return NHC_PROPAGATOR_LEGACY
    elseif propagator === :gromacs
        return NHC_PROPAGATOR_GROMACS
    elseif propagator === :lammps
        return NHC_PROPAGATOR_LAMMPS
    end
    throw(ArgumentError("Unsupported NHC propagator $(propagator). Expected :legacy, :gromacs, or :lammps."))
end

@inline function _nhc_propagator_name(propagator::UInt8)
    if propagator == NHC_PROPAGATOR_LEGACY
        return :legacy
    elseif propagator == NHC_PROPAGATOR_GROMACS
        return :gromacs
    elseif propagator == NHC_PROPAGATOR_LAMMPS
        return :lammps
    end
    return :unknown
end

NHCParams{T}(mass::T,
             target_temperature::Vector{T},
             tau::Vector{T},
             substeps::Int,
             chain_length::Int,
             chain_masses::Matrix{T}) where {T<:AbstractFloat} =
    NHCParams{T}(mass,
                 target_temperature,
                 tau,
                 substeps,
                 chain_length,
                 chain_masses,
                 NHC_PROPAGATOR_GROMACS)

NHCParams(mass::T,
          target_temperature::Vector{T},
          tau::Vector{T},
          substeps::Int,
          chain_length::Int,
          chain_masses::Matrix{T}) where {T<:AbstractFloat} =
    NHCParams{T}(mass,
                 target_temperature,
                 tau,
                 substeps,
                 chain_length,
                 chain_masses)

@inline function _default_nhc_chain_masses(::Type{T},
                                           dof::Int,
                                           target_temperature::T,
                                           tau::T,
                                           chain_length::Int) where {T<:AbstractFloat}
    @assert chain_length >= 1
    base = target_temperature * tau * tau
    masses = Vector{T}(undef, chain_length)
    masses[1] = max(one(T), T(dof)) * base
    @inbounds for j in 2:chain_length
        masses[j] = base
    end
    return masses
end

@inline function _nhc_chain_masses_signature(masses::AbstractArray{T}) where {T<:AbstractFloat}
    sig = hash(size(masses))
    @inbounds for q in masses
        sig = hash(q, sig)
    end
    return sig
end

@inline function _new_nhc_workspace(backend::Backends.AbstractBackend,
                                    ::Type{T},
                                    chain_length::Int,
                                    nbaths::Int,
                                    nparticles::Int) where {T<:AbstractFloat}
    return NHCWorkspace{T}(Backends.zeros_matrix(backend, T, chain_length, nbaths),
                           Backends.zeros_matrix(backend, T, chain_length, nbaths),
                           Backends.zeros_matrix(backend, T, chain_length, nbaths),
                           Backends.zeros_matrix(backend, T, chain_length, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.fill_vector(backend, Int32(1), nparticles),
                           Backends.zeros_vector(backend, Int32, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.zeros_vector(backend, T, nbaths),
                           Backends.fill_vector(backend, one(T), nbaths),
                           UInt64(0),
                           true,
                           false)
end

@inline function _new_csvr_workspace(backend::Backends.AbstractBackend,
                                     ::Type{T},
                                     nbaths::Int,
                                     nparticles::Int) where {T<:AbstractFloat}
    return CSVRWorkspace{T}(Backends.zeros_vector(backend, T, nbaths),
                            Backends.zeros_vector(backend, T, nbaths),
                            Backends.fill_vector(backend, Int32(1), nparticles),
                            Backends.zeros_vector(backend, Int32, nbaths),
                            Backends.zeros_vector(backend, T, nbaths),
                            Backends.zeros_vector(backend, T, nbaths),
                            Backends.zeros_vector(backend, T, nbaths),
                            Backends.fill_vector(backend, one(T), nbaths),
                            true,
                            false)
end

"""
    nosehooverchain(st; temperature=1, tau=1, chain_length=5, substeps=5,
                    mass=st.mass, chain_masses=nothing, propagator=:gromacs) -> NHCSpec

Create a deterministic NVT integrator specification using a Nose-Hoover Chain.
When `chain_masses` is omitted, masses are initialized from `(dof, T, tau)`
using the standard `Q₁ = g T τ²`, `Qⱼ = T τ² (j>1)` rule. `propagator`
selects the chain update scheme: `:gromacs` is the package default and uses
a GPU port of the reversible Suzuki-Yoshida propagator used by GROMACS,
`:lammps` uses a GPU port of the reversible `tloop` chain update used by
LAMMPS, and `:legacy` preserves the original package implementation. For
`propagator=:gromacs`, `substeps=5` matches the default fifth-order outer
repetition used there; for `propagator=:lammps`, `substeps` corresponds to
LAMMPS `tloop` and `substeps=1` matches the LAMMPS default.
"""
function nosehooverchain(st::SimulationState{T};
                         temperature::Union{Nothing,Real}=nothing,
                         tau::Union{Nothing,Real}=nothing,
                         temperatures::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         taus::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         chain_length::Integer=5,
                         substeps::Integer=5,
                         mass::Real=st.mass,
                         chain_masses::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}}=nothing,
                         propagator::Symbol=:gromacs) where {T<:AbstractFloat}
    chain_length >= 1 || throw(ArgumentError("chain_length must be >= 1, got $(chain_length)."))
    substeps >= 1 || throw(ArgumentError("substeps must be >= 1, got $(substeps)."))
    massT = T(mass)
    massT > zero(T) || throw(ArgumentError("NHC mass must be > 0."))
    propagator_id = _nhc_propagator_id(propagator)

    if temperatures !== nothing && temperature !== nothing
        throw(ArgumentError("Provide either `temperature` or `temperatures`, not both."))
    end
    if taus !== nothing && tau !== nothing
        throw(ArgumentError("Provide either `tau` or `taus`, not both."))
    end

    temp_vec = if temperatures === nothing
        [T(something(temperature, one(T)))]
    else
        T.(temperatures)
    end
    tau_vec = if taus === nothing
        [T(something(tau, one(T)))]
    else
        T.(taus)
    end

    length(temp_vec) == length(tau_vec) ||
        throw(ArgumentError("temperatures and taus must have identical lengths."))
    nbaths = length(temp_vec)
    nbaths >= 1 || throw(ArgumentError("NHC requires at least one bath."))

    @inbounds for (b, Tb) in pairs(temp_vec)
        Tb > zero(T) || throw(ArgumentError("NHC target temperature for bath $(b) must be > 0."))
    end
    @inbounds for (b, τb) in pairs(tau_vec)
        τb > zero(T) || throw(ArgumentError("NHC tau for bath $(b) must be > 0."))
    end

    dof_total = (_is_3d(st) ? 3 : 2) * length(st.rx)
    dof_guess = max(1, cld(dof_total, nbaths))

    masses = if chain_masses === nothing
        out = Matrix{T}(undef, Int(chain_length), nbaths)
        @inbounds for b in 1:nbaths
            col = _default_nhc_chain_masses(T, dof_guess, temp_vec[b], tau_vec[b], Int(chain_length))
            out[:, b] = col
        end
        out
    else
        if chain_masses isa AbstractVector
            length(chain_masses) == chain_length ||
                throw(ArgumentError("Vector chain_masses length must equal chain_length ($(chain_length))."))
            v = T.(chain_masses)
            repeat(reshape(v, Int(chain_length), 1), 1, nbaths)
        else
            cm = T.(chain_masses)
            size(cm, 1) == chain_length ||
                throw(ArgumentError("Matrix chain_masses first dimension must equal chain_length ($(chain_length))."))
            size(cm, 2) == nbaths ||
                throw(ArgumentError("Matrix chain_masses second dimension must equal number of baths ($(nbaths))."))
            cm
        end
    end
    @inbounds for j in axes(masses, 1), b in axes(masses, 2)
        masses[j, b] > zero(T) ||
            throw(ArgumentError("NHC chain mass Q[$(j), bath=$(b)] must be > 0."))
    end

    params = NHCParams{T}(massT,
                          temp_vec,
                          tau_vec,
                          Int(substeps),
                          Int(chain_length),
                          masses,
                          propagator_id)
    return NHCSpec{T}(params, _new_nhc_workspace(Backends.storage_backend(st), T, Int(chain_length), nbaths, length(st.rx)))
end

"""
    csvr(st; temperature=1, tau=1, mass=st.mass) -> CSVRSpec
    csvr(st; temperatures, taus, mass=st.mass) -> CSVRSpec

Create a deterministic MD integrator using the Bussi canonical-sampling through
velocity rescaling (CSVR) thermostat. The thermostat acts on one or more baths
defined by filter assignments, with one global velocity-rescaling factor drawn
per bath and per timestep.
"""
function csvr(st::SimulationState{T};
              temperature::Union{Nothing,Real}=nothing,
              tau::Union{Nothing,Real}=nothing,
              temperatures::Union{Nothing,AbstractVector{<:Real}}=nothing,
              taus::Union{Nothing,AbstractVector{<:Real}}=nothing,
              mass::Real=st.mass) where {T<:AbstractFloat}
    if temperatures !== nothing && temperature !== nothing
        throw(ArgumentError("Provide either `temperature` or `temperatures`, not both."))
    end
    if taus !== nothing && tau !== nothing
        throw(ArgumentError("Provide either `tau` or `taus`, not both."))
    end

    massT = T(mass)
    massT > zero(T) || throw(ArgumentError("CSVR mass must be > 0."))

    temp_vec = if temperatures === nothing
        [T(something(temperature, one(T)))]
    else
        T.(temperatures)
    end
    tau_vec = if taus === nothing
        [T(something(tau, one(T)))]
    else
        T.(taus)
    end

    length(temp_vec) == length(tau_vec) ||
        throw(ArgumentError("temperatures and taus must have identical lengths."))
    nbaths = length(temp_vec)
    nbaths >= 1 || throw(ArgumentError("CSVR requires at least one bath."))

    @inbounds for (b, Tb) in pairs(temp_vec)
        Tb > zero(T) || throw(ArgumentError("CSVR target temperature for bath $(b) must be > 0."))
    end
    @inbounds for (b, τb) in pairs(tau_vec)
        τb > zero(T) || throw(ArgumentError("CSVR tau for bath $(b) must be > 0."))
    end

    params = CSVRParams{T}(massT, temp_vec, tau_vec)
    return CSVRSpec{T}(params, _new_csvr_workspace(Backends.storage_backend(st), T, nbaths, length(st.rx)))
end
