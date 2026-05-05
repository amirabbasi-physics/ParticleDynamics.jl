# -----------------------------------------------------------------------------
# NHC multi-bath controls
# -----------------------------------------------------------------------------

function _nhc_ensure_particle_bath_buffer!(spec::NHCSpec, st::SimulationState)
    N = length(st.rx)
    ws = spec.workspace
    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    return ws.particle_bath_id
end

function _nhc_resize_baths!(spec::NHCSpec{T},
                            st::SimulationState,
                            nbaths::Int) where {T<:AbstractFloat}
    nbaths >= 1 || throw(ArgumentError("NHC requires at least one bath."))
    p = spec.params
    ws = spec.workspace
    old_nbaths = length(p.target_temperature)
    old_target = copy(p.target_temperature)
    old_tau = copy(p.tau)
    old_masses = copy(p.chain_masses)

    p.target_temperature = Vector{T}(undef, nbaths)
    p.tau = Vector{T}(undef, nbaths)
    p.chain_masses = Matrix{T}(undef, p.chain_length, nbaths)

    @inbounds for b in 1:nbaths
        src = min(b, old_nbaths)
        p.target_temperature[b] = old_target[src]
        p.tau[b] = old_tau[src]
        p.chain_masses[:, b] .= old_masses[:, src]
    end

    if size(ws.xi) != (p.chain_length, nbaths)
        ws.xi = CUDA.zeros(T, p.chain_length, nbaths)
        ws.eta = CUDA.zeros(T, p.chain_length, nbaths)
        ws.chain_force = CUDA.zeros(T, p.chain_length, nbaths)
        ws.chain_masses = CUDA.zeros(T, p.chain_length, nbaths)
    end
    if length(ws.target_temperature) != nbaths
        ws.target_temperature = CUDA.zeros(T, nbaths)
    end
    if length(ws.bath_counts) != nbaths
        ws.bath_counts = CUDA.zeros(Int32, nbaths)
    end
    if length(ws.dof_per_bath) != nbaths
        ws.dof_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.kinetic_total_per_bath) != nbaths
        ws.kinetic_total_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.kinetic_stage_start_per_bath) != nbaths
        ws.kinetic_stage_start_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.cumulative_energy_exchange_per_bath) != nbaths
        ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.thermostat_kinetic_per_bath) != nbaths
        ws.thermostat_kinetic_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.thermostat_potential_per_bath) != nbaths
        ws.thermostat_potential_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.last_velocity_scale_per_bath) != nbaths
        ws.last_velocity_scale_per_bath = CUDA.fill(one(T), nbaths)
    end

    _nhc_ensure_particle_bath_buffer!(spec, st)
    fill!(ws.cumulative_energy_exchange_per_bath, zero(T))
    fill!(ws.last_velocity_scale_per_bath, one(T))
    ws.chain_masses_signature = UInt64(0)
    ws.kinetic_initialized = false
    ws.dof_dirty = true
    return spec
end

function _nhc_selected_baths(spec::NHCSpec, st::SimulationState, filter::Filter)
    idx = resolve_gpu(filter, st)
    if length(idx) == 0
        return Int[]
    end
    _nhc_ensure_particle_bath_buffer!(spec, st)
    selected = gather(spec.workspace.particle_bath_id, idx)
    return unique(Int.(selected))
end

function _nhc_apply_tau_to_baths!(spec::NHCSpec{T},
                                  bath_ids::AbstractVector{<:Integer},
                                  tau_new::T;
                                  rescale_chain_masses::Bool=true) where {T<:AbstractFloat}
    p = spec.params
    tau_new > zero(T) || throw(ArgumentError("NHC timescale tau must be > 0."))
    @inbounds for b in bath_ids
        1 <= b <= length(p.tau) || continue
        tau_old = p.tau[b]
        if rescale_chain_masses
            α = (tau_new / tau_old)^2
            p.chain_masses[:, b] .*= α
        end
        p.tau[b] = tau_new
    end
    ws = spec.workspace
    ws.chain_masses_signature = UInt64(0)
    ws.kinetic_initialized = false
    ws.dof_dirty = true
    return spec
end

"""
    assign_nhc_baths!(spec, st, filter=>bath_id, ...)

Assign per-particle NHC bath ids from filter pairs. Every particle must be
assigned by at least one pair; later pairs overwrite earlier ones.
"""
function assign_nhc_baths!(spec::NHCSpec{T},
                           st::SimulationState,
                           pairs::Pair{<:Filter,<:Integer}...) where {T<:AbstractFloat}
    isempty(pairs) && return spec
    nbaths = maximum(last.(pairs))
    _nhc_resize_baths!(spec, st, nbaths)

    ws = spec.workspace
    fill!(ws.particle_bath_id, Int32(0))
    for (f, bath_id) in pairs
        1 <= bath_id <= nbaths || throw(ArgumentError("NHC bath id $(bath_id) out of range 1:$(nbaths)."))
        idx = resolve_gpu(f, st)
        assign_scalar!(ws.particle_bath_id, idx, Int32(bath_id))
    end

    n_unassigned = Int(CUDA.sum(Int32.(ws.particle_bath_id .== Int32(0))))
    n_unassigned == 0 ||
        throw(ArgumentError("NHC bath assignment left $(n_unassigned) particles unassigned. Provide a complete filter partition."))

    ws.kinetic_initialized = false
    ws.dof_dirty = true
    fill!(ws.cumulative_energy_exchange_per_bath, zero(T))
    fill!(ws.last_velocity_scale_per_bath, one(T))
    return spec
end

"""
    set_thermostat_temperature!(spec::NHCSpec, T)

Set all NHC bath target temperatures to `T`.
"""
function set_thermostat_temperature!(spec::NHCSpec{T}, temperature::Real) where {T<:AbstractFloat}
    Ttarget = T(temperature)
    Ttarget > zero(T) || throw(ArgumentError("NHC target temperature must be > 0."))
    fill!(spec.params.target_temperature, Ttarget)
    if length(spec.workspace.target_temperature) == length(spec.params.target_temperature)
        fill!(spec.workspace.target_temperature, Ttarget)
    end
    spec.workspace.kinetic_initialized = false
    spec.workspace.dof_dirty = true
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    fill!(spec.workspace.last_velocity_scale_per_bath, one(T))
    return spec
end

function set_thermostat_temperature!(spec::NHCSpec{T},
                                     st::SimulationState,
                                     temperature::Real;
                                     filter::Filter=All()) where {T<:AbstractFloat}
    if filter isa All
        return set_thermostat_temperature!(spec, temperature)
    end
    Ttarget = T(temperature)
    Ttarget > zero(T) || throw(ArgumentError("NHC target temperature must be > 0."))
    for b in _nhc_selected_baths(spec, st, filter)
        if 1 <= b <= length(spec.params.target_temperature)
            spec.params.target_temperature[b] = Ttarget
        end
    end
    spec.workspace.kinetic_initialized = false
    spec.workspace.dof_dirty = true
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    fill!(spec.workspace.last_velocity_scale_per_bath, one(T))
    return spec
end

function set_thermostat_temperature!(st::SimulationState,
                                     spec::NHCSpec,
                                     temperature::Real;
                                     filter::Filter=All())
    return set_thermostat_temperature!(spec, st, temperature; filter=filter)
end

function set_thermostat_temperature!(spec::NHCSpec{T},
                                     st::SimulationState,
                                     pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    return set_temperature!(spec, st, one(T), pairs...)
end

function set_thermostat_temperature!(st::SimulationState,
                                     spec::NHCSpec,
                                     pairs::Pair{<:Filter,<:Real}...)
    return set_thermostat_temperature!(spec, st, pairs...)
end

"""
    set_thermostat_timescale!(spec::NHCSpec, tau; rescale_chain_masses=true)

Set all NHC bath response timescales. By default chain masses are rescaled by
`(tau_new / tau_old)^2` per updated bath.
"""
function set_thermostat_timescale!(spec::NHCSpec{T},
                                   tau::Real;
                                   rescale_chain_masses::Bool=true) where {T<:AbstractFloat}
    tau_new = T(tau)
    return _nhc_apply_tau_to_baths!(spec, collect(eachindex(spec.params.tau)), tau_new;
                                    rescale_chain_masses=rescale_chain_masses)
end

function set_thermostat_timescale!(spec::NHCSpec{T},
                                   st::SimulationState,
                                   tau::Real;
                                   filter::Filter=All(),
                                   rescale_chain_masses::Bool=true) where {T<:AbstractFloat}
    bath_ids = if filter isa All
        collect(eachindex(spec.params.tau))
    else
        _nhc_selected_baths(spec, st, filter)
    end
    return _nhc_apply_tau_to_baths!(spec, bath_ids, T(tau);
                                    rescale_chain_masses=rescale_chain_masses)
end

function set_thermostat_timescale!(st::SimulationState,
                                   spec::NHCSpec,
                                   tau::Real;
                                   filter::Filter=All(),
                                   rescale_chain_masses::Bool=true)
    return set_thermostat_timescale!(spec, st, tau;
                                     filter=filter,
                                     rescale_chain_masses=rescale_chain_masses)
end

function set_thermostat_timescale!(spec::NHCSpec{T},
                                   st::SimulationState,
                                   pairs::Pair{<:Filter,<:Real}...;
                                   rescale_chain_masses::Bool=true) where {T<:AbstractFloat}
    for (f, τval) in pairs
        set_thermostat_timescale!(spec, st, τval;
                                  filter=f,
                                  rescale_chain_masses=rescale_chain_masses)
    end
    return spec
end

function set_thermostat_timescale!(st::SimulationState,
                                   spec::NHCSpec,
                                   pairs::Pair{<:Filter,<:Real}...;
                                   rescale_chain_masses::Bool=true)
    return set_thermostat_timescale!(spec, st, pairs...;
                                     rescale_chain_masses=rescale_chain_masses)
end

# Deterministic NHC controls are global and do not expose stochastic knobs.
set_noise_scale!(spec::NHCSpec, value::Real) =
    throw(ArgumentError("NHC is deterministic and has no noise scale. Use set_thermostat_temperature! and set_thermostat_timescale!."))
set_noise_scale!(spec::NHCSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("NHC is deterministic and has no noise scale."))
set_friction!(spec::NHCSpec, value::Real) =
    throw(ArgumentError("NHC has no Langevin friction coefficient. Use set_thermostat_timescale!."))
set_friction!(spec::NHCSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("NHC has no Langevin friction coefficient."))
set_corr_time!(spec::NHCSpec, value::Real) =
    throw(ArgumentError("NHC has no OU correlation-time parameter."))
set_corr_time!(spec::NHCSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("NHC has no OU correlation-time parameter."))

function set_temperature!(spec::NHCSpec{T}, dt::Real, temperature::Real) where {T<:AbstractFloat}
    return set_thermostat_temperature!(spec, temperature)
end

function set_temperature!(spec::NHCSpec{T},
                          st::SimulationState,
                          dt::Real,
                          temperature::Real;
                          filter::Filter=All()) where {T<:AbstractFloat}
    return set_thermostat_temperature!(spec, st, temperature; filter=filter)
end

function set_temperature!(spec::NHCSpec{T},
                          st::SimulationState,
                          dt::Real,
                          pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    isempty(pairs) && return spec
    nbaths = length(pairs)
    _nhc_resize_baths!(spec, st, nbaths)
    ws = spec.workspace
    fill!(ws.particle_bath_id, Int32(0))

    # Reinitialize chain masses when redefining bath layout from filter pairs.
    D = st.rz === nothing ? 2 : 3
    dof_guess = max(1, cld(D * length(st.rx), nbaths))
    @inbounds for b in 1:nbaths
        Tb = T(pairs[b].second)
        Tb > zero(T) || throw(ArgumentError("NHC target temperature for bath $(b) must be > 0."))
        spec.params.target_temperature[b] = Tb
        spec.params.chain_masses[:, b] .= Simulation._default_nhc_chain_masses(T,
                                                                                dof_guess,
                                                                                Tb,
                                                                                spec.params.tau[b],
                                                                                spec.params.chain_length)
    end

    @inbounds for b in 1:nbaths
        f = pairs[b].first
        idx = resolve_gpu(f, st)
        assign_scalar!(ws.particle_bath_id, idx, Int32(b))
    end

    n_unassigned = Int(CUDA.sum(Int32.(ws.particle_bath_id .== Int32(0))))
    n_unassigned == 0 ||
        throw(ArgumentError("NHC temperature assignment left $(n_unassigned) particles unassigned. Provide a complete filter partition."))

    ws.chain_masses_signature = UInt64(0)
    fill!(ws.cumulative_energy_exchange_per_bath, zero(T))
    fill!(ws.last_velocity_scale_per_bath, one(T))
    ws.kinetic_initialized = false
    ws.dof_dirty = true
    return spec
end

function set_temperature!(st::SimulationState,
                          spec::NHCSpec,
                          dt::Real,
                          pairs::Pair{<:Filter,<:Real}...)
    return set_temperature!(spec, st, dt, pairs...)
end

function set_temperature!(spec::NHCSpec,
                          st::SimulationState,
                          dt::Real,
                          mapping::AbstractDict{<:Filter,<:Real})
    return set_temperature!(spec, st, dt, collect(pairs(mapping))...)
end

function set_temperature!(st::SimulationState,
                          spec::NHCSpec,
                          dt::Real,
                          mapping::AbstractDict{<:Filter,<:Real})
    return set_temperature!(spec, st, dt, mapping)
end

# -----------------------------------------------------------------------------
# CSVR multi-bath controls
# -----------------------------------------------------------------------------

function _csvr_ensure_particle_bath_buffer!(spec::CSVRSpec, st::SimulationState)
    N = length(st.rx)
    ws = spec.workspace
    if length(ws.particle_bath_id) != N
        ws.particle_bath_id = CUDA.fill(Int32(1), N)
        ws.kinetic_initialized = false
        ws.dof_dirty = true
    end
    return ws.particle_bath_id
end

function _csvr_resize_baths!(spec::CSVRSpec{T},
                             st::SimulationState,
                             nbaths::Int) where {T<:AbstractFloat}
    nbaths >= 1 || throw(ArgumentError("CSVR requires at least one bath."))
    p = spec.params
    ws = spec.workspace
    old_nbaths = length(p.target_temperature)
    old_target = copy(p.target_temperature)
    old_tau = copy(p.tau)

    p.target_temperature = Vector{T}(undef, nbaths)
    p.tau = Vector{T}(undef, nbaths)

    @inbounds for b in 1:nbaths
        src = min(b, old_nbaths)
        p.target_temperature[b] = old_target[src]
        p.tau[b] = old_tau[src]
    end

    if length(ws.target_temperature) != nbaths
        ws.target_temperature = CUDA.zeros(T, nbaths)
    end
    if length(ws.tau) != nbaths
        ws.tau = CUDA.zeros(T, nbaths)
    end
    if length(ws.bath_counts) != nbaths
        ws.bath_counts = CUDA.zeros(Int32, nbaths)
    end
    if length(ws.dof_per_bath) != nbaths
        ws.dof_per_bath = CUDA.zeros(T, nbaths)
    end
    if length(ws.kinetic_total_per_bath) != nbaths
        ws.kinetic_total_per_bath = CUDA.zeros(T, nbaths)
    end
    ws.cumulative_energy_exchange_per_bath = CUDA.zeros(T, nbaths)
    ws.last_velocity_scale_per_bath = CUDA.fill(one(T), nbaths)

    _csvr_ensure_particle_bath_buffer!(spec, st)
    ws.kinetic_initialized = false
    ws.dof_dirty = true
    return spec
end

function _csvr_selected_baths(spec::CSVRSpec, st::SimulationState, filter::Filter)
    idx = resolve_gpu(filter, st)
    if length(idx) == 0
        return Int[]
    end
    _csvr_ensure_particle_bath_buffer!(spec, st)
    selected = gather(spec.workspace.particle_bath_id, idx)
    return unique(Int.(selected))
end

function _csvr_apply_tau_to_baths!(spec::CSVRSpec{T},
                                   bath_ids::AbstractVector{<:Integer},
                                   tau_new::T) where {T<:AbstractFloat}
    p = spec.params
    tau_new > zero(T) || throw(ArgumentError("CSVR timescale tau must be > 0."))
    @inbounds for b in bath_ids
        1 <= b <= length(p.tau) || continue
        p.tau[b] = tau_new
    end
    spec.workspace.kinetic_initialized = false
    spec.workspace.dof_dirty = true
    return spec
end

"""
    assign_csvr_baths!(spec, st, filter=>bath_id, ...)

Assign per-particle CSVR bath ids from filter pairs. Every particle must be
assigned by at least one pair; later pairs overwrite earlier ones.
"""
function assign_csvr_baths!(spec::CSVRSpec{T},
                            st::SimulationState,
                            pairs::Pair{<:Filter,<:Integer}...) where {T<:AbstractFloat}
    isempty(pairs) && return spec
    nbaths = maximum(last.(pairs))
    _csvr_resize_baths!(spec, st, nbaths)

    ws = spec.workspace
    fill!(ws.particle_bath_id, Int32(0))
    for (f, bath_id) in pairs
        1 <= bath_id <= nbaths || throw(ArgumentError("CSVR bath id $(bath_id) out of range 1:$(nbaths)."))
        idx = resolve_gpu(f, st)
        assign_scalar!(ws.particle_bath_id, idx, Int32(bath_id))
    end

    n_unassigned = Int(CUDA.sum(Int32.(ws.particle_bath_id .== Int32(0))))
    n_unassigned == 0 ||
        throw(ArgumentError("CSVR bath assignment left $(n_unassigned) particles unassigned. Provide a complete filter partition."))

    ws.kinetic_initialized = false
    ws.dof_dirty = true
    fill!(ws.cumulative_energy_exchange_per_bath, zero(T))
    fill!(ws.last_velocity_scale_per_bath, one(T))
    return spec
end

function set_thermostat_temperature!(spec::CSVRSpec{T}, temperature::Real) where {T<:AbstractFloat}
    Ttarget = T(temperature)
    Ttarget > zero(T) || throw(ArgumentError("CSVR target temperature must be > 0."))
    fill!(spec.params.target_temperature, Ttarget)
    if length(spec.workspace.target_temperature) == length(spec.params.target_temperature)
        fill!(spec.workspace.target_temperature, Ttarget)
    end
    spec.workspace.kinetic_initialized = false
    spec.workspace.dof_dirty = true
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    fill!(spec.workspace.last_velocity_scale_per_bath, one(T))
    return spec
end

function set_thermostat_temperature!(spec::CSVRSpec{T},
                                     st::SimulationState,
                                     temperature::Real;
                                     filter::Filter=All()) where {T<:AbstractFloat}
    if filter isa All
        return set_thermostat_temperature!(spec, temperature)
    end
    Ttarget = T(temperature)
    Ttarget > zero(T) || throw(ArgumentError("CSVR target temperature must be > 0."))
    for b in _csvr_selected_baths(spec, st, filter)
        if 1 <= b <= length(spec.params.target_temperature)
            spec.params.target_temperature[b] = Ttarget
        end
    end
    spec.workspace.kinetic_initialized = false
    spec.workspace.dof_dirty = true
    fill!(spec.workspace.cumulative_energy_exchange_per_bath, zero(T))
    fill!(spec.workspace.last_velocity_scale_per_bath, one(T))
    return spec
end

function set_thermostat_temperature!(st::SimulationState,
                                     spec::CSVRSpec,
                                     temperature::Real;
                                     filter::Filter=All())
    return set_thermostat_temperature!(spec, st, temperature; filter=filter)
end

function set_thermostat_temperature!(spec::CSVRSpec{T},
                                     st::SimulationState,
                                     pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    return set_temperature!(spec, st, one(T), pairs...)
end

function set_thermostat_temperature!(st::SimulationState,
                                     spec::CSVRSpec,
                                     pairs::Pair{<:Filter,<:Real}...)
    return set_thermostat_temperature!(spec, st, pairs...)
end

function set_thermostat_timescale!(spec::CSVRSpec{T},
                                   tau::Real) where {T<:AbstractFloat}
    tau_new = T(tau)
    return _csvr_apply_tau_to_baths!(spec, collect(eachindex(spec.params.tau)), tau_new)
end

function set_thermostat_timescale!(spec::CSVRSpec{T},
                                   st::SimulationState,
                                   tau::Real;
                                   filter::Filter=All()) where {T<:AbstractFloat}
    bath_ids = if filter isa All
        collect(eachindex(spec.params.tau))
    else
        _csvr_selected_baths(spec, st, filter)
    end
    return _csvr_apply_tau_to_baths!(spec, bath_ids, T(tau))
end

function set_thermostat_timescale!(st::SimulationState,
                                   spec::CSVRSpec,
                                   tau::Real;
                                   filter::Filter=All())
    return set_thermostat_timescale!(spec, st, tau; filter=filter)
end

function set_thermostat_timescale!(spec::CSVRSpec{T},
                                   st::SimulationState,
                                   pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, τval) in pairs
        set_thermostat_timescale!(spec, st, τval; filter=f)
    end
    return spec
end

function set_thermostat_timescale!(st::SimulationState,
                                   spec::CSVRSpec,
                                   pairs::Pair{<:Filter,<:Real}...)
    return set_thermostat_timescale!(spec, st, pairs...)
end

set_noise_scale!(spec::CSVRSpec, value::Real) =
    throw(ArgumentError("CSVR is not a per-particle stochastic thermostat. Use set_thermostat_temperature! and set_thermostat_timescale!."))
set_noise_scale!(spec::CSVRSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("CSVR is not a per-particle stochastic thermostat."))
set_friction!(spec::CSVRSpec, value::Real) =
    throw(ArgumentError("CSVR has no Langevin friction coefficient. Use set_thermostat_timescale!."))
set_friction!(spec::CSVRSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("CSVR has no Langevin friction coefficient."))
set_corr_time!(spec::CSVRSpec, value::Real) =
    throw(ArgumentError("CSVR has no OU correlation-time parameter."))
set_corr_time!(spec::CSVRSpec, st::SimulationState, value::Real; filter::Filter=All()) =
    throw(ArgumentError("CSVR has no OU correlation-time parameter."))

function set_temperature!(spec::CSVRSpec{T}, dt::Real, temperature::Real) where {T<:AbstractFloat}
    return set_thermostat_temperature!(spec, temperature)
end

function set_temperature!(spec::CSVRSpec{T},
                          st::SimulationState,
                          dt::Real,
                          temperature::Real;
                          filter::Filter=All()) where {T<:AbstractFloat}
    return set_thermostat_temperature!(spec, st, temperature; filter=filter)
end

function set_temperature!(spec::CSVRSpec{T},
                          st::SimulationState,
                          dt::Real,
                          pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    isempty(pairs) && return spec
    nbaths = length(pairs)
    _csvr_resize_baths!(spec, st, nbaths)
    ws = spec.workspace
    fill!(ws.particle_bath_id, Int32(0))

    @inbounds for b in 1:nbaths
        Tb = T(pairs[b].second)
        Tb > zero(T) || throw(ArgumentError("CSVR target temperature for bath $(b) must be > 0."))
        spec.params.target_temperature[b] = Tb
    end

    @inbounds for b in 1:nbaths
        f = pairs[b].first
        idx = resolve_gpu(f, st)
        assign_scalar!(ws.particle_bath_id, idx, Int32(b))
    end

    n_unassigned = Int(CUDA.sum(Int32.(ws.particle_bath_id .== Int32(0))))
    n_unassigned == 0 ||
        throw(ArgumentError("CSVR temperature assignment left $(n_unassigned) particles unassigned. Provide a complete filter partition."))

    fill!(ws.cumulative_energy_exchange_per_bath, zero(T))
    fill!(ws.last_velocity_scale_per_bath, one(T))
    ws.kinetic_initialized = false
    ws.dof_dirty = true
    return spec
end

function set_temperature!(st::SimulationState,
                          spec::CSVRSpec,
                          dt::Real,
                          pairs::Pair{<:Filter,<:Real}...)
    return set_temperature!(spec, st, dt, pairs...)
end

function set_temperature!(spec::CSVRSpec,
                          st::SimulationState,
                          dt::Real,
                          mapping::AbstractDict{<:Filter,<:Real})
    return set_temperature!(spec, st, dt, collect(pairs(mapping))...)
end

function set_temperature!(st::SimulationState,
                          spec::CSVRSpec,
                          dt::Real,
                          mapping::AbstractDict{<:Filter,<:Real})
    return set_temperature!(spec, st, dt, mapping)
end
