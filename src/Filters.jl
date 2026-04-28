"""
Particle-selection helpers used to assign per-group parameters (temperatures,
frictions, correlation times) as in `examples/TwoT_2D_LD_VV.jl`.
"""
module Filters

using CUDA
using CUDA: CuArray, CuDeviceVector
import ..Simulation
using ..Definitions
using ..Simulation: SimulationState, IntegratorSpec, VVSpec, BAOABSpec, BAOASpec, GSMSpec, BrownianSpec, EMSpec, NHCSpec, CSVRSpec
using ..BrownianIntegrators
using ..LangevinIntegrators

export Filter, All, TypeIDs, Indices, Selection,
       resolve, resolve_gpu, selection, count,
       assign_scalar!, assign_values!, gather, sum,
       set_noise_scale!, set_temperature!, set_friction!, set_corr_time!, set_ou_spectrum!,
       set_thermostat_temperature!, set_thermostat_timescale!, assign_nhc_baths!, assign_csvr_baths!,
       freeze_particles!, unfreeze_particles!

"""
Base type for particle selections that operate on `SimulationState`.
"""
abstract type Filter end

"""
Select all particles.
"""
struct All <: Filter end

"""
Filter by type IDs (`st.typeid`). Accepts a single `Int` or a vector.
"""
struct TypeIDs{I<:Integer} <: Filter
    ids::Vector{I}
end
TypeIDs(ids::AbstractVector{<:Integer}) = TypeIDs{Int}(Int.(ids))
TypeIDs(id::Integer) = TypeIDs([Int(id)])

"""
Explicit list of particle indices.
"""
struct Indices{I<:Integer} <: Filter
    idx::Vector{I}
end
Indices(idx::AbstractVector{<:Integer}) = Indices{Int}(Int.(idx))
Indices(id::Integer) = Indices([Int(id)])

"""
GPU/host selection pair returned by [`selection`](@ref). Access `.host` for
CPU arrays and `.device` for `CuArray{Int32}` indexing.
"""
struct Selection
    host::Vector{Int}
    device::CuArray{Int32,1}
end

function Selection(host::Vector{Int})
    dev = CuArray(Int32.(host))
    return Selection(host, dev)
end

count(sel::Selection) = length(sel.host)

# -----------------------------------------------------------------------------
# Index resolution helpers
# -----------------------------------------------------------------------------

function resolve(::All, st::SimulationState)
    N = length(st.rx)
    return collect(1:N)
end

function resolve(f::Indices, st::SimulationState)
    idx = Int.(f.idx)
    _validate_indices(idx, length(st.rx))
    return idx
end

function resolve(f::TypeIDs, st::SimulationState)
    return Int.(Array(resolve_gpu(f, st)))
end

"""
    resolve(filter, st) -> Vector{Int}

Return host indices matching `filter`. Used in the Filters unit tests to verify
type-based selections.
"""
resolve(st::SimulationState, f::Filter) = resolve(f, st)

"""
    resolve_gpu(filter, st) -> CuArray{Int32}

GPU version of [`resolve`](@ref), used by `assign_scalar!` and friends when
updating `CuArray` buffers directly.
"""
function _resolve_typeids_gpu(f::TypeIDs, st::SimulationState)
    N = length(st.rx)
    N == 0 && return CUDA.zeros(Int32, 0)
    ids = unique(Int32.(f.ids))
    isempty(ids) && return CUDA.zeros(Int32, 0)

    mask = CUDA.fill(false, N)
    @inbounds for id in ids
        mask .|= (st.typeid .== id)
    end

    idx64 = findall(mask)
    return Int32.(idx64)
end

function resolve_gpu(f::All, st::SimulationState)
    N = length(st.rx)
    return CuArray(Int32.(collect(1:N)))
end

function resolve_gpu(f::Indices, st::SimulationState)
    idx = Int.(f.idx)
    _validate_indices(idx, length(st.rx))
    return CuArray(Int32.(idx))
end

resolve_gpu(f::TypeIDs, st::SimulationState) = _resolve_typeids_gpu(f, st)

function resolve_gpu(f::Filter, st::SimulationState)
    host = resolve(f, st)
    return CuArray(Int32.(host))
end

resolve_gpu(st::SimulationState, f::Filter) = resolve_gpu(f, st)

"""
    selection(st, filter) -> Selection

Allocate a [`Selection`](@ref) (host+device indices) for repeated use.
"""
function selection(st::SimulationState, f::Filter)
    host = resolve(f, st)
    return Selection(host)
end

function selection(st::SimulationState, f::TypeIDs)
    dev = resolve_gpu(f, st)
    host = Int.(Array(dev))
    return Selection(host, dev)
end

function _validate_indices(idx::Vector{Int}, N::Int)
    for (k, i) in enumerate(idx)
        @assert 1 <= i <= N "Index $(i) at position $(k) out of bounds 1:$(N)"
    end
    nothing
end

"""
    count(filter, st)

Number of particles matched by `filter`.
"""
count(f::Filter, st::SimulationState) = length(resolve(f, st))
count(f::TypeIDs, st::SimulationState) = length(resolve_gpu(f, st))
count(st::SimulationState, f::Filter) = count(f, st)

# -----------------------------------------------------------------------------
# GPU kernels
# -----------------------------------------------------------------------------

function _assign_scalar_kernel!(dest::CuDeviceVector{T}, idx::CuDeviceVector{Int32}, value::T) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    dest[j] = value
    return
end

function _assign_vector_kernel!(dest::CuDeviceVector{T}, idx::CuDeviceVector{Int32}, values::CuDeviceVector{T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    dest[j] = values[i]
    return
end

function _gather_kernel!(out::CuDeviceVector{T}, src::CuDeviceVector{T}, idx::CuDeviceVector{Int32}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    out[i] = src[j]
    return
end

function _set_noise_gamma_kernel!(noise_scale::CuDeviceVector{T},
                                  gamma::CuDeviceVector{T},
                                  idx::CuDeviceVector{Int32},
                                  dt::T, temperature::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(idx)
    i > N && return
    j = Int(idx[i])
    g = gamma[j]
    noise_scale[j] = sqrt(T(2) * g * temperature * dt)
    return
end

function _set_noise_from_gamma!(noise_scale::CuArray{T,1},
                                 gamma::CuArray{T,1},
                                 idx::CuArray{Int32,1},
                                 dt::T, temperature::T) where {T<:AbstractFloat}
    N = length(idx)
    N == 0 && return
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _set_noise_gamma_kernel!(noise_scale, gamma, idx, dt, temperature)
    CUDA.@sync k(noise_scale, gamma, idx, dt, temperature; threads, blocks)
    return
end

function _ensure_corr_time_array(bp::BrownianIntegrators.BrownianParams{T}) where {T<:AbstractFloat}
    if bp.corr_time === nothing
        corr = CUDA.fill(zero(T), length(bp.gamma))
        return BrownianIntegrators.BrownianParams{T}(bp.gamma, bp.dt, bp.noise_scale, corr, bp.ou)
    end
    return bp
end

function _ensure_corr_time_array(em::BrownianIntegrators.EMParams{T}) where {T<:AbstractFloat}
    if em.corr_time === nothing
        corr = CUDA.fill(zero(T), length(em.gamma))
        return BrownianIntegrators.EMParams{T}(em.gamma, em.dt, em.noise_scale, corr, em.ou)
    end
    return em
end

# -----------------------------------------------------------------------------
# Integrator-spec parameter accessors
# -----------------------------------------------------------------------------

_gamma_view(spec::VVSpec) = spec.params.gamma
_gamma_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.gamma
_gamma_view(spec::BrownianSpec) = spec.params.gamma
_gamma_view(spec::EMSpec) = spec.params.gamma
_gamma_view(spec::NHCSpec) =
    throw(ArgumentError("NHC integrator has no per-particle friction buffer. Use set_thermostat_timescale!."))
_gamma_view(spec::CSVRSpec) =
    throw(ArgumentError("CSVR integrator has no per-particle friction buffer. Use set_thermostat_timescale!."))

_noise_scale_view(spec::VVSpec) = spec.params.noise_scale
_noise_scale_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.noise_scale
_noise_scale_view(spec::BrownianSpec) = spec.params.noise_scale
_noise_scale_view(spec::EMSpec) = spec.params.noise_scale
_noise_scale_view(spec::NHCSpec) =
    throw(ArgumentError("NHC integrator has no stochastic noise scale."))
_noise_scale_view(spec::CSVRSpec) =
    throw(ArgumentError("CSVR integrator has no stochastic noise scale."))

_corr_time_view(spec::VVSpec) = spec.params.corr_time
_corr_time_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.corr_time
_corr_time_view(spec::BrownianSpec) = spec.params.corr_time
_corr_time_view(spec::EMSpec) = spec.params.corr_time
_corr_time_view(spec::NHCSpec) = nothing
_corr_time_view(spec::CSVRSpec) = nothing

_ou_view(spec::VVSpec) = spec.params.ou
_ou_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.ou
_ou_view(spec::BrownianSpec) = spec.params.ou
_ou_view(spec::EMSpec) = spec.params.ou
_ou_view(spec::NHCSpec) = nothing
_ou_view(spec::CSVRSpec) = nothing

_dt_view(spec::VVSpec) = spec.params.dt
_dt_view(spec::Union{BAOABSpec,BAOASpec,GSMSpec}) = spec.params.dt
_dt_view(spec::BrownianSpec) = spec.params.dt
_dt_view(spec::EMSpec) = spec.params.dt

function _set_corr_time_view!(spec::VVSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=corr, ou=p.ou)
    return spec
end

function _set_corr_time_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=corr, ou=p.ou)
    return spec
end

function _set_corr_time_view!(spec::BrownianSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, p.dt, p.noise_scale, corr, p.ou)
    return spec
end

function _set_corr_time_view!(spec::EMSpec{T}, corr::Union{Nothing,CuArray{T,1}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, p.dt, p.noise_scale, corr, p.ou)
    return spec
end

function _set_ou_view!(spec::VVSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=p.corr_time, ou=ou)
    return spec
end

function _set_ou_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=p.dt, corr_time=p.corr_time, ou=ou)
    return spec
end

function _set_ou_view!(spec::BrownianSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, p.dt, p.noise_scale, p.corr_time, ou)
    return spec
end

function _set_ou_view!(spec::EMSpec{T}, ou::Union{Nothing,Definitions.OUSpectrum{T}}) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, p.dt, p.noise_scale, p.corr_time, ou)
    return spec
end

function _set_dt_view!(spec::VVSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.VVParams{T}(p.gamma, p.mass, p.noise_scale; dt=dt, corr_time=p.corr_time, ou=p.ou)
    return spec
end

function _set_dt_view!(spec::Union{BAOABSpec{T},BAOASpec{T},GSMSpec{T}}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = LangevinIntegrators.BAOABParams{T}(p.gamma, p.mass, p.noise_scale; dt=dt, corr_time=p.corr_time, ou=p.ou)
    return spec
end

function _set_dt_view!(spec::BrownianSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.BrownianParams{T}(p.gamma, dt, p.noise_scale, p.corr_time, p.ou)
    return spec
end

function _set_dt_view!(spec::EMSpec{T}, dt::T) where {T<:AbstractFloat}
    p = spec.params
    spec.params = BrownianIntegrators.EMParams{T}(p.gamma, dt, p.noise_scale, p.corr_time, p.ou)
    return spec
end

function _ensure_corr_time_array(spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    corr = _corr_time_view(spec)
    if corr === nothing
        corr = CUDA.fill(zero(T), length(_gamma_view(spec)))
        _set_corr_time_view!(spec, corr)
    end
    return corr
end

function _rebuild_single_mode_ou!(spec::IntegratorSpec{T}) where {T<:AbstractFloat}
    corr = _corr_time_view(spec)
    if corr === nothing
        _set_ou_view!(spec, nothing)
        return spec
    end
    ou = Simulation._build_single_mode_ou(T, _noise_scale_view(spec), corr, _dt_view(spec))
    _set_ou_view!(spec, ou)
    return spec
end

# -----------------------------------------------------------------------------
# Assign helpers
# -----------------------------------------------------------------------------

"""
    assign_scalar!(dest, st[, filter], value)

Fill elements of `dest` referenced by `filter` (or all particles) with `value`.
This underpins `set_friction!` and the temperature setup in
`examples/TwoT_2D_LD_VV.jl`.
"""
function assign_scalar!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, value::Real) where {T}
    N = length(idx)
    N == 0 && return dest
    val = T(value)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_scalar_kernel!(dest, idx, val)
    CUDA.@sync k(dest, idx, val; threads, blocks)
    return dest
end

function assign_scalar!(dest::CuArray{T,1}, sel::Selection, value::Real) where {T}
    return assign_scalar!(dest, sel.device, value)
end

function assign_scalar!(dest::CuArray{T,1}, st::SimulationState, f::Filter, value::Real) where {T}
    idx = resolve_gpu(f, st)
    assign_scalar!(dest, idx, value)
    return idx
end

function assign_scalar!(dest::CuArray{T,1}, st::SimulationState; filter::Filter=All(), value::Real) where {T}
    return assign_scalar!(dest, st, filter, value)
end

function assign_scalar!(dest::AbstractVector{T}, idx::Vector{Int}, value::Real) where {T}
    val = T(value)
    @inbounds for j in idx
        dest[j] = val
    end
    return dest
end

function assign_scalar!(dest::AbstractVector{T}, st::SimulationState, f::Filter, value::Real) where {T}
    idx = resolve(f, st)
    assign_scalar!(dest, idx, value)
    return idx
end

function assign_scalar!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T}
    return assign_scalar!(dest, st, filter, value)
end

"""
    assign_values!(dest, st[, filter], values)

Assign distinct values per particle according to `values`. Useful when setting
custom noise scales per lattice site.
"""
function assign_values!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, values::CuArray{T,1}) where {T}
    N = length(idx)
    @assert length(values) == N "values length $(length(values)) must match index length $(N)"
    N == 0 && return dest
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _assign_vector_kernel!(dest, idx, values)
    CUDA.@sync k(dest, idx, values; threads, blocks)
    return dest
end

function assign_values!(dest::CuArray{T,1}, sel::Selection, values::AbstractVector{<:Real}) where {T}
    vals_gpu = CuArray(T.(values))
    return assign_values!(dest, sel.device, vals_gpu)
end

function assign_values!(dest::CuArray{T,1}, idx::CuArray{Int32,1}, values::AbstractVector{<:Real}) where {T}
    vals_gpu = CuArray(T.(values))
    return assign_values!(dest, idx, vals_gpu)
end

function assign_values!(dest::CuArray{T,1}, st::SimulationState, f::Filter, values::AbstractVector{<:Real}) where {T}
    idx = resolve_gpu(f, st)
    assign_values!(dest, idx, values)
    return idx
end

function assign_values!(dest::CuArray{T,1}, st::SimulationState; filter::Filter=All(), values::AbstractVector{<:Real}) where {T}
    return assign_values!(dest, st, filter, values)
end

function assign_values!(dest::AbstractVector{T}, idx::Vector{Int}, values::AbstractVector{<:Real}) where {T}
    @assert length(values) == length(idx)
    @inbounds for (k, j) in enumerate(idx)
        dest[j] = T(values[k])
    end
    return dest
end

function assign_values!(dest::AbstractVector{T}, st::SimulationState, f::Filter, values::AbstractVector{<:Real}) where {T}
    idx = resolve(f, st)
    assign_values!(dest, idx, values)
    return idx
end

function assign_values!(dest::AbstractVector{T}, st::SimulationState; filter::Filter=All(), values::AbstractVector{<:Real}) where {T}
    return assign_values!(dest, st, filter, values)
end

# -----------------------------------------------------------------------------
# Gather / sum
# -----------------------------------------------------------------------------

"""
    gather(src, st[, filter]) -> Vector

Collect the values referenced by `filter` onto the host. Used in the tests to
verify that per-type energies match expectations.
"""
function gather(src::CuArray{T,1}, idx::CuArray{Int32,1}) where {T}
    N = length(idx)
    N == 0 && return T[]
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, idx)
    CUDA.@sync k(tmp, src, idx; threads, blocks)
    return Array(tmp)
end

function gather(src::CuArray{T,1}, sel::Selection) where {T}
    return gather(src, sel.device)
end

function gather(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T}
    idx = resolve_gpu(f, st)
    return gather(src, idx)
end

function gather(src::AbstractVector{T}, idx::Vector{Int}) where {T}
    return T.(src[idx])
end

function gather(src::AbstractVector{T}, sel::Selection) where {T}
    return gather(src, sel.host)
end

function gather(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T}
    idx = resolve(f, st)
    return gather(src, idx)
end

"""
    sum(src, st[, filter])

Sum the selected entries of `src`, returning a scalar on the host. Mirrors the
heat/energy aggregation in `examples/TwoT_2D_LD_VV.jl`.
"""
function sum(src::CuArray{T,1}, idx::CuArray{Int32,1}) where {T<:Real}
    N = length(idx)
    N == 0 && return zero(T)
    tmp = CuArray{T}(undef, N)
    threads = min(256, N)
    blocks = cld(N, threads)
    k = CUDA.@cuda launch=false _gather_kernel!(tmp, src, idx)
    CUDA.@sync k(tmp, src, idx; threads, blocks)
    return Base.sum(tmp)
end

function sum(src::CuArray{T,1}, sel::Selection) where {T<:Real}
    return sum(src, sel.device)
end

function sum(src::CuArray{T,1}, st::SimulationState, f::Filter) where {T<:Real}
    idx = resolve_gpu(f, st)
    return sum(src, idx)
end

function sum(src::AbstractVector{T}, idx::Vector{Int}) where {T<:Real}
    acc = zero(T)
    @inbounds for j in idx
        acc += src[j]
    end
    return acc
end

function sum(src::AbstractVector{T}, sel::Selection) where {T<:Real}
    return sum(src, sel.host)
end

function sum(src::AbstractVector{T}, st::SimulationState, f::Filter) where {T<:Real}
    idx = resolve(f, st)
    return sum(src, idx)
end

# -----------------------------------------------------------------------------
# Convenience APIs
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

"""
    set_noise_scale!(spec, value)
    set_noise_scale!(spec, st, value; filter=All())
    set_noise_scale!(st, spec, value; filter=All())

Integrator-parameter control for the stochastic noise amplitude.
"""
function set_noise_scale!(spec::IntegratorSpec, value::Real)
    fill!(_noise_scale_view(spec), eltype(_noise_scale_view(spec))(value))
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, value::Real; filter::Filter=All())
    idx = resolve_gpu(filter, st)
    assign_scalar!(_noise_scale_view(spec), idx, value)
    _rebuild_single_mode_ou!(spec)
    return idx
end

function set_noise_scale!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_noise_scale!(spec, st, value; filter=filter)
end

set_noise_scale!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_noise_scale!(spec, st, mapping)
set_noise_scale!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_noise_scale!(spec, st, pairs...)

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_noise_scale!(spec, st, val; filter=f)
    end
    return spec
end

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_noise_scale!(spec, st, val; filter=f)
    end
    return spec
end

"""
    set_friction!(spec, γ)
    set_friction!(spec, st, γ; filter=All())
    set_friction!(st, spec, γ; filter=All())

Integrator-parameter control for per-particle friction coefficients.
"""
function set_friction!(spec::IntegratorSpec, value::Real)
    fill!(_gamma_view(spec), eltype(_gamma_view(spec))(value))
    return spec
end

function set_friction!(spec::IntegratorSpec, st::SimulationState, value::Real; filter::Filter=All())
    idx = resolve_gpu(filter, st)
    assign_scalar!(_gamma_view(spec), idx, value)
    return idx
end

function set_friction!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_friction!(spec, st, value; filter=filter)
end

set_friction!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_friction!(spec, st, mapping)
set_friction!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_friction!(spec, st, pairs...)

function set_friction!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_friction!(spec, st, val; filter=f)
    end
    return spec
end

function set_friction!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_friction!(spec, st, val; filter=f)
    end
    return spec
end

"""
    set_temperature!(spec, dt, T)
    set_temperature!(spec, st, dt, T; filter=All())
    set_temperature!(st, spec, dt, T; filter=All())

Set the effective thermostat temperature through the integrator-owned
`noise_scale` and `gamma` buffers.
"""
function set_temperature!(spec::IntegratorSpec{T}, dt::Real, temperature::Real) where {T<:AbstractFloat}
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    noise = _noise_scale_view(spec)
    gamma = _gamma_view(spec)
    @. noise = sqrt(T(2) * gamma * Tval * Δt)
    _set_dt_view!(spec, Δt)
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_temperature!(spec::IntegratorSpec{T}, st::SimulationState, dt::Real, temperature::Real; filter::Filter=All()) where {T<:AbstractFloat}
    idx = resolve_gpu(filter, st)
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(_noise_scale_view(spec), _gamma_view(spec), idx, Δt, Tval)
    _set_dt_view!(spec, Δt)
    _rebuild_single_mode_ou!(spec)
    return idx
end

function set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, temperature::Real; filter::Filter=All())
    return set_temperature!(spec, st, dt, temperature; filter=filter)
end

set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, mapping::AbstractDict{<:Filter,<:Real}) =
    set_temperature!(spec, st, dt, mapping)
set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, pairs::Pair{<:Filter,<:Real}...) =
    set_temperature!(spec, st, dt, pairs...)

function set_temperature!(spec::IntegratorSpec, st::SimulationState, dt::Real, mapping::AbstractDict{<:Filter,<:Real})
    for (f, temp) in mapping
        set_temperature!(spec, st, dt, temp; filter=f)
    end
    return spec
end

function set_temperature!(spec::IntegratorSpec, st::SimulationState, dt::Real, pairs::Pair{<:Filter,<:Real}...)
    for (f, temp) in pairs
        set_temperature!(spec, st, dt, temp; filter=f)
    end
    return spec
end

"""
    set_corr_time!(spec, τ)
    set_corr_time!(spec, st, τ; filter=All())
    set_corr_time!(st, spec, τ; filter=All())

Set per-particle correlation times for OU noise processes on the integrator
spec.
"""
function set_corr_time!(spec::IntegratorSpec{T}, value::Real) where {T<:AbstractFloat}
    corr = _ensure_corr_time_array(spec)
    fill!(corr, T(value))
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_corr_time!(spec::IntegratorSpec{T}, st::SimulationState, value::Real; filter::Filter=All()) where {T<:AbstractFloat}
    corr = _ensure_corr_time_array(spec)
    assign_scalar!(corr, st; filter=filter, value=value)
    _rebuild_single_mode_ou!(spec)
    return corr
end

function set_corr_time!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_corr_time!(spec, st, value; filter=filter)
end

set_corr_time!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_corr_time!(spec, st, mapping)
set_corr_time!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_corr_time!(spec, st, pairs...)

function set_corr_time!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_corr_time!(spec, st, val; filter=f)
    end
    return spec
end

function set_corr_time!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_corr_time!(spec, st, val; filter=f)
    end
    return spec
end

@inline function _compat_corr_time_for_selection(::Type{T},
                                                 N::Integer,
                                                 idx::CuArray{Int32,1},
                                                 taus::AbstractVector{T},
                                                 scales::AbstractVector{T}) where {T<:AbstractFloat}
    length(taus) == 1 && length(scales) == 1 || return nothing
    corr = CUDA.fill(zero(T), N)
    assign_scalar!(corr, idx, taus[1])
    return corr
end

"""
    set_ou_spectrum!(spec, st, taus, scales; filter=All(), dt=spec.params.dt)
    set_ou_spectrum!(st, spec, taus, scales; filter=All(), dt=spec.params.dt)

Configure a generalized OU spectrum on the selected particles. Scalars are
treated as one-mode spectra, so the new implementation reduces exactly to the
legacy single-exponential process when `taus` and `scales` are scalars.
"""
function set_ou_spectrum!(spec::IntegratorSpec{T},
                          st::SimulationState,
                          taus::Union{AbstractVector{<:Real},Real},
                          scales::Union{AbstractVector{<:Real},Real};
                          filter::Filter=All(),
                          dt::Real=_dt_view(spec)) where {T<:AbstractFloat}
    sel = selection(st, filter)
    dtT = convert(T, dt)
    tau_vals, scale_vals = Simulation._canonical_mode_vectors(T, taus, scales)
    if length(tau_vals) == 1 && length(scale_vals) == 1
        assign_scalar!(_noise_scale_view(spec), sel.device, scale_vals[1])
    end
    ou = Simulation._build_mode_ou(T, sel.device, tau_vals, scale_vals, dtT)
    corr = _compat_corr_time_for_selection(T, length(_gamma_view(spec)), sel.device, tau_vals, scale_vals)
    _set_dt_view!(spec, dtT)
    _set_corr_time_view!(spec, corr)
    _set_ou_view!(spec, ou)
    return spec
end

function set_ou_spectrum!(st::SimulationState,
                          spec::IntegratorSpec,
                          taus::Union{AbstractVector{<:Real},Real},
                          scales::Union{AbstractVector{<:Real},Real};
                          filter::Filter=All(),
                          dt::Real=_dt_view(spec))
    return set_ou_spectrum!(spec, st, taus, scales; filter=filter, dt=dt)
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, sel::Selection, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.noise_scale, sel.device, value)
    return sel
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, idx::CuArray{Int32,1}, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.noise_scale, idx, value)
    return idx
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, filter::Filter, value::Real) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_noise_scale!(bp, sel, value)
    return sel
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T<:AbstractFloat}
    return set_noise_scale!(bp, st, filter, value)
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, val) in mapping
        set_noise_scale!(bp, st, f, val)
    end
    return bp
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, val) in pairs
        set_noise_scale!(bp, st, f, val)
    end
    return bp
end

function set_corr_time!(bp::BrownianIntegrators.BrownianParams{T}, value::Real) where {T<:AbstractFloat}
    bp2 = _ensure_corr_time_array(bp)
    fill!(bp2.corr_time, T(value))
    ou = Simulation._build_single_mode_ou(T, bp2.noise_scale, bp2.corr_time, bp2.dt)
    return BrownianIntegrators.BrownianParams{T}(bp2.gamma, bp2.dt, bp2.noise_scale, bp2.corr_time, ou)
end

function set_corr_time!(em::BrownianIntegrators.EMParams{T}, value::Real) where {T<:AbstractFloat}
    em2 = _ensure_corr_time_array(em)
    fill!(em2.corr_time, T(value))
    ou = Simulation._build_single_mode_ou(T, em2.noise_scale, em2.corr_time, em2.dt)
    return BrownianIntegrators.EMParams{T}(em2.gamma, em2.dt, em2.noise_scale, em2.corr_time, ou)
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, sel::Selection, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.gamma, sel.device, value)
    return sel
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, idx::CuArray{Int32,1}, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.gamma, idx, value)
    return idx
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, filter::Filter, value::Real) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_friction!(bp, sel, value)
    return sel
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T<:AbstractFloat}
    return set_friction!(bp, st, filter, value)
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, val) in mapping
        set_friction!(bp, st, f, val)
    end
    return bp
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, val) in pairs
        set_friction!(bp, st, f, val)
    end
    return bp
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real; filter::Filter=All()) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_temperature!(bp, st, dt, temperature, sel)
    return sel
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real, sel::Selection) where {T<:AbstractFloat}
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(bp.noise_scale, bp.gamma, sel.device, Δt, Tval)
    return sel
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real, idx::CuArray{Int32,1}) where {T<:AbstractFloat}
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(bp.noise_scale, bp.gamma, idx, Δt, Tval)
    return idx
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, temp) in mapping
        set_temperature!(bp, st, dt, temp; filter=f)
    end
    return bp
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, temp) in pairs
        set_temperature!(bp, st, dt, temp; filter=f)
    end
    return bp
end

"""
    freeze_particles!(st; filter=All(), mode=:hold, steps=nothing, k=0, include_energy=true)

Freeze the selected particles for a fixed number of steps.
- `mode=:hold` clamps positions while allowing velocities to update (Langevin).
- `mode=:spring` adds a harmonic tether with stiffness `k` to the current positions.
- `steps`: number of steps to keep the freeze active; `nothing` keeps it on until
  [`unfreeze_particles!`](@ref) is called.
- `include_energy`: add tether energy to `Epot` when `mode=:spring`.
"""
function freeze_particles!(st::SimulationState{T};
                           filter::Filter=All(),
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    sel = selection(st, filter)
    freeze_particles!(st, sel; mode, steps, k, include_energy)
    return sel
end

function freeze_particles!(st::SimulationState{T}, sel::Selection;
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    freeze_particles!(st, sel.device; mode, steps, k, include_energy)
    return sel
end

function freeze_particles!(st::SimulationState{T}, idx::CuArray{Int32,1};
                           mode::Symbol=:hold,
                           steps::Union{Nothing,Integer}=nothing,
                           k::Real=0,
                           include_energy::Bool=true) where {T<:AbstractFloat}
    if steps !== nothing
        steps < 0 && throw(ArgumentError("steps must be >= 0"))
    end
    if mode === :hold
        st.freeze_mode = Simulation.FREEZE_HOLD
        st.freeze_k = zero(T)
    elseif mode === :spring
        k <= 0 && throw(ArgumentError("spring k must be > 0"))
        st.freeze_mode = Simulation.FREEZE_SPRING
        st.freeze_k = T(k)
    else
        throw(ArgumentError("mode must be :hold or :spring"))
    end
    st.freeze_include_energy = include_energy
    st.freeze_until = steps === nothing ? -1 : st.step + Int(steps)

    if length(idx) == 0
        st.freeze_mode = Simulation.FREEZE_NONE
        st.freeze_until = -1
        return idx
    end

    if st.freeze_mask === nothing || length(st.freeze_mask) != length(st.rx)
        st.freeze_mask = CUDA.fill(UInt8(0), length(st.rx))
    else
        fill!(st.freeze_mask, UInt8(0))
    end
    assign_scalar!(st.freeze_mask, idx, UInt8(1))

    if st.freeze_rx === nothing || length(st.freeze_rx) != length(st.rx)
        st.freeze_rx = similar(st.rx)
        st.freeze_ry = similar(st.ry)
        st.freeze_rz = st.rz === nothing ? nothing : similar(st.rz)
    elseif st.rz !== nothing && st.freeze_rz === nothing
        st.freeze_rz = similar(st.rz)
    end

    copyto!(st.freeze_rx, st.rx)
    copyto!(st.freeze_ry, st.ry)
    if st.rz !== nothing
        copyto!(st.freeze_rz, st.rz)
    end
    return idx
end

"""
    unfreeze_particles!(st)

Disable any active freeze/tethering.
"""
function unfreeze_particles!(st::SimulationState)
    st.freeze_mode = Simulation.FREEZE_NONE
    st.freeze_until = -1
    return st
end

end # module Filters
