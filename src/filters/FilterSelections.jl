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

Selection(group::ParticleGroups.ParticleGroup) = Selection(group.host, group.device)

count(sel::Selection) = length(sel.host)

_to_particle_selection(::All) = ParticleGroups.All()
_to_particle_selection(f::TypeIDs) = ParticleGroups.TypeIDs(f.ids)
_to_particle_selection(f::Indices) = ParticleGroups.Indices(f.idx)

_to_particle_group(sel::Selection) = ParticleGroups.ParticleGroup(sel.host, sel.device)

# -----------------------------------------------------------------------------
# Index resolution helpers
# -----------------------------------------------------------------------------

function resolve(::All, st::SimulationState)
    return ParticleGroups.resolve(_to_particle_selection(All()), st)
end

function resolve(f::Indices, st::SimulationState)
    return ParticleGroups.resolve(_to_particle_selection(f), st)
end

function resolve(f::TypeIDs, st::SimulationState)
    return ParticleGroups.resolve(_to_particle_selection(f), st)
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
    return ParticleGroups.resolve_gpu(_to_particle_selection(f), st)
end

function resolve_gpu(f::Indices, st::SimulationState)
    return ParticleGroups.resolve_gpu(_to_particle_selection(f), st)
end

resolve_gpu(f::TypeIDs, st::SimulationState) = ParticleGroups.resolve_gpu(_to_particle_selection(f), st)

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
    return Selection(ParticleGroups.materialize(_to_particle_selection(f), st))
end

function selection(st::SimulationState, f::TypeIDs)
    return Selection(ParticleGroups.materialize(_to_particle_selection(f), st))
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
count(f::Filter, st::SimulationState) = ParticleGroups.count(_to_particle_selection(f), st)
count(f::TypeIDs, st::SimulationState) = ParticleGroups.count(_to_particle_selection(f), st)
count(st::SimulationState, f::Filter) = count(f, st)
