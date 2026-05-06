abstract type AbstractSelectionSpec end

struct AllSelection <: AbstractSelectionSpec end

struct TypeSelection{T} <: AbstractSelectionSpec
    value::T
end

TypeSelection(value::Symbol) = TypeSelection{Symbol}(value)
TypeSelection(value::Integer) = TypeSelection{Int}(Int(value))

struct IndexSelection <: AbstractSelectionSpec
    indices::Vector{Int}
    function IndexSelection(indices::AbstractVector{<:Integer})
        host = Int.(indices)
        all(>(0), host) || throw(ArgumentError("IndexSelection indices must be positive."))
        new(host)
    end
end

struct Group{S<:AbstractSelectionSpec}
    name::Symbol
    domain::Symbol
    selection::S
end

Group(name::Symbol, selection::AbstractSelectionSpec; domain::Symbol=:particles) =
    Group{typeof(selection)}(name, domain, selection)

struct Groups
    entries::Vector{Group}
    byname::Dict{Symbol,Int}
    function Groups(entries::AbstractVector{<:Group})
        byname = Dict{Symbol,Int}()
        groups = Group[entries...]
        for (idx, group) in pairs(groups)
            haskey(byname, group.name) &&
                throw(ArgumentError("Duplicate group name $(group.name)."))
            byname[group.name] = idx
        end
        new(groups, byname)
    end
end

Groups(groups::Group...) = Groups(collect(groups))

Base.length(groups::Groups) = length(groups.entries)
Base.iterate(groups::Groups, state::Int=1) =
    state > length(groups.entries) ? nothing : (groups.entries[state], state + 1)
Base.getindex(groups::Groups, name::Symbol) = groups.entries[groups.byname[name]]
Base.getindex(groups::Groups, idx::Int) = groups.entries[idx]
Base.keys(groups::Groups) = keys(groups.byname)

_particle_count(system::ParticleSystem) = length(system.positions)

function _selection_typeid(system::ParticleSystem, value::Symbol)
    idx = findfirst(==(value), system.types)
    idx === nothing &&
        throw(ArgumentError("Unknown particle type $(value). Known types: $(join(string.(system.types), ", "))."))
    return idx
end

_selection_typeid(::ParticleSystem, value::Integer) = Int(value)

function _selection_indices(system::ParticleSystem, ::AllSelection)
    return collect(1:_particle_count(system))
end

function _selection_indices(system::ParticleSystem, selection::TypeSelection)
    system.typeids === nothing &&
        throw(ArgumentError("TypeSelection requires `ParticleSystem.typeids` to be defined."))
    tid = _selection_typeid(system, selection.value)
    return findall(==(tid), Int.(system.typeids))
end

function _selection_indices(system::ParticleSystem, selection::IndexSelection)
    n = _particle_count(system)
    for idx in selection.indices
        1 <= idx <= n ||
            throw(ArgumentError("IndexSelection index $(idx) is out of bounds for $(n) particles."))
    end
    return selection.indices
end

materialize_selection(::ParticleSystem, ::AllSelection) = Filters.All()

function materialize_selection(system::ParticleSystem, selection::TypeSelection)
    return Filters.TypeIDs(_selection_typeid(system, selection.value))
end

materialize_selection(::ParticleSystem, selection::IndexSelection) = Filters.Indices(selection.indices)

function materialize_group(system::ParticleSystem, group::Group)
    group.domain == :particles ||
        throw(ArgumentError("Only particle groups are supported right now; got domain=$(group.domain)."))
    indices = _selection_indices(system, group.selection)
    isempty(indices) && throw(ArgumentError("Group $(group.name) selects no particles."))
    return materialize_selection(system, group.selection)
end
