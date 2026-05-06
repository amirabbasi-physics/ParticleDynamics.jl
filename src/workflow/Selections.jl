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
