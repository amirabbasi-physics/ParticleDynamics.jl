using StaticArrays: SVector, StaticVector
using ..Writers: read_gsd_frame!, GSDFrameData, GSDTopology

"""
    PeriodicBox(lengths)

High-level periodic box descriptor for a workflow [`ParticleSystem`](@ref).
Supports 2D and 3D boxes.
"""
struct PeriodicBox{T,N}
    lengths::NTuple{N,T}
end

function PeriodicBox(lengths::NTuple{N,T}) where {N,T<:AbstractFloat}
    N in (2, 3) || throw(ArgumentError("PeriodicBox only supports 2D or 3D boxes; got N=$(N)."))
    all(>(zero(T)), lengths) || throw(ArgumentError("PeriodicBox lengths must be positive."))
    return PeriodicBox{T,N}(lengths)
end

function PeriodicBox(lengths::NTuple{N,<:Real}) where {N}
    T = float(promote_type(map(typeof, lengths)...))
    return PeriodicBox(ntuple(i -> T(lengths[i]), N))
end

PeriodicBox(lengths::AbstractVector{<:Real}) = PeriodicBox(Tuple(lengths))

Base.length(box::PeriodicBox) = length(box.lengths)
Base.getindex(box::PeriodicBox, idx::Int) = box.lengths[idx]
Base.Tuple(box::PeriodicBox) = box.lengths

function _metadata_dict(metadata)
    dict = Dict{Symbol,Any}()
    metadata === nothing && return dict
    for (k, v) in pairs(metadata)
        dict[Symbol(k)] = v
    end
    return dict
end

function _normalize_pair_tuples(values, label::AbstractString)
    tuples = Tuple{Int32,Int32}[]
    sizehint!(tuples, length(values))
    for item in values
        length(item) == 2 || throw(ArgumentError("Each $(label) entry must have length 2."))
        i = Int32(item[1])
        j = Int32(item[2])
        i > 0 && j > 0 || throw(ArgumentError("$(label) indices must be positive."))
        push!(tuples, (i, j))
    end
    return tuples
end

function _normalize_ntuples(values, ::Val{C}, label::AbstractString) where {C}
    tuples = NTuple{C,Int32}[]
    sizehint!(tuples, length(values))
    for item in values
        length(item) == C || throw(ArgumentError("Each $(label) entry must have length $(C)."))
        tupleC = ntuple(i -> Int32(item[i]), C)
        all(>(Int32(0)), tupleC) || throw(ArgumentError("$(label) indices must be positive."))
        push!(tuples, tupleC)
    end
    return tuples
end

function _normalize_molecules(values)
    molecules = Vector{Int32}[]
    sizehint!(molecules, length(values))
    for mol in values
        host = Int32.(collect(mol))
        all(>(Int32(0)), host) || throw(ArgumentError("Molecule indices must be positive."))
        push!(molecules, host)
    end
    return molecules
end

"""
    Topology(; bonds=[], bond_types=[], angles=[], dihedrals=[], impropers=[], exclusions=[], molecules=[], metadata=Dict())

Future-ready topology container used by the workflow API. Current compiled
support covers bond connectivity and bond forces; higher-order terms are stored
as metadata for future force-field work.
"""
struct Topology
    bonds::Vector{Tuple{Int32,Int32}}
    bond_types::Vector{Symbol}
    angles::Vector{NTuple{3,Int32}}
    dihedrals::Vector{NTuple{4,Int32}}
    impropers::Vector{NTuple{4,Int32}}
    exclusions::Vector{Tuple{Int32,Int32}}
    molecules::Vector{Vector{Int32}}
    metadata::Dict{Symbol,Any}
end

function Topology(; bonds=Tuple{Int32,Int32}[],
                    bond_types=Symbol[],
                    angles=NTuple{3,Int32}[],
                    dihedrals=NTuple{4,Int32}[],
                    impropers=NTuple{4,Int32}[],
                    exclusions=Tuple{Int32,Int32}[],
                    molecules=Vector{Int32}[],
                    metadata=Dict{Symbol,Any}())
    bonds32 = _normalize_pair_tuples(bonds, "bond")
    bond_types_sym = Symbol.(collect(bond_types))
    isempty(bond_types_sym) || length(bond_types_sym) == length(bonds32) ||
        throw(ArgumentError("bond_types must be empty or match the number of bonds."))
    return Topology(
        bonds32,
        bond_types_sym,
        _normalize_ntuples(angles, Val(3), "angle"),
        _normalize_ntuples(dihedrals, Val(4), "dihedral"),
        _normalize_ntuples(impropers, Val(4), "improper"),
        _normalize_pair_tuples(exclusions, "exclusion"),
        _normalize_molecules(molecules),
        _metadata_dict(metadata),
    )
end

function _default_box(dim::Integer, ::Type{T}=Float64) where {T<:AbstractFloat}
    dim in (2, 3) || throw(ArgumentError("ParticleSystem only supports 2D or 3D positions; got dim=$(dim)."))
    return PeriodicBox(ntuple(_ -> one(T), dim))
end

_coerce_box(box::PeriodicBox) = box
_coerce_box(box::Tuple) = PeriodicBox(box)
_coerce_box(box::AbstractVector{<:Real}) = PeriodicBox(box)

function _vector_coordinate_type(data)
    found = false
    T = Float64
    for point in data
        for value in point
            Tv = typeof(float(value))
            T = found ? promote_type(T, Tv) : Tv
            found = true
        end
    end
    found || throw(ArgumentError("ParticleSystem requires at least one position to infer the spatial dimension."))
    return T
end

function _normalize_position_data(data)
    if data isa AbstractMatrix
        rows, cols = size(data)
        if cols in (2, 3)
            D = cols
            N = rows
            T = typeof(float(zero(eltype(data))))
            positions = Vector{SVector{D,T}}(undef, N)
            for i in 1:N
                positions[i] = SVector{D,T}(ntuple(j -> T(data[i, j]), D))
            end
            return positions, T, D
        elseif rows in (2, 3)
            D = rows
            N = cols
            T = typeof(float(zero(eltype(data))))
            positions = Vector{SVector{D,T}}(undef, N)
            for i in 1:N
                positions[i] = SVector{D,T}(ntuple(j -> T(data[j, i]), D))
            end
            return positions, T, D
        else
            throw(ArgumentError("Position matrices must be N×2, N×3, 2×N, or 3×N."))
        end
    elseif data isa AbstractVector
        isempty(data) && throw(ArgumentError("ParticleSystem requires at least one position to infer the spatial dimension."))
        first_point = first(data)
        D = length(first_point)
        D in (2, 3) || throw(ArgumentError("ParticleSystem only supports 2D or 3D positions; got dim=$(D)."))
        T = _vector_coordinate_type(data)
        positions = Vector{SVector{D,T}}(undef, length(data))
        for (idx, point) in pairs(data)
            length(point) == D || throw(ArgumentError("All positions must have the same dimension."))
            positions[idx] = SVector{D,T}(ntuple(i -> T(point[i]), D))
        end
        return positions, T, D
    else
        throw(ArgumentError("Unsupported position container $(typeof(data)). Use a vector of tuples/SVectors or an N×D matrix."))
    end
end

function _normalize_velocity_data(data, ::Type{T}, D::Int, N::Int) where {T<:AbstractFloat}
    if data isa AbstractMatrix
        rows, cols = size(data)
        if rows == N && cols == D
            return [SVector{D,T}(ntuple(j -> T(data[i, j]), D)) for i in 1:N]
        elseif rows == D && cols == N
            return [SVector{D,T}(ntuple(j -> T(data[j, i]), D)) for i in 1:N]
        else
            throw(ArgumentError("Velocity matrices must match the particle count and dimension ($(N)×$(D) or $(D)×$(N))."))
        end
    elseif data isa AbstractVector
        length(data) == N || throw(ArgumentError("Velocity data must have one entry per particle."))
        velocities = Vector{SVector{D,T}}(undef, N)
        for (idx, point) in pairs(data)
            length(point) == D || throw(ArgumentError("Velocity entries must have dimension $(D)."))
            velocities[idx] = SVector{D,T}(ntuple(i -> T(point[i]), D))
        end
        return velocities
    else
        throw(ArgumentError("Unsupported velocity container $(typeof(data)). Use a vector of tuples/SVectors or an N×D matrix."))
    end
end

function _normalize_types_and_ids(typeids, types, N::Int)
    type_symbols = types === nothing ? Symbol[] : Symbol.(collect(types))
    if typeids === nothing
        if isempty(type_symbols)
            return fill(Int32(1), N), Symbol[:A]
        elseif length(type_symbols) == 1
            return fill(Int32(1), N), type_symbols
        else
            throw(ArgumentError("ParticleSystem requires `typeids` when more than one particle type is declared."))
        end
    end

    ids = Int32.(collect(typeids))
    length(ids) == N || throw(ArgumentError("typeids must have length $(N)."))
    all(>(Int32(0)), ids) || throw(ArgumentError("typeids must be positive 1-based integers."))

    if isempty(type_symbols)
        ntypes = Int(maximum(ids))
        type_symbols = [Symbol("T$(i)") for i in 1:ntypes]
    end

    maximum(ids) <= length(type_symbols) ||
        throw(ArgumentError("typeids reference $(maximum(ids)) types, but only $(length(type_symbols)) names were provided."))

    return ids, type_symbols
end

function _normalize_masses(masses, types::Vector{Symbol}, typeids::Vector{Int32}, ::Type{T}, N::Int) where {T<:AbstractFloat}
    masses === nothing && return nothing
    if masses isa AbstractDict
        dict = Dict{Symbol,T}()
        for (k, v) in pairs(masses)
            key = if k isa Symbol
                k
            elseif k isa AbstractString
                Symbol(k)
            elseif k isa Integer
                idx = Int(k)
                1 <= idx <= length(types) || throw(ArgumentError("Mass key $(k) is out of bounds for $(length(types)) types."))
                types[idx]
            else
                throw(ArgumentError("Mass dictionaries must be keyed by Symbol, String, or Integer type ids."))
            end
            dict[key] = T(v)
        end
        used = unique(types[Int(id)] for id in typeids)
        missing = filter(sym -> !haskey(dict, sym), used)
        isempty(missing) || throw(ArgumentError("Mass dictionary is missing entries for used particle types: $(join(string.(missing), ", "))."))
        return dict
    elseif masses isa AbstractVector
        host = T.(collect(masses))
        length(host) == N || throw(ArgumentError("Mass vectors must have length $(N)."))
        return host
    elseif masses isa Real
        return T(masses)
    else
        throw(ArgumentError("Unsupported masses container $(typeof(masses))."))
    end
end

function _section_types(section)
    if !haskey(section, :typeid) || isempty(section.typeid) || isempty(section.types)
        return Symbol[]
    end
    type_names = String.(section.types)
    mapped = Vector{Symbol}(undef, length(section.typeid))
    for i in eachindex(section.typeid)
        tid = Int(section.typeid[i])
        1 <= tid <= length(type_names) || throw(ArgumentError("Topology type id $(tid) is out of bounds for $(length(type_names)) type names."))
        mapped[i] = Symbol(type_names[tid])
    end
    return mapped
end

function _topology_from_gsd(topology::GSDTopology)
    metadata = Dict{Symbol,Any}()
    angle_types = _section_types(topology.angles)
    isempty(angle_types) || (metadata[:angle_types] = angle_types)
    dihedral_types = _section_types(topology.dihedrals)
    isempty(dihedral_types) || (metadata[:dihedral_types] = dihedral_types)
    improper_types = _section_types(topology.impropers)
    isempty(improper_types) || (metadata[:improper_types] = improper_types)
    topology.constraints.N == 0 || (metadata[:constraints] = topology.constraints)
    topology.special_pairs.N == 0 || (metadata[:special_pairs] = topology.special_pairs)

    return Topology(
        bonds = topology.bonds.tuples,
        bond_types = _section_types(topology.bonds),
        angles = topology.angles.tuples,
        dihedrals = topology.dihedrals.tuples,
        impropers = topology.impropers.tuples,
        metadata = metadata,
    )
end

"""
    ParticleSystem(data; box, types, typeids, masses, velocities, topology=Topology(), metadata=Dict())

High-level particle/topology container used to build a workflow
[`Simulation`](@ref). `data` may be positions directly, an initializer result
with `.positions`/`.box`, or a GSD frame.
"""
struct ParticleSystem
    positions
    box::PeriodicBox
    types::Vector{Symbol}
    typeids::Vector{Int32}
    masses
    velocities
    topology::Topology
    metadata::Dict{Symbol,Any}
end

Base.length(system::ParticleSystem) = length(system.positions)

function _particle_system_from_positions(positions;
                                         box,
                                         types=nothing,
                                         typeids=nothing,
                                         masses=nothing,
                                         velocities=nothing,
                                         topology::Topology=Topology(),
                                         metadata=Dict{Symbol,Any}())
    positions_norm, T, D = _normalize_position_data(positions)
    box_norm = _coerce_box(box)
    length(box_norm) == D ||
        throw(ArgumentError("Position dimension $(D) does not match box dimension $(length(box_norm))."))
    ids_norm, types_norm = _normalize_types_and_ids(typeids, types, length(positions_norm))
    velocities_norm = velocities === nothing ? nothing : _normalize_velocity_data(velocities, T, D, length(positions_norm))
    masses_norm = _normalize_masses(masses, types_norm, ids_norm, T, length(positions_norm))
    return ParticleSystem(
        positions_norm,
        box_norm,
        types_norm,
        ids_norm,
        masses_norm,
        velocities_norm,
        topology,
        _metadata_dict(metadata),
    )
end

function ParticleSystem(data;
                        box=nothing,
                        types=nothing,
                        typeids=nothing,
                        masses=nothing,
                        velocities=nothing,
                        topology::Topology=Topology(),
                        metadata=Dict{Symbol,Any}())
    if data isa GSDFrameData
        return ParticleSystem_from_gsd(data;
                                       box=box,
                                       types=types,
                                       typeids=typeids,
                                       masses=masses,
                                       velocities=velocities,
                                       topology=topology,
                                       metadata=metadata)
    end

    if box === nothing
        hasproperty(data, :positions) || throw(ArgumentError("ParticleSystem(data; ...) requires positions or cfg.positions."))
        hasproperty(data, :box) || throw(ArgumentError("ParticleSystem(data; ...) requires box or cfg.box."))
        positions = getproperty(data, :positions)
        box_data = getproperty(data, :box)
        auto_metadata = Dict{Symbol,Any}()
        for name in propertynames(data)
            name in (:positions, :box) && continue
            auto_metadata[Symbol(name)] = getproperty(data, name)
        end
    else
        positions = data
        box_data = box
        auto_metadata = Dict{Symbol,Any}()
    end

    return _particle_system_from_positions(positions;
                                           box=box_data,
                                           types=types,
                                           typeids=typeids,
                                           masses=masses,
                                           velocities=velocities,
                                           topology=topology,
                                           metadata=merge(auto_metadata, _metadata_dict(metadata)))
end

function ParticleSystem_from_gsd(frame::GSDFrameData;
                                 box=nothing,
                                 types=nothing,
                                 typeids=nothing,
                                 masses=nothing,
                                 velocities=nothing,
                                 topology::Topology=_topology_from_gsd(frame.topology),
                                 metadata=Dict{Symbol,Any}())
    D = frame.D
    T = eltype(frame.rx)
    positions = if D == 2
        [SVector{2,T}(frame.rx[i], frame.ry[i]) for i in eachindex(frame.rx)]
    else
        @assert frame.rz !== nothing
        [SVector{3,T}(frame.rx[i], frame.ry[i], frame.rz[i]) for i in eachindex(frame.rx)]
    end
    velocities_data = if velocities === nothing
        if D == 2
            [SVector{2,T}(frame.vx[i], frame.vy[i]) for i in eachindex(frame.vx)]
        else
            @assert frame.vz !== nothing
            [SVector{3,T}(frame.vx[i], frame.vy[i], frame.vz[i]) for i in eachindex(frame.vx)]
        end
    else
        velocities
    end
    masses_data = if masses === nothing && haskey(frame.particle_properties, :mass)
        frame.particle_properties[:mass]
    else
        masses
    end
    types_data = types === nothing ? Symbol.(frame.types) : types
    typeids_data = typeids === nothing ? frame.typeid : typeids
    auto_metadata = Dict{Symbol,Any}(
        :step => frame.step,
        :particle_properties => frame.particle_properties,
        :per_type_properties => frame.per_type_properties,
        :configuration => frame.configuration,
    )
    if haskey(frame.particle_properties, :diameter)
        auto_metadata[:diameters] = frame.particle_properties[:diameter]
    end
    if haskey(frame.per_type_properties, :diameter)
        auto_metadata[:diameter] = frame.per_type_properties[:diameter]
    end
    return _particle_system_from_positions(positions;
                                           box=box === nothing ? frame.box : box,
                                           types=types_data,
                                           typeids=typeids_data,
                                           masses=masses_data,
                                           velocities=velocities_data,
                                           topology=topology,
                                           metadata=merge(auto_metadata, _metadata_dict(metadata)))
end

"""
    ParticleSystem.from_gsd(path; frame=-1)

Load a workflow [`ParticleSystem`](@ref) from a GSD file. By default the last
frame is used.
"""
function from_gsd(::Type{ParticleSystem}, path::AbstractString; frame::Integer=-1)
    frame_data = frame == -1 ? read_gsd_frame!(path) : read_gsd_frame!(path; step=frame)
    return ParticleSystem_from_gsd(frame_data; metadata=Dict(:source_path => path))
end

struct ParticleSystemFromGSDAccessor end

(::ParticleSystemFromGSDAccessor)(path::AbstractString; frame::Integer=-1) =
    from_gsd(ParticleSystem, path; frame=frame)

const _PARTICLE_SYSTEM_FROM_GSD = ParticleSystemFromGSDAccessor()

function Base.getproperty(::Type{ParticleSystem}, name::Symbol)
    name === :from_gsd && return _PARTICLE_SYSTEM_FROM_GSD
    return getfield(ParticleSystem, name)
end

"""
    Particles

Alias for [`ParticleSystem`](@ref).
"""
const Particles = ParticleSystem
