@kwdef struct PeriodicBox{T,N}
    lengths::NTuple{N,T}
end

PeriodicBox(lengths::NTuple{N,<:Real}) where {N} =
    PeriodicBox{Float64,N}(ntuple(i -> Float64(lengths[i]), N))

PeriodicBox(lengths::AbstractVector{<:Real}) = PeriodicBox(Tuple(lengths))

Base.length(box::PeriodicBox) = length(box.lengths)
Base.getindex(box::PeriodicBox, idx::Int) = box.lengths[idx]
Base.Tuple(box::PeriodicBox) = box.lengths

@kwdef struct Topology
    bonds::Vector{Tuple{Int32,Int32}} = Tuple{Int32,Int32}[]
    bond_types::Vector{Symbol} = Symbol[]
    angles::Vector{NTuple{3,Int32}} = NTuple{3,Int32}[]
    dihedrals::Vector{NTuple{4,Int32}} = NTuple{4,Int32}[]
    impropers::Vector{NTuple{4,Int32}} = NTuple{4,Int32}[]
    exclusions::Vector{Tuple{Int32,Int32}} = Tuple{Int32,Int32}[]
    molecules::Vector{Vector{Int32}} = Vector{Int32}[]
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

_default_box() = PeriodicBox((1.0, 1.0))

_coerce_box(box::PeriodicBox) = box
_coerce_box(box::Tuple) = PeriodicBox(box)
_coerce_box(box::AbstractVector{<:Real}) = PeriodicBox(box)

@kwdef struct ParticleSystem
    positions
    box = _default_box()
    types::Vector{Symbol} = Symbol[:A]
    typeids = nothing
    masses = nothing
    velocities = nothing
    topology::Topology = Topology()
    metadata::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

function ParticleSystem(data;
                        box=nothing,
                        types::AbstractVector{<:Symbol}=Symbol[:A],
                        typeids=nothing,
                        masses=nothing,
                        velocities=nothing,
                        topology::Topology=Topology(),
                        metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())
    if box === nothing
        hasproperty(data, :positions) || throw(ArgumentError("ParticleSystem(data; ...) requires positions or cfg.positions."))
        hasproperty(data, :box) || throw(ArgumentError("ParticleSystem(data; ...) requires box or cfg.box."))
        positions = getproperty(data, :positions)
        box_data = getproperty(data, :box)
    else
        positions = data
        box_data = box
    end

    return ParticleSystem(positions=positions,
                          box=_coerce_box(box_data),
                          types=Symbol[types...],
                          typeids=typeids,
                          masses=masses,
                          velocities=velocities,
                          topology=topology,
                          metadata=metadata)
end

const Particles = ParticleSystem
