using CUDA
using ..Definitions
using ..SimulationCore
using ..NeighborLists

"""
    Force

Abstract workflow force descriptor. Concrete force objects describe physical
interactions and compile onto the existing low-level engine.
"""
abstract type Force end

"""
    ForceField(; forces=Force[])

Future-compatible container for a set of workflow force objects.
"""
@kwdef mutable struct ForceField
    forces::Vector{Force} = Force[]
end

Base.length(ff::ForceField) = length(ff.forces)
Base.iterate(ff::ForceField, state::Int=1) =
    state > length(ff.forces) ? nothing : (ff.forces[state], state + 1)

function add!(ff::ForceField, force::Force)
    push!(ff.forces, force)
    return ff
end

"""
    CellList(; buffer=0.4, capacity=96, rebuild_interval=20)

Neighbor-list policy for workflow nonbonded forces.
"""
@kwdef struct CellList
    buffer::Float64 = 0.4
    capacity::Int = 96
    rebuild_interval::Int = 20
end

"""
    PairTable(; sigma, epsilon, cutoff, type_names)

Per-type parameter table for workflow Lennard-Jones and WCA forces.
"""
struct PairTable
    sigma
    epsilon
    cutoff
    type_names::Vector{Symbol}
end

function PairTable(; sigma, epsilon, cutoff, type_names)
    names = Symbol.(collect(type_names))
    isempty(names) && throw(ArgumentError("PairTable.type_names must not be empty."))
    return PairTable(sigma, epsilon, cutoff, names)
end

"""
    LennardJones(; epsilon, sigma, cutoff, pairs=:all, mode=:standard, pair_table=nothing, neighborlist=nothing)

Workflow Lennard-Jones force descriptor.
"""
@kwdef struct LennardJones <: Force
    epsilon = nothing
    sigma = nothing
    cutoff = nothing
    pairs = :all
    mode = :standard
    pair_table = nothing
    neighborlist = nothing
end

"""
    WCA(; epsilon, sigma, cutoff=nothing, pairs=:all, mode=:standard, pair_table=nothing, neighborlist=nothing)

Workflow Weeks-Chandler-Andersen force descriptor.
"""
@kwdef struct WCA <: Force
    epsilon = nothing
    sigma = nothing
    cutoff = nothing
    pairs = :all
    mode = :standard
    pair_table = nothing
    neighborlist = nothing
end

"""
    SoftRepulsive(; epsilon, sigma, cutoff, params=nothing, pairs=:all, pair_table=nothing, neighborlist=nothing)

Workflow soft-repulsive force descriptor.
"""
@kwdef struct SoftRepulsive <: Force
    epsilon = nothing
    sigma = nothing
    cutoff = nothing
    params = nothing
    pairs = :all
    pair_table = nothing
    neighborlist = nothing
end

"""
    HarmonicBondForce(; k, r0, type=:default)

Workflow harmonic bond force descriptor.
"""
@kwdef struct HarmonicBondForce <: Force
    k
    r0
    type = :default
end

"""
    FENEBondForce(; k, R0, type=:default)

Workflow FENE bond force descriptor.
"""
@kwdef struct FENEBondForce <: Force
    k
    R0
    type = :default
end

struct CompiledForces
    build_kwargs::Dict{Symbol,Any}
    post_build!::Function
    metadata::Dict{Symbol,Any}
end

post_build!(compiled::CompiledForces, st) = compiled.post_build!(st)

_wca_cutoff(sigma) = sigma * SimulationCore.WCA_RC_FACTOR

function _force_list(forces)
    if forces === nothing
        return Force[]
    elseif forces isa ForceField
        return collect(forces.forces)
    elseif forces isa Force
        return Force[forces]
    else
        host = collect(forces)
        all(force -> force isa Force, host) ||
            throw(ArgumentError("forces must be a Force, a ForceField, or a collection of Force objects."))
        return Force[host...]
    end
end

function _precision_type(precision)
    if precision === :f32 || precision === Float32
        return Float32
    elseif precision === :f64 || precision === Float64
        return Float64
    else
        throw(ArgumentError("Unsupported workflow precision $(precision). Use :f32, :f64, Float32, or Float64."))
    end
end

function _pair_table_dimension(x, ntypes::Int, label::AbstractString)
    if x isa AbstractMatrix
        size(x) == (ntypes, ntypes) ||
            throw(ArgumentError("PairTable $(label) matrix must have size ($(ntypes), $(ntypes))."))
        return :matrix
    elseif x isa AbstractVector
        length(x) == ntypes ||
            throw(ArgumentError("PairTable $(label) vector must have length $(ntypes)."))
        return :vector
    elseif x isa Real
        return :scalar
    else
        throw(ArgumentError("Unsupported PairTable $(label) container $(typeof(x))."))
    end
end

function _matrix_from_pair_data(values, ntypes::Int, ::Type{T}, label::AbstractString) where {T<:AbstractFloat}
    shape = _pair_table_dimension(values, ntypes, label)
    if shape === :matrix
        return T.(Matrix(values))
    elseif shape === :vector
        vec = T.(collect(values))
        return T.(0.5) .* (vec .+ permutedims(vec))
    else
        return fill(T(values), ntypes, ntypes)
    end
end

function _pair_table_for_system(table::PairTable, types::Vector{Symbol}, ::Type{T}) where {T<:AbstractFloat}
    nsys = length(types)
    ntable = length(table.type_names)
    nsys == ntable ||
        throw(ArgumentError("PairTable defines $(ntable) types, but the ParticleSystem uses $(nsys)."))

    sigma_src = _matrix_from_pair_data(table.sigma, ntable, T, "sigma")
    epsilon_src = _matrix_from_pair_data(table.epsilon, ntable, T, "epsilon")
    cutoff_src = _matrix_from_pair_data(table.cutoff, ntable, T, "cutoff")

    order = Vector{Int}(undef, nsys)
    for (idx, name) in pairs(types)
        match = findfirst(==(name), table.type_names)
        match === nothing &&
            throw(ArgumentError("PairTable is missing an entry for particle type $(name). Known types: $(join(string.(table.type_names), ", "))."))
        order[idx] = match
    end

    sigma = sigma_src[order, order]
    epsilon = epsilon_src[order, order]
    cutoff = cutoff_src[order, order]
    return sigma, epsilon, cutoff
end

function _effective_pair_cutoff(cutoff_pair::AbstractMatrix{T},
                                epsilon_pair::AbstractMatrix{T}) where {T<:AbstractFloat}
    size(cutoff_pair) == size(epsilon_pair) ||
        throw(ArgumentError("Pair cutoff and epsilon matrices must have the same shape."))
    effective = Matrix{T}(cutoff_pair)
    @inbounds for j in axes(effective, 2), i in axes(effective, 1)
        epsilon_pair[i, j] > zero(T) || (effective[i, j] = zero(T))
    end
    return effective
end

function _neighbor_kwargs(force)
    kwargs = Dict{Symbol,Any}()
    if force.pairs === :all
        kwargs[:use_neighborlist] = false
        return kwargs, false
    elseif force.pairs === :neighborlist
        kwargs[:use_neighborlist] = true
    else
        throw(ArgumentError("Unsupported force traversal mode $(force.pairs). Use :all or :neighborlist."))
    end

    spec = force.neighborlist
    if spec === nothing
        spec = CellList()
    elseif !(spec isa CellList)
        throw(ArgumentError("Only CellList neighbor specifications are supported right now; got $(typeof(spec))."))
    end
    spec.capacity > 0 || throw(ArgumentError("CellList capacity must be positive."))
    spec.buffer >= 0 || throw(ArgumentError("CellList buffer must be nonnegative."))
    spec.rebuild_interval > 0 || throw(ArgumentError("CellList rebuild_interval must be positive."))

    kwargs[:skin] = Float64(spec.buffer)
    kwargs[:cap] = Int32(spec.capacity)
    kwargs[:neigh_interval] = Int(spec.rebuild_interval)
    return kwargs, true
end

function _uniform_lj_kwargs(force, nonbonded::Symbol, ::Type{T}) where {T<:AbstractFloat}
    force.mode == :standard ||
        throw(ArgumentError("Only mode=:standard is supported for $(typeof(force))."))
    force.pair_table === nothing ||
        throw(ArgumentError("Use PairTable through the `pair_table=` keyword only for type-dependent mappings."))
    force.epsilon === nothing && throw(ArgumentError("$(typeof(force)) requires `epsilon` unless `pair_table` is provided."))
    force.sigma === nothing && throw(ArgumentError("$(typeof(force)) requires `sigma` unless `pair_table` is provided."))

    epsilon = T(force.epsilon)
    sigma = T(force.sigma)
    cutoff = if nonbonded === :wca
        force.cutoff === nothing ? _wca_cutoff(sigma) : T(force.cutoff)
    else
        force.cutoff === nothing && throw(ArgumentError("LennardJones requires an explicit `cutoff`."))
        T(force.cutoff)
    end

    kwargs = Dict{Symbol,Any}(
        :nonbonded => nonbonded,
        :epsilon => epsilon,
        :sigma => sigma,
        :cutoff => cutoff,
    )
    merge!(kwargs, _neighbor_kwargs(force)[1])
    return kwargs, (st -> st), Dict{Symbol,Any}(:pair_style => :uniform)
end

function _pair_table_kwargs(system::ParticleSystem, force, nonbonded::Symbol, ::Type{T}) where {T<:AbstractFloat}
    table = force.pair_table
    table isa PairTable || throw(ArgumentError("Type-dependent $(typeof(force)) requires `pair_table=PairTable(...)`."))
    sigma_pair, epsilon_pair, cutoff_pair = _pair_table_for_system(table, system.types, T)
    effective_cutoff_pair = _effective_pair_cutoff(cutoff_pair, epsilon_pair)
    kwargs, use_typed_neighbors = _neighbor_kwargs(force)
    kwargs[:nonbonded] = nonbonded
    kwargs[:epsilon] = maximum(epsilon_pair)
    kwargs[:sigma] = maximum(sigma_pair)
    kwargs[:cutoff] = maximum(effective_cutoff_pair)

    if use_typed_neighbors
        kwargs[:use_neighborlist] = true
    end

    system_typeids = copy(system.typeids)
    box_tuple = Tuple(system.box)
    cap = get(kwargs, :cap, Int32(96))
    skin = get(kwargs, :skin, 0.4)

    function _post_build(st)
        copyto!(st.typeid, system_typeids)
        st.sigma_pair = CuArray(T.(sigma_pair))
        st.epsilon_pair = CuArray(T.(epsilon_pair))
        st.rcut_pair = CuArray(T.(effective_cutoff_pair))
        if use_typed_neighbors
            if length(box_tuple) == 2
                st.nbh = NeighborLists.build_neighbors_stencil_by_types!(st.rx, st.ry;
                                                                         box=st.box2,
                                                                         typeid=st.typeid,
                                                                         rcut_pair=effective_cutoff_pair,
                                                                         cap=cap,
                                                                         skin=skin)
            else
                st.nbh = NeighborLists.build_neighbors_stencil_by_types!(st.rx, st.ry, st.rz;
                                                                         box=st.box3,
                                                                         typeid=st.typeid,
                                                                         rcut_pair=effective_cutoff_pair,
                                                                         cap=cap,
                                                                         skin=skin)
            end
        end
        return st
    end

    return kwargs, _post_build, Dict{Symbol,Any}(
        :pair_style => :pair_table,
        :sigma_pair => sigma_pair,
        :epsilon_pair => epsilon_pair,
        :cutoff_pair => effective_cutoff_pair,
    )
end

function _compile_nonbonded(system::ParticleSystem, force::LennardJones, ::Type{T}) where {T<:AbstractFloat}
    if force.pair_table === nothing
        return _uniform_lj_kwargs(force, :lj, T)
    end
    return _pair_table_kwargs(system, force, :lj, T)
end

function _compile_nonbonded(system::ParticleSystem, force::WCA, ::Type{T}) where {T<:AbstractFloat}
    if force.pair_table === nothing
        return _uniform_lj_kwargs(force, :wca, T)
    end
    return _pair_table_kwargs(system, force, :wca, T)
end

function _compile_nonbonded(system::ParticleSystem, force::SoftRepulsive, ::Type{T}) where {T<:AbstractFloat}
    force.pair_table === nothing || throw(ArgumentError("SoftRepulsive pair tables are not implemented yet."))
    force.epsilon === nothing && throw(ArgumentError("SoftRepulsive requires `epsilon`."))
    force.sigma === nothing && throw(ArgumentError("SoftRepulsive requires `sigma`."))
    force.cutoff === nothing && throw(ArgumentError("SoftRepulsive requires an explicit `cutoff`."))

    params = if force.params === nothing
        Definitions.SoftRepulsiveParams{T}(T(force.epsilon), T(force.sigma))
    elseif force.params isa Definitions.SoftRepulsiveParams
        Definitions.SoftRepulsiveParams{T}(T(force.params.ϵ), T(force.params.σ))
    else
        throw(ArgumentError("SoftRepulsive params must be a Definitions.SoftRepulsiveParams or `nothing`."))
    end

    kwargs = Dict{Symbol,Any}(
        :nonbonded => :soft_repulsive,
        :epsilon => T(force.epsilon),
        :sigma => T(force.sigma),
        :cutoff => T(force.cutoff),
        :softrep_params => params,
    )
    merge!(kwargs, _neighbor_kwargs(force)[1])
    return kwargs, (st -> st), Dict{Symbol,Any}(:pair_style => :uniform)
end

function _compile_bonded(system::ParticleSystem, force::HarmonicBondForce, ::Type{T}) where {T<:AbstractFloat}
    force.type == :default || throw(ArgumentError("Type-specific harmonic bond parameters are not implemented yet."))
    isempty(system.topology.bonds) &&
        throw(ArgumentError("HarmonicBondForce requires `system.topology.bonds` to be populated."))
    unique_types = unique(system.topology.bond_types)
    isempty(unique_types) || length(unique_types) == 1 ||
        throw(ArgumentError("Multiple bond types are not implemented yet; current engine supports one bonded parameter set per simulation."))
    return Dict{Symbol,Any}(
        :bonds => system.topology.bonds,
        :bonding => Definitions.HarmonicBond{T}(Definitions.HarmonicBondParams{T}(T(force.k), T(force.r0))),
    ), Dict{Symbol,Any}(:bond_style => :harmonic)
end

function _compile_bonded(system::ParticleSystem, force::FENEBondForce, ::Type{T}) where {T<:AbstractFloat}
    force.type == :default || throw(ArgumentError("Type-specific FENE bond parameters are not implemented yet."))
    isempty(system.topology.bonds) &&
        throw(ArgumentError("FENEBondForce requires `system.topology.bonds` to be populated."))
    unique_types = unique(system.topology.bond_types)
    isempty(unique_types) || length(unique_types) == 1 ||
        throw(ArgumentError("Multiple bond types are not implemented yet; current engine supports one bonded parameter set per simulation."))
    return Dict{Symbol,Any}(
        :bonds => system.topology.bonds,
        :bonding => Definitions.FENEBond{T}(Definitions.FENEParams{T}(T(force.k), T(force.R0))),
    ), Dict{Symbol,Any}(:bond_style => :fene)
end

function compile_forces(system::ParticleSystem, forces; precision=:f64)
    T = _precision_type(precision)
    force_list = _force_list(forces)

    nonbonded = Force[]
    bonded = Force[]
    for force in force_list
        if force isa Union{LennardJones,WCA,SoftRepulsive}
            push!(nonbonded, force)
        elseif force isa Union{HarmonicBondForce,FENEBondForce}
            push!(bonded, force)
        else
            throw(ArgumentError("Unsupported workflow force type $(typeof(force))."))
        end
    end

    length(nonbonded) <= 1 ||
        throw(ArgumentError("Only one nonbonded force family may be active per simulation with the current engine."))
    length(bonded) <= 1 ||
        throw(ArgumentError("Only one bonded force family may be active per simulation with the current engine."))

    build_kwargs = Dict{Symbol,Any}()
    post_build_hooks = Function[]
    metadata = Dict{Symbol,Any}()

    if !isempty(nonbonded)
        nb_kwargs, nb_post, nb_meta = _compile_nonbonded(system, only(nonbonded), T)
        merge!(build_kwargs, nb_kwargs)
        push!(post_build_hooks, nb_post)
        merge!(metadata, nb_meta)
    end

    if !isempty(bonded)
        bond_kwargs, bond_meta = _compile_bonded(system, only(bonded), T)
        merge!(build_kwargs, bond_kwargs)
        merge!(metadata, bond_meta)
    end

    function _post_build(st)
        for hook in post_build_hooks
            hook(st)
        end
        return st
    end

    return CompiledForces(build_kwargs, _post_build, metadata)
end
