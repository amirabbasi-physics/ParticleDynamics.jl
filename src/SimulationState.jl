# =========================
#   Simulation state (SoA)
# =========================
"""
    SimulationState{T}

Structure-of-arrays storage for all GPU-resident buffers required to evolve a
non-equilibrium simulation. Users normally do not construct this type manually;
[`build_simulation`](@ref) assembles an allocated instance with concrete
CuArray fields and consistent dimensionality inferred from the `box` argument.

Key fields that user code may read or mutate:
- `rx, ry[, rz]`, `vx, vy[, vz]`, `fx, fy[, fz]`: positions, velocities, and
  force accumulators in GPU memory. Arrays are sized `N` and remain allocated
  during ordinary stepping; optional spatial reordering can replace them.
  After manual position/force-model edits, call `invalidate_forces!(st)`.
- `rx_unwrap, ry_unwrap[, rz_unwrap]`: optional unwrapped positions that track
  continuous motion across periodic boundaries (enabled via `unwrapped_positions`).
- `nbh`: neighbor matrix (either dense cell list or sentinel all-pairs) built
  with the `NeighborLists.build_neighbors_*` family using the requested cutoff,
  skin, and capacity.
- `mass`, `dt`: nominal build-time timestep plus the scalar fast-path mass used
  when all particles share one mass. Heterogeneous-mass states additionally
  populate `mass_particle` and `inv_mass_particle` with per-particle data.
- `Epot`, `Ekin`, `virial`, `dq`, `dU` plus the corresponding `*_accum` buffers:
  energy, virial, and heat observables that can be sampled directly on the GPU.
- `virial_nonbonded`, `virial_bonded`, `virial_tensor`, `virial_tensor_accum`:
  GPU-resident configurational virial tensors. These store the raw pairwise
  virial `W = Σ r_ij ⊗ F_ij` using the same minimum-image displacement used in
  the force kernels. The scalar `virial` field stores the trace of `W` for
  backward compatibility with existing examples. These buffers are refreshed on
  force evaluations with `compute_energy=true`.
- `virial_global`, `virial_global_accum`: global virial terms without a
  per-particle decomposition (the reciprocal-space electrostatic tensor).
  System totals ([`virial_tensor`](@ref), [`collect_step_observables`](@ref))
  include them.
- `vdw_modifier`: Lennard-Jones energy convention at the cutoff, `:none` or
  `:potential_shift` (see [`set_vdw_modifier!`](@ref)).
- `nonbonded_backend`, `cluster`, `cluster_config`: short-range traversal
  selected with [`set_nonbonded_backend!`](@ref) and its lazily built engine.
- `charge`: optional per-particle charges; `electrostatics`: optional
  long-range Coulomb solver attached with [`attach_electrostatics!`](@ref).
- `energy_valid`: true only while the energy and virial buffers belong to the
  current coordinates, i.e. after a force evaluation with
  `compute_energy=true` and before any position change or
  `invalidate_forces!`. Force-only evaluations clear it. Workflow observables
  refresh stale energies before sampling.
- `typeid`: per-particle type ids used by `Filters` and mixed-size LJ kernels.
- `freeze_*`: optional buffers used by the freeze/tether helpers in `Filters`.
- `coll_*`: optional buffers that appear when collision counting is enabled via
  `enable_collision_counting!`.

All other fields are internal implementation details (neighbor rebuild state,
bond lists, cached LJ parameters, etc.) and should be treated as read-only.
Current backend abstraction is storage-level only: `SimulationState` remains
CuArray-backed today, and the execution kernels are still CUDA-specific.
"""
mutable struct SimulationState{T<:AbstractFloat}
    # SoA arrays
    rx::CuArray{T,1}; ry::CuArray{T,1}
    rz::Union{Nothing,CuArray{T,1}}
    rx_unwrap::Union{Nothing,CuArray{T,1}}
    ry_unwrap::Union{Nothing,CuArray{T,1}}
    rz_unwrap::Union{Nothing,CuArray{T,1}}
    vx::CuArray{T,1}; vy::CuArray{T,1}
    vz::Union{Nothing,CuArray{T,1}}
    fx::CuArray{T,1}; fy::CuArray{T,1}
    fz::Union{Nothing,CuArray{T,1}}

    # previous forces (to avoid per-step allocations)
    f0x::CuArray{T,1}; f0y::CuArray{T,1}
    f0z::Union{Nothing,CuArray{T,1}}

    # per-particle type id
    typeid::CuArray{Int32,1}

    # box (stored directly; no splatting)
    box2::Union{Definitions.Box2{T},Nothing}
    box3::Union{Definitions.Box3{T},Nothing}

    # neighbor list
    nbh::NeighborLists.AbstractNeighborMatrix
    neigh_interval::Int

    # pair params (global LJ)
    pair_lj::Definitions.LJParams{T}
    # optional per-particle size (mixed interactions)
    sigma_particle::Union{Nothing,CuArray{T,1}}
    rcut_factor::T
    # optional per-type pair parameters
    sigma_pair::Union{Nothing,CuArray{T,2}}
    epsilon_pair::Union{Nothing,CuArray{T,2}}
    rcut_pair::Union{Nothing,CuArray{T,2}}

    # bonded interactions (optional)
    bonds::Union{Nothing,BondedForces.BondList}
    bonding::Union{Nothing,Definitions.BondPotential{T}}

    # integration metadata
    mass::T
    mass_particle::Union{Nothing,CuArray{T,1}}
    inv_mass_particle::Union{Nothing,CuArray{T,1}}
    dt::T

    # observables buffers
    Epot::CuArray{T,1}
    dq::CuArray{T,1}
    dU::CuArray{T,1}
    Ekin::CuArray{T,1}
    virial::CuArray{T,1}
    virial_nonbonded::CuArray{T,2}
    virial_bonded::CuArray{T,2}
    virial_tensor::CuArray{T,2}
    # interval accumulators (GPU) to avoid host reductions each step
    Epot_accum::CuArray{T,1}
    Ekin_accum::CuArray{T,1}
    virial_accum::CuArray{T,1}
    virial_tensor_accum::CuArray{T,2}

    # misc
    step::Int
    # last integrator used: 1=Langevin, 2=Brownian, 3=NHC, 4=CSVR, 5=NVE, 0=unknown
    last_integrator::UInt8
    nb_kind::UInt8
    softrep::Union{Nothing,Definitions.SoftRepulsiveParams{T}}
    # freeze controls (optional)
    freeze_mode::UInt8
    freeze_until::Int
    freeze_include_energy::Bool
    freeze_mask::Union{Nothing,CuArray{UInt8,1}}
    freeze_k::T
    freeze_rx::Union{Nothing,CuArray{T,1}}
    freeze_ry::Union{Nothing,CuArray{T,1}}
    freeze_rz::Union{Nothing,CuArray{T,1}}
    # Collision counting (optional)
    coll_enabled::Bool
    coll_prev::Union{Nothing,CuArray{UInt8,1}}
    coll_counts::Union{Nothing,CuArray{Int64,1}}
    coll_bins::Union{Nothing,CuArray{Int32,2}}
    # Spatial-reorder identity map: `tag[k]` is the original (build-time) id of
    # the particle currently stored in slot `k`. `nothing` disables reordering.
    tag::Union{Nothing,CuArray{Int32,1}}
    # Step of the last spatial reorder (-1 = none yet) and the minimum number
    # of steps between reorders. Locality decays diffusively over thousands of
    # steps, so reordering at every rebuild would waste the gather cost.
    last_reorder_step::Int
    reorder_interval::Int
    # External potential provider (opt-in): when set (non-nothing), it replaces
    # every internal force term in `evaluate_forces_into_f!`. Typed `Any` on
    # purpose — it is dispatched once per force evaluation, never in kernels.
    external_potential::Any
    # When false, bonded pairs also feel the nonbonded potential (e.g. the
    # canonical Kremer-Grest FENE+WCA bond). Default true: bonded pairs are
    # excluded from nonbonded interactions.
    exclude_bonded::Bool
    # Validity of the active force slot at the current physical coordinates.
    force_valid::Bool
    force_freeze_spring::Bool
    # Last sampled collision coordinates, independent of neighbor row ordering.
    coll_ref_x::Union{Nothing,CuArray{T,1}}
    coll_ref_y::Union{Nothing,CuArray{T,1}}
    coll_ref_z::Union{Nothing,CuArray{T,1}}
    # Per-particle charges in the caller's charge unit (`nothing` = uncharged).
    charge::Union{Nothing,CuArray{T,1}}
    # Long-range electrostatics solver (opt-in). Typed `Any` on purpose: it is
    # dispatched once per force evaluation behind a function barrier.
    electrostatics::Any
    # Global (not per-particle) configurational virial contributions, currently
    # the reciprocal-space electrostatic tensor, in `virial_components` order.
    virial_global::CuArray{T,1}
    virial_global_accum::CuArray{T,1}
    # True only while `Epot`/virial buffers describe the current coordinates
    # (set by force evaluations with `compute_energy=true`).
    energy_valid::Bool
    # Short-range nonbonded traversal: `:ell` (per-particle neighbor rows) or
    # `:cluster` (cluster-pair engine, `ClusterPairs`). The engine is typed
    # `Any` and created lazily behind a function barrier; `cluster_config`
    # keeps the skin, prune buffer and the dense-list settings to restore.
    nonbonded_backend::Symbol
    cluster::Any
    cluster_config::Any
    # Typed bonded families (`BondedForces.BondedTerms`) added after the
    # legacy uniform bond kernels; `nothing` when absent. Typed `Any` and
    # dispatched behind a function barrier once per force evaluation.
    bonded::Any
    # Optional rigid-water integration workspace; conservative forces stay
    # separate from constraint impulses and the pressure estimator.
    rigid_water::Any
    # Device-resident engine (`ParticleDynamics.DeviceEngine`) currently advancing
    # this state through captured CUDA graphs, or `nothing`. While set, it owns
    # the particle arrays' update order; staged entry points release it first.
    engine::Any
    # Lennard-Jones energy at the cutoff: `:none` (plain truncation) or
    # `:potential_shift` (u(r) − u(r_c) per pair; forces and virials unchanged).
    vdw_modifier::Symbol
    # Inverse of `tag` (build-time particle id → storage slot) for sorted states with
    # topology: bonded tables keep ids; kernels map them through `tag`/`slot_of`.
    # Allocated on demand (`_slot_of!`); the device engine refreshes it at every sort.
    slot_of::Union{Nothing,CuArray{Int32,1}}
    # Device-resident pressure control (`attach_pressure_control!`): a barostat the device
    # engine applies inside its step graphs every `interval` steps, or `nothing`.
    pressure_control::Any
end

"""
    invalidate_forces!(st) -> st

Invalidate cached forces after manually changing coordinates, force parameters,
particle types, topology, or an external provider's internal state. The next
step initializes its first force before using it, independently of `st.step`.
This does not resize or reconfigure neighbor ranges: parameter/box changes must
also update the appropriate neighbor container. Velocity-only edits do not
require force invalidation.
"""
function invalidate_forces!(st::SimulationState)
    st.force_valid = false
    st.energy_valid = false
    _engine_invalidated!(st.engine)
    return st
end

const VDW_MODIFIERS = (:none, :potential_shift)

"""
    set_vdw_modifier!(st, modifier) -> st

Lennard-Jones energy convention at the cutoff (GROMACS `vdw-modifier`):
`:none` truncates the pair energy at `r_c`; `:potential_shift` uses
`u(r) − u(r_c)` for every pair inside its cutoff (per pair cutoff with pair
matrices). Forces and virials are identical for both; energies differ by
`u(r_c)` per pair inside the cutoff. Energy-driven samplers (Monte Carlo volume
moves) and energy comparisons depend on the choice. Only Lennard-Jones states
take `:potential_shift` (cell-list and cluster backends). Energies are invalidated.
"""
function set_vdw_modifier!(st::SimulationState, modifier::Symbol)
    modifier in VDW_MODIFIERS || throw(ArgumentError("vdw_modifier must be one of $(VDW_MODIFIERS)"))
    modifier === :none || st.nb_kind == NB_KIND_LJ ||
        throw(ArgumentError("vdw_modifier=:$(modifier) applies to Lennard-Jones (nonbonded=:lj) only"))
    st.vdw_modifier === modifier && return st
    _drop_engine!(st.engine, st)
    st.vdw_modifier = modifier
    invalidate_forces!(st)
    return st
end
vdw_modifier(st::SimulationState) = st.vdw_modifier

function _slot_of_kernel!(slot_of, tag)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(tag) && (@inbounds slot_of[tag[i]] = Int32(i))
    return nothing
end
"""`slot_of` filled from the current `tag` (allocated on first use); `nothing` without tags."""
function _slot_of!(st::SimulationState)
    st.tag === nothing && return nothing
    N = length(st.tag)
    (st.slot_of === nothing || length(st.slot_of) != N) && (st.slot_of = CuArray{Int32}(undef, N))
    CUDA.@cuda threads=256 blocks=cld(N, 256) _slot_of_kernel!(st.slot_of, st.tag)
    return st.slot_of
end
"""Particle topology (legacy bonds or typed bonded terms) present."""
_has_topology(st::SimulationState) = st.bonds !== nothing || st.bonded !== nothing
_lj_shifted(st::SimulationState) = st.vdw_modifier === :potential_shift

@inline backend(st::SimulationState) = storage_backend(st.rx)
Backends.storage_backend(st::SimulationState) = backend(st)
@inline has_uniform_particle_mass(st::SimulationState) = st.mass_particle === nothing

"""
    zero_forces!(st)

Fill `st.fx`, `st.fy`, (`st.fz` if present) with zeros. Called by the tests
before running isolated force kernels and safe to use between manual force
evaluations.
"""
function zero_forces!(st::SimulationState{T}) where {T<:AbstractFloat}
    invalidate_forces!(st)
    fill!(st.fx, zero(T)); fill!(st.fy, zero(T))
    st.fz === nothing || fill!(st.fz, zero(T))
    return nothing
end

"""
    sync_unwrapped!(st)

Copy the wrapped positions into the unwrapped buffers. Call this after manually
overwriting `st.rx`/`st.ry`/`st.rz` when `unwrapped_positions=true`.
"""
function sync_unwrapped!(st::SimulationState{T}) where {T<:AbstractFloat}
    st.rx_unwrap === nothing && return st
    copyto!(st.rx_unwrap, st.rx)
    copyto!(st.ry_unwrap, st.ry)
    if st.rz !== nothing && st.rz_unwrap !== nothing
        copyto!(st.rz_unwrap, st.rz)
    end
    return st
end

@inline function _host_mic(dx::T, L::T) where {T<:AbstractFloat}
    return dx - L * round(dx / L)
end

"""
    initialize_unwrapped_from_bonds!(st, bonds)

Seed `st.rx_unwrap`/`st.ry_unwrap`/`st.rz_unwrap` from the wrapped coordinates
and bond topology by traversing each connected component with minimum-image
bond displacements. This is useful for bonded systems that already straddle the
periodic box at initialization, where a plain `sync_unwrapped!` copy would make
chains appear artificially stretched in unwrapped output.
"""
function initialize_unwrapped_from_bonds!(st::SimulationState{T},
                                          bonds::AbstractVector{<:Tuple{<:Integer,<:Integer}}) where {T<:AbstractFloat}
    st.rx_unwrap === nothing && return st
    isempty(bonds) && return sync_unwrapped!(st)

    N = length(st.rx)
    adjacency = [Int32[] for _ in 1:N]
    for (i_raw, j_raw) in bonds
        i = Int(i_raw)
        j = Int(j_raw)
        1 <= i <= N || throw(ArgumentError("Bond index $(i) is out of bounds for N=$(N)."))
        1 <= j <= N || throw(ArgumentError("Bond index $(j) is out of bounds for N=$(N)."))
        push!(adjacency[i], Int32(j))
        push!(adjacency[j], Int32(i))
    end

    rx = Array(st.rx)
    ry = Array(st.ry)
    rz = st.rz === nothing ? nothing : Array(st.rz)
    rxu = copy(rx)
    ryu = copy(ry)
    rzu = rz === nothing ? nothing : copy(rz)

    if st.box3 === nothing
        st.box2 === nothing && throw(ArgumentError("SimulationState is missing box metadata."))
        Lx, Ly = st.box2
        visited = falses(N)
        stack = Int32[]
        sizehint!(stack, N)
        for root in 1:N
            visited[root] && continue
            visited[root] = true
            push!(stack, Int32(root))
            while !isempty(stack)
                i = Int(pop!(stack))
                xi = rxu[i]
                yi = ryu[i]
                xwi = rx[i]
                ywi = ry[i]
                for j32 in adjacency[i]
                    j = Int(j32)
                    visited[j] && continue
                    rxu[j] = xi + _host_mic(rx[j] - xwi, Lx)
                    ryu[j] = yi + _host_mic(ry[j] - ywi, Ly)
                    visited[j] = true
                    push!(stack, j32)
                end
            end
        end
    else
        rz === nothing && throw(ArgumentError("3D SimulationState is missing z coordinates."))
        rzu === nothing && throw(ArgumentError("3D SimulationState is missing z unwrapped coordinates."))
        Lx, Ly, Lz = st.box3
        visited = falses(N)
        stack = Int32[]
        sizehint!(stack, N)
        for root in 1:N
            visited[root] && continue
            visited[root] = true
            push!(stack, Int32(root))
            while !isempty(stack)
                i = Int(pop!(stack))
                xi = rxu[i]
                yi = ryu[i]
                zi = rzu[i]
                xwi = rx[i]
                ywi = ry[i]
                zwi = rz[i]
                for j32 in adjacency[i]
                    j = Int(j32)
                    visited[j] && continue
                    rxu[j] = xi + _host_mic(rx[j] - xwi, Lx)
                    ryu[j] = yi + _host_mic(ry[j] - ywi, Ly)
                    rzu[j] = zi + _host_mic(rz[j] - zwi, Lz)
                    visited[j] = true
                    push!(stack, j32)
                end
            end
        end
    end

    copyto!(st.rx_unwrap, rxu)
    copyto!(st.ry_unwrap, ryu)
    if rzu !== nothing && st.rz_unwrap !== nothing
        copyto!(st.rz_unwrap, rzu)
    end
    return st
end
