# =========================
#   Simulation state (SoA)
# =========================
"""
    SimulationState{T}

Structure-of-arrays storage for all GPU-resident buffers required to evolve a
non-equilibrium simulation. Users normally do not construct this type manually;
[`build_simulation`](@ref) assembles a fully initialized instance with concrete
CuArray fields and consistent dimensionality inferred from the `box` argument.

Key fields that user code may read or mutate:
- `rx, ry[, rz]`, `vx, vy[, vz]`, `fx, fy[, fz]`: positions, velocities, and
  force accumulators in GPU memory. Arrays are sized `N` and remain allocated
  for the entire simulation so that stepping is allocation-free.
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
    # last integrator used: 1=Langevin, 2=Brownian, 3=NHC, 4=CSVR, 0=unknown
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
end

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
