# =========================
#   Dynamic cell: scaling, geometry and trial transactions
# =========================
#
# `set_cell_and_positions!` is the supported dynamic-cell mutation: it scales
# every coordinate representation with the box and then refreshes geometry
# through `set_box!` (neighbor grid, electrostatic mesh coefficients),
# invalidating forces, energies and virials.

_current_box(st::SimulationState{T}) where {T} =
    _is_3d(st) ? (st.box3::Definitions.Box3{T}) : (st.box2::Definitions.Box2{T})

"""
    cell_volume(st) -> Float64

Box volume (area in 2D).
"""
cell_volume(st::SimulationState) = prod(Float64, _current_box(st))

# Largest pair cutoff of the short-range term (all-pairs traversal relies on
# the minimum-image convention, which needs rc ≤ L/2).
function _max_pair_cutoff(st::SimulationState{T}) where {T}
    st.nb_kind == NB_KIND_NONE && return zero(T)
    st.rcut_pair !== nothing && return T(maximum(st.rcut_pair))
    st.sigma_particle !== nothing && return T(st.rcut_factor * maximum(st.sigma_particle))
    st.nb_kind == NB_KIND_SOFTREP && return T(st.softrep.σ)
    return T(st.pair_lj.rcut)
end

# In-place neighbor geometry refresh: the ELL rows, counts and references do
# not depend on the box; only the cell grid does. Reallocate the (small) cell
# arrays when the grid changes and force a rebuild at the next evaluation.
function _refresh_neighbor_grid!(nbh::NeighborLists.NeighborMatrix{T}, newbox) where {T}
    D = Int(nbh.D)
    nx, ny, nz = NeighborLists._choose_grid(newbox, nbh.cell_size, D)
    NeighborLists._validate_neighbor_grid(newbox, nbh.cell_size, nx, ny, nz, D, "Dense cell-list neighbor builder")
    if (nx, ny, nz) != (nbh.nx, nbh.ny, nbh.nz)
        ncell = Int(nx) * Int(ny) * Int(nz)
        nbh.cell_offsets = CUDA.fill(Int32(1), ncell + 1)
        nbh.cell_counts = CUDA.zeros(Int32, ncell)
        nbh.nx, nbh.ny, nbh.nz = nx, ny, nz
    end
    nbh.valid = false
    return nbh
end

"""
    cell_supported(st, box) -> Union{Nothing,String}

`nothing` when `box` is a valid periodic cell for the state's neighbor
traversal and cutoffs, otherwise the reason it is not. Never mutates `st`.
"""
function cell_supported(st::SimulationState{T}, box) where {T}
    D = _is_3d(st) ? 3 : 2
    length(box) == D || return "expected $(D) box lengths"
    newbox = ntuple(d -> T(box[d]), D)
    all(L -> isfinite(L) && L > zero(T), newbox) || return "box lengths must be finite and positive"
    nbh = st.nbh
    if nbh isa NeighborLists.StencilNeighborMatrix
        return "typed stencil neighbor lists cannot follow a changing box"
    elseif nbh isa NeighborLists.NeighborMatrix
        nx, ny, nz = NeighborLists._choose_grid(newbox, nbh.cell_size, D)
        (nx >= 3 && ny >= 3 && (D == 2 || nz >= 3)) ||
            return "box $(newbox) leaves fewer than 3 neighbor cells (cutoff+skin = $(nbh.cell_size))"
    else
        rc = _max_pair_cutoff(st)
        2 * rc <= minimum(newbox) || return "pair cutoff $(rc) exceeds half the box $(newbox)"
    end
    if st.electrostatics !== nothing
        2 * st.electrostatics.spec.real_cutoff <= minimum(newbox) ||
            return "real-space Coulomb cutoff exceeds half the box $(newbox)"
    end
    return nothing
end

function _scale_coords_kernel!(rx, ry, rz, s::NTuple{D,T}, L::NTuple{D,T}, ::Val{WRAP}) where {D,T,WRAP}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        x = rx[i] * s[1]; y = ry[i] * s[2]
        rx[i] = WRAP ? _wrap_coord(x, L[1]) : x
        ry[i] = WRAP ? _wrap_coord(y, L[2]) : y
        if D == 3
            z = rz[i] * s[D]
            rz[i] = WRAP ? _wrap_coord(z, L[D]) : z
        end
    end
    return
end

function _scale_coords!(st::SimulationState{T}, x, y, z, s, L, wrap::Bool) where {T}
    x === nothing && return nothing
    threads, blocks = _launch1d(length(x))
    zz = z === nothing ? x : z
    CUDA.@cuda threads=threads blocks=blocks _scale_coords_kernel!(x, y, zz, s, L, Val(wrap))
    return nothing
end

"""
    set_cell_and_positions!(st, box) -> st

Change the periodic box to `box` and scale every particle coordinate with it
(fractional coordinates are preserved; velocities are unchanged). Wrapped,
unwrapped and freeze-anchor coordinates are scaled consistently; neighbor
geometry and an attached electrostatics solver are refreshed through
[`set_box!`](@ref); forces, energies and virials are invalidated. An
unsupported cell throws before anything is modified.
"""
function set_cell_and_positions!(st::SimulationState{T}, box) where {T<:AbstractFloat}
    st.rigid_water === nothing ||
        throw(ArgumentError("Atom-wise scaling would stretch rigid water; use set_cell_and_molecules!"))
    reason = cell_supported(st, box)
    reason === nothing || throw(DomainError(box, reason))
    _engine_boundary!(st.engine, st)   # device engine: earlier steps settle in the old cell first
    D = _is_3d(st) ? 3 : 2
    old = _current_box(st)
    newbox = ntuple(d -> T(box[d]), D)
    s = ntuple(d -> newbox[d] / old[d], D)
    _scale_coords!(st, st.rx, st.ry, st.rz, s, newbox, true)
    _scale_coords!(st, st.rx_unwrap, st.ry_unwrap, st.rz_unwrap, s, newbox, false)
    st.freeze_rx === nothing || _scale_coords!(st, st.freeze_rx, st.freeze_ry, st.freeze_rz, s, newbox, true)
    set_box!(st, newbox)
    _collisions_snapshot!(st)   # the engine re-initializes contact flags at the rebuild the cell change requests
    return st
end

# Molecular scaling (rigid water): particles outside rigid waters are scaled
# atom-wise; each water's centre of mass is scaled and the molecule translated
# rigidly, so bond lengths, angles and (constrained) velocities are unchanged.
function _scale_free_kernel!(rx, ry, rz, in_water, s::NTuple{3,T}, L::NTuple{3,T}, ::Val{WRAP}) where {T,WRAP}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    (i > length(rx) || @inbounds(in_water[i])) && return
    @inbounds begin
        x = rx[i] * s[1]; y = ry[i] * s[2]; z = rz[i] * s[3]
        rx[i] = WRAP ? _wrap_coord(x, L[1]) : x
        ry[i] = WRAP ? _wrap_coord(y, L[2]) : y
        rz[i] = WRAP ? _wrap_coord(z, L[3]) : z
    end
    return
end

function _scale_water_kernel!(rx, ry, rz, atoms, inverse_masses, s::NTuple{3,T}, Lold::NTuple{3,T},
                              L::NTuple{3,T}, ::Val{WRAP}) where {T,WRAP}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return
    @inbounds begin
        ids = atoms[i]; w = inverse_masses[i]
        raw = RigidWaterConstraints._load_water(rx, ry, rz, ids)
        # wrapped coordinates: the molecule made whole around its oxygen (minimum image);
        # unwrapped ones are whole already
        x = WRAP ? RigidWaterConstraints.local_water(raw, raw[1], Lold) : (raw[1] - raw[1], raw[2] - raw[1], raw[3] - raw[1])
        m1 = one(T) / w[1]; m2 = one(T) / w[2]; m3 = one(T) / w[3]
        com = raw[1] + (m1 * x[1] + m2 * x[2] + m3 * x[3]) / (m1 + m2 + m3)
        d = (s[1] - one(T)) * com[1], (s[2] - one(T)) * com[2], (s[3] - one(T)) * com[3]
        for k in 1:3
            j = ids[k]; p = raw[1] + x[k]
            rx[j] = WRAP ? _wrap_coord(p[1] + d[1], L[1]) : p[1] + d[1]
            ry[j] = WRAP ? _wrap_coord(p[2] + d[2], L[2]) : p[2] + d[2]
            rz[j] = WRAP ? _wrap_coord(p[3] + d[3], L[3]) : p[3] + d[3]
        end
    end
    return
end

function _scale_molecules!(st, x, y, z, s, Lold, L, wrap::Bool)
    x === nothing && return nothing
    ws = st.rigid_water.workspace; n = length(ws.atoms)
    threads, blocks = _launch1d(length(x))
    CUDA.@cuda threads=threads blocks=blocks _scale_free_kernel!(x, y, z, st.rigid_water.in_water, s, L, Val(wrap))
    CUDA.@cuda threads=128 blocks=cld(n, 128) _scale_water_kernel!(x, y, z, ws.atoms, ws.inverse_masses, s, Lold, L, Val(wrap))
    return nothing
end

"""
    set_cell_and_molecules!(st, box) -> st

Molecular counterpart of [`set_cell_and_positions!`](@ref) for states with rigid
water: each water's centre of mass is scaled with the cell and the molecule is
translated rigidly (bond lengths, angles and velocities unchanged); every other
particle is scaled atom-wise. Wrapped and unwrapped coordinates are scaled
consistently, the constraint reference is re-saved, geometry is refreshed
through [`set_box!`](@ref) and forces, energies and virials are invalidated.
Without rigid water this is `set_cell_and_positions!`. An unsupported cell
throws before anything is modified.
"""
function set_cell_and_molecules!(st::SimulationState{T}, box) where {T<:AbstractFloat}
    st.rigid_water === nothing && return set_cell_and_positions!(st, box)
    reason = cell_supported(st, box)
    reason === nothing || throw(DomainError(box, reason))
    _engine_boundary!(st.engine, st)   # device engine: earlier steps settle in the old cell first
    old = _current_box(st)
    newbox = ntuple(d -> T(box[d]), 3)
    s = ntuple(d -> newbox[d] / old[d], 3)
    _scale_molecules!(st, st.rx, st.ry, st.rz, s, old, newbox, true)
    _scale_molecules!(st, st.rx_unwrap, st.ry_unwrap, st.rz_unwrap, s, old, newbox, false)
    set_box!(st, newbox)
    _save_rigid_reference!(st)         # SHAKE reference at the new cell
    return st
end

# Cell change of a barostat event: molecular with rigid water, atom-wise otherwise.
_set_cell_scaled!(st, box) = st.rigid_water === nothing ? set_cell_and_positions!(st, box) : set_cell_and_molecules!(st, box)
"""Number of independently scaled units: particles, with each rigid water counted once."""
scaled_units(st) = length(st.rx) - (st.rigid_water === nothing ? 0 : 2length(st.rigid_water.workspace.atoms))

# ---------------------------- trial transactions ----------------------------

"""
    TrialWorkspace

Preallocated copy of the committed configuration (cell, coordinates, force
slot, energy/virial buffers and their validity) used to restore the exact
state after a rejected trial.
"""
mutable struct TrialWorkspace{T<:AbstractFloat}
    pos::Vector{CuVector{T}}
    unwrap::Vector{CuVector{T}}
    force::Vector{CuVector{T}}
    Epot::CuVector{T}
    virial::CuVector{T}
    virial_nonbonded::CuMatrix{T}
    virial_bonded::CuMatrix{T}
    virial_tensor::CuMatrix{T}
    virial_global::CuVector{T}
    box::Any
    force_valid::Bool
    energy_valid::Bool
    force_freeze_spring::Bool
end

function TrialWorkspace(st::SimulationState{T}) where {T<:AbstractFloat}
    st.freeze_rx === nothing ||
        throw(ArgumentError("trial transactions do not support freeze anchors"))
    # Sorted states are fine: save, trial and restore happen at one step boundary, where
    # nothing sorts (evaluations never permute storage).
    d3 = _is_3d(st)
    N = length(st.rx)
    v() = CuVector{T}(undef, N)
    pos = d3 ? [v(), v(), v()] : [v(), v()]
    unwrap = st.rx_unwrap === nothing ? CuVector{T}[] : (d3 ? [v(), v(), v()] : [v(), v()])
    force = d3 ? [v(), v(), v()] : [v(), v()]
    return TrialWorkspace{T}(pos, unwrap, force, v(), v(), similar(st.virial_nonbonded), similar(st.virial_bonded),
                             similar(st.virial_tensor), similar(st.virial_global), _current_box(st), false, false, false)
end

_state_pos(st) = _is_3d(st) ? (st.rx, st.ry, st.rz) : (st.rx, st.ry)
_state_unwrap(st) = _is_3d(st) ? (st.rx_unwrap, st.ry_unwrap, st.rz_unwrap) : (st.rx_unwrap, st.ry_unwrap)
_state_force(st) = _is_3d(st) ? (st.fx, st.fy, st.fz) : (st.fx, st.fy)

"""
    save_trial_state!(ws, st)

Copy the committed configuration and its outputs into `ws`.
"""
function save_trial_state!(ws::TrialWorkspace, st::SimulationState)
    foreach(copyto!, ws.pos, _state_pos(st))
    isempty(ws.unwrap) || foreach(copyto!, ws.unwrap, _state_unwrap(st))
    foreach(copyto!, ws.force, _state_force(st))
    copyto!(ws.Epot, st.Epot); copyto!(ws.virial, st.virial)
    copyto!(ws.virial_nonbonded, st.virial_nonbonded); copyto!(ws.virial_bonded, st.virial_bonded)
    copyto!(ws.virial_tensor, st.virial_tensor); copyto!(ws.virial_global, st.virial_global)
    ws.box = _current_box(st)
    ws.force_valid, ws.energy_valid, ws.force_freeze_spring = st.force_valid, st.energy_valid, st.force_freeze_spring
    return ws
end

"""
    restore_trial_state!(ws, st)

Restore exactly the configuration saved by [`save_trial_state!`](@ref): cell
geometry, coordinates, force slot, energy/virial buffers and validity flags.
"""
function restore_trial_state!(ws::TrialWorkspace, st::SimulationState)
    _current_box(st) == ws.box || set_box!(st, ws.box)
    foreach(copyto!, _state_pos(st), ws.pos)
    isempty(ws.unwrap) || foreach(copyto!, _state_unwrap(st), ws.unwrap)
    foreach(copyto!, _state_force(st), ws.force)
    copyto!(st.Epot, ws.Epot); copyto!(st.virial, ws.virial)
    copyto!(st.virial_nonbonded, ws.virial_nonbonded); copyto!(st.virial_bonded, ws.virial_bonded)
    copyto!(st.virial_tensor, ws.virial_tensor); copyto!(st.virial_global, ws.virial_global)
    st.force_valid, st.energy_valid, st.force_freeze_spring = ws.force_valid, ws.energy_valid, ws.force_freeze_spring
    _collisions_snapshot!(st)   # the engine re-initializes contact flags at the rebuild the cell change requests
    _save_rigid_reference!(st)   # rigid water: the trial re-saved the SHAKE reference
    return st
end
