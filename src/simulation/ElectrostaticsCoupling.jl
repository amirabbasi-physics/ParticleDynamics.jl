# =========================
#   Long-range electrostatics: state coupling
# =========================


# slot_of[id] = storage slot currently holding build-time particle `id`.
function _slot_of_ids(st::SimulationState)
    N = length(st.rx)
    st.tag === nothing && return collect(1:N)
    tag = Array(st.tag)
    slot = Vector{Int}(undef, N)
    @inbounds for k in 1:N
        slot[tag[k]] = k
    end
    return slot
end

# Host charges in build-time order → storage order.
function _charges_to_storage(st::SimulationState{T}, q::AbstractVector{<:Real}) where {T}
    N = length(st.rx)
    length(q) == N || throw(ArgumentError("charges must have length $(N)."))
    host = T.(collect(q))
    all(isfinite, host) || throw(ArgumentError("charges must be finite."))
    st.tag === nothing && return host
    tag = Array(st.tag)
    return host[tag]
end


"""
    attach_electrostatics!(st, spec::Electrostatics.ElectrostaticsSpec;
                           charges=nothing, exceptions=()) -> st

Attach a long-range Coulomb solver to a 3D state. `charges` (build-time
particle order, caller's charge unit) replace `st.charge`; when omitted the
state must already carry charges. `exceptions` lists `(i, j, scale)` triples
in build-time ids: the pair's home-image interaction `C qᵢqⱼ/r` is scaled by
`scale` (0 excludes it) while periodic images keep the full interaction.

Setup validates the box (`real_cutoff` below half the shortest box length,
orthorhombic 3D), plans cuFFT transforms and computes the influence function.
The state moves to the cluster-pair backend (charged systems), which keeps its
storage order unsorted.
"""
function attach_electrostatics!(st::SimulationState{T}, spec::Electrostatics.ElectrostaticsSpec;
                                charges=nothing, exceptions=()) where {T<:AbstractFloat}
    _is_3d(st) || throw(ArgumentError("Long-range electrostatics supports 3D periodic boxes only; 2D and slab geometries are not implemented."))
    st.external_potential === nothing ||
        throw(ArgumentError("External potentials replace all internal forces and cannot be combined with long-range electrostatics."))
    N = length(st.rx)
    if charges !== nothing
        host = _charges_to_storage(st, charges)
        if st.charge === nothing
            st.charge = CuArray(host)
        else
            copyto!(st.charge, host)
        end
    end
    st.charge === nothing && throw(ArgumentError("attach_electrostatics! requires per-particle charges."))
    qstore = Array(st.charge::CuArray{T,1})
    exc = Electrostatics.normalize_exceptions(exceptions, N)
    solver = Electrostatics.ElectrostaticsSolver(spec, T, st.box3::Definitions.Box3{T}, qstore, exc,
                                                 _slot_of_ids(st))
    st.electrostatics = solver
    fill!(st.virial_global, zero(T))
    _charged_backend!(st)
    invalidate_forces!(st)
    return st
end

# User rule (13 Oct 2026): charged systems run on the cluster-pair backend (ELL for
# uncharged ones). The cluster backend takes LJ (uniform, pair table, per-particle σ),
# WCA and no vdW term, with bond exclusions and collision counting; small
# boxes cap the cluster skin at L/2 − r_max. Charged soft-repulsion and 2D states are
# not supported (the device engine rejects them; there is no staged path).
function _charged_backend!(st::SimulationState)
    st.nonbonded_backend === :cluster && return st
    (try _cluster_supported(st); true catch; false end) || return st
    rmax = max(_cluster_lj_table(st)[2], Float64(st.electrostatics.spec.real_cutoff))
    set_nonbonded_backend!(st, :cluster)
    # small boxes: the list radius stays within half the box (one image per pair)
    skin = min(st.cluster_config.skin, max(0.999 * (minimum(Float64.(st.box3)) / 2 - rmax), 0.0))
    st.cluster_config = merge(st.cluster_config, (auto=true, skin=skin, prune_buffer=min(st.cluster_config.prune_buffer, skin)))
    st.cluster = nothing   # rebuilt with the configuration
    return st
end

# Real-space Coulomb (pairs, exception corrections, self energy) of `solver` into scratch
# buffers, for diagnostics and accuracy targeting: a Coulomb-only cluster list with the
# solver's own split (exact list, no buffer), whatever the state's backend. The buffers
# are overwritten.
function _scratch_real_space!(st::SimulationState{T}, solver, fx, fy, fz, E, V, compute_energy::Bool) where {T}
    q = st.charge::CuArray{T,1}
    rc = Float64(solver.spec.real_cutoff)
    coulomb = ClusterPairs.coulomb_polynomial(solver.spec.split, Float64(solver.C); T=T, cutoff=rc,
                                              eshift=Float64(solver.eshift))
    exceptions = length(solver.exj) > 0 ? (solver.exo, solver.exj, solver.exs) : nothing
    eng = ClusterPairs.ClusterPairEngine(T, length(st.rx), st.box3; coulomb, selfc=Float64(solver.selfc),
                                         rmax=rc, skin=0.0, prune_buffer=0.0, exceptions)
    ClusterPairs.compute!(eng, st.rx, st.ry, st.rz, q, st.typeid, fx, fy, fz, E, V, st.box3, compute_energy)
    return nothing
end

"""
    detach_electrostatics!(st) -> st

Remove the long-range solver (charges stay stored) and clear the global
virial contribution.
"""
function detach_electrostatics!(st::SimulationState{T}) where {T}
    st.electrostatics = nothing
    cfg = st.cluster_config
    # an uncharged state returns to ELL when the cluster backend came from attaching charges
    (cfg !== nothing && get(cfg, :auto, false)) && set_nonbonded_backend!(st, :ell)
    fill!(st.virial_global, zero(T))
    fill!(st.virial_global_accum, zero(T))
    invalidate_forces!(st)
    return st
end

"""
    set_charges!(st, q) -> st

Replace the per-particle charges (`q` in build-time particle order). With an
attached solver the net-charge policy is re-checked and the background term
refreshed. Forces are invalidated.
"""
function set_charges!(st::SimulationState{T}, q::AbstractVector{<:Real}) where {T<:AbstractFloat}
    host = _charges_to_storage(st, q)
    if st.electrostatics !== nothing
        Electrostatics.set_solver_charges!(st.electrostatics, host)
    end
    if st.charge === nothing
        st.charge = CuArray(host)
    else
        copyto!(st.charge, host)
    end
    invalidate_forces!(st)
    return st
end

"""
    charges(st) -> Vector or nothing

Host copy of the charges in build-time particle order.
"""
function charges(st::SimulationState)
    st.charge === nothing && return nothing
    q = Array(st.charge)
    st.tag === nothing && return q
    out = similar(q)
    out[Array(st.tag)] = q
    return out
end

"""
    electrostatic_energy(st) -> NamedTuple

Diagnostic breakdown of the electrostatic energy at the current coordinates:
`real` (real-space pairs plus exception corrections), `self`, `reciprocal`
(mesh), `background` and `total`, plus the reciprocal virial tensor
`reciprocal_virial` `(xx, yy, zz, xy, xz, yz)`. Uses scratch buffers; the
state's force and energy buffers are untouched. Not for the hot path.
"""
function electrostatic_energy(st::SimulationState{T}) where {T<:AbstractFloat}
    solver = st.electrostatics
    solver === nothing && throw(ArgumentError("No electrostatics solver is attached."))
    N = length(st.rx)
    fx = CUDA.zeros(T, N); fy = CUDA.zeros(T, N); fz = CUDA.zeros(T, N)
    E = CUDA.zeros(T, N); V = CUDA.zeros(T, N, 6); Vg = CUDA.zeros(T, 6)
    q = st.charge::CuArray{T,1}
    _scratch_real_space!(st, solver, fx, fy, fz, E, V, true)
    Ereal_self = Float64(sum(Float64, Array(E)))
    Eself = Electrostatics.self_energy(solver)
    fill!(E, zero(T))
    Electrostatics.compute_mesh!(solver, st.rx, st.ry, st.rz, q, fx, fy, fz, E, Vg, true, st.nbh)
    Emesh = Electrostatics.mesh_energy(solver)
    Ebg = Electrostatics.background_energy(solver)
    W = Float64.(Array(Vg))
    total = Ereal_self + Emesh + Ebg
    return (real = Ereal_self - Eself, self = Eself, reciprocal = Emesh, background = Ebg,
            total = total, reciprocal_virial = (xx=W[1], yy=W[2], zz=W[3], xy=W[4], xz=W[5], yz=W[6]))
end

"""
    set_box!(st, box) -> st

Replace the periodic box lengths (same dimensionality). Coordinates are left
unchanged; [`set_cell_and_positions!`](@ref) scales them with the box. A dense
cell list keeps its storage, cutoff, skin and capacity: only its cell grid is
recomputed (the small per-cell arrays are reallocated when the grid changes)
and it is rebuilt at the next force evaluation (all-pairs traversal needs no
change). An attached electrostatics
solver keeps its mesh dimensions and FFT plans and recomputes the
box-dependent influence function, virial coefficients and background term.
Forces and energies are invalidated. Typed stencil lists must be rebuilt by
the caller.
"""
function set_box!(st::SimulationState{T}, box) where {T<:AbstractFloat}
    D = _is_3d(st) ? 3 : 2
    length(box) == D || throw(ArgumentError("set_box! expects $(D) box lengths."))
    newbox = ntuple(d -> T(box[d]), D)
    all(L -> isfinite(L) && L > zero(T), newbox) || throw(ArgumentError("Box lengths must be finite and positive."))
    nbh = st.nbh
    if nbh isa NeighborLists.StencilNeighborMatrix
        throw(ArgumentError("set_box! cannot resize typed stencil neighbor lists; rebuild them with the new box."))
    elseif nbh isa NeighborLists.NeighborMatrix
        _refresh_neighbor_grid!(nbh, newbox)
    end
    if st.electrostatics !== nothing
        Electrostatics.refresh_geometry!(st.electrostatics, Float64.(newbox))
    end
    if D == 3
        st.box3 = newbox
    else
        st.box2 = newbox
    end
    invalidate_forces!(st)
    _engine_set_box!(st.engine, st, newbox)   # device engine geometry; forces invalid only by the move
    return st
end
