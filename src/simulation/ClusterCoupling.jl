# =========================
#   Cluster-pair backend: state coupling
# =========================

"""
    set_nonbonded_backend!(st, backend; prune_buffer=nothing, skin=nothing, list_intervals=nothing) -> st

Select the short-range nonbonded traversal of a 3D state.

- `:ell` (default): per-particle neighbor rows; the vdW kernel and the
  real-space Coulomb kernel each traverse them.
- `:cluster`: the fused cluster-pair engine (`ClusterPairs`) evaluates
  Lennard-Jones and the real-space Coulomb term (exclusions and scaled pairs
  included) in one kernel over a Newton-3 cluster-pair list; the mesh part
  of an attached electrostatics solver is unchanged. Supported vdW terms:
  none, uniform LJ and pair-table LJ, with bond exclusions of the LJ term. The dense
  neighbor list is released while the backend is active.

The cluster list radius is `max(vdW cutoff, real_cutoff) + skin` (the skin of
the dense list the state was built with). Between rebuilds the list is pruned
to the interaction cutoff plus `prune_buffer` (default `skin/2`) and
re-pruned when a particle has moved more than half of it; `prune_buffer=0`
gives an exact list for fixed coordinates. `skin` replaces the list skin.

`list_intervals=(nstlist, nstprune)` (opt-in) replaces the displacement triggers by fixed
intervals, as GROMACS's dual pair list with a Verlet buffer: the list is rebuilt every
`nstlist` and pruned every `nstprune` steps (0: only at rebuilds). Exactness then rests on the
buffers being large enough for the motion over those intervals (GROMACS sizes them from
`verlet-buffer-tolerance`); a pair can enter the cutoff unseen when they are not.
"""
function set_nonbonded_backend!(st::SimulationState{T}, backend::Symbol;
                                prune_buffer::Union{Nothing,Real}=nothing, skin::Union{Nothing,Real}=nothing,
                                list_intervals::Union{Nothing,Tuple{Integer,Integer}}=nothing) where {T}
    backend in (:ell, :cluster) || throw(ArgumentError("nonbonded backend must be :ell or :cluster."))
    # charged states may use the dense list when the solver has no Coulomb exceptions
    # (excluded pairs are corrected on the cluster backend only)
    backend === :ell && st.electrostatics !== nothing && length(st.electrostatics.exj) > 0 &&
        throw(ArgumentError("long-range electrostatics with Coulomb exceptions run on the cluster-pair backend only."))
    if backend === :cluster
        _cluster_supported(st)
        cfg = st.cluster_config
        if cfg === nothing
            nbh = st.nbh
            dense = nbh isa NeighborLists.NeighborMatrix
            cfg = (skin=Float64(nbh.skin), dense=dense,
                   cutoff=dense ? Float64(nbh.cutoff) : 0.0, cap=dense ? nbh.cap : Int32(0),
                   target_interval=dense ? nbh.target_interval : 20, prune_buffer=0.5 * Float64(nbh.skin))
        end
        skin === nothing || (cfg = merge(cfg, (skin=Float64(skin),)))
        pb = prune_buffer === nothing ? cfg.prune_buffer : Float64(prune_buffer)
        0 <= pb <= cfg.skin || throw(ArgumentError("prune_buffer must lie in [0, skin = $(cfg.skin)]."))
        if list_intervals !== nothing
            nl, np = list_intervals
            nl >= 1 && 0 <= np <= nl || throw(ArgumentError("list_intervals = (nstlist ≥ 1, 0 ≤ nstprune ≤ nstlist)."))
        end
        st.cluster_config = merge(cfg, (prune_buffer=pb, intervals=list_intervals === nothing ? (0, 0) : Tuple(Int.(list_intervals))))
        if !(st.nbh isa NeighborLists.AllPairsNeighborMatrix)
            st.nbh = NeighborLists.AllPairsNeighborMatrix{T}(T(cfg.skin), Int32(length(st.rx)), Int32(3))
        end
        st.nonbonded_backend = :cluster
        st.cluster = nothing
    else
        cfg = st.cluster_config
        if st.nonbonded_backend === :cluster && cfg !== nothing && cfg.dense
            nbh = NeighborLists._alloc_neighbor_matrix(T, length(st.rx), 3, st.box3, T(cfg.cutoff), T(cfg.skin), cfg.cap)
            nbh.target_interval = cfg.target_interval
            st.nbh = nbh
        end
        st.nonbonded_backend = :ell
        st.cluster = nothing
        st.cluster_config = nothing
    end
    invalidate_forces!(st)
    return st
end

"""
    nonbonded_backend(st) -> Symbol

The short-range traversal of `st`: `:ell` or `:cluster`.
"""
nonbonded_backend(st::SimulationState) = st.nonbonded_backend

function _cluster_supported(st::SimulationState)
    _is_3d(st) || throw(ArgumentError("The cluster-pair backend supports 3D states only."))
    st.external_potential === nothing ||
        throw(ArgumentError("External potentials replace all internal forces; the cluster-pair backend does not apply."))
    # collision counting: the device engine keeps contact rows for it
    st.nb_kind in (NB_KIND_LJ, NB_KIND_WCA, NB_KIND_NONE) ||
        throw(ArgumentError("The cluster-pair backend supports LJ, WCA or no vdW term; soft repulsion runs on the :ell backend (without electrostatics)."))
    return nothing
end

# LJ description as a type table: pair tables directly, uniform LJ as a
# constant table over all type ids in use.
# WCA (charged WCA systems run on the cluster backend) is LJ truncated at its cutoff
# and shifted by +ε: the table's energy shift is −ε (the kernel subtracts it).
function _cluster_lj_table(st::SimulationState{T}) where {T}
    st.nb_kind in (NB_KIND_LJ, NB_KIND_WCA) || return nothing, 0.0
    wca = st.nb_kind == NB_KIND_WCA
    if st.sigma_particle !== nothing   # per-particle σ: rc = factor·(σi + σj)/2
        f = Float64(st.rcut_factor); e = Float64(st.pair_lj.ϵ)
        esh = wca ? -e : _lj_shifted(st) ? 4e * (f^-12 - f^-6) : 0.0
        return (mixed=(epsilon=e, factor=f, sigma=st.sigma_particle, energy_shift=esh), ntypes=1),
               f * Float64(maximum(st.sigma_particle))
    end
    if st.sigma_pair !== nothing
        sig = Array(st.sigma_pair); eps = Array(st.epsilon_pair); rc = Array(st.rcut_pair)
    else
        n = max(1, Int(maximum(st.typeid)))
        p = st.pair_lj
        rcu = wca ? sqrt(1.2599211) * Float64(p.σ) : Float64(p.rcut)   # the engine's uniform WCA cutoff
        sig = fill(Float64(p.σ), n, n); eps = fill(Float64(p.ϵ), n, n); rc = fill(rcu, n, n)
    end
    rmax = maximum(Float64(rc[a, b]) for a in axes(rc, 1), b in axes(rc, 2) if eps[a, b] != 0; init=0.0)
    return (table=ClusterPairs.lj_type_table(sig, eps, rc; T=T, shift=_lj_shifted(st), wca), ntypes=size(sig, 1)), rmax
end

# The box is not part of the key: a box change (set_box!, barostats) only
# rebuilds the list, which `update_list!` does when the box differs.
function _cluster_source(st::SimulationState)
    return (st.electrostatics, st.nb_kind, st.pair_lj, st.sigma_pair, st.epsilon_pair, st.rcut_pair,
            st.cluster_config, st.vdw_modifier, st.bonds, st.exclude_bonded, st.typeid, st.sigma_particle, st.rcut_factor)
end

_same_source(a, b) = length(a) == length(b) && all(x -> x[1] === x[2] || (isbits(x[1]) && x[1] == x[2]), zip(a, b))

# Engine for the current state configuration (rebuilt when the solver, the
# vdW parameters' arrays, the box or the backend settings change).
function _cluster_engine!(st::SimulationState{T}) where {T}
    src = _cluster_source(st)
    eng = st.cluster
    if eng !== nothing && _same_source(eng.source, src)
        return eng
    end
    _cluster_supported(st)
    cfg = st.cluster_config
    lj, rlj = _cluster_lj_table(st)
    s = st.electrostatics
    coulomb = nothing
    exceptions = nothing
    selfc = 0.0
    rcoul = 0.0
    if s !== nothing
        rcoul = s.spec.real_cutoff
        coulomb = ClusterPairs.coulomb_polynomial(s.spec.split, Float64(s.C); T=T, cutoff=rcoul,
                                                  eshift=Float64(s.eshift))
        selfc = Float64(s.selfc)
        length(s.exj) > 0 && (exceptions = (s.exo, s.exj, s.exs))
    end
    rmax = max(rlj, rcoul)
    rmax > 0 || throw(ArgumentError("The cluster-pair backend needs a vdW term or an electrostatics solver."))
    # Bond exclusions of the vdW term: one tile mask for the bonded pairs and the
    # Coulomb exceptions, with corrections for what the shared mask removes beyond them.
    lj_excluded = false
    corrections = nothing
    if lj !== nothing && st.bonds !== nothing && st.exclude_bonded
        exceptions, corrections = ClusterPairs.bond_exclusion_rows(T, length(st.rx), _bond_pairs(st.bonds, _slot_of!(st)),
            s === nothing ? () : _exception_pairs(s); coulomb=s !== nothing, typ=st.typeid)
        lj_excluded = true
    end
    st.cluster = ClusterPairs.ClusterPairEngine(T, length(st.rx), st.box3; lj=lj, coulomb=coulomb, selfc=selfc,
                                                rmax=rmax, skin=cfg.skin, prune_buffer=cfg.prune_buffer,
                                                exceptions=exceptions, lj_excluded=lj_excluded,
                                                corrections=corrections, source=src)
    nl, np = get(cfg, :intervals, (0, 0))
    st.cluster.list.nstlist = Int32(nl); st.cluster.list.nstprune = Int32(np)
    return st.cluster
end

# Host pair lists (storage indices) of the bond list and of the solver's exception rows.
function _bond_pairs(b, slot_of)
    idx = Array(b.index); flat = Array(b.flat); cnt = Array(b.counts)
    ids = [(i, Int(flat[idx[i] + k])) for i in eachindex(cnt) for k in 1:cnt[i] if i < flat[idx[i] + k]]
    slot_of === nothing && return ids
    s = Array(slot_of)     # bond tables hold particle ids
    return [(Int(s[i]), Int(s[j])) for (i, j) in ids]
end
function _exception_pairs(s)
    o = Array(s.exo); js = Array(s.exj); ss = Array(s.exs)
    return [(i, Int(js[p]), Float64(ss[p])) for i in 1:length(o)-1 for p in o[i]:(o[i + 1] - 1) if i < js[p]]
end
