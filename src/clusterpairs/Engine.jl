# =========================
#   Engine: list maintenance and one force evaluation
# =========================

"""
    ClusterPairEngine

Cluster-pair nonbonded engine for one 3D state: the list, the LJ
description (kind, type table or per-type separable factors) and the Coulomb
polynomial. The list is rebuilt when a particle moved more than half the skin
since the build and re-pruned when it moved more than half the prune buffer
since the last prune. `rmax` is the largest interaction cutoff; the list
radius is `rmax + skin` and the prune radius `rmax + prune_buffer`.
"""
mutable struct ClusterPairEngine{T,CP}
    list::ClusterPairList{T}
    ljk::Int
    lj_excluded::Bool
    ljp::CuVector{NTuple{4,T}}
    ntypes::Int32
    s6t::CuVector{T}
    s12t::CuVector{T}
    lrc2u::T
    irc6::T
    coulomb::Bool
    coul::CP
    selfc::T
    rmax::T
    skin::T
    prune_buffer::T
    exo::Union{Nothing,CuVector{Int32}}
    exj::Union{Nothing,CuVector{Int32}}
    exs::Union{Nothing,CuVector{T}}
    fc::CuMatrix{T}
    Ec::CuVector{T}
    Vc::CuMatrix{T}
    source::Any            # identity of the objects the engine was built from
    corrections::Any       # nothing or bond-exclusion corrections (see `bond_exclusion_rows`)
    vtrace::CuVector{Float64}   # virial-trace accumulator Σ r·f of trace passes (barostat sampling)
end

"""
    ClusterPairEngine(T, N, box; lj, coulomb, selfc, rmax, skin, prune_buffer, exceptions, lj_excluded, corrections, source)

`lj` is `nothing` or a named tuple `(table=Vector{NTuple{4}}, ntypes)` built by
[`lj_type_table`](@ref); a table that factorises as `c6_ab = s6_a s6_b`,
`c12_ab = s12_a s12_b` with one cutoff runs the separable kernel. `coulomb`
is `nothing` or a [`CoulombPolynomial`](@ref). `exceptions` is `nothing` or
the `(exo, exj, exs)` CSR rows in storage indices. `lj_excluded` drops the
Lennard-Jones term of every pair in those rows; `corrections` (from
`bond_exclusion_rows`) then restores what the shared tile mask removes
beyond the bonded pairs.
"""
function ClusterPairEngine(::Type{T}, N::Integer, box::NTuple{3,<:Real}; lj=nothing, coulomb=nothing,
                           selfc::Real=0.0, rmax::Real, skin::Real, prune_buffer::Real=0.5 * skin,
                           exceptions=nothing, lj_excluded::Bool=false, corrections=nothing,
                           source=nothing) where {T}
    skin >= 0 && prune_buffer >= 0 && prune_buffer <= skin ||
        throw(ArgumentError("cluster pairs need 0 ≤ prune_buffer ≤ skin."))
    ljk = LJ_NONE
    ljp = CuVector{NTuple{4,T}}(undef, 1)
    ntypes = Int32(1)
    s6t = CUDA.zeros(T, 1); s12t = CUDA.zeros(T, 1)
    lrc2u = zero(T); irc6 = zero(T)
    if lj !== nothing && haskey(lj, :mixed)
        # per-particle σ: s6t holds σ by particle; one entry (4ε, –, factor², shift)
        m = lj.mixed
        ljk = LJ_MIXED
        s6t = m.sigma
        f = Float64(m.factor); e = Float64(m.epsilon)
        esh = Float64(m.energy_shift)   # potential shift, −ε for WCA, or 0
        ljp = CuVector{NTuple{4,T}}([NTuple{4,T}((4e, 0.0, f^2, esh))])
    elseif lj !== nothing
        table = lj.table
        n = Int(lj.ntypes)
        ntypes = Int32(n)
        sep = _separable(table, n)
        if sep !== nothing
            ljk = LJ_SEPARABLE
            s6t = CuVector{T}(sep[1]); s12t = CuVector{T}(sep[2])
            lrc2u = T(sep[3])
            # energy shift only when the table carries one (potential-shift modifier)
            irc6 = (sep[3] > 0 && any(x -> x[4] != 0, table)) ? T(1 / sep[3]^3) : zero(T)
        else
            ljk = n * n <= MAX_SHARED_TABLE ? LJ_TABLE_SHARED : LJ_TABLE_GLOBAL
        end
        ljp = CuVector{NTuple{4,T}}(NTuple{4,T}.(table))
    end
    coul = coulomb === nothing ? CoulombPolynomial{T,1,1,true}(zero(T), zero(T), one(T), zero(T), false, (zero(T),), (zero(T),)) :
           coulomb
    exo, exj, exs = exceptions === nothing ? (nothing, nothing, nothing) : exceptions
    list = ClusterPairList(N, box, T)
    list.prune_only = PRUNE_ONLY[]
    return ClusterPairEngine{T,typeof(coul)}(list, ljk, lj_excluded, ljp, ntypes, s6t, s12t, lrc2u, irc6,
                                             coulomb !== nothing, coul, T(selfc), T(rmax), T(skin), T(prune_buffer),
                                             exo, exj, exs, CuMatrix{T}(undef, 0, 3), CuVector{T}(undef, 0),
                                             CuMatrix{T}(undef, 0, 6), source, corrections, CUDA.zeros(Float64, 1))
end

# (√c6, √c12, cutoff²) per type when the table factorises with one cutoff for
# all pairs that interact; `nothing` otherwise.
function _separable(table, n::Int)
    c6 = [Float64(table[(a - 1) * n + b][1]) for a in 1:n, b in 1:n]
    c12 = [Float64(table[(a - 1) * n + b][2]) for a in 1:n, b in 1:n]
    rc2 = [Float64(table[(a - 1) * n + b][3]) for a in 1:n, b in 1:n]
    any(<(0), [c6[a, a] for a in 1:n]) && return nothing
    any(<(0), [c12[a, a] for a in 1:n]) && return nothing
    s6 = sqrt.([c6[a, a] for a in 1:n]); s12 = sqrt.([c12[a, a] for a in 1:n])
    rcs = unique(rc2[a, b] for a in 1:n, b in 1:n if c6[a, b] != 0 || c12[a, b] != 0)
    length(rcs) <= 1 || return nothing
    for a in 1:n, b in 1:n
        isapprox(c6[a, b], s6[a] * s6[b]; rtol=1e-6, atol=0.0) || return nothing
        isapprox(c12[a, b], s12[a] * s12[b]; rtol=1e-6, atol=0.0) || return nothing
    end
    return s6, s12, isempty(rcs) ? 0.0 : rcs[1]
end

function _pack!(eng::ClusterPairEngine{T}, rx, ry, rz, q, typ) where {T}
    L = eng.list
    th, bl = _launch1(L.nslot)
    charged = eng.coulomb && q !== nothing
    qq = charged ? q : L.qs
    @cuda threads=th blocks=bl _pack_kernel!(L.xs, L.ys, L.zs, L.qs, L.ts, L.s6, L.s12, L.perm, rx, ry, rz, qq, typ,
                                             eng.s6t, eng.s12t, L.xw0, L.yw0, L.zw0, L.rx0, L.ry0, L.rz0,
                                             L.geom, Val(charged), Val(eng.ljk == LJ_SEPARABLE), Val(false),
                                             Val(eng.ljk == LJ_MIXED))
    return nothing
end

# Geometry and storage for this box (host boundary, allocates when it changes).
function setup!(eng::ClusterPairEngine{T}, box) where {T}
    L = eng.list
    changed = setup!(L, box, Float64(eng.rmax + eng.skin), eng.exj === nothing ? 0 : length(eng.exj))
    if size(eng.fc, 1) != L.nslot
        eng.fc = CUDA.zeros(T, L.nslot, 3)
        eng.Ec = CUDA.zeros(T, L.nslot)
        eng.Vc = CUDA.zeros(T, L.nslot, 6)
    end
    return changed
end

function _rebuild!(eng::ClusterPairEngine{T}, rx, ry, rz, box) where {T}
    L = eng.list
    setup!(eng, box)
    n = L.nbuild
    build!(L, rx, ry, rz; exo=eng.exo, exj=eng.exj, exs=eng.exs)
    # staged path: boundary check of the outer-list capacity after every build
    while ensure_capacity!(L)
        build!(L, rx, ry, rz; exo=eng.exo, exj=eng.exj, exs=eng.exs)
    end
    L.nbuild = n + 1                   # one logical build, however often it was resized
    return nothing
end

"""
    update_list!(eng, rx, ry, rz, q, typ, box) -> Symbol

Keep the list valid for the current coordinates and pack them in cluster
order. Returns `:built`, `:pruned` or `:kept`.
"""
function update_list!(eng::ClusterPairEngine{T}, rx, ry, rz, q, typ, box) where {T}
    L = eng.list
    action = :kept
    if L.nbuild == 0 || L.box != T.(box)
        _rebuild!(eng, rx, ry, rz, box)
        action = :built
    else
        action = list_action(L, rx, ry, rz, eng.skin / 2, eng.prune_buffer / 2)
        action === :built && _rebuild!(eng, rx, ry, rz, box)
    end
    _pack!(eng, rx, ry, rz, q, typ)
    if action !== :kept
        request_prune!(L)              # the next force pass prunes the full list
        mark_pruned!(L, rx, ry, rz)
    end
    return action
end

"""
    compute!(eng, rx, ry, rz, q, typ, fx, fy, fz, Epot, V, box, compute_energy)

Overwrite the force slot (and, with energies, `Epot` and the per-particle
virial rows) with the short-range nonbonded forces: Lennard-Jones, real-space
Coulomb including exclusion corrections and the Coulomb self energy.
"""
function compute!(eng::ClusterPairEngine{T}, rx, ry, rz, q, typ, fx, fy, fz, Epot, V, box,
                  compute_energy::Bool) where {T}
    update_list!(eng, rx, ry, rz, q, typ, box)
    compute_forces!(eng, rx, ry, rz, q, fx, fy, fz, Epot, V, compute_energy)
    return nothing
end

"""
    compute_forces!(eng, rx, ry, rz, q, fx, fy, fz, Epot, V, compute_energy)

The force part of [`compute!`](@ref) for a list already updated and packed by
[`update_list!`](@ref) at the current coordinates.
"""
function compute_forces!(eng::ClusterPairEngine{T}, rx, ry, rz, q, fx, fy, fz, Epot, V,
                         compute_energy::Bool) where {T}
    L = eng.list
    if L.prune_pending
        enqueue_pairs!(eng, compute_energy, Val(true))     # whole list, writing the pruned tile masks
        finish_prune!(L)                                   # then compact
    else
        enqueue_pairs!(eng, compute_energy, Val(false))
    end
    enqueue_scatter!(eng, rx, ry, rz, q, fx, fy, fz, Epot, V, compute_energy)
    return nothing
end

# Pair pass over the working list (`Val(false)`) or the whole outer list with
# pruning (`Val(true)`). `skip` > 0: the pass does nothing when `counts[skip]`
# is set on the device (graph steps: the plain pass yields to the pruning pass).
function enqueue_pairs!(eng::ClusterPairEngine{T}, compute_energy::Bool, ::Val{PRUNE}; skip::Integer=0,
                        trace::Bool=false) where {T,PRUNE}
    L = eng.list
    rp2 = (eng.rmax + eng.prune_buffer)^2
    lst = PRUNE ? (L.scio, L.cjf, L.m8f, L.xido) : (L.sciw, L.cjw, L.m8w, L.xidw)
    args = (eng.fc, eng.Ec, eng.Vc, L.xs, L.ys, L.zs, L.qs, L.ts, L.s6, L.s12, lst..., L.masks,
            L.counts, Int32(skip), L.geom, eng.ljp, eng.ntypes, eng.lrc2u, eng.irc6, eng.coul)
    _launch_cluster_kernel!(args, L.m8o, rp2, Val(eng.ljk), Val(eng.lj_excluded), Val(eng.coulomb),
                            Val(compute_energy), Val(true), Val(PRUNE), L.NSmax, eng.vtrace, Val(trace))
    return nothing
end

# Slot buffers → storage order (overwrites the force slot), self energy, the
# scaled/out-of-list exception pairs and the bond-exclusion corrections.
function enqueue_scatter!(eng::ClusterPairEngine{T}, rx, ry, rz, q, fx, fy, fz, Epot, V, compute_energy::Bool;
                          trace::Bool=false, add::Bool=false) where {T}
    L = eng.list
    th, bl = _launch1(L.nslot)
    charged = eng.coulomb && q !== nothing
    @cuda threads=th blocks=bl _scatter_kernel!(fx, fy, fz, Epot, V, eng.fc, eng.Ec, eng.Vc, L.perm,
                                                charged ? q : L.qs, T(0.5) * eng.coul.C * eng.selfc,
                                                Val(compute_energy), Val(charged), Val(add))
    if charged && L.npmax > 0
        th, bl = _launch1(L.npmax)
        @cuda threads=th blocks=bl _scaled_pairs_kernel!(fx, fy, fz, Epot, V, L.pi, L.pj, L.pc, rx, ry, rz, q,
                                                         eng.coul.C, L.geom, L.counts, Val(compute_energy),
                                                         trace ? eng.vtrace : nothing)
    end
    c = eng.corrections
    if c !== nothing
        if c.lj !== nothing
            th, bl = _launch1(length(c.lj[1]))
            mixed = eng.ljk == LJ_MIXED
            @cuda threads=th blocks=bl _lj_restore_kernel!(fx, fy, fz, Epot, V, c.lj..., rx, ry, rz,
                                                           mixed ? eng.s6t : c.typ, eng.ljp, eng.ntypes, L.geom,
                                                           L.counts, Val(compute_energy), Val(mixed))
        end
        if compute_energy && charged && c.coul_shift !== nothing && eng.coul.eshift != zero(T)
            th, bl = _launch1(length(c.coul_shift[1]))
            @cuda threads=th blocks=bl _coulomb_shift_kernel!(Epot, c.coul_shift..., rx, ry, rz, q, eng.coul.C,
                                                              eng.coul.rc2, eng.coul.eshift, L.geom, L.counts)
        end
    end
    return nothing
end

"""
    bond_exclusion_rows(T, N, bonded, exceptions; coulomb, typ) -> (exceptions, corrections)

Exclusion rows and corrections for Lennard-Jones bond exclusions on one tile
mask shared with the Coulomb exceptions. `bonded` are the `(i, j)` pairs whose LJ term
is dropped (storage indices), `exceptions` the Coulomb `(i, j, scale)` pairs. The mask
covers the union; with `lj_excluded` the kernel drops LJ for every masked pair, so

- a Coulomb exception that is not bonded gets its LJ back (`corrections.lj`);
- a bonded pair that is not a Coulomb exception keeps its full Coulomb term: scale 1
  (exact forces) plus the potential shift it would carry as an ordinary pair
  (`corrections.coul_shift`, energy only). Without `coulomb` its scale is 0.

The rows are symmetric CSR in storage indices, as the electrostatics solver's.
"""
function bond_exclusion_rows(::Type{T}, N::Integer, bonded, exceptions; coulomb::Bool, typ) where {T}
    key(i, j) = i < j ? (Int32(i), Int32(j)) : (Int32(j), Int32(i))
    B = Set(key(i, j) for (i, j) in bonded)
    C = Dict(key(i, j) => Float64(s) for (i, j, s) in exceptions)
    pairs = [(k..., s) for (k, s) in C]
    restore = Tuple{Int32,Int32}[]
    for k in B
        haskey(C, k) && continue
        push!(pairs, (k..., coulomb ? 1.0 : 0.0))
        coulomb && push!(restore, k)
    end
    ljback = [k for k in keys(C) if !(k in B)]
    sort!(pairs); sort!(restore); sort!(ljback)
    rows(v) = isempty(v) ? nothing : (CuVector{Int32}(first.(v)), CuVector{Int32}(last.(v)))
    exo, exj, exs = ES._exception_rows(T, Int(N), pairs)
    return (exo, exj, exs), (lj=rows(ljback), coul_shift=rows(restore), typ=typ)
end

# ---------- device-decided list maintenance ----------
# One step = decide → [rebuild] → pack → [pruning pass + compaction] → plain
# pass (skipped after a pruning pass) → scatter, with the bracketed parts as
# conditional graph nodes on `counts[CL_BUILD]` / `counts[CL_PRUNE]`
# (`NeighborLists.flag_conditional_graph`). Same decisions as `update_list!`:
# rebuild when a particle moved more than skin/2 since the build (or on
# request), prune when one moved more than prune_buffer/2 since the last prune
# or after a rebuild.

function _decide_kernel!(counts, flags, nstlist::Int32=Int32(0), nstprune::Int32=Int32(0), steps=nothing,
                        sidx::Int32=Int32(0))
    @inbounds begin
        anyr = reinterpret(Float32, flags[1]); br = reinterpret(Float32, flags[2])
        if nstlist > Int32(0)
            # fixed intervals (statistical buffer, GROMACS-style): rebuild every nstlist and prune every
            # nstprune decisions; requests (setup, box changes) and non-finite motion still rebuild.
            # With `steps`, the schedule follows that step counter (host-scheduled steps: the host
            # launches graphs without a decision between scheduled steps).
            k = counts[CL_DSTEP]; counts[CL_DSTEP] = k + Int32(1)
            steps === nothing || (k = Int32(steps[sidx] % Int64(nstlist * nstprune)))
            build = (k % nstlist == Int32(0)) | (counts[CL_REQUEST] != Int32(0)) | !isfinite(br)
            prune = build | (nstprune > Int32(0) && k % nstprune == Int32(0))
        else
            build = (br > 1.0f0) | (counts[CL_REQUEST] != Int32(0))
            prune = build | (anyr > 1.0f0)
        end
        counts[CL_BUILD] = Int32(build); counts[CL_PRUNE] = Int32(prune)
        counts[CL_REQUEST] = Int32(0)
        counts[CL_NBUILD] += Int32(build); counts[CL_NPRUNE] += Int32(prune)
        flags[1] = UInt32(0); flags[2] = UInt32(0)        # ready for the next step's atomic max
    end
    return
end

"""Device rebuild/prune decision at the current coordinates (no host read). `fixed=false`: the
displacement triggers even with fixed list intervals (evaluations outside the step schedule)."""
function enqueue_decide!(eng::ClusterPairEngine{T}, rx, ry, rz; fixed::Bool=true, step=nothing) where {T}
    L = eng.list
    th, bl = _launch1(L.N)
    @cuda threads=th blocks=bl _displacement_kernel!(L.flags, rx, ry, rz, L.rx0, L.ry0, L.rz0, L.px0, L.py0, L.pz0,
                                                     L.geom, T(eng.skin / 2)^2, T(eng.prune_buffer / 2)^2)
    @cuda threads=1 blocks=1 _decide_kernel!(L.counts, L.flags, fixed ? L.nstlist : Int32(0), L.nstprune, _steps(step)...)
    return nothing
end
_steps(::Nothing) = (nothing, Int32(0))
_steps(step::Tuple) = (step[1], Int32(step[2]))

"""Device rebuild/prune decision from displacement maxima a fused integration kernel
already reduced into `flags` (the rigid-water fast path)."""
function enqueue_decide_only!(eng::ClusterPairEngine; step=nothing)
    L = eng.list
    @cuda threads=1 blocks=1 _decide_kernel!(L.counts, L.flags, L.nstlist, L.nstprune, _steps(step)...)
    return nothing
end

"""Rebuild body (conditional on `counts[CL_BUILD]`)."""
enqueue_build!(eng::ClusterPairEngine, rx, ry, rz) = (build!(eng.list, rx, ry, rz; exo=eng.exo, exj=eng.exj, exs=eng.exs); nothing)

"""Pruning by a distance-only pass (then the plain pass computes forces on the new working list)
instead of a force pass over the whole outer list (default for newly created engines; `false`
restores the fused force-and-prune pass). Its working list is a superset of the fused pass's
(widened radius, see `_prune_only_kernel!`). 10k water NPT, five paired runs: 1.02× faster with
buffered lists, 1.03× with exact lists."""
const PRUNE_ONLY = Ref(true)

"""Pruning body (conditional on `counts[CL_PRUNE]`): prune reference, pruning pass (forces over the
outer list, or distances only with `eng.list.prune_only`), compaction. With distance-only
pruning the caller must subsequently enqueue the plain pair pass before scattering."""
function enqueue_prune!(eng::ClusterPairEngine{T}, rx, ry, rz, compute_energy::Bool; trace::Bool=false,
                        mark::Union{Nothing,Bool}=nothing) where {T}
    L = eng.list
    # Fixed list intervals decide by schedule, so graph steps skip the prune references.
    # Evaluations outside the schedule decide by displacement from them (`enqueue_decide!(…;
    # fixed=false)`) and pass `mark=true`.
    (mark === nothing ? L.nstlist == 0 : mark) && mark_pruned!(L, rx, ry, rz)
    if L.prune_only
        NS = L.NSmax; k = _split_factor(NS)
        @cuda threads=SUPER blocks=NS * k _prune_only_kernel!(L.m8o, L.xs, L.ys, L.zs, L.scio, L.cjf, L.m8f, L.xido, L.masks,
                                                              L.counts, L.geom, T(eng.rmax + eng.prune_buffer)^2, Int32(k))
    else
        enqueue_pairs!(eng, compute_energy, Val(true); trace)
    end
    compact!(L, L.m8o, L.sciw, L.cjw, L.m8w, L.xidw)
    return nothing
end

"""Graph sections and suffix of one device-decided short-range evaluation."""
function graph_sections(eng::ClusterPairEngine, rx, ry, rz, q, typ, fx, fy, fz, Epot, V, compute_energy::Bool)
    L = eng.list
    sections = [(() -> enqueue_decide!(eng, rx, ry, rz), L.counts, CL_BUILD, () -> enqueue_build!(eng, rx, ry, rz)),
                (() -> _pack!(eng, rx, ry, rz, q, typ), L.counts, CL_PRUNE, () -> enqueue_prune!(eng, rx, ry, rz, compute_energy))]
    suffix = () -> (enqueue_pairs!(eng, compute_energy, Val(false); skip=L.prune_only ? 0 : CL_PRUNE);
                    enqueue_scatter!(eng, rx, ry, rz, q, fx, fy, fz, Epot, V, compute_energy))
    return sections, suffix
end

const _SM_COUNT = Ref(0)

# Blocks per super-cluster: enough blocks for about two full waves of
# 64-thread blocks (at most 16 per SM), never more than 32 per super-cluster.
function _split_factor(NS::Integer)
    if _SM_COUNT[] == 0
        _SM_COUNT[] = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    end
    return Int32(clamp(cld(32 * _SM_COUNT[], max(NS, 1)), 1, 32))
end

function _launch_cluster_kernel!(args, m8out, rp2, vlj, vljx, vc, ve, vs, vp, NS, vtr, vt=Val(false))
    NS == 0 && return nothing
    k = _split_factor(NS)
    if k == 1
        @cuda threads=SUPER blocks=NS _cluster_pair_kernel!(args..., k, m8out, rp2, Val(false), vlj, vljx, vc, ve, vs, vp, vtr, vt)
    else
        @cuda threads=SUPER blocks=NS * k _cluster_pair_kernel!(args..., k, m8out, rp2, Val(true), vlj, vljx, vc, ve, vs, vp, vtr, vt)
    end
    return nothing
end

"""
    mesh_order(eng) -> CuVector{Int32}

Particles in cluster order, a spatially sorted traversal for mesh spreading
and gathering.
"""
mesh_order(eng::ClusterPairEngine) = eng.list.order

"""
    statistics(eng) -> NamedTuple

Super-clusters, outer and working list entries, and pair evaluations per
particle of the working list.
"""
function statistics(eng::ClusterPairEngine)
    L = eng.list
    c = Array(L.counts); sw = Array(L.sciw)
    nw = Int(sw[Int(c[CL_NS]) + 1])
    bits = sum(count_ones, Array(L.m8w)[1:2nw]; init=0)
    return (superclusters=Int(c[CL_NS]), outer_entries=Int(c[CL_NE]), entries=nw,
            # staged (host) plus device-decided (graph engine) totals
            pairs_per_particle=32 * bits / max(L.N, 1), builds=L.nbuild + Int(c[CL_NBUILD]),
            prunes=L.nprune + Int(c[CL_NPRUNE]),
            scaled_pairs=Int(c[CL_NP]), outer_capacity=L.necap)
end
