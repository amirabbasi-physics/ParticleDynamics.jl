# =========================
#   Energy minimization (fixed box)
# =========================
#
# Minimizes the declared conservative potential at fixed box: no thermostat,
# barostat, noise, collision counting, heat/work accounting or clock advance.
# Pinned coordinates (a mobile mask and FREEZE_HOLD particles) keep their
# interactions but never move. All O(N) work stays on the device; each
# iteration transfers a handful of Float64 reduction results.

"""
    SteepestDescent(; initial_step, armijo=1e-4, growth=1.2, shrink=0.5)

Backtracking steepest descent. The trial step moves the particle with the
largest force by `h` (length units, `h ≤ max_displacement`) along the force;
a trial is accepted when `U_trial ≤ U - armijo·h·Σ|F|²/Fmax`, otherwise `h` is
multiplied by `shrink` and retried from the same accepted coordinates.
"""
Base.@kwdef struct SteepestDescent
    initial_step::Float64
    armijo::Float64 = 1e-4
    growth::Float64 = 1.2
    shrink::Float64 = 0.5
end

"""
    FIRE(; dt_initial, dt_max, fictitious_mass=1.0, alpha0=0.1, f_inc=1.1,
         f_dec=0.5, f_alpha=0.99, n_min=5, restore_best=true)

Velocity-Verlet FIRE with one fictitious mass for all mobile coordinates.
`dt_initial`, `dt_max` are pseudotimes in the caller's time unit
(time = L·√(M/E)); the dimensionless defaults follow Bitzek et al.
"""
Base.@kwdef struct FIRE
    dt_initial::Float64
    dt_max::Float64
    fictitious_mass::Float64 = 1.0
    alpha0::Float64 = 0.1
    f_inc::Float64 = 1.1
    f_dec::Float64 = 0.5
    f_alpha::Float64 = 0.99
    n_min::Int = 5
    restore_best::Bool = true
end

"""
    MinimizationResult

Outcome of [`minimize!`](@ref). `converged` is true only when the maximum
mobile force reached `force_tolerance`; `reason` is one of `:force_tolerance`,
`:max_iterations`, `:max_evaluations`, `:step_too_small`,
`:line_search_failed`, `:invalid_initial_configuration`,
`:nonfinite_evaluation` or `:no_mobile_coordinates`. Energies are Float64
whole-system totals; forces refer to mobile coordinates of the final state.
"""
struct MinimizationResult
    converged::Bool
    reason::Symbol
    iterations::Int
    evaluations::Int
    initial_energy::Float64
    final_energy::Float64
    best_energy::Float64
    max_force::Float64
    rms_force::Float64
    last_step::Float64
    elapsed::Float64
    message::String
end

function Base.show(io::IO, r::MinimizationResult)
    print(io, "MinimizationResult(", r.converged ? "converged" : "not converged", ", reason=:", r.reason,
          ", iterations=", r.iterations, ", evaluations=", r.evaluations,
          ", U: ", r.initial_energy, " → ", r.final_energy, ", max|F|=", r.max_force, ")")
end

const _MIN_THREADS = 256
const _MIN_MAX_BLOCKS = 512

mutable struct MinimizerWorkspace{T<:AbstractFloat}
    mask::CuVector{UInt8}
    nmobile::Int
    sx::CuVector{T}; sy::CuVector{T}; sz::CuVector{T}            # saved positions
    sxu::CuVector{T}; syu::CuVector{T}; szu::CuVector{T}         # saved unwrapped positions
    px::CuVector{T}; py::CuVector{T}; pz::CuVector{T}            # accepted forces
    ux::CuVector{T}; uy::CuVector{T}; uz::CuVector{T}            # FIRE velocities
    bx::CuVector{T}; by::CuVector{T}; bz::CuVector{T}            # best positions
    bxu::CuVector{T}; byu::CuVector{T}; bzu::CuVector{T}
    partials::CuMatrix{Float64}
    out::CuVector{Float64}
    host::Vector{Float64}
    flag::CuVector{Int32}
    # last reduction: E, Σ|F|², max|F|², Σ u·F, Σ|u|², max|u|²
    E::Float64; fsq::Float64; fmax2::Float64; P::Float64; usq::Float64; umax2::Float64
    message::String
end

function MinimizerWorkspace(st::SimulationState{T}, mask::CuVector{UInt8}) where {T<:AbstractFloat}
    N = length(st.rx)
    z() = CUDA.zeros(T, N)
    D3 = _is_3d(st)
    zz() = D3 ? CUDA.zeros(T, N) : CUDA.zeros(T, 0)
    uw = st.rx_unwrap !== nothing
    zu() = uw ? CUDA.zeros(T, N) : CUDA.zeros(T, 0)
    zzu() = (uw && D3) ? CUDA.zeros(T, N) : CUDA.zeros(T, 0)
    nmobile = Int(CUDA.mapreduce(x -> Int(x != 0), +, mask))
    return MinimizerWorkspace{T}(mask, nmobile,
        z(), z(), zz(), zu(), zu(), zzu(),
        z(), z(), zz(), z(), z(), zz(),
        z(), z(), zz(), zu(), zu(), zzu(),
        CUDA.zeros(Float64, _MIN_MAX_BLOCKS, 6), CUDA.zeros(Float64, 6), zeros(Float64, 6),
        CUDA.zeros(Int32, 1),
        NaN, NaN, NaN, NaN, NaN, NaN, "")
end

# ---------------------------- device kernels ----------------------------

# Wrap into [-L/2, L/2) only when needed: re-wrapping an inside coordinate is
# not exact in floating point and would move pinned/converged coordinates.
@inline function _wrap_coord(x::T, L::T) where {T}
    h = T(0.5) * L
    (x >= -h && x < h) && return x
    y = x + h
    y -= floor(y / L) * L
    return y - h
end

function _min_partials_kernel!(partials, fx, fy, fz, E, ux, uy, uz, mask, ::Val{D}, ::Val{WITHU}) where {D,WITHU}
    shm = CuStaticSharedArray(Float64, (_MIN_THREADS, 6))
    tid = threadIdx().x
    N = length(fx)
    e = 0.0; fsq = 0.0; fmax2 = 0.0; p = 0.0; usq = 0.0; umax2 = 0.0
    i = (blockIdx().x - 1) * blockDim().x + tid
    stride = gridDim().x * blockDim().x
    @inbounds while i <= N
        e += Float64(E[i])
        if mask[i] != 0
            f1 = Float64(fx[i]); f2 = Float64(fy[i]); f3 = D == 3 ? Float64(fz[i]) : 0.0
            f2i = f1 * f1 + f2 * f2 + f3 * f3
            fsq += f2i; fmax2 = max(fmax2, f2i)
            if WITHU
                u1 = Float64(ux[i]); u2 = Float64(uy[i]); u3 = D == 3 ? Float64(uz[i]) : 0.0
                u2i = u1 * u1 + u2 * u2 + u3 * u3
                p += u1 * f1 + u2 * f2 + u3 * f3
                usq += u2i; umax2 = max(umax2, u2i)
            end
        end
        i += stride
    end
    @inbounds begin
        shm[tid, 1] = e; shm[tid, 2] = fsq; shm[tid, 3] = fmax2
        shm[tid, 4] = p; shm[tid, 5] = usq; shm[tid, 6] = umax2
    end
    sync_threads()
    s = _MIN_THREADS ÷ 2
    while s > 0
        if tid <= s
            @inbounds begin
                shm[tid, 1] += shm[tid + s, 1]; shm[tid, 2] += shm[tid + s, 2]
                shm[tid, 3] = max(shm[tid, 3], shm[tid + s, 3])
                shm[tid, 4] += shm[tid + s, 4]; shm[tid, 5] += shm[tid + s, 5]
                shm[tid, 6] = max(shm[tid, 6], shm[tid + s, 6])
            end
        end
        sync_threads()
        s >>= 1
    end
    if tid == 1
        @inbounds for k in 1:6
            partials[blockIdx().x, k] = shm[1, k]
        end
    end
    return
end

function _min_finish_kernel!(out, partials, nb::Int)
    if threadIdx().x == 1 && blockIdx().x == 1
        @inbounds begin
            a1 = 0.0; a2 = 0.0; a3 = 0.0; a4 = 0.0; a5 = 0.0; a6 = 0.0
            for b in 1:nb
                a1 += partials[b, 1]; a2 += partials[b, 2]; a3 = max(a3, partials[b, 3])
                a4 += partials[b, 4]; a5 += partials[b, 5]; a6 = max(a6, partials[b, 6])
            end
            out[1] = a1; out[2] = a2; out[3] = a3; out[4] = a4; out[5] = a5; out[6] = a6
        end
    end
    return
end

function _min_reduce!(ws::MinimizerWorkspace, st::SimulationState, withu::Bool)
    N = length(st.rx)
    D = _is_3d(st) ? 3 : 2
    fz = D == 3 ? st.fz : st.fx
    uz = D == 3 ? ws.uz : ws.ux
    nb = min(cld(N, _MIN_THREADS), _MIN_MAX_BLOCKS)
    CUDA.@cuda threads=_MIN_THREADS blocks=nb _min_partials_kernel!(ws.partials, st.fx, st.fy, fz, st.Epot,
        ws.ux, ws.uy, uz, ws.mask, Val(D), Val(withu))
    CUDA.@cuda threads=1 blocks=1 _min_finish_kernel!(ws.out, ws.partials, nb)
    copyto!(ws.host, ws.out)
    ws.E, ws.fsq, ws.fmax2, ws.P, ws.usq, ws.umax2 = ws.host
    return ws
end

# r = s + scale·p on mobile coordinates (pinned ones are restored from s).
function _min_displace_kernel!(rx, ry, rz, rxu, ryu, rzu, sx, sy, sz, sxu, syu, szu,
                               px, py, pz, mask, scale, L, ::Val{D}, ::Val{UW}) where {D,UW}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        if mask[i] != 0
            dx = scale * px[i]; dy = scale * py[i]
            rx[i] = _wrap_coord(sx[i] + dx, L[1]); ry[i] = _wrap_coord(sy[i] + dy, L[2])
            if UW
                rxu[i] = sxu[i] + dx; ryu[i] = syu[i] + dy
            end
            if D == 3
                dz = scale * pz[i]
                rz[i] = _wrap_coord(sz[i] + dz, L[D])
                UW && (rzu[i] = szu[i] + dz)
            end
        else
            # pinned: restore the saved coordinates bit for bit
            rx[i] = sx[i]; ry[i] = sy[i]
            UW && (rxu[i] = sxu[i]; ryu[i] = syu[i])
            if D == 3
                rz[i] = sz[i]
                UW && (rzu[i] = szu[i])
            end
        end
    end
    return
end

# u ← a·u + b·F on mobile coordinates; pinned velocities stay zero.
function _min_axpby_kernel!(ux, uy, uz, fx, fy, fz, mask, a, b, ::Val{D}) where {D}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(ux) && return
    @inbounds begin
        if mask[i] != 0
            ux[i] = a * ux[i] + b * fx[i]; uy[i] = a * uy[i] + b * fy[i]
            D == 3 && (uz[i] = a * uz[i] + b * fz[i])
        else
            ux[i] = zero(a); uy[i] = zero(a)
            D == 3 && (uz[i] = zero(a))
        end
    end
    return
end

# Exact coincidence check along stored neighbor rows (pair kernels skip r² = 0).
function _min_overlap_ell_kernel!(flag, rx, ry, rz, flat, counts, ::Val{D}) where {D}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    i > N && return
    @inbounds for t in 0:(Int(counts[i]) - 1)
        j = flat[t * N + i]
        same = rx[i] == rx[j] && ry[i] == ry[j] && (D == 2 || rz[i] == rz[j])
        same && CUDA.atomic_min!(pointer(flag, 1), Int32(i))
    end
    return
end

function _min_overlap_allpairs_kernel!(flag, rx, ry, rz, ::Val{D}) where {D}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    i > N && return
    @inbounds for j in (i + 1):N
        same = rx[i] == rx[j] && ry[i] == ry[j] && (D == 2 || rz[i] == rz[j])
        same && CUDA.atomic_min!(pointer(flag, 1), Int32(i))
    end
    return
end

# ---------------------------- host helpers ----------------------------

@inline _launch1d(N) = (threads = min(256, N); (threads, cld(N, threads)))

function _min_box(st::SimulationState{T}) where {T}
    return _is_3d(st) ? (st.box3::Definitions.Box3{T}) : (st.box2::Definitions.Box2{T})
end

function _min_displace!(st::SimulationState{T}, ws::MinimizerWorkspace{T}, p, scale) where {T}
    D = _is_3d(st) ? 3 : 2
    uw = st.rx_unwrap !== nothing
    px, py, pz = p
    rz = D == 3 ? st.rz : st.rx
    rxu = uw ? st.rx_unwrap : st.rx; ryu = uw ? st.ry_unwrap : st.ry
    rzu = (uw && D == 3) ? st.rz_unwrap : rxu
    sz = D == 3 ? ws.sz : ws.sx
    sxu = uw ? ws.sxu : ws.sx; syu = uw ? ws.syu : ws.sy; szu = (uw && D == 3) ? ws.szu : sxu
    pzz = D == 3 ? pz : px
    threads, blocks = _launch1d(length(st.rx))
    CUDA.@cuda threads=threads blocks=blocks _min_displace_kernel!(st.rx, st.ry, rz, rxu, ryu, rzu,
        ws.sx, ws.sy, sz, sxu, syu, szu, px, py, pzz, ws.mask, T(scale), _min_box(st), Val(D), Val(uw))
    invalidate_forces!(st)
    return nothing
end

function _min_axpby!(st::SimulationState{T}, ws::MinimizerWorkspace{T}, a, b) where {T}
    D = _is_3d(st) ? 3 : 2
    uz = D == 3 ? ws.uz : ws.ux; fz = D == 3 ? st.fz : st.fx
    threads, blocks = _launch1d(length(st.rx))
    CUDA.@cuda threads=threads blocks=blocks _min_axpby_kernel!(ws.ux, ws.uy, uz, st.fx, st.fy, fz,
        ws.mask, T(a), T(b), Val(D))
    return nothing
end

_copy3!(dst, src, d3::Bool) = (copyto!(dst[1], src[1]); copyto!(dst[2], src[2]); d3 && copyto!(dst[3], src[3]); nothing)

_positions(st) = (st.rx, st.ry, st.rz)
_unwrapped(st) = (st.rx_unwrap, st.ry_unwrap, st.rz_unwrap)
_forces(st) = (st.fx, st.fy, st.fz)

function _min_save!(st::SimulationState, ws::MinimizerWorkspace)
    d3 = _is_3d(st)
    _copy3!((ws.sx, ws.sy, ws.sz), _positions(st), d3)
    st.rx_unwrap === nothing || _copy3!((ws.sxu, ws.syu, ws.szu), _unwrapped(st), d3)
    return nothing
end

function _min_restore!(st::SimulationState, ws::MinimizerWorkspace)
    d3 = _is_3d(st)
    _copy3!(_positions(st), (ws.sx, ws.sy, ws.sz), d3)
    st.rx_unwrap === nothing || _copy3!(_unwrapped(st), (ws.sxu, ws.syu, ws.szu), d3)
    invalidate_forces!(st)
    return nothing
end

# Evaluate forces and energy; domain violations are reported, not thrown. Only
# coordinates change between evaluations (`moved`): on the device engine (an
# evaluation-only engine is created when none owns the state), whose
# evaluations never reorder storage, so the slot-indexed workspace stays valid.
function _min_evaluate!(st::SimulationState, ws::MinimizerWorkspace, spring::Bool; reduce::Bool=true)
    try
        evaluate_current_configuration!(st, true; freeze_spring=spring, moved=true, create=true)
    catch err
        err isa DomainError || rethrow()
        invalidate_forces!(st)
        ws.message = sprint(showerror, err)
        return false
    end
    reduce || return true
    _min_reduce!(ws, st, false)
    return _min_finite(ws)
end

function _min_finite(ws::MinimizerWorkspace)
    ok = isfinite(ws.E) && isfinite(ws.fsq) && isfinite(ws.P) && isfinite(ws.usq)
    ok || (ws.message = "nonfinite energy or force")
    return ok
end

function _min_overlap_index(st::SimulationState, ws::MinimizerWorkspace)
    fill!(ws.flag, typemax(Int32))
    D = _is_3d(st) ? 3 : 2
    rz = D == 3 ? st.rz : st.rx
    N = length(st.rx)
    threads, blocks = _launch1d(N)
    nbh = st.nbh
    rows = _engine_neighbor_rows(st.engine)          # the engine's list when it owns the state
    if rows !== nothing
        CUDA.@cuda threads=threads blocks=blocks _min_overlap_ell_kernel!(ws.flag, st.rx, st.ry, rz,
            rows..., Val(D))
    elseif nbh isa NeighborLists.NeighborMatrix
        CUDA.@cuda threads=threads blocks=blocks _min_overlap_ell_kernel!(ws.flag, st.rx, st.ry, rz,
            nbh.neighbors_flat, nbh.counts, Val(D))
    elseif nbh isa NeighborLists.AllPairsNeighborMatrix
        CUDA.@cuda threads=threads blocks=blocks _min_overlap_allpairs_kernel!(ws.flag, st.rx, st.ry, rz, Val(D))
    else
        return 0   # typed stencil lists: not checked
    end
    i = only(Array(ws.flag))
    return i == typemax(Int32) ? 0 : Int(i)
end

function _min_mask(st::SimulationState, mobile)
    N = length(st.rx)
    host = if mobile === nothing
        ones(UInt8, N)
    elseif mobile isa AbstractVector{Bool} || mobile isa BitVector
        length(mobile) == N || throw(ArgumentError("mobile mask must have length $(N)"))
        UInt8.(collect(mobile))
    elseif mobile isa AbstractVector{<:Integer}
        m = zeros(UInt8, N)
        for i in mobile
            1 <= i <= N || throw(ArgumentError("mobile index $(i) out of range"))
            m[i] = 0x01
        end
        m
    else
        throw(ArgumentError("mobile must be nothing, a Bool mask or a vector of particle indices"))
    end
    mask = CuArray(host)
    if _freeze_active!(st) && st.freeze_mode == FREEZE_HOLD && st.freeze_mask !== nothing
        mask .= mask .* (st.freeze_mask .== 0x00)
    end
    return mask
end

function _min_result(converged, reason, iters, evals, E0, ws, best, step, t0)
    fmax = sqrt(max(ws.fmax2, 0.0))
    rms = ws.nmobile > 0 ? sqrt(max(ws.fsq, 0.0) / ws.nmobile) : 0.0
    return MinimizationResult(converged, reason, iters, evals, E0, ws.E, best, fmax, rms, step,
                              (time_ns() - t0) / 1e9, ws.message)
end

# ---------------------------- drivers ----------------------------

"""
    minimize!(st; method, force_tolerance, max_displacement, max_iterations=10_000,
              max_evaluations=20_000, mobile=nothing, velocities=:zero) -> MinimizationResult

Minimize the potential energy of `st` at fixed box with [`SteepestDescent`](@ref)
or [`FIRE`](@ref). `force_tolerance` (E/L) applies to the largest mobile
force; `max_displacement` (L) caps any coordinate's move per step. `mobile` is
`nothing` (all particles), a Bool mask or a vector of indices; FREEZE_HOLD
particles are always pinned. On return the forces and energies describe the
final coordinates; with `velocities=:zero` (default) physical velocities are
zeroed, with `:preserve` they are left untouched (not optimized).
"""
function minimize!(st::SimulationState{T};
                   method::Union{SteepestDescent,FIRE},
                   force_tolerance::Real,
                   max_displacement::Real,
                   max_iterations::Integer=10_000,
                   max_evaluations::Integer=20_000,
                   mobile=nothing,
                   velocities::Symbol=:zero) where {T<:AbstractFloat}
    st.rigid_water === nothing || throw(ArgumentError("Constrained minimization is not implemented"))
    force_tolerance > 0 && isfinite(force_tolerance) || throw(ArgumentError("force_tolerance must be positive"))
    max_displacement > 0 && isfinite(max_displacement) || throw(ArgumentError("max_displacement must be positive"))
    velocities in (:zero, :preserve) || throw(ArgumentError("velocities must be :zero or :preserve"))
    st.external_potential === nothing ||
        throw(ArgumentError("minimize! does not support external potential providers"))
    t0 = time_ns()
    # Invalid forces may signal changes the engine cannot see (attachments): settle and
    # drop the stepping engine; the minimizer evaluates with evaluation-only engines.
    st.force_valid || _drop_engine!(st.engine, st)
    spring = _freeze_active!(st) && st.freeze_mode == FREEZE_SPRING
    ws = MinimizerWorkspace(st, _min_mask(st, mobile))
    result = if ws.nmobile == 0
        _min_evaluate!(st, ws, spring)
        _min_result(false, :no_mobile_coordinates, 0, 1, ws.E, ws, ws.E, 0.0, t0)
    else
        _minimize!(st, ws, method, spring, Float64(force_tolerance), Float64(max_displacement),
                   Int(max_iterations), Int(max_evaluations), t0)
    end
    velocities === :zero && (fill!(st.vx, zero(T)); fill!(st.vy, zero(T)); st.vz === nothing || fill!(st.vz, zero(T)))
    _collisions_snapshot!(st)   # contact flags are re-initialized from it at the next engine rebuild
    return result
end

function _min_initial!(st, ws, spring)
    coords_ok = all(c -> c === nothing || CUDA.mapreduce(isfinite, &, c), _positions(st))
    if !coords_ok
        ws.message = "nonfinite coordinates"
        return false
    end
    _min_evaluate!(st, ws, spring) || return false
    # Pair kernels skip r² = 0; bonded overlaps are caught by the bonded status.
    (st.nb_kind == NB_KIND_NONE && st.electrostatics === nothing) && return true
    i = _min_overlap_index(st, ws)
    if i != 0
        ws.message = "particle $(i) exactly overlaps a neighbor"
        return false
    end
    return true
end

function _minimize!(st::SimulationState{T}, ws, m::SteepestDescent, spring, ftol, maxdisp,
                    max_iter, max_eval, t0) where {T}
    0 < m.armijo < 1 && 0 < m.shrink < 1 && m.growth >= 1 && m.initial_step > 0 ||
        throw(ArgumentError("invalid SteepestDescent parameters"))
    d3 = _is_3d(st)
    evals = 1
    _min_initial!(st, ws, spring) ||
        return _min_result(false, :invalid_initial_configuration, 0, evals, ws.E, ws, ws.E, 0.0, t0)
    E0 = ws.E
    U = ws.E; fsq = ws.fsq; fmax = sqrt(ws.fmax2)
    h = min(m.initial_step, maxdisp)
    hmin = 4 * eps(T) * maximum(_min_box(st))
    iters = 0; last = 0.0
    p = (ws.px, ws.py, ws.pz)
    while true
        fmax <= ftol && return _min_result(true, :force_tolerance, iters, evals, E0, ws, U, last, t0)
        iters >= max_iter && return _min_result(false, :max_iterations, iters, evals, E0, ws, U, last, t0)
        _min_save!(st, ws)
        _copy3!(p, _forces(st), d3)
        accepted = false
        while !accepted
            if evals >= max_eval || h < hmin
                reason = evals >= max_eval ? :max_evaluations : :step_too_small
                _min_restore!(st, ws)
                evals += 1
                _min_evaluate!(st, ws, spring)
                return _min_result(false, reason, iters, evals, E0, ws, U, last, t0)
            end
            _min_displace!(st, ws, p, h / fmax)
            evals += 1
            ok = _min_evaluate!(st, ws, spring)
            if ok && ws.E <= U - m.armijo * h * fsq / fmax
                accepted = true
                U = ws.E; fsq = ws.fsq; fmax = sqrt(ws.fmax2)
                last = h
                h = min(h * m.growth, maxdisp)
            else
                h *= m.shrink
            end
        end
        iters += 1
    end
end

function _minimize!(st::SimulationState{T}, ws, m::FIRE, spring, ftol, maxdisp,
                    max_iter, max_eval, t0) where {T}
    m.dt_initial > 0 && m.dt_max >= m.dt_initial && m.fictitious_mass > 0 &&
        0 < m.f_dec < 1 && m.f_inc >= 1 && 0 < m.alpha0 < 1 && 0 < m.f_alpha <= 1 && m.n_min >= 0 ||
        throw(ArgumentError("invalid FIRE parameters"))
    d3 = _is_3d(st)
    uw = st.rx_unwrap !== nothing
    evals = 1
    fill!(ws.ux, zero(T)); fill!(ws.uy, zero(T)); d3 && fill!(ws.uz, zero(T))
    _min_initial!(st, ws, spring) ||
        return _min_result(false, :invalid_initial_configuration, 0, evals, ws.E, ws, ws.E, 0.0, t0)
    E0 = ws.E
    best = ws.E
    m.restore_best && (_copy3!((ws.bx, ws.by, ws.bz), _positions(st), d3);
                       uw && _copy3!((ws.bxu, ws.byu, ws.bzu), _unwrapped(st), d3))
    invm = 1 / m.fictitious_mass
    h = m.dt_initial
    hmin = m.dt_initial * 1e-8
    a = m.alpha0
    npos = 0
    iters = 0; last = 0.0
    U = ws.E; fmax = sqrt(ws.fmax2); umax = 0.0
    zero_u!() = (fill!(ws.ux, zero(T)); fill!(ws.uy, zero(T)); d3 && fill!(ws.uz, zero(T)))

    finish(conv, reason) = begin
        if !conv && m.restore_best && best < U
            _copy3!(_positions(st), (ws.bx, ws.by, ws.bz), d3)
            uw && _copy3!(_unwrapped(st), (ws.bxu, ws.byu, ws.bzu), d3)
            invalidate_forces!(st)
            evals += 1
            _min_evaluate!(st, ws, spring)
        elseif !st.force_valid || !st.energy_valid
            evals += 1
            _min_evaluate!(st, ws, spring)
        end
        _min_result(conv, reason, iters, evals, E0, ws, best, last, t0)
    end

    while true
        fmax <= ftol && return finish(true, :force_tolerance)
        iters >= max_iter && return finish(false, :max_iterations)
        evals >= max_eval && return finish(false, :max_evaluations)
        h < hmin && return finish(false, :step_too_small)
        # 1. one global step reduction so that no coordinate moves more than maxdisp
        acc = fmax * invm
        bound = h * umax + 0.5 * h * h * acc
        if bound > maxdisp
            h = acc > 0 ? (-umax + sqrt(umax * umax + 2 * acc * maxdisp)) / acc : maxdisp / umax
        end
        # 2. save, first kick, drift
        _min_save!(st, ws)
        _copy3!((ws.px, ws.py, ws.pz), _forces(st), d3)
        _min_axpby!(st, ws, 1, 0.5 * h * invm)
        _min_displace!(st, ws, (ws.ux, ws.uy, ws.uz), h)
        # 3. new forces
        evals += 1
        ok = _min_evaluate!(st, ws, spring; reduce=false)
        if ok
            _min_axpby!(st, ws, 1, 0.5 * h * invm)
            _min_reduce!(ws, st, true)
            ok = _min_finite(ws)
        end
        if !ok
            # domain recovery: restore the finite state and its forces, stop, shrink
            _min_restore!(st, ws)
            _copy3!(_forces(st), (ws.px, ws.py, ws.pz), d3)
            st.force_valid = true
            zero_u!(); umax = 0.0
            h *= m.f_dec; a = m.alpha0; npos = 0
            ws.E = U
            continue
        end
        iters += 1
        last = h * sqrt(max(ws.umax2, 0.0))
        U = ws.E; fmax = sqrt(ws.fmax2)
        if U < best
            best = U
            m.restore_best && (_copy3!((ws.bx, ws.by, ws.bz), _positions(st), d3);
                               uw && _copy3!((ws.bxu, ws.byu, ws.bzu), _unwrapped(st), d3))
        end
        fmax <= ftol && return finish(true, :force_tolerance)
        # 4./5. FIRE mixing and timestep control
        if ws.P > 0
            ratio = sqrt(ws.usq / ws.fsq)
            _min_axpby!(st, ws, 1 - a, a * ratio)
            # upper bound on the mixed max|u| for the next displacement cap
            umax = (1 - a) * sqrt(max(ws.umax2, 0.0)) + a * ratio * fmax
            npos += 1
            if npos > m.n_min
                h = min(h * m.f_inc, m.dt_max)
                a *= m.f_alpha
            end
        else
            zero_u!(); umax = 0.0
            npos = 0; a = m.alpha0
            h *= m.f_dec
        end
    end
end
