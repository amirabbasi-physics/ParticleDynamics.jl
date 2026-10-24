# Device-resident pressure control.
#
# An isotropic C-rescale event (Bernetti & Bussi 2020, the formulation of
# `SimulationCore.stochastic_cell_rescale!`) runs as device kernels inside the
# captured step graph, every `interval` steps, with no host synchronization:
#
#   1. at the end of every sampled step (forces and positions agree there): the
#      configurational virial trace without the constraint part, minus the
#      intramolecular term Σ_i (r_i − R_M)·F_i of every rigid water (molecular virial),
#      overwritten into the barostat buffer; the step before an event is sampled;
#   2. one thread: P = N kT/V + W/(3V), λ' = λ + h κλ/(2τ)(P − P0 + kT/(2V)) +
#      √h √(kT κ/(2τ)) R with a Philox Gaussian keyed on the event counter, the
#      scale s = ∛(λ'²/V), the new box written into the device geometry;
#   3. molecular scaling (waters translated with their centres of mass, other
#      particles scaled), wrapped and unwrapped coordinates;
#   4. the mesh influence function at the new box.
#
# Exact (displacement-triggered) lists are rebuilt after every cell change: the displacement
# check measures motion in the current box only, and a contraction can bring a pair from
# outside the list radius inside the cutoff while every particle stays within half the skin.
# The event moves the list references a quarter box away, so the
# step's decision rebuilds. Buffered lists (fixed intervals) keep their list across scaling,
# as GROMACS does; their error is bounded statistically, not by the displacement check. The
# pair kernels image every pair with the current device box.
#
# The event sits after the drift and the position constraints and before the list
# decision and the force evaluation of the step, so the forces of the step are at the
# scaled positions (GROMACS order: scale after the update). Velocities are not scaled.
# Host mirrors of the box are refreshed at host boundaries only.

const _BARO_W = 1        # virial accumulator (reset after each event)
const _BARO_S = 2        # scale factor of the last event
const _BARO_EVENTS = 3   # events so far
const _BARO_P = 4        # last instantaneous pressure (molecular)
const _BARO_DV = 5       # accumulated volume change
const _BARO_REF = 6      # 6:8 box lengths at the last host setup (grid validity)
const _BARO_LEN = 8
const _BARO_STREAM = UInt32(0x50524553)   # Philox stream of the barostat ("PRES")
const DN_BAROSTAT_ERROR = Int64(-31)      # latched: nonpositive volume or grid limit exceeded

struct DeviceCRescale
    pressure::Float64
    kT::Float64
    compressibility::Float64
    tau::Float64
    h::Float64          # elapsed time per event (interval·dt)
    seed::UInt64
    units::Float64      # scaled units: molecules (each rigid water once)
    interval::Int
    buf::CuVector{Float64}
    waters::Any         # (atoms, inverse_masses) of the rigid waters, or nothing
    in_water::Any       # per-particle mask, or nothing
    unwrap::Any         # unwrapped coordinates (tuple), or nothing
end

function DeviceCRescale(st, cr, interval::Integer, dt)
    rw = st.rigid_water
    buf = CUDA.zeros(Float64, _BARO_LEN)
    copyto!(view(buf, _BARO_REF:_BARO_REF+2), Float64.(collect(st.box3)))
    copyto!(view(buf, _BARO_EVENTS:_BARO_EVENTS), Float64[cr.events])
    return DeviceCRescale(cr.pressure, cr.kT, cr.compressibility, cr.tau, Float64(interval) * Float64(dt), cr.seed,
        Float64(SC.scaled_units(st)), Int(interval), buf,
        rw === nothing ? nothing : (rw.workspace.atoms, rw.workspace.inverse_masses),
        rw === nothing ? nothing : rw.in_water,
        st.rx_unwrap === nothing ? nothing : (st.rx_unwrap, st.ry_unwrap, st.rz_unwrap))
end

# ---- 1. molecular virial -------------------------------------------------------------
function _baro_zero_kernel!(buf)
    @inbounds buf[_BARO_W] = 0.0
    return nothing
end

function _baro_rows_kernel!(buf, virial, fx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    w = 0.0
    if i <= size(virial, 1)
        @inbounds begin
            vb = _find(fx, Val(:vbond))   # bonded rows (feature), never the constraint rows
            w = _obs_virial(vb, virial, i, 1) + _obs_virial(vb, virial, i, 2) + _obs_virial(vb, virial, i, 3)
        end
    end
    w = _block_sum(w)
    threadIdx().x == 1 && CUDA.@atomic buf[_BARO_W] += w
    return nothing
end

function _baro_water_kernel!(buf, rx, ry, rz, fx, fy, fz, atoms, inverse_masses, geom)
    T = eltype(rx)
    box = Definitions.box_lengths(geom, Val(3))
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    acc = 0.0
    if i <= length(atoms)
        @inbounds begin
            ids = atoms[i]; w = inverse_masses[i]
            raw = RWC._load_water(rx, ry, rz, ids)
            x = RWC.local_water(raw, raw[1], box)
            m1 = one(T) / w[1]; m2 = one(T) / w[2]; m3 = one(T) / w[3]
            c = (m1 * x[1] + m2 * x[2] + m3 * x[3]) / (m1 + m2 + m3)
            for k in 1:3
                j = ids[k]; d = x[k] - c
                acc += Float64(d[1]) * Float64(fx[j]) + Float64(d[2]) * Float64(fy[j]) + Float64(d[3]) * Float64(fz[j])
            end
        end
    end
    acc = _block_sum(acc)
    threadIdx().x == 1 && CUDA.@atomic buf[_BARO_W] -= acc
    return nothing
end

# Block sum of one Float64 per thread (blockDim ≤ 1024, multiple of 32).
@inline function _block_sum(x::Float64)
    lane = (threadIdx().x - 1) % 32; wid = (threadIdx().x - 1) ÷ 32
    for off in (16, 8, 4, 2, 1)
        x += CUDA.shfl_down_sync(0xffffffff, x, off)
    end
    sh = CuStaticSharedArray(Float64, 32)
    lane == 0 && (@inbounds sh[wid + 1] = x)
    sync_threads()
    nw = (blockDim().x + 31) ÷ 32
    x = (threadIdx().x <= nw) ? (@inbounds sh[threadIdx().x]) : 0.0
    if wid == 0
        for off in (16, 8, 4, 2, 1)
            x += CUDA.shfl_down_sync(0xffffffff, x, off)
        end
    end
    return x
end

# ---- 2. the event (one thread) -----------------------------------------------------------
function _baro_event_kernel!(buf, geom, vglobal, state, P0::Float64, kT::Float64, kappa::Float64, tau::Float64,
                             h::Float64, units::Float64, seed::UInt64, shrink::Float64)
    T = eltype(geom)
    @inbounds begin
        W = buf[_BARO_W]
        if vglobal !== nothing
            W += Float64(vglobal[1]) + Float64(vglobal[2]) + Float64(vglobal[3])
        end
        L1 = Float64(geom[1]); L2 = Float64(geom[2]); L3 = Float64(geom[3])
        V = L1 * L2 * L3
        P = units * kT / V + W / (3V)
        lam = sqrt(V)
        drift = kappa * lam / (2tau) * (P - P0 + kT / (2V))
        amp = sqrt(kT * kappa / (2tau))
        k = unsafe_trunc(UInt64, buf[_BARO_EVENTS])
        w0, w1, w2, _ = _philox_words(seed, k, 0, _BARO_STREAM)
        R, _ = _box_muller(_open01(Float64, w0, w1), (Float64(w2) + 0.5) * 2.0^-32)
        lam2 = lam + h * drift + sqrt(h) * amp * R
        s = lam2 > 0 ? cbrt(lam2 * lam2 / V) : -1.0
        # the cluster grid covers boxes down to `shrink` × the box of its host setup
        ok = isfinite(s) && s > 0 && s * L1 >= shrink * buf[_BARO_REF] && s * L2 >= shrink * buf[_BARO_REF + 1] &&
             s * L3 >= shrink * buf[_BARO_REF + 2]
        if !ok
            state[NL.DN_ERROR] == 0 && (state[NL.DN_ERROR] = DN_BAROSTAT_ERROR)
            s = 1.0
        end
        geom[1] = T(s * L1); geom[2] = T(s * L2); geom[3] = T(s * L3)
        buf[_BARO_S] = s
        buf[_BARO_P] = P
        buf[_BARO_DV] += (s^3 - 1) * V
        buf[_BARO_EVENTS] += 1
    end
    return nothing
end

# ---- 3. molecular scaling ---------------------------------------------------------------
function _baro_scale_free_kernel!(rx, ry, rz, in_water, buf, geom, ::Val{WRAP}) where {WRAP}
    T = eltype(rx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(rx) && return nothing
    @inbounds begin
        (in_water !== nothing && in_water[i]) && return nothing
        s = T(buf[_BARO_S]); L = Definitions.box_lengths(geom, Val(3))
        x = rx[i] * s; y = ry[i] * s; z = rz[i] * s
        rx[i] = WRAP ? _wrap(x, L[1]) : x
        ry[i] = WRAP ? _wrap(y, L[2]) : y
        rz[i] = WRAP ? _wrap(z, L[3]) : z
    end
    return nothing
end

function _baro_scale_water_kernel!(rx, ry, rz, atoms, inverse_masses, buf, geom, ::Val{WRAP}) where {WRAP}
    T = eltype(rx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(atoms) && return nothing
    @inbounds begin
        s = T(buf[_BARO_S]); L = Definitions.box_lengths(geom, Val(3)); Lold = map(x -> x / s, L)
        ids = atoms[i]; w = inverse_masses[i]
        raw = RWC._load_water(rx, ry, rz, ids)
        # wrapped coordinates: the molecule made whole around its oxygen in the old box
        x = WRAP ? RWC.local_water(raw, raw[1], Lold) : (raw[1] - raw[1], raw[2] - raw[1], raw[3] - raw[1])
        m1 = one(T) / w[1]; m2 = one(T) / w[2]; m3 = one(T) / w[3]
        com = raw[1] + (m1 * x[1] + m2 * x[2] + m3 * x[3]) / (m1 + m2 + m3)
        d = (s - one(T)) * com
        for k in 1:3
            j = ids[k]; p = raw[1] + x[k] + d
            rx[j] = WRAP ? _wrap(p[1], L[1]) : p[1]
            ry[j] = WRAP ? _wrap(p[2], L[2]) : p[2]
            rz[j] = WRAP ? _wrap(p[3], L[3]) : p[3]
        end
    end
    return nothing
end

"""Molecular virial of the completed step into the barostat buffer (end of every sampled
step of a run with device pressure control); `launch=false` compiles only."""
function enqueue_virial_sample!(run, b; launch::Bool=true)
    bar = b.barostat; bar === nothing && return nothing
    ws = run.neighbors; geom = ws.geom; N = length(run.r[1]); state = ws.state
    kw(k, args, n; threads=256) = (launch ? k(args...; threads, blocks=cld(max(n, 1), threads)) : nothing)
    k0 = @cuda launch=false NL._device_if_valid!(state, _baro_zero_kernel!, bar.buf)
    launch && k0(state, _baro_zero_kernel!, bar.buf; threads=1, blocks=1)
    k1 = @cuda launch=false NL._device_if_valid!(state, _baro_rows_kernel!, bar.buf, run.virial, run.fx)
    kw(k1, (state, _baro_rows_kernel!, bar.buf, run.virial, run.fx), N)
    if bar.waters !== nothing
        atoms, im = bar.waters
        k2 = @cuda launch=false NL._device_if_valid!(state, _baro_water_kernel!, bar.buf, run.r..., run.f..., atoms, im, geom)
        kw(k2, (state, _baro_water_kernel!, bar.buf, run.r..., run.f..., atoms, im, geom), length(atoms))
    end
    return nothing
end

function _baro_trace_kernel!(buf, vtr)
    @inbounds begin buf[_BARO_W] = vtr[1]; vtr[1] = 0.0 end
    return nothing
end

"""Molecular virial from the pairs' virial trace of a trace step (no energies, no per-particle
virial rows): Σ r·f of the pair kernels, minus the rigid waters' intramolecular term; the event
adds the mesh's global virial. `launch=false` compiles only."""
function enqueue_virial_trace_sample!(run, b; launch::Bool=true)
    bar = b.barostat; bar === nothing && return nothing
    ws = run.neighbors; geom = ws.geom; state = ws.state; vtr = b.eng.vtrace
    k0 = @cuda launch=false NL._device_if_valid!(state, _baro_trace_kernel!, bar.buf, vtr)
    launch && k0(state, _baro_trace_kernel!, bar.buf, vtr; threads=1, blocks=1)
    if bar.waters !== nothing
        atoms, im = bar.waters
        k2 = @cuda launch=false NL._device_if_valid!(state, _baro_water_kernel!, bar.buf, run.r..., run.f..., atoms, im, geom)
        launch && k2(state, _baro_water_kernel!, bar.buf, run.r..., run.f..., atoms, im, geom;
                     threads=256, blocks=cld(max(length(atoms), 1), 256))
    end
    return nothing
end

"""Can the step before an event use a trace step? Not with bonded virial rows or LJ
bond-exclusion corrections (they keep per-particle virial rows only)."""
virial_trace_ok(run) = run.cluster !== nothing && run.cluster.eng.corrections === nothing &&
    all(f -> f isa Union{UnwrapFeature,MassFeature,RigidWaterFeature}, run.features)

"""Enqueue one device C-rescale event (inside a capture or eagerly) after the drift and
the position constraints of an event step; it reads the virial of the sampled step
before. `launch=false` compiles only."""
function enqueue_crescale!(run, b; launch::Bool=true)
    bar = b.barostat; bar === nothing && return nothing
    ws = run.neighbors; geom = ws.geom; N = length(run.r[1]); state = ws.state
    kw(k, args, n; threads=256) = (launch ? k(args...; threads, blocks=cld(max(n, 1), threads)) : nothing)
    vg = b.Vglobal
    k3 = @cuda launch=false NL._device_if_valid!(state, _baro_event_kernel!, bar.buf, geom, vg, state, bar.pressure, bar.kT,
        bar.compressibility, bar.tau, bar.h, bar.units, bar.seed, CP.GEOMETRY_SHRINK)
    launch && k3(state, _baro_event_kernel!, bar.buf, geom, vg, state, bar.pressure, bar.kT, bar.compressibility, bar.tau,
                 bar.h, bar.units, bar.seed, CP.GEOMETRY_SHRINK; threads=1, blocks=1)
    for (coords, wrap) in ((run.r, Val(true)), (bar.unwrap, Val(false)))
        coords === nothing && continue
        k4 = @cuda launch=false NL._device_if_valid!(state, _baro_scale_free_kernel!, coords..., bar.in_water, bar.buf, geom, wrap)
        kw(k4, (state, _baro_scale_free_kernel!, coords..., bar.in_water, bar.buf, geom, wrap), N)
        if bar.waters !== nothing
            atoms, im = bar.waters
            k5 = @cuda launch=false NL._device_if_valid!(state, _baro_scale_water_kernel!, coords..., atoms, im, bar.buf, geom, wrap)
            kw(k5, (state, _baro_scale_water_kernel!, coords..., atoms, im, bar.buf, geom, wrap), length(atoms))
        end
    end
    for refs in _exact_list_refs(run, b)
        k6 = @cuda launch=false NL._device_if_valid!(state, _baro_invalidate_refs_kernel!, refs, run.r, bar.buf, geom)
        kw(k6, (state, _baro_invalidate_refs_kernel!, refs, run.r, bar.buf, geom), N)
    end
    b.solver === nothing || ES.device_influence!(b.solver, geom; launch)
    return nothing
end

# Reference positions of the run's exact lists: the ELL rows (always displacement-triggered) and
# a cluster list without fixed intervals.
function _exact_list_refs(run, b)
    out = Any[]
    nb = run.neighbors.list
    (nb.rref_z !== nothing && length(nb.rref_x) == length(run.r[1])) && push!(out, (nb.rref_x, nb.rref_y, nb.rref_z))
    eng = b.eng
    (eng !== nothing && eng.list.nstlist == 0) && push!(out, (eng.list.rx0, eng.list.ry0, eng.list.rz0))
    return out
end

# After a cell change (s ≠ 1): references a quarter box from the scaled positions, so the next
# displacement check exceeds any half skin (list radius < L/2) and requests a rebuild.
function _baro_invalidate_refs_kernel!(refs, r, buf, geom)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(r[1]) && return nothing
    @inbounds begin
        buf[_BARO_S] == 1.0 && return nothing
        L = Definitions.box_lengths(geom, Val(3)); T = eltype(r[1])
        refs[1][i] = r[1][i] + T(0.25) * L[1]
        refs[2][i] = r[2][i] + T(0.25) * L[2]
        refs[3][i] = r[3][i] + T(0.25) * L[3]
    end
    return nothing
end

"""The barostat context of a run: the cluster backend, or for an ELL run (no mesh, no
cluster list) a stand-in with the same fields."""
_baro_ctx(run) = run.cluster !== nothing ? run.cluster :
    (barostat=run.barostat, Vglobal=run.mesh === nothing ? nothing : run.mesh.Vglobal,
     solver=run.mesh === nothing ? nothing : run.mesh.solver, eng=nothing)
_has_barostat(run) = (run.cluster !== nothing && run.cluster.barostat !== nothing) || run.barostat !== nothing

"""Host mirrors of the device box after device events (boundaries only: one small read)."""
function sync_device_box!(e, st)
    _has_barostat(e.run) || return nothing
    if e.run.cluster === nothing   # ELL run
        g = Array(e.run.neighbors.geom); box = (g[1], g[2], g[3])
        box == st.box3 && return nothing
        st.box3 = box; e.run.neighbors.box = box; e.graph_box = box
        e.run.mesh === nothing || (e.run.mesh.solver.box = Float64.(box))
        return nothing
    end
    b = e.run.cluster
    g = Array(e.run.neighbors.geom)
    box = (g[1], g[2], g[3])
    box == st.box3 && return nothing
    st.box3 = box
    e.run.neighbors.box = box
    b.eng.list.box = box
    b.solver === nothing || (b.solver.box = Float64.(box))
    e.graph_box = box   # the graphs read the device box: no recapture
    return nothing
end
