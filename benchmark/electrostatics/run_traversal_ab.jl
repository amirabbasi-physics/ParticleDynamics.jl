# Pair-traversal comparison for charged systems: the device engine's ELL rows against the
# cluster-pair backend. Water, 1 M NaCl(aq)
# and molten NaCl from 24k to 1M atoms, identical physics on every path: LJ (amber99sb-ildn
# SPC/E water and ions, Lorentz–Berthelot) + real-space Coulomb with the ESP split,
# r_c = 1.0 nm, ESP mesh at Δ = 1e-3, Float32, molecular exclusions.
#   cluster — the cluster-pair backend (8×8 tiles, pruned working list), as it runs today
#   engine  — the device engine's ELL rows and cooperative pair kernel (G lanes per atom),
#             with LJ + Coulomb fused in one pair model defined below (an experiment, no
#             package source changes); exclusions through the engine's exclusion feature,
#             the Ewald exclusion correction in a small kernel, the mesh from the same solver
#   staged  — the staged ELL path (LJ and Coulomb as two ELL traversals), reference only
# Two measurements per system:
#   frozen — at the equilibrated configuration: pair-force, list-build, prune and mesh
#            times, and agreement of the full forces (mesh included) of all three paths
#   replay — 300 consecutive MD steps (GROMACS, 2 fs, every frame) replayed frame by frame
#            per skin: each path checks its list, rebuilds or prunes by its own policy and
#            evaluates the short-range forces. The engine runs every frame as a captured
#            conditional graph (displacement check → rebuild node → forces), the cluster
#            backend through its staged list update (displacement readback → rebuild or
#            prune → forces). The final frame's short-range forces of both paths are compared.
# Skins (user: two, physics-backed, close to GROMACS): per system, the two Verlet buffers
# GROMACS derived for it (tolerance 0.005 kJ/mol/ps per atom, 80–100-step list lifetime):
# its cluster-list buffer and the equivalent classical 1x1 (per-atom) list buffer, both read
# from the equilibration log. Both paths run at both skins with PD's strict policies: rebuild
# when a particle moved more than skin/2, cluster re-prune at prune buffer skin/2.
# Usage: julia --project=. benchmark/electrostatics/run_traversal_ab.jl SYSTEMS_DIR OUT_DIR [names…]
# Environment: PD_E6_SKINS (nm, comma-separated) overrides the GROMACS buffers.
using CUDA, ParticleDynamics, TOML, Statistics, Dates, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "Benchmark.jl"))
using .Fixtures, .PDHarness
const DE = ParticleDynamics.DeviceEngine
const NL = ParticleDynamics.NeighborLists
const ES = ParticleDynamics.Electrostatics
const SC = ParticleDynamics.SimulationCore
const BF = ParticleDynamics.BondedForces
const CP = ParticleDynamics.ClusterPairs
const GMX = "/net/storage/abbaa90/gromacs-gpu/bin/gmx"

# amber99sb-ildn (SPC/E water, Åqvist ions), Lorentz–Berthelot mixing; σ nm, ε kJ/mol
const LJTYPES = ["OW" => (0.316557, 0.650629), "HW" => (0.0, 0.0), "NA" => (0.332840, 0.0115897),
                 "CL" => (0.440104, 0.4184)]
_typeidx(name) = startswith(name, "OW") ? 1 : startswith(name, "HW") ? 2 : startswith(name, "NA") ? 3 :
                 startswith(name, "CL") ? 4 : error("unknown atom name $name")
function lj_matrices()
    n = length(LJTYPES); σ = zeros(n, n); ε = zeros(n, n)
    for a in 1:n, b in 1:n
        (sa, ea), (sb, eb) = LJTYPES[a][2], LJTYPES[b][2]
        ε[a, b] = sqrt(ea * eb); σ[a, b] = ε[a, b] > 0 ? 0.5 * (sa + sb) : 0.0
    end
    return σ, ε
end
gro_names(path) = [strip(l[11:15]) for l in readlines(path)[3:end-1]]

# --- the engine experiment: LJ (type matrix) + real-space Coulomb in one pair model ---
struct LJCoulomb{T,K} <: DE.PairModel
    C::T      # Coulomb prefactor
    rc2::T    # common cutoff²
    k::K      # ES device pair kernel of the solver's split
end
@inline DE.pair_context(::LJCoulomb, data, i) = (@inbounds (data[1][i], data[4][i]))
@inline function DE.pair_eval(m::LJCoulomb{T}, data, ctx, j, r2::T) where {T}
    ti, qi = ctx
    @inbounds begin
        tj = data[1][j]; σ = data[2][ti, tj]; ε = data[3][ti, tj]; qj = data[4][j]
    end
    inside = (r2 > zero(T)) & (r2 < m.rc2)
    invr2 = one(T) / r2; s2 = σ * σ * invr2; s6 = s2 * s2 * s2; s12 = s6 * s6
    flj = T(24) * ε * (T(2) * s12 - s6) * invr2; ulj = T(4) * ε * (s12 - s6)
    r = sqrt(r2); invr = one(T) / r
    e, f = ES.pair_kernel(m.k, r, invr)
    cq = m.C * qi * qj
    return inside, flj + cq * f, ulj + cq * e
end

# Ewald exclusion correction: the mesh includes excluded pairs; remove their long-range
# part, −C qᵢqⱼ (1/r³ − f_s(r)) d per pair (thread per particle, its exclusion row).
function _exclusion_kernel!(fx, fy, fz, rx, ry, rz, q, bidx, bflat, bcnt, C::T, k, L::NTuple{3,T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        ax = zero(T); ay = zero(T); az = zero(T)
        for t in 0:(Int(bcnt[i]) - 1)
            j = bflat[bidx[i] + t + 1]
            dx = rx[i] - rx[j]; dy = ry[i] - ry[j]; dz = rz[i] - rz[j]
            dx -= L[1] * round(dx / L[1]); dy -= L[2] * round(dy / L[2]); dz -= L[3] * round(dz / L[3])
            r2 = dx * dx + dy * dy + dz * dz; r = sqrt(r2); invr = one(T) / r
            _, fs = ES.pair_kernel(k, r, invr)
            g = -C * q[i] * q[j] * (invr * invr * invr - fs)
            ax += g * dx; ay += g * dy; az += g * dz
        end
        fx[i] += ax; fy[i] += ay; fz[i] += az
    end
    return
end

function engine_run(f::Fixture, types, σ, ε, spec; G::Int, B::Int, skin::Real)
    T = Float32; N = length(f); box = ntuple(d -> T(f.box[d]), 3)
    r = ntuple(d -> CuArray(T.(mod.(f.R[d, :], f.box[d]) .- f.box[d] / 2)), 3)
    v = ntuple(_ -> CUDA.zeros(T, N), 3)
    bl = BF.build_bondlist(N, [(Int32(i), Int32(j)) for (i, j, s) in f.exceptions if s == 0])
    K = ES.device_pair_kernel(spec.split, T)
    rc = spec.real_cutoff
    model = LJCoulomb{T,typeof(K)}(T(f.prefactor), T(rc)^2, K)
    data = (CuArray(Int32.(types)), CuArray(T.(σ)), CuArray(T.(ε)), CuArray(T.(f.q)))
    cap = ceil(Int, 1.3 * 4pi / 3 * (rc + skin)^3 * N / prod(f.box) + 64)
    run = DE.Run(r, v, box; cutoff=T(rc), skin=T(skin), cap, dt=T(0.002), mass=T(18), sigma=T(1), epsilon=T(1),
                 force_lanes=G, build_lanes=B, rebuild_policy=:displacement, fused=false, model, model_data=data,
                 features=(DE.BondExclusionFeature(bl.index, bl.flat, bl.counts),))
    exclusions!() = (@cuda threads=256 blocks=cld(N, 256) _exclusion_kernel!(run.f..., run.r..., data[4], bl.index,
                         bl.flat, bl.counts, T(f.prefactor), K, box))
    exclusions!()
    return run, exclusions!, data[4], bl
end

# Cluster backend as MD runs it: list at r_c + skin, working list pruned to r_c + skin/2.
function cluster_state(f::Fixture, spec, types, σ, ε; skin::Real)
    st = pd_state_lj(f, spec; types, sigma=σ, epsilon=ε, rc_lj=1.0, skin, backend=:cluster)
    set_nonbonded_backend!(st, :cluster; prune_buffer=skin / 2)
    SC.invalidate_forces!(st); SC.evaluate_forces_into_f!(st, false)
    return st
end
_cluster_forces!(st) = CP.compute_forces!(st.cluster, st.rx, st.ry, st.rz, st.charge, st.fx, st.fy, st.fz, st.Epot,
                                          st.virial_nonbonded, false)
_cluster_update!(st) = CP.update_list!(st.cluster, st.rx, st.ry, st.rz, st.charge, st.typeid, st.box3)
function _cluster_built!(st)          # what `update_list!` does for :built, plus the pruning force pass
    eng = st.cluster; L = eng.list
    CP._rebuild!(eng, st.rx, st.ry, st.rz, st.box3)
    CP._pack!(eng, st.rx, st.ry, st.rz, st.charge, st.typeid)
    CP.request_prune!(L); CP.mark_pruned!(L, st.rx, st.ry, st.rz)
    _cluster_forces!(st)
end

timed(f) = PDHarness._timed(f; reps=30, batches=5).median            # ms, CUDA events

# --- GROMACS .trr frames (single precision, XDR big-endian), streamed ---
struct TRRReader
    io::IOStream
    N::Int
    buf::Vector{UInt32}
end
TRRReader(path::AbstractString, N::Integer) = TRRReader(open(path, "r"), Int(N), Vector{UInt32}(undef, 3N))
_rdi(io) = Int(ntoh(read(io, Int32)))
function next_frame!(t::TRRReader, X::Matrix{Float32})
    eof(t.io) && return nothing
    _rdi(t.io) == 1993 || error("not a .trr frame")
    _rdi(t.io); n = _rdi(t.io); skip(t.io, 4 * cld(n, 4))          # version string
    h = [_rdi(t.io) for _ in 1:13]   # ir e box vir pres top sym x v f natoms step nre
    h[11] == t.N || error(".trr has $(h[11]) atoms, expected $(t.N)")
    h[3] == 36 && h[8] == 12 * t.N && h[1] == h[2] == h[6] == h[7] == 0 ||
        error("only single-precision .trr frames with box and coordinates are read")
    skip(t.io, 8)                                                   # t, λ
    b = [reinterpret(Float32, ntoh(read(t.io, UInt32))) for _ in 1:9]
    skip(t.io, h[4] + h[5])
    read!(t.io, t.buf); t.buf .= ntoh.(t.buf)
    copyto!(X, reinterpret(Float32, t.buf))
    skip(t.io, h[9] + h[10])
    return h[12], (Float64(b[1]), Float64(b[5]), Float64(b[9]))
end
# fixture rows (spatially sorted) of a frame in original atom order; engine: centered
function frame_rows!(h, X, ids, box, centered::Bool)
    @inbounds for (row, id) in enumerate(ids), d in 1:3
        L = box[d]; x = mod(Float64(X[d, id]), L)
        h[d][row] = centered ? Float32(x - L / 2) : Float32(x)
    end
    return h
end

function engine_replay(run, exclusions!, trr, f::Fixture)
    ws = run.neighbors; N = length(f)
    X = Matrix{Float32}(undef, 3, N); h = ntuple(_ -> Vector{Float32}(undef, N), 3)
    t = TRRReader(trr, N)
    upload!() = (frame_rows!(h, X, f.ids, f.box, true); foreach(d -> copyto!(run.r[d], h[d]), 1:3))
    next_frame!(t, X); upload!()                          # frame 0: a fresh list
    NL.rearm_capacity_rebuild!(ws); DE.enqueue_rebuild!(run; reorder=false)
    DE.enqueue_force!(run; energy=false); exclusions!(); NL.advance_device_neighbor_step!(ws)
    NL.device_neighbor_status(ws)
    graph = CUDA.instantiate(NL.device_conditional_graph(
        () -> DE.decide_policy!(ws, run.r, Val(:displacement)),
        () -> DE.enqueue_rebuild!(run; reorder=false),
        () -> (DE.enqueue_force!(run; energy=false); exclusions!(); NL.advance_device_neighbor_step!(ws)),
        ws.state))
    CUDA.launch(graph)                                    # warm-up at frame 0 (no rebuild)
    nb = NL.device_neighbor_status(ws).builds
    GC.gc(); CUDA.synchronize()
    times = Float64[]; rebuilt = Bool[]
    while next_frame!(t, X) !== nothing
        upload!()
        push!(times, 1e3 * CUDA.@elapsed CUDA.launch(graph))
        b = NL.device_neighbor_status(ws).builds          # throws on a latched device error
        push!(rebuilt, b > nb); nb = b
    end
    close(t.io)
    st = NL.device_neighbor_status(ws); d = NL.device_neighbor_diagnostics(ws)
    F = vcat((Float64.(Array(x))' for x in run.f)...)
    return (; times, rebuilt, F, max_row=Int(st.required), mean_row=d.total_accepted / max(d.builds, 1) / N)
end

function cluster_replay(st, trr, f::Fixture)
    N = length(f)
    X = Matrix{Float32}(undef, 3, N); h = ntuple(_ -> Vector{Float32}(undef, N), 3)
    t = TRRReader(trr, N)
    upload!() = (frame_rows!(h, X, f.ids, f.box, false); copyto!(st.rx, h[1]); copyto!(st.ry, h[2]); copyto!(st.rz, h[3]))
    next_frame!(t, X); upload!(); _cluster_built!(st); CUDA.synchronize()     # frame 0: a fresh list
    GC.gc(); CUDA.synchronize()
    times = Float64[]; actions = Symbol[]; act = Ref(:kept)
    while next_frame!(t, X) !== nothing
        upload!()
        push!(times, 1e3 * CUDA.@elapsed (act[] = _cluster_update!(st); _cluster_forces!(st)))
        push!(actions, act[])
    end
    close(t.io)
    return (; times, actions, F=pd_forces(st))
end

# GROMACS's Verlet buffers for this system, from its log: the cluster pair list ("outer list:
# updated every n steps, buffer b nm") and the "equivalent classical 1x1 list" at the same tolerance.
function gromacs_buffers(log)
    lines = readlines(log)
    i = findfirst(l -> occursin("outer list: updated every", l), lines)
    j = findfirst(l -> occursin("equivalent classical 1x1 list", l), lines)
    (i === nothing || j === nothing) && error("no Verlet buffer report in $log")
    buf(l) = parse(Float64, match(r"buffer ([0-9.]+) nm", l).captures[1])
    every(l) = parse(Int, match(r"updated every +([0-9]+) steps", l).captures[1])
    return (cluster=buf(lines[i]), atom=buf(lines[j + 1]), lifetime=every(lines[i]))
end

function gromacs_ms_per_step(log)
    isfile(log) || return NaN
    for l in eachline(log)
        startswith(l, "Performance:") && return 172.8 / parse(Float64, split(l)[2])   # ns/day at 2 fs → ms/step
    end
    return NaN
end

_mean(x) = isempty(x) ? NaN : mean(x)
_median(x) = isempty(x) ? NaN : median(x)
# Per-step cost of a replay: frames grouped by list action (kept / pruned / rebuilt), each
# group at its median frame time, weighted by its count. Robust to rare host pauses (a GC
# pause from the cluster path's per-build device allocations cost one dry-run frame ~50 ms);
# the raw mean and the slowest frame are recorded next to it.
function step_cost(times, groups)
    return sum(count(==(g), groups) * median(times[groups .== g]) for g in unique(groups)) / length(times)
end

function measure(name, sysdir, outdir, skins_override)
    gro = joinpath(sysdir, name * ".gro"); top = joinpath(sysdir, name * ".top")
    trr = joinpath(sysdir, name * ".replay.trr"); log = joinpath(sysdir, name * ".log")
    gb = gromacs_buffers(log)
    skins = skins_override === nothing ? [gb.cluster, gb.atom] : skins_override
    rec = Dict{String,Any}("system" => name, "created_utc" => string(now(UTC)), "gpu" => CUDA.name(CUDA.device()),
                           "gromacs_md_ms_per_step" => gromacs_ms_per_step(log),
                           "gromacs_buffers_nm" => Dict("cluster_list" => gb.cluster, "per_atom_list" => gb.atom,
                                                        "list_lifetime_steps" => gb.lifetime),
                           "skins_nm" => skins)
    f0 = gromacs_fixture(gro; top, gmx=GMX, workdir=mktempdir())
    names = gro_names(gro)
    f = spatially_sorted(f0; cell=1.1)
    types = [_typeidx(names[id]) for id in f.ids]
    σ, ε = lj_matrices()
    N = length(f); rec["N"] = N; rec["box_nm"] = collect(f.box)
    rec["species"] = Dict(String(LJTYPES[t][1]) => count(==(t), types) for t in 1:4)
    spec = esp_spec(f; rc=1.0, tolerance=1e-3)
    rec["mesh"] = collect(spec.mesh)
    @printf("%s: N=%d box=%.3f nm mesh=%s\n", name, N, f.box[1], spec.mesh); flush(stdout)

    # ---- frozen: staged ELL reference, mesh, full-force agreement at skin 0.1 ----
    ste = pd_state_lj(f, spec; types, sigma=σ, epsilon=ε, rc_lj=1.0, backend=:ell)
    rec["staged_full_ms"] = pd_nonbonded_time(ste).median
    s = ste.electrostatics
    rec["mesh_ms"] = timed(() -> ES.compute_mesh!(s, ste.rx, ste.ry, ste.rz, ste.charge, ste.fx, ste.fy, ste.fz,
                                                  ste.Epot, ste.virial_global, false))
    SC.invalidate_forces!(ste); SC.evaluate_forces_into_f!(ste, false)
    Fs = pd_forces(ste)
    ste = nothing; GC.gc(); CUDA.reclaim()

    rec["skin"] = Dict{String,Any}()
    for skin in skins
        key = @sprintf("%.3f", skin); sr = Dict{String,Any}(); rec["skin"][key] = sr
        # -- cluster backend at this skin: frozen stage times, then the replay --
        stc = cluster_state(f, spec, types, σ, ε; skin)
        L = stc.cluster.list
        c = Dict{String,Any}()
        c["full_ms"] = pd_nonbonded_time(stc).median           # check + pack + pairs, mesh overlapped
        c["check_ms"] = timed(() -> CP.list_action(L, stc.rx, stc.ry, stc.rz, skin / 2, skin / 4))  # incl. readback
        th, bl = CP._launch1(L.N)                                   # the check's kernel alone (a graph port keeps only this)
        c["check_kernel_ms"] = timed(() -> (fill!(L.flags, UInt32(0));
            @cuda threads=th blocks=bl CP._displacement_kernel!(L.flags, stc.rx, stc.ry, stc.rz, L.rx0, L.ry0, L.rz0,
                L.px0, L.py0, L.pz0, L.geom, Float32(skin / 2)^2, Float32(skin / 4)^2)))
        c["pack_ms"] = timed(() -> CP._pack!(stc.cluster, stc.rx, stc.ry, stc.rz, stc.charge, stc.typeid))
        c["pairs_ms"] = timed(() -> _cluster_forces!(stc))                            # pruned working list
        c["prune_pass_ms"] = timed(() -> (CP.request_prune!(L); _cluster_forces!(stc)))  # forces over the outer list + compaction
        c["rebuild_pass_ms"] = timed(() -> _cluster_built!(stc))                      # rebuild + pack + prune pass
        if skin == first(skins)
            SC.invalidate_forces!(stc); SC.evaluate_forces_into_f!(stc, false)
            rec["cluster_vs_staged_full"] = relerr(pd_forces(stc), Fs)
        end
        rp = cluster_replay(stc, trr, f)
        Fc = rp.F
        c["replay_ms_per_step"] = step_cost(rp.times, rp.actions)
        c["replay_mean_ms"] = mean(rp.times); c["replay_max_ms"] = maximum(rp.times)
        for a in (:kept, :pruned, :built)
            sel = rp.actions .== a
            c["replay_$(a)"] = count(sel); c["replay_$(a)_median_ms"] = _median(rp.times[sel])
        end
        # the same list work with the host readback of the check replaced by its kernel (graph-port estimate)
        c["graph_port_est_ms_per_step"] = c["replay_ms_per_step"] - (c["check_ms"] - c["check_kernel_ms"])
        sr["cluster"] = c
        stc = nothing; GC.gc(); CUDA.reclaim()
        # -- engine at this skin: lanes, builder, then the graph replay with the best pair --
        e = Dict{String,Any}()
        for (G, B) in ((8, 8), (4, 8), (8, 4), (8, 1))
            run, exclusions!, q, _ = engine_run(f, types, σ, ε, spec; G, B, skin)
            # pair forces (+ exclusion correction) per lane count; a forced rebuild per builder width
            B == 8 && (e["pairs_G$(G)_ms"] = timed(() -> (DE.enqueue_force!(run; energy=false); exclusions!())))
            G == 8 && (e["build_B$(B)_ms"] = timed(() -> (NL.rearm_capacity_rebuild!(run.neighbors);
                                                          DE.enqueue_rebuild!(run; reorder=false))))
            if skin == first(skins) && (G, B) == (8, 8)
                DE.enqueue_force!(run; energy=false); exclusions!()
                ES.compute_mesh!(s, run.r..., q, run.f..., run.energy, CUDA.zeros(Float32, 9), false)
                Fe = vcat((Float64.(Array(x))' for x in run.f)...)
                rec["engine_vs_staged_full"] = relerr(Fe, Fs)
            end
            run = nothing; GC.gc(); CUDA.reclaim()
        end
        Gb = e["pairs_G4_ms"] < e["pairs_G8_ms"] ? 4 : 8
        Bb = argmin(b -> e["build_B$(b)_ms"], (1, 4, 8))
        e["G"] = Gb; e["B"] = Bb
        run, exclusions!, _, _ = engine_run(f, types, σ, ε, spec; G=Gb, B=Bb, skin)
        rp = engine_replay(run, exclusions!, trr, f)
        e["replay_ms_per_step"] = step_cost(rp.times, rp.rebuilt)
        e["replay_mean_ms"] = mean(rp.times); e["replay_max_ms"] = maximum(rp.times)
        e["replay_rebuilds"] = count(rp.rebuilt)
        e["replay_rebuild_median_ms"] = _median(rp.times[rp.rebuilt])
        e["replay_kept_median_ms"] = _median(rp.times[.!rp.rebuilt])
        e["max_row"] = rp.max_row; e["mean_row"] = rp.mean_row
        e["final_frame_vs_cluster"] = relerr(rp.F, Fc)
        sr["engine"] = e
        run = nothing; GC.gc(); CUDA.reclaim()
        @printf("  skin %.3f  cluster %.3f ms/step (built %d, pruned %d; mean %.3f, max %.2f)  engine %.3f ms/step (G%d B%d, rebuilds %d; mean %.3f, max %.2f)  final-frame relerr %.1e\n",
                skin, c["replay_ms_per_step"], c["replay_built"], c["replay_pruned"], c["replay_mean_ms"], c["replay_max_ms"],
                e["replay_ms_per_step"], Gb, Bb, e["replay_rebuilds"], e["replay_mean_ms"], e["replay_max_ms"],
                e["final_frame_vs_cluster"]); flush(stdout)
    end
    best(p) = argmin(k -> rec["skin"][k][p]["replay_ms_per_step"], collect(keys(rec["skin"])))
    bc = best("cluster"); be = best("engine")
    tc = rec["skin"][bc]["cluster"]["replay_ms_per_step"]; te = rec["skin"][be]["engine"]["replay_ms_per_step"]
    rec["summary"] = Dict("cluster_best_skin" => bc, "cluster_ms_per_step" => tc,
                          "cluster_graph_port_est_ms_per_step" => rec["skin"][bc]["cluster"]["graph_port_est_ms_per_step"],
                          "engine_best_skin" => be, "engine_ms_per_step" => te, "engine_over_cluster" => te / tc,
                          "mesh_ms" => rec["mesh_ms"], "cluster_full_ms_frozen" => rec["skin"][bc]["cluster"]["full_ms"])
    @printf("  best: cluster %.3f ms (skin %s), engine %.3f ms (skin %s) → engine/cluster %.2f; mesh %.3f ms\n",
            tc, bc, te, be, te / tc, rec["mesh_ms"]); flush(stdout)
    open(io -> TOML.print(io, rec; sorted=true), joinpath(outdir, name * ".toml"), "w")
    return rec
end

function main(args)
    sysdir = abspath(args[1]); outdir = abspath(args[2]); mkpath(outdir)
    names = length(args) > 2 ? args[3:end] :
        ["water_24k", "water_192k", "nacl_aq_192k", "molten_200k", "water_1M", "nacl_aq_1M", "molten_1M"]
    skins = haskey(ENV, "PD_E6_SKINS") ? parse.(Float64, split(ENV["PD_E6_SKINS"], ",")) : nothing
    CUDA.allowscalar(false)
    for n in names
        isfile(joinpath(sysdir, n * ".replay.trr")) || (println("skip $n (no replay trajectory)"); continue)
        try
            measure(n, sysdir, outdir, skins)
        catch err
            println("FAILED $n: ", sprint(showerror, err)[1:min(end, 800)]); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
