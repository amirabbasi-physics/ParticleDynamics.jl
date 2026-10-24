# Molten NaCl benchmark: one ParticleDynamics run of the shared GROMACS state,
# the model and settings of gromacs_run.sh: LJ (σ, ε per type from the .top [atomtypes],
# Lorentz–Berthelot) with potential shift at 1 nm, no dispersion correction, SPME with
# β from erfc(β·r_c) = 1e-5 (GROMACS ewald-rtol), order 4, the GROMACS grid (spacing 0.12 nm,
# or mesh=nx,ny,nz from its tpr), real masses, dt = 2 fs, Float32, cluster-pair backend.
#   ensemble=nve | nvt (CSVR, τ_t 0.1 ps, 1500 K) | npt (+ device C-rescale, τ_p 5 ps, every 100 steps)
# Timing as pd_npt.jl: setup (state, first forces, the first step's graph capture) outside;
# exactly N steps timed after explicit warm-up with one host synchronization at the end.
# Usage: julia --project=<PD> pd_run.jl gro= top= ensemble= out=JSON [steps=10000 skin=0.2 prune=0.1 mesh= profile=0]
#   method=pme (default) | esp (tolerance=…) | pppm (pppm_mesh= pppm_order= pppm_diff=ik|ad); reorder=0: no spatial sorting (ELL)
#   rc_coul=1.0: real-space Coulomb cutoff in nm (LJ stays at 1 nm; the list covers the larger)
using ParticleDynamics, CUDA, Printf, JSON3
const SC = ParticleDynamics.SimulationCore
const DE = ParticleDynamics.DeviceEngine
const ES = ParticleDynamics.Electrostatics
const KB = 0.0083144626; const BAR = 0.0602214076
opts = Dict(split(a, "=", limit=2) for a in ARGS)
opt(k, d) = haskey(opts, k) ? opts[k] : d
include(joinpath(@__DIR__, "..", "common", "metadata.jl"))

function read_gro(path)
    lines = readlines(path); n = parse(Int, strip(lines[2]))
    t = lines[3][21:end]; d1 = findfirst('.', t); w = findnext('.', t, d1 + 1) - d1
    R = zeros(3, n); V = zeros(3, n); names = String[]
    for i in 1:n
        l = lines[2 + i]; push!(names, strip(l[11:15])); s = l[21:end]
        for k in 1:3
            R[k, i] = parse(Float64, s[(k - 1) * w + 1:k * w])
            V[k, i] = parse(Float64, s[3w + (k - 1) * (w + 1) + 1:3w + k * (w + 1)])
        end
    end
    return R, V, Tuple(parse.(Float64, split(lines[3 + n])[1:3])), names
end
function read_atomtypes(top)
    t = Dict{String,NTuple{4,Float64}}(); sec = ""
    for l in readlines(top)
        s = strip(split(l, ';')[1]); isempty(s) && continue
        startswith(s, "[") && (sec = strip(s, ['[', ']', ' ']); continue)
        sec == "atomtypes" || continue
        f = split(s)   # name at.num mass charge ptype sigma epsilon
        t[f[1]] = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[6]), parse(Float64, f[7]))
    end
    return t
end
function ewald_beta(rc, rtol)
    lo, hi = 0.0, 20.0
    for _ in 1:200
        mid = (lo + hi) / 2
        ES._host_erfc(mid * rc) > rtol ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# State construction shared by main() and other benchmark tools.
function salt_state()
    R, V, box, names = read_gro(opts["gro"]); N = size(R, 2)
    at = read_atomtypes(opts["top"]); tn = sort(collect(keys(at)))   # type ids by name
    tid = Int32[findfirst(==(n), tn) for n in names]
    m = [at[n][1] for n in names]; q = [at[n][2] for n in names]
    # layer ablation (user: which layers make PD slower than GROMACS): charges=0 drops the
    # charges and the mesh (LJ only), masses=uniform uses the mean mass, backend=ell the dense list
    charged = opt("charges", "1") == "1"
    uniform = opt("masses", "real") == "uniform"
    uniform && (m = fill(sum(m) / N, N))
    backend = Symbol(opt("backend", "cluster"))
    sig = [(at[a][3] + at[b][3]) / 2 for a in tn, b in tn]; eps = [sqrt(at[a][4] * at[b][4]) for a in tn, b in tn]
    ens = opts["ensemble"]; steps = parse(Int, opt("steps", "10000")); T = Float32
    temperature = 1500.0; kT = KB * temperature; rc = 1.0
    rcc = parse(Float64, opt("rc_coul", string(rc)))   # real-space Coulomb cutoff (LJ stays at rc)
    beta = ewald_beta(rcc, parse(Float64, opt("rtol", "1e-5")))   # GROMACS ewald-rtol
    mesh = haskey(opts, "mesh") ? Tuple(parse.(Int, split(opts["mesh"], ","))) :
           ntuple(d -> ES.fft_friendly(ceil(Int, box[d] / 0.12 - 1e-9)), 3)
    t0 = time()
    # dense-list engine skin (default: the engine's size/density table)
    haskey(opts, "engine_skin") && (DE.ENGINE_SKIN[] = parse(Float64, opts["engine_skin"]))
    haskey(opts, "cell_sub") && (DE.CELL_SUB[] = parse(Int, opts["cell_sub"]))   # device cell grid subdivision
    haskey(opts, "spread_stride") && (DE.SPREAD_STRIDE[] = opts["spread_stride"] == "1")   # strided spread order
    st = SC.build_simulation(; N, box, nonbonded=:lj, cutoff=T(rc), skin=T(parse(Float64, opt("skin", "0.2"))),
        cap=Int32(16), epsilon=T(maximum(eps)), sigma=T(maximum(sig)), mass=uniform ? T(m[1]) : T.(m), dt=T(0.002), precision=:f32,
        charges=charged ? q : nothing, spatial_reorder=(backend === :ell && opt("reorder", "1") == "1"), neigh_interval=backend === :ell ? parse(Int, opt("sort", "20")) : 10_000,
        vdw_modifier=:potential_shift)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    copyto!(st.typeid, tid)
    st.sigma_pair = CuArray(T.(sig)); st.epsilon_pair = CuArray(T.(eps)); st.rcut_pair = CuArray(fill(T(rc), size(sig)))
    # method=esp tolerance=ε (match_accuracy.jl: ε = 1.328e-3 meets GROMACS PME's Δ on salt 10k):
    # prolate split and window, order and mesh from the ESP rules (benchmark/electrostatics esp_spec);
    # esp_mesh=nx,ny,nz / esp_order=P override them
    spec = if opt("method", "pme") == "esp"
        tol = parse(Float64, opt("tolerance", "1.328e-3")); c = ES.prolate_shape_for_tolerance(tol)
        P = haskey(opts, "esp_order") ? parse(Int, opts["esp_order"]) : ParticleDynamics.Workflow.esp_order_for_tolerance(tol)
        # esp_s: mesh oversampling (esp_scan.jl: P = 4, tolerance 1e-3, s = 1.5 meets the salt target)
        so = parse(Float64, opt("esp_s", "1.0"))
        em = haskey(opts, "esp_mesh") ? Tuple(parse.(Int, split(opts["esp_mesh"], ","))) :
            ntuple(d -> max(ES.fft_friendly(ceil(Int, box[d] * c * so / (pi * rcc))), P), 3)
        mesh = em
        cw = max(ES.prolate_shape_for_tolerance(tol / 2), pi * (P + 0.5) / 2 + 1e-3)   # window: at least the minimum for P
        ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rcc, split=ES.ProlateSplit(rcc, c),
            window=ES.ProlateWindow(P, cw), mesh=em, influence=:esp)
    elseif opt("method", "pme") == "pppm"
        # method=pppm: P3M with the Ewald β of `rtol`; pppm_mesh=nx,ny,nz (default: the PME mesh),
        # pppm_order (default 5), pppm_diff=ik|ad (default ik)
        pm = haskey(opts, "pppm_mesh") ? Tuple(parse.(Int, split(opts["pppm_mesh"], ","))) : mesh
        mesh = pm
        ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rcc, split=ES.GaussianSplit(beta),
            window=ES.BSplineWindow(parse(Int, opt("pppm_order", "5"))), mesh=pm,
            influence=opt("pppm_diff", "ik") == "ad" ? :p3m_ad : :p3m_ik)
    else
        ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rcc, split=ES.GaussianSplit(beta),
            window=ES.BSplineWindow(parse(Int, opt("pme_order", "4"))), mesh=mesh, influence=:spme)
    end
    charged && attach_electrostatics!(st, spec)
    # attaching charges moves the state to the cluster backend: backend=ell returns it to the dense list
    (charged && backend === :ell) && set_nonbonded_backend!(st, :ell)
    # intervals=nstlist,nstprune: fixed list intervals on the cluster backend (opt-in, statistical buffer)
    iv = haskey(opts, "intervals") ? Tuple(parse.(Int, split(opts["intervals"], ","))) : nothing
    backend === :cluster && set_nonbonded_backend!(st, :cluster; prune_buffer=parse(Float64, opt("prune", "0.1")), list_intervals=iv)
    sp = ens == "nve" ? SC.nve(st) : SC.csvr(st; temperature=T(kT), tau=T(0.1), interval=parse(Int, opt("nsttcouple", "1")))
    if ens == "npt"
        cr = SC.StochasticCellRescalingState(st; pressure=1.01325 * BAR, kT, compressibility=4.5e-5 / BAR, tau=5.0, seed=29)
        SC.attach_pressure_control!(st, cr; interval=100)
    end
    initial_model = BenchmarkMetadata.initial_model(st;mixing="Lorentz-Berthelot",kinetic_dof=SC.kinetic_dof(st))
    solver_model = BenchmarkMetadata.solver_settings(st.electrostatics)
    SC.evaluate_forces_into_f!(st, true)
    return (; st, sp, N, m, beta, mesh, rc, charged, backend, ens, steps, t0, initial_model, solver_model)
end

function main()
    provenance = BenchmarkMetadata.begin_case([opts["gro"], opts["top"]])
    (; st, sp, N, m, beta, mesh, rc, charged, backend, ens, steps, t0, initial_model, solver_model) = salt_state()
    SC.step!(st, sp, st.dt; compute_energy=false)
    # warm the I/O path (its host functions compile on first use: 0.6–4.8 s, which a timed
    # loop must not contain; warmed, one I/O point costs 0.4–0.6 ms)
    SC.step!(st, sp, st.dt; compute_energy=true); pilot=SC.collect_step_observables(st, sp)
    BenchmarkMetadata.warm_output!(opts["out"],pilot,st.box3[1])
    # Warm-up includes both pilot steps; default 6000 aligns with 2000-step cadence, as GROMACS reset (graph uploads, initial sorting, GPU clocks)
    warm = parse(Int, opt("warm", "6000")); warm >= 2 || error("warm must include at least two pilot steps")
    for i in 3:warm; SC.step!(st, sp, st.dt; compute_energy=(i % 2000 == 0)); end
    DE.synchronize_state!(st)
    advance! = sampled -> SC.step!(st,sp,st.dt;compute_energy=sampled)
    observe = () -> SC.collect_step_observables(st,sp)
    length_x = () -> st.box3[1]
    finish! = () -> (DE.synchronize_state!(st); CUDA.synchronize())
    runwindow = (path,offset,n) -> BenchmarkMetadata.run_observations!(advance!,observe,length_x,finish!,path,offset,n)
    runwindow(opts["out"]*".loop-warmup.csv",warm,0) # host compilation only; no added simulation steps
    profile = opt("profile","0")
    profile in ("0","1","cupti") || error("unsupported profile mode: $profile")
    audit = opt("audit_window","0")=="1"
    window_before = audit ? BenchmarkMetadata.window_frame(st,opts["out"]*".window-before") : nothing
    work_audit = opt("work_audit","0")=="1"
    device_before = work_audit ? BenchmarkMetadata.work_snapshot(st,ParticleDynamics) : nothing
    phase_before = BenchmarkMetadata.phase_snapshot(st,ParticleDynamics,opts["out"]*".phase-before.bin")
    CUDA.synchronize(); setup = time() - t0
    work = nothing
    t1 = time()
    if profile == "cupti"
        res = CUDA.@profile trace=false begin
            work = runwindow(opts["out"]*".io.csv",warm,steps)
        end
        open(io -> show(IOContext(io, :displaysize => (2000, 300)), res), opts["out"]*".cupti.txt", "w")
    elseif profile == "1" # nsys capture includes the same transfers/output
        CUDA.@profile external=true begin
            work = runwindow(opts["out"]*".io.csv",warm,steps)
        end
    else
        work = runwindow(opts["out"]*".io.csv",warm,steps)
    end
    loop = time() - t1
    device_work = work_audit ? BenchmarkMetadata.work_delta(device_before,BenchmarkMetadata.work_snapshot(st,ParticleDynamics),steps) : nothing
    phase_work = BenchmarkMetadata.phase_delta(phase_before,BenchmarkMetadata.phase_snapshot(st,ParticleDynamics,opts["out"]*".phase-after.bin"),steps)
    window_after = audit ? BenchmarkMetadata.window_frame(st,opts["out"]*".window-after") : nothing
    window_audit = audit ? Dict("before"=>window_before,"after"=>window_after) : nothing
    st.engine isa DE.StateEngine || error("the device engine was not used")
    L = Float64.(st.box3)
    stats = backend === :cluster ? ParticleDynamics.ClusterPairs.statistics(st.cluster) :
            (pairs_per_particle=-1.0, builds=-1, prunes=-1)
    rec = Dict("engine" => "ParticleDynamics", "ensemble" => ens, "n_ions" => N, "steps" => steps, "timed_steps" => steps,
        "ms_per_step" => 1e3 * loop / steps, "ns_per_day" => steps * 0.002e-3 / (loop / 86400),
        "setup_s" => setup, "loop_s" => loop, "beta" => beta, "mesh" => collect(mesh), "final_box" => L,
        "density_kg_m3" => sum(m) / 6.02214076e23 * 1e-3 / (prod(L) * 1e-27),
        "pairs_per_particle" => stats.pairs_per_particle, "builds" => stats.builds, "prunes" => stats.prunes,
        "skin" => parse(Float64, opt("skin", "0.2")), "prune_buffer" => parse(Float64, opt("prune", "0.1")),
        "charges" => charged, "masses" => opt("masses", "real"), "backend" => String(backend),
        "method" => opt("method", "pme"), "tolerance" => opt("tolerance", ""))
    settings = Dict{String,Any}("n_atoms"=>N,"n_molecules"=>N,"dt"=>Float64(st.dt),
        "ensemble"=>ens,"backend"=>String(nonbonded_backend(st)),"method"=>opt("method","pme"),
        "mesh"=>charged ? collect(st.electrostatics.ws.mesh) : nothing,
        "lj_cutoff"=>rc,"lj_modifier"=>"potential-shift","dispersion_correction"=>"no",
        "com_removal"=>"none","thermostat_interval"=>ens=="nve" ? 0 : parse(Int,opt("nsttcouple","1")),
        "energy_interval"=>2000,"log_interval"=>2000,"requested_observable_phase"=>mod(-warm,2000),
        "graphs"=>st.engine isa DE.StateEngine,"offload"=>"gpu",
        "cadence_unavailable_reason"=>"actual K/virial/reduction/transfer work requires event instrumentation; output is not computation")
    settings = merge(initial_model,settings,solver_model)
    BenchmarkMetadata.attach!(rec,provenance,settings,CUDA;warm,steps,state=st,window_audit=window_audit,device_work=device_work,phase_work=phase_work,
        profiled=profile!="0",observed_work=work,requested_events=count(i->(warm+i)%2000==0,1:steps),requested_backend=String(backend))
    open(io -> JSON3.pretty(io, rec), opts["out"], "w")
    println(JSON3.write(rec))
end
abspath(PROGRAM_FILE) == @__FILE__() && main()
