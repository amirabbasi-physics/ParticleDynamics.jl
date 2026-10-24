# Argon NVE/NVT/NPT comparison, ParticleDynamics side, every setting matched to gromacs_run.sh:
# plain LJ cutoff (σ, ε, mass from the .top), no dispersion correction, dt 0.00539087 ps,
# Float32, energies computed and one observables line written every 2000 steps (as GROMACS), CSVR τ_t 0.5 ps at 101.8 K every step, C-rescale
# τ_p 5 ps, compressibility 4.5e-5 bar⁻¹, every 10 steps, at PRESSURE bar. Pair list: the dense
# (ELL, sorted) or the cluster-pair backend, rebuilt on displacement (skin 0.55σ).
# Timing: setup (state, first forces, first step's graph capture) outside; exactly N steps timed after explicit warm-up.
# Usage: julia --project=<PD> pd_run.jl gro= top= pressure= ensemble= backend=ell|cluster out=JSON [steps=10000]
using ParticleDynamics, CUDA, Printf, JSON3
const SC = ParticleDynamics.SimulationCore
const DE = ParticleDynamics.DeviceEngine
const KB = 0.0083144626; const BAR = 0.0602214076
opts = Dict(split(a, "=", limit=2) for a in ARGS)
opt(k, d) = haskey(opts, k) ? opts[k] : d
include(joinpath(@__DIR__, "..", "molten_salt", "pd_run.jl"))   # read_gro, read_atomtypes (main() not run)

function argon_state()
    R, V, box, names = read_gro(opts["gro"]); N = size(R, 2)
    at = read_atomtypes(opts["top"]); m, _, sig, eps = at["AR"]
    ens = opts["ensemble"]; backend = Symbol(opt("backend", "ell")); steps = parse(Int, opt("steps", "10000"))
    T = Float32; rc = 2.5 * sig; skin = haskey(opts, "skin") ? parse(Float64, opts["skin"]) : 0.55 * sig; dt = 0.00539087; kT = KB * 101.8
    dens = N / prod(box)
    cap = Int32(ceil(1.5 * 4pi / 3 * (rc + skin)^3 * dens + 64))
    t0 = time()
    st = SC.build_simulation(; N, box, nonbonded=:lj, cutoff=T(rc), skin=T(skin), cap, epsilon=T(eps), sigma=T(sig),
        mass=T(m), dt=T(dt), precision=:f32, spatial_reorder=(backend === :ell), neigh_interval=20, temperature=0)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    # intervals=nstlist,nstprune: fixed list intervals on the cluster backend (opt-in, statistical buffer)
    iv = haskey(opts, "intervals") ? Tuple(parse.(Int, split(opts["intervals"], ","))) : nothing
    backend === :cluster && set_nonbonded_backend!(st, :cluster; prune_buffer=parse(Float64, opt("prune", string(0.5 * skin))), list_intervals=iv)
    sp = ens == "nve" ? SC.nve(st) : SC.csvr(st; temperature=T(kT), tau=T(0.5), interval=parse(Int, opt("nsttcouple", "1")))
    if ens == "npt"
        cr = SC.StochasticCellRescalingState(st; pressure=parse(Float64, opts["pressure"]) * BAR, kT,
            compressibility=4.5e-5 / BAR, tau=5.0, seed=29)
        SC.attach_pressure_control!(st, cr; interval=10)
    end
    initial_model = BenchmarkMetadata.initial_model(st;mixing="single LJ type",kinetic_dof=SC.kinetic_dof(st))
    SC.evaluate_forces_into_f!(st, true)
    return st, sp, (; N, ens, backend, steps, skin, m, dt, kT, t0, initial_model)
end

function argon_main()
    provenance = BenchmarkMetadata.begin_case([opts["gro"], opts["top"]])
    st, sp, c = argon_state(); N = c.N; ens = c.ens; backend = c.backend; steps = c.steps; skin = c.skin; t0 = c.t0
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
    profile in ("0","cupti") || error("unsupported profile mode: $profile")
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
    rec = Dict("engine" => "ParticleDynamics", "ensemble" => ens, "backend" => String(backend), "n_atoms" => N,
        "steps" => steps, "timed_steps" => steps, "ms_per_step" => 1e3 * loop / steps, "setup_s" => setup,
        "loop_s" => loop, "final_box" => L, "density_nm3" => N / prod(L), "skin_nm" => skin)
    if backend === :cluster
        s = ParticleDynamics.ClusterPairs.statistics(st.cluster)
        rec["builds"] = s.builds; rec["prunes"] = s.prunes; rec["pairs_per_particle"] = s.pairs_per_particle
    end
    settings = Dict{String,Any}("n_atoms"=>N,"n_molecules"=>N,"dt"=>Float64(st.dt),"ensemble"=>ens,
        "backend"=>String(nonbonded_backend(st)),"method"=>"none","energy_interval"=>2000,"log_interval"=>2000,
        "com_removal"=>"none","thermostat_interval"=>ens=="nve" ? 0 : parse(Int,opt("nsttcouple","1")),
        "requested_observable_phase"=>mod(-warm,2000),"graphs"=>st.engine isa DE.StateEngine,"offload"=>"gpu",
        "cadence_unavailable_reason"=>"actual K/virial/reduction/transfer work requires event instrumentation")
    settings = merge(c.initial_model,settings)
    BenchmarkMetadata.attach!(rec,provenance,settings,CUDA;warm,steps,state=st,window_audit=window_audit,device_work=device_work,phase_work=phase_work,
        profiled=profile!="0",observed_work=work,requested_events=count(i->(warm+i)%2000==0,1:steps),requested_backend=String(backend))
    open(io -> JSON3.pretty(io, rec), opts["out"], "w")
    println(JSON3.write(rec))
end
function argon_observables()
    st, sp, c = argon_state(); N = c.N; m = c.m; h = c.dt / 2
    rows = NTuple{5,Float64}[]
    for i in 1:c.steps
        SC.step!(st, sp, st.dt; compute_energy=(i % 100 == 0))
        if i % 100 == 0
            obs = SC.collect_step_observables(st, sp)
            L = Float64.(st.box3); V = prod(L)
            v = [Float64.(Array(x)) for x in (st.vx, st.vy, st.vz)]; f = [Float64.(Array(x)) for x in (st.fx, st.fy, st.fz)]
            K = 0.5 * m * sum(sum(abs2, v[d]) + sum(abs2, (h / m) .* f[d]) for d in 1:3)
            P = (2K + Float64(obs.virial_total)) / (3V) / BAR
            push!(rows, (i * c.dt, Float64(obs.Epot_total) / N, 2K / (3N * KB), P, N / V))
        end
    end
    open(opts["out"] * ".obs.csv", "w") do io
        println(io, "t_ps,U_per_atom,T_K,P_bar,density_nm3")
        for r in rows; println(io, join(r, ",")); end
    end
    DE.synchronize_state!(st)
    L = Float64.(st.box3)
    open(opts["out"] * ".final.gro", "w") do io
        println(io, "ParticleDynamics Argon final"); println(io, N)
        x = [Float64.(Array(a)) for a in (st.rx, st.ry, st.rz)]
        for i in 1:N
            @printf(io, "%5d%-5s%5s%5d%11.6f%11.6f%11.6f\n", mod(i, 100000), "AR", "AR", mod(i, 100000), x[1][i], x[2][i], x[3][i])
        end
        @printf(io, "%12.7f%12.7f%12.7f\n", L...)
    end
    println("observables written: ", opts["out"])
end
# Run only as a script, so other tools can include argon_state().
abspath(PROGRAM_FILE) == @__FILE__() && (get(opts, "obs", "0") == "1" ? argon_observables() : argon_main())
