# Rigid SPC/E water NPT with ParticleDynamics.
# Shared GROMACS state (.gro with velocities, any precision) -> Float32 device engine:
# SPC/E (SETTLE geometry via SHAKE/RATTLE), LJ with potential shift, no dispersion
# correction, r_c = 1 nm, PME (SPME) or ESP on the cluster-pair backend, CSVR thermostat,
# molecular C-rescale every `nstpcouple` steps inside the device step graphs.
# Requested global observations/host CSV output every 2000 steps; coupling performs its required work.
# Default 6000 warm steps precede exactly `steps` measured steps; final frame/JSON follow timing.
# Usage: julia --project=<PD> pd_npt.jl key=value ...
#   gro= out= json= steps=10000 method=pme|esp|pppm
#   pme=alpha,nx,ny,nz,order   (alpha in 1/nm)      esp=tolerance[,nx,ny,nz,P]
#   pppm=alpha,nx,ny,nz,order[,ik|ad]  (P3M influence function, ik or analytical differentiation)
#   or tuned settings with the grid resolved for the box from a spacing (tune_pd_electrostatics.jl):
#   pme_tuned=alpha,order,spacing                    esp_tuned=split_shape,window_shape,order,spacing
#   pppm_tuned=alpha,order,spacing[,ik|ad]
#   rc_coul=1.0 (real-space Coulomb cutoff, nm; LJ stays at 1 nm)
#   skin=0.2 prune=0.1 nstpcouple=100 tau_p=5.0 tau_t=0.1 temperature=298 pressure=1.01325 seed=29
using ParticleDynamics, CUDA, Printf, JSON3
const SC = ParticleDynamics.SimulationCore
const DE = ParticleDynamics.DeviceEngine
const ES = ParticleDynamics.Electrostatics
const KB = 0.0083144626; const BAR = 0.0602214076   # kJ/mol/K; kJ/mol/nm^3 per bar
const QO = -0.8476; const QH = 0.4238; const SIG = 0.316557; const EPS = 0.650629
const MO = 15.9994; const MH = 1.008; const OH = 0.1; const HH = 0.16330

opts = Dict(split(a, "=", limit=2) for a in ARGS)
opt(k, d) = haskey(opts, k) ? opts[k] : d
include(joinpath(@__DIR__, "..", "common", "metadata.jl"))

"""GROMACS .gro with any precision (field width = distance between decimal points)."""
function read_gro(path)
    lines = readlines(path); n = parse(Int, strip(lines[2]))
    t = lines[3][21:end]; d1 = findfirst('.', t); w = findnext('.', t, d1 + 1) - d1
    R = zeros(3, n); V = zeros(3, n)
    for i in 1:n
        s = lines[2 + i][21:end]
        for k in 1:3
            R[k, i] = parse(Float64, s[(k - 1) * w + 1:k * w])
        end
        if length(s) >= 3w + 3(w + 1)
            for k in 1:3
                V[k, i] = parse(Float64, s[3w + (k - 1) * (w + 1) + 1:3w + k * (w + 1)])
            end
        end
    end
    box = Tuple(parse.(Float64, split(lines[3 + n])[1:3]))
    return R, V, box
end

function write_gro(path, R, V, box, title)
    open(path, "w") do io
        n = size(R, 2); println(io, title); println(io, n)
        for i in 1:n
            nm = (i - 1) % 3 == 0 ? "OW" : (i - 1) % 3 == 1 ? "HW1" : "HW2"
            @printf(io, "%5d%-5s%5s%5d%11.6f%11.6f%11.6f%12.7f%12.7f%12.7f\n", ((i - 1) ÷ 3 + 1) % 100000, "SOL", nm,
                    i % 100000, R[1, i], R[2, i], R[3, i], V[1, i], V[2, i], V[3, i])
        end
        @printf(io, "%12.7f%12.7f%12.7f\n", box...)
    end
end

friendly_mesh(box, h) = ntuple(d -> ES.fft_friendly(ceil(Int, box[d] / h - 1e-9)), 3)
function electrostatics_spec(box)
    rc = parse(Float64, opt("rc_coul", "1.0"))   # real-space Coulomb cutoff (LJ stays at 1 nm)
    if haskey(opts, "pme_tuned")
        a, P, h = parse.(Float64, split(opts["pme_tuned"], ","))
        return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.GaussianSplit(a),
            window=ES.BSplineWindow(Int(P)), mesh=friendly_mesh(box, h), influence=:spme)
    elseif haskey(opts, "pppm_tuned")
        f = split(opts["pppm_tuned"], ","); a, P, h = parse.(Float64, f[1:3])
        return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.GaussianSplit(a),
            window=ES.BSplineWindow(Int(P)), mesh=friendly_mesh(box, h),
            influence=(length(f) >= 4 && f[4] == "ad") ? :p3m_ad : :p3m_ik)
    elseif haskey(opts, "esp_tuned")
        c, c1, P, h = parse.(Float64, split(opts["esp_tuned"], ","))
        return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.ProlateSplit(rc, c),
            window=ES.ProlateWindow(Int(P), c1), mesh=friendly_mesh(box, h), influence=:esp)
    end
    if opt("method", "pme") == "pppm"
        f = split(opts["pppm"], ","); p = parse.(Float64, f[1:5])
        return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.GaussianSplit(p[1]),
            window=ES.BSplineWindow(Int(p[5])), mesh=(Int(p[2]), Int(p[3]), Int(p[4])),
            influence=(length(f) >= 6 && f[6] == "ad") ? :p3m_ad : :p3m_ik)
    end
    if opt("method", "pme") == "pme"
        p = parse.(Float64, split(opts["pme"], ","))
        return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.GaussianSplit(p[1]),
            window=ES.BSplineWindow(Int(p[5])), mesh=(Int(p[2]), Int(p[3]), Int(p[4])), influence=:spme)
    end
    e = parse.(Float64, split(opts["esp"], ","))
    tol = e[1]; c = ES.prolate_shape_for_tolerance(tol)
    P = length(e) >= 5 ? Int(e[5]) : ParticleDynamics.Workflow.esp_order_for_tolerance(tol)
    mesh = length(e) >= 4 ? (Int(e[2]), Int(e[3]), Int(e[4])) :
        ntuple(d -> max(ES.fft_friendly(ceil(Int, box[d] * c / (pi * rc))), P), 3)
    return ES.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=rc, split=ES.ProlateSplit(rc, c),
        window=ES.ProlateWindow(P, ES.prolate_shape_for_tolerance(tol / 2)), mesh=mesh, influence=:esp)
end

function water_state(R, V, box; skin)
    T = Float32; N = size(R, 2); nw = N ÷ 3
    q = repeat([QO, QH, QH], nw); m = repeat([MO, MH, MH], nw); ty = repeat(Int32[1, 2, 2], nw)
    ex = Tuple{Int,Int,Float64}[]
    for w in 1:nw
        o = 3w - 2; push!(ex, (o, o + 1, 0.0), (o, o + 2, 0.0), (o + 1, o + 2, 0.0))
    end
    # cap: the dense per-particle table of build_simulation is unused by the cluster backend that
    # charged states run on (N × 800 entries overflow Int32 at 3M atoms)
    st = SC.build_simulation(; N, box, nonbonded=:lj, cutoff=T(1.0), skin=T(skin), cap=Int32(16),
        epsilon=T(EPS), sigma=T(SIG), mass=T.(m), dt=T(parse(Float64, opt("dt", "0.002"))), precision=:f32,
        charges=q, spatial_reorder=false, neigh_interval=10_000, vdw_modifier=:potential_shift)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    copyto!(st.typeid, ty)
    st.sigma_pair = CuArray(T.([SIG 0.0; 0.0 0.0])); st.epsilon_pair = CuArray(T.([EPS 0.0; 0.0 0.0]))
    st.rcut_pair = CuArray(T.([1.0 0.0; 0.0 0.0]))
    attach_electrostatics!(st, electrostatics_spec(box); exceptions=ex)
    attach_rigid_water!(st, [(3w - 2, 3w - 1, 3w) for w in 1:nw]; oh=OH, hh=HH)
    return st
end

function main()
    provenance = BenchmarkMetadata.begin_case([opts["gro"]])
    R, V, box = read_gro(opts["gro"]); N = size(R, 2); nw = N ÷ 3
    steps = parse(Int, opt("steps", "10000")); nst = parse(Int, opt("nstpcouple", "100"))
    temperature = parse(Float64, opt("temperature", "298")); kT = KB * temperature
    t0 = time()
    st = water_state(R, V, box; skin=parse(Float64, opt("skin", "0.2")))
    # working-list buffer of the cluster backend (tuned at 100k: skin 0.2, prune buffer 0.1)
    # intervals=nstlist,nstprune: fixed list intervals (GROMACS-style; with GROMACS' buffers from its log)
    iv = haskey(opts, "intervals") ? Tuple(parse.(Int, split(opts["intervals"], ","))) : nothing
    set_nonbonded_backend!(st, :cluster; prune_buffer=parse(Float64, opt("prune", "0.1")), list_intervals=iv)
    sp = SC.csvr(st; temperature=Float32(kT), tau=Float32(parse(Float64, opt("tau_t", "0.1"))), interval=parse(Int,opt("nsttcouple","1")))
    cr = SC.StochasticCellRescalingState(st; pressure=parse(Float64, opt("pressure", "1.01325")) * BAR, kT,
        compressibility=4.5e-5 / BAR, tau=parse(Float64, opt("tau_p", "5.0")), seed=parse(Int, opt("seed", "29")))
    SC.attach_pressure_control!(st, cr; interval=nst)
    initial_model = BenchmarkMetadata.initial_model(st;mixing="explicit pair table; zero LJ on hydrogen",kinetic_dof=SC.kinetic_dof(st))
    solver_model = BenchmarkMetadata.solver_settings(st.electrostatics)
    SC.evaluate_forces_into_f!(st, true)          # list build, compilation and first forces before timing
    SC.step!(st, sp, st.dt; compute_energy=false) # capture of the step graphs (counted below)
    SC.step!(st,sp,st.dt;compute_energy=true); pilot=SC.collect_step_observables(st,sp)
    BenchmarkMetadata.warm_output!(opts["json"],pilot,st.box3[1])
    warm = parse(Int,opt("warm","6000")); warm >= 2 || error("warm must include at least two pilot steps")
    for i in 3:warm; SC.step!(st, sp, st.dt; compute_energy=(i % 2000 == 0)); end
    DE.synchronize_state!(st)
    advance! = sampled -> SC.step!(st,sp,st.dt;compute_energy=sampled)
    observe = () -> SC.collect_step_observables(st,sp)
    length_x = () -> st.box3[1]
    finish! = () -> (DE.synchronize_state!(st); CUDA.synchronize())
    runwindow = (path,offset,n) -> BenchmarkMetadata.run_observations!(advance!,observe,length_x,finish!,path,offset,n)
    runwindow(opts["json"]*".loop-warmup.csv",warm,0) # host compilation only; no added simulation steps
    profile = opt("profile","0")
    profile in ("0","cupti") || error("unsupported profile mode: $profile")
    audit = opt("audit_window","0")=="1"
    window_before = audit ? BenchmarkMetadata.window_frame(st,opts["json"]*".window-before") : nothing
    work_audit = opt("work_audit","0")=="1"
    device_before = work_audit ? BenchmarkMetadata.work_snapshot(st,ParticleDynamics) : nothing
    phase_before = BenchmarkMetadata.phase_snapshot(st,ParticleDynamics,opts["json"]*".phase-before.bin")
    CUDA.synchronize(); setup = time() - t0
    work = nothing
    t1 = time()
    if profile == "cupti"
        res = CUDA.@profile trace=false begin
            work = runwindow(opts["json"]*".io.csv",warm,steps)
        end
        open(io -> show(IOContext(io, :displaysize => (2000, 300)), res), opts["json"]*".cupti.txt", "w")
    else
        work = runwindow(opts["json"]*".io.csv",warm,steps)
    end
    loop = time() - t1
    device_work = work_audit ? BenchmarkMetadata.work_delta(device_before,BenchmarkMetadata.work_snapshot(st,ParticleDynamics),steps) : nothing
    phase_work = BenchmarkMetadata.phase_delta(phase_before,BenchmarkMetadata.phase_snapshot(st,ParticleDynamics,opts["json"]*".phase-after.bin"),steps)
    window_after = audit ? BenchmarkMetadata.window_frame(st,opts["json"]*".window-after") : nothing
    window_audit = audit ? Dict("before"=>window_before,"after"=>window_after) : nothing
    st.engine isa DE.StateEngine || error("the device engine was not used")
    L = Float64.(st.box3)
    Rf = vcat((Float64.(Array(x))' for x in (st.rx, st.ry, st.rz))...)
    Vf = vcat((Float64.(Array(x))' for x in (st.vx, st.vy, st.vz))...)
    write_gro(opts["out"], Rf, Vf, L, "ParticleDynamics final frame step $steps")
    vol = prod(L)
    eskey = first(k for k in ("pme_tuned", "esp_tuned", "pppm_tuned", "pme", "esp", "pppm") if haskey(opts, k))
    rec = Dict("engine" => "ParticleDynamics", "precision" => "Float32", "method" => opt("method", "pme"),
        "electrostatics" => eskey * "=" * opts[eskey], "mesh" => collect(st.electrostatics.ws.mesh),
        "skin" => parse(Float64, opt("skin", "0.2")), "prune_buffer" => parse(Float64, opt("prune", "0.1")),
        "n_atoms" => N, "n_waters" => nw,
        "steps" => steps, "timed_steps" => steps, "dt" => Float64(st.dt), "nstpcouple" => nst,
        "barostat_events" => cr.events, "barostat" => "C-rescale (device)",
        "setup_s" => setup, "loop_s" => loop, "ms_per_step" => 1e3 * loop / steps,
        "ns_per_day" => steps * Float64(st.dt) * 1e-3 / (loop / 86400), "final_box" => L,
        "final_density_kg_m3" => nw * (MO + 2MH) / 6.02214076e23 * 1e-3 / (vol * 1e-27),
        "backend" => String(nonbonded_backend(st)))
    settings=Dict{String,Any}("n_atoms"=>N,"n_molecules"=>nw,"dt"=>Float64(st.dt),"ensemble"=>"npt",
        "backend"=>String(nonbonded_backend(st)),"method"=>opt("method","pme"),"mesh"=>collect(st.electrostatics.ws.mesh),
        "lj_cutoff"=>1.0,"lj_modifier"=>"potential-shift","dispersion_correction"=>"no","com_removal"=>"none",
        "thermostat_interval"=>parse(Int,opt("nsttcouple","1")),"barostat_interval"=>nst,
        "energy_interval"=>2000,"log_interval"=>2000,"requested_observable_phase"=>mod(-warm,2000),
        "graphs"=>st.engine isa DE.StateEngine,"offload"=>"gpu",
        "cadence_unavailable_reason"=>"actual K/virial/reduction/transfer work requires event instrumentation; output is not computation")
    settings = merge(initial_model,settings,solver_model)
    BenchmarkMetadata.attach!(rec,provenance,settings,CUDA;warm,steps,state=st,window_audit=window_audit,device_work=device_work,phase_work=phase_work,
        profiled=profile!="0",observed_work=work,requested_events=count(i->(warm+i)%2000==0,1:steps),requested_backend="cluster")
    open(io -> JSON3.pretty(io, rec), opts["json"], "w")
    println(JSON3.write(rec))
end
abspath(PROGRAM_FILE) == @__FILE__() && main()
