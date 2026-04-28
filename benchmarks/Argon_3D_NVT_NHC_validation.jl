using NonEqSimGPU
using NonEqSimGPU: step!, collect_step_observables
using CUDA
using Printf
using Random

include(joinpath(@__DIR__, "argon_nvt_common.jl"))

CUDA.allowscalar(false)
Random.seed!(0xB47E)

mutable struct RunningMoments
    n::Int
    mean::Float64
    m2::Float64
end

RunningMoments() = RunningMoments(0, 0.0, 0.0)

function update!(s::RunningMoments, x::Real)
    s.n += 1
    δ = float(x) - s.mean
    s.mean += δ / s.n
    s.m2 += δ * (float(x) - s.mean)
    return s
end

function std_error(s::RunningMoments)
    s.n <= 1 && return 0.0
    return sqrt(s.m2 / (s.n - 1)) / sqrt(s.n)
end

function main()
    # -------------------------------------------------------------------------
    # Rigorous literature benchmark:
    # NIST LJ NVT Monte Carlo reference table (rc=3σ + standard LRC).
    # Default state point: T*=0.90, rho*=0.005 (stable dilute vapor).
    # Optional dense liquid benchmark: T0.90_RHO0.820.
    # -------------------------------------------------------------------------
    ref_key = get(ENV, "NEQSIM_ARGON_VAL_NIST_POINT", "T0.90_RHO0.005")
    ref = nist_lj_mc_reference(ref_key)

    N = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_N", "10000"))
    dt_star = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_DT_STAR", "0.001"))
    warmup_steps = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_WARMUP_STEPS", "20000"))
    sample_steps = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_SAMPLE_STEPS", "1000000"))
    sample_stride = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_SAMPLE_STRIDE", "100000"))

    nhc_tau_star = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_NHC_TAU_STAR", "1.0"))
    nhc_chain_length = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_NHC_CHAIN_LENGTH", "2"))
    nhc_substeps = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_NHC_SUBSTEPS", "3"))

    write_gsd = parse_bool_env("NEQSIM_ARGON_VAL_WRITE_GSD", true)
    gsd_interval = parse(Int, get(ENV, "NEQSIM_ARGON_VAL_GSD_INTERVAL", string(sample_stride)))
    gsd_sync_on_write = parse_bool_env("NEQSIM_ARGON_VAL_GSD_SYNC_ON_WRITE", true)

    # Benchmark table uses truncation rc=3σ and standard LRC.
    rcut_star = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_RCUT_STAR", "3.0"))

    u_rel_tol = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_U_REL_TOL", "0.05"))
    u_abs_tol = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_U_ABS_TOL", "0.03"))
    p_rel_tol = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_P_REL_TOL", "0.08"))
    p_abs_tol = parse(Float64, get(ENV, "NEQSIM_ARGON_VAL_P_ABS_TOL", "5e-4"))

    T_star = ref.T_star
    rho_star = ref.rho_star
    L_star = argon_box_length_reduced(N, rho_star)

    precision_tag = Symbol(lowercase(strip(get(ENV, "NEQSIM_ARGON_VAL_PRECISION", "f32"))))
    T = if precision_tag === :f64
        Float64
    elseif precision_tag === :f32
        Float32
    else
        error("Unsupported precision tag $(precision_tag). Use f32 or f64.")
    end
    box = (T(L_star), T(L_star), T(L_star))

    println("Argon NVT validation against NIST LJ benchmark")
    @printf("  Reference key: %s\n", ref_key)
    @printf("  Source: %s\n", ref.source)
    @printf("  Target state: T* = %.5f, rho* = %.5f, N = %d, L* = %.5f\n", T_star, rho_star, N, L_star)
    @printf("  Target values: U* = %.7f +/- %.2e, P* = %.7f +/- %.2e\n", ref.U_ref, ref.U_sigma, ref.P_ref, ref.P_sigma)
    @printf("  Sim setup: dt* = %.4g, warmup = %d, sample_steps = %d, stride = %d, rc* = %.2f\n",
            dt_star, warmup_steps, sample_steps, sample_stride, rcut_star)
    @printf("  Precision: %s\n", precision_tag)

    st = build_simulation(
        N=N,
        box=box,
        cutoff=T(rcut_star),
        skin=T(0.5),
        cap=Int32(96),
        neigh_interval=5,
        epsilon=T(1),
        sigma=T(1),
        mass=T(1),
        gamma=T(1),
        temperature=T(T_star),
        dt=T(dt_star),
        nonbonded=:lj,
        precision=precision_tag,
    )

    initialize_simple_cubic_lattice!(st, box; jitter_frac=T(0.10))

    spec, integrator_label, _ = select_nvt_integrator(
        st;
        temperature_reduced=T_star,
        dt=dt_star,
        nhc_tau_reduced=nhc_tau_star,
        nhc_chain_length=nhc_chain_length,
        nhc_substeps=nhc_substeps,
    )
    println("Integrator: $(integrator_label)")

    obs_path = joinpath(@__DIR__, "obs_argon_nvt_nhc_validation.csv")
    gsd_path = joinpath(@__DIR__, "traj_argon_nvt_nhc_validation.gsd")
    if isfile(obs_path)
        rm(obs_path; force=true)
    end
    if write_gsd && isfile(gsd_path)
        rm(gsd_path; force=true)
    end

    u_tail = lj_tail_energy_per_particle(rho_star, rcut_star)
    p_tail = lj_tail_pressure(rho_star, rcut_star)
    @printf("  Applied standard LRC: u_tail* = %.6e, p_tail* = %.6e\n", u_tail, p_tail)

    u_stats = RunningMoments()
    p_stats = RunningMoments()
    t_stats = RunningMoments()

    function run_validation!(gsdh)
        println("Warmup...")
        for _ in 1:warmup_steps
            step!(st, spec, T(dt_star); compute_energy=false)
        end

        println("Sampling...")
        for i in 1:sample_steps
            step!(st, spec, T(dt_star); compute_energy=true)

            if gsdh !== nothing && i % gsd_interval == 0
                write_gsd_frame!(gsdh, st; diameter=1.0, types_names=["Ar"], step=st.step, sync_on_write=gsd_sync_on_write)
            end

            if i % sample_stride == 0
                obs = collect_step_observables(st, spec)
                T_inst, P_raw, _ = instantaneous_compressibility_factor(st, rho_star)
                U_raw = obs.Epot_total / N
                U_corr = U_raw + u_tail
                P_corr = P_raw + p_tail

                update!(u_stats, U_corr)
                update!(p_stats, P_corr)
                update!(t_stats, T_inst)

                write_observables_csv!(obs_path, st, spec)
            end
        end
    end

    if write_gsd
        println("Writing trajectory: $(gsd_path)")
        gsd_open(gsd_path) do gsdh
            write_gsd_frame!(gsdh, st; diameter=1.0, types_names=["Ar"], step=st.step, sync_on_write=gsd_sync_on_write)
            run_validation!(gsdh)
        end
    else
        run_validation!(nothing)
    end

    U_mean = u_stats.mean
    P_mean = p_stats.mean
    T_mean = t_stats.mean
    U_se = std_error(u_stats)
    P_se = std_error(p_stats)
    T_se = std_error(t_stats)

    du = abs(U_mean - ref.U_ref)
    dp = abs(P_mean - ref.P_ref)
    u_tol = max(u_abs_tol, u_rel_tol * abs(ref.U_ref))
    p_tol = max(p_abs_tol, p_rel_tol * abs(ref.P_ref))

    println("Validation summary (NIST MC benchmark)")
    @printf("  Samples: %d\n", u_stats.n)
    @printf("  <T*>      = %.6f +/- %.3e (target %.6f)\n", T_mean, T_se, T_star)
    @printf("  <U*>      = %.6f +/- %.3e (ref %.6f)\n", U_mean, U_se, ref.U_ref)
    @printf("  <P*>      = %.6f +/- %.3e (ref %.6f)\n", P_mean, P_se, ref.P_ref)
    @printf("  |ΔU*|     = %.6e (tol %.6e)\n", du, u_tol)
    @printf("  |ΔP*|     = %.6e (tol %.6e)\n", dp, p_tol)

    pass_u = du <= u_tol
    pass_p = dp <= p_tol
    if pass_u && pass_p
        println("PASS: LJ benchmark is within configured tolerances.")
    else
        println("WARN: LJ benchmark is outside configured tolerances.")
        println("      Increase warmup/sample length, reduce dt*, or tune NHC tau/substeps.")
    end

    println("CSV observables: $(obs_path)")
    if write_gsd
        println("GSD trajectory: $(gsd_path)")
    end
end

main()
