using NonEqSimGPU
using NonEqSimGPU: step!, collect_step_observables
using CUDA
using Random
using Printf
using Dates

include(joinpath(@__DIR__, "argon_nvt_common.jl"))

CUDA.allowscalar(false)
Random.seed!(0xB47E)

mutable struct RunningMoments
    n::Int
    mean::Float64
    m2::Float64
end

RunningMoments() = RunningMoments(0, 0.0, 0.0)

function update!(stats::RunningMoments, x::Real)
    stats.n += 1
    delta = float(x) - stats.mean
    stats.mean += delta / stats.n
    stats.m2 += delta * (float(x) - stats.mean)
    return stats
end

function std_error(stats::RunningMoments)
    stats.n <= 1 && return 0.0
    return sqrt(stats.m2 / (stats.n - 1)) / sqrt(stats.n)
end

function parse_float_list_env(key::AbstractString, default::Vector{Float64})
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    vals = Float64[]
    for token in split(raw, ",")
        t = strip(token)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    isempty(vals) ? default : vals
end

function parse_string_list_env(key::AbstractString, default::Vector{String})
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    vals = String[]
    for token in split(raw, ",")
        t = strip(token)
        isempty(t) && continue
        push!(vals, t)
    end
    isempty(vals) ? default : vals
end

@inline function minimum_image_host(dx::Float64, L::Float64)
    half = 0.5 * L
    if dx > half
        return dx - L
    elseif dx < -half
        return dx + L
    end
    return dx
end

"""
    count_pairs_within_cutoff(st, rcut_star)

Count interacting pairs `(i, j)` with `i < j` and `r_ij < rcut_star` for the
current snapshot, using the same minimum-image convention as the force kernels.
"""
function count_pairs_within_cutoff(st, rcut_star::Real)
    st.rz === nothing && error("Pair counting helper currently expects a 3D state.")
    box = st.box3
    box === nothing && error("3D box not found in simulation state.")

    rx = Float64.(Array(st.rx))
    ry = Float64.(Array(st.ry))
    rz = Float64.(Array(st.rz))
    Lx = Float64(box[1])
    Ly = Float64(box[2])
    Lz = Float64(box[3])
    rc2 = float(rcut_star)^2

    N = length(rx)
    npairs = 0
    @inbounds for i in 1:(N - 1)
        xi = rx[i]
        yi = ry[i]
        zi = rz[i]
        for j in (i + 1):N
            dx = minimum_image_host(xi - rx[j], Lx)
            dy = minimum_image_host(yi - ry[j], Ly)
            dz = minimum_image_host(zi - rz[j], Lz)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2
                npairs += 1
            end
        end
    end
    return npairs
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n == length(y) || error("x/y length mismatch in linear_fit.")
    n >= 2 || error("Need at least two points for linear_fit.")
    mx = sum(x) / n
    my = sum(y) / n
    sxx = 0.0
    sxy = 0.0
    @inbounds for i in eachindex(x)
        dx = x[i] - mx
        sxx += dx * dx
        sxy += dx * (y[i] - my)
    end
    slope = sxy / sxx
    intercept = my - slope * mx
    return intercept, slope
end

function recommended_neighbor_cap(rho_star::Real)
    if rho_star >= 0.80
        return Int32(512)
    elseif rho_star >= 0.70
        return Int32(384)
    elseif rho_star >= 0.20
        return Int32(192)
    end
    return Int32(96)
end

"""
    run_argon_nvt_case(; kwargs...) -> NamedTuple

Run a single NHC-NVT LJ case and return summary statistics and sampled series.
All energies/pressures in the returned summary are corrected with standard LJ
tail terms (`rcut_star` dependent) to match NIST benchmark conventions.
"""
function run_argon_nvt_case(;
                            ref_key::String,
                            N::Int,
                            dt_star::Float64,
                            warmup_steps::Int,
                            sample_steps::Int,
                            sample_stride::Int,
                            nhc_tau_star::Float64,
                            nhc_chain_length::Int,
                            nhc_substeps::Int,
                            rcut_star::Float64=3.0,
                            cap::Int32=Int32(96),
                            collect_ekin::Bool=false,
                            collect_extended_h::Bool=false,
                            compute_pair_count::Bool=false)
    ref = nist_lj_mc_reference(ref_key)
    rho_star = ref.rho_star
    T_star = ref.T_star
    L_star = argon_box_length_reduced(N, rho_star)

    T = Float32
    box = (T(L_star), T(L_star), T(L_star))

    st = build_simulation(
        N=N,
        box=box,
        cutoff=T(rcut_star),
        skin=T(0.55),
        cap=cap,
        neigh_interval=2,
        epsilon=T(1),
        sigma=T(1),
        mass=T(1),
        gamma=T(1),
        temperature=T(T_star),
        dt=T(dt_star),
        nonbonded=:lj,
        precision=:f32,
    )

    initialize_simple_cubic_lattice!(st, box; jitter_frac=T(0.10))
    spec, _, _ = select_nvt_integrator(
        st;
        temperature_reduced=T_star,
        dt=dt_star,
        nhc_tau_reduced=nhc_tau_star,
        nhc_chain_length=nhc_chain_length,
        nhc_substeps=nhc_substeps,
    )

    for _ in 1:warmup_steps
        step!(st, spec, T(dt_star); compute_energy=false)
    end

    nsamples_target = sample_steps ÷ sample_stride
    sample_steps_series = Int[]
    T_series = Float64[]
    U_raw_series = Float64[]
    U_corr_series = Float64[]
    P_corr_series = Float64[]
    K_series = Float64[]
    Hext_series = Float64[]

    sizehint!(sample_steps_series, nsamples_target)
    sizehint!(T_series, nsamples_target)
    sizehint!(U_raw_series, nsamples_target)
    sizehint!(U_corr_series, nsamples_target)
    sizehint!(P_corr_series, nsamples_target)
    collect_ekin && sizehint!(K_series, nsamples_target)
    collect_extended_h && sizehint!(Hext_series, nsamples_target)

    u_tail = lj_tail_energy_per_particle(rho_star, rcut_star)
    p_tail = lj_tail_pressure(rho_star, rcut_star)

    T_stats = RunningMoments()
    Uraw_stats = RunningMoments()
    Ucorr_stats = RunningMoments()
    Pcorr_stats = RunningMoments()

    for i in 1:sample_steps
        step!(st, spec, T(dt_star); compute_energy=true)
        if i % sample_stride != 0
            continue
        end

        obs = collect_step_observables(st, spec)
        T_inst, P_raw, _ = instantaneous_compressibility_factor(st, rho_star)
        U_raw = obs.Epot_total / N
        U_corr = U_raw + u_tail
        P_corr = P_raw + p_tail

        update!(T_stats, T_inst)
        update!(Uraw_stats, U_raw)
        update!(Ucorr_stats, U_corr)
        update!(Pcorr_stats, P_corr)

        push!(sample_steps_series, st.step)
        push!(T_series, float(T_inst))
        push!(U_raw_series, float(U_raw))
        push!(U_corr_series, float(U_corr))
        push!(P_corr_series, float(P_corr))
        if collect_ekin
            push!(K_series, float(obs.Ekin_total))
        end
        if collect_extended_h
            push!(Hext_series, float(obs.extended_hamiltonian))
        end
    end

    pair_count = compute_pair_count ? count_pairs_within_cutoff(st, rcut_star) : -1

    return (
        ref_key=ref_key,
        ref=ref,
        N=N,
        dt_star=dt_star,
        warmup_steps=warmup_steps,
        sample_steps=sample_steps,
        sample_stride=sample_stride,
        nhc_tau_star=nhc_tau_star,
        nhc_chain_length=nhc_chain_length,
        nhc_substeps=nhc_substeps,
        rcut_star=rcut_star,
        cap=Int(cap),
        u_tail=u_tail,
        p_tail=p_tail,
        T_mean=T_stats.mean,
        T_se=std_error(T_stats),
        U_raw_mean=Uraw_stats.mean,
        U_raw_se=std_error(Uraw_stats),
        U_corr_mean=Ucorr_stats.mean,
        U_corr_se=std_error(Ucorr_stats),
        P_corr_mean=Pcorr_stats.mean,
        P_corr_se=std_error(Pcorr_stats),
        sample_steps_series=sample_steps_series,
        T_series=T_series,
        U_raw_series=U_raw_series,
        U_corr_series=U_corr_series,
        P_corr_series=P_corr_series,
        K_series=K_series,
        Hext_series=Hext_series,
        pair_count=pair_count,
    )
end

function push_metric!(rows::Vector{NamedTuple};
                      section::String,
                      test::String,
                      metric::String,
                      value::Float64,
                      unit::String="",
                      status::String="",
                      note::String="")
    push!(rows, (
        section=section,
        test=test,
        metric=metric,
        value=value,
        unit=unit,
        status=status,
        note=note,
    ))
    return nothing
end

function write_metrics_csv(path::String, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, "section,test,metric,value,unit,status,note")
        for r in rows
            note_clean = replace(r.note, "," => ";")
            @printf(io, "%s,%s,%s,%.10e,%s,%s,%s\n",
                    r.section, r.test, r.metric, r.value, r.unit, r.status, note_clean)
        end
    end
end

function write_text_report(path::String, lines::Vector{String})
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function main()
    mode = lowercase(strip(get(ENV, "NEQSIM_NHC_BENCH_MODE", "quick")))
    full_mode = mode == "full"

    dt_grid_default = full_mode ? [1.0e-3, 5.0e-4, 2.5e-4] : [1.0e-3, 5.0e-4]
    dt_grid = parse_float_list_env("NEQSIM_NHC_BENCH_DT_GRID", dt_grid_default)
    multi_point_keys_default = ["T0.90_RHO0.005", "T0.85_RHO0.776", "T0.90_RHO0.820"]
    multi_point_keys = parse_string_list_env("NEQSIM_NHC_BENCH_POINTS", multi_point_keys_default)

    base_ref_key = get(ENV, "NEQSIM_NHC_BENCH_BASE_POINT", "T0.90_RHO0.005")
    nhc_tau_star = parse(Float64, get(ENV, "NEQSIM_NHC_BENCH_TAU", "1.0"))
    nhc_chain_length = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CHAIN_LENGTH", "5"))
    nhc_substeps = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_SUBSTEPS", "10"))
    rcut_star = parse(Float64, get(ENV, "NEQSIM_NHC_BENCH_RCUT", "3.0"))

    potential_N = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_POT_N", full_mode ? "8192" : "2048"))
    potential_warmup = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_POT_WARMUP", full_mode ? "10000" : "3000"))
    potential_sample = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_POT_SAMPLE", full_mode ? "200000" : "30000"))
    potential_stride = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_POT_STRIDE", full_mode ? "5000" : "1000"))

    conv_N = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CONV_N", full_mode ? "8192" : "2048"))
    conv_warmup = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CONV_WARMUP", full_mode ? "12000" : "3000"))
    conv_sample = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CONV_SAMPLE", full_mode ? "240000" : "60000"))
    conv_stride = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CONV_STRIDE", full_mode ? "6000" : "2000"))

    canon_N = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CANON_N", full_mode ? "10000" : "4096"))
    canon_warmup = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CANON_WARMUP", full_mode ? "20000" : "5000"))
    canon_sample = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CANON_SAMPLE", full_mode ? "500000" : "120000"))
    canon_stride = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_CANON_STRIDE", full_mode ? "5000" : "1000"))
    drift_rel_tol_default = full_mode ? "1.5e-1" : "1e-1"
    drift_rel_tol = parse(Float64, get(ENV, "NEQSIM_NHC_BENCH_DRIFT_REL_TOL", drift_rel_tol_default))

    eos_N = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_EOS_N", full_mode ? "8192" : "2048"))
    eos_warmup = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_EOS_WARMUP", full_mode ? "10000" : "3000"))
    eos_sample = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_EOS_SAMPLE", full_mode ? "200000" : "60000"))
    eos_stride = parse(Int, get(ENV, "NEQSIM_NHC_BENCH_EOS_STRIDE", full_mode ? "5000" : "2000"))
    eos_u_abs_tol = parse(Float64, get(ENV, "NEQSIM_NHC_BENCH_EOS_U_ABS_TOL", full_mode ? "5e-3" : "1e-2"))
    eos_p_abs_tol = parse(Float64, get(ENV, "NEQSIM_NHC_BENCH_EOS_P_ABS_TOL", full_mode ? "1.5e-2" : "2e-2"))

    report_path = joinpath(@__DIR__, "nhc_validation_report.txt")
    metrics_path = joinpath(@__DIR__, "nhc_validation_metrics.csv")

    lines = String[]
    rows = NamedTuple[]

    push!(lines, "NonEqSimGPU NHC Validation Benchmark Suite")
    push!(lines, "Generated at: $(Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "Mode: $(mode)")
    if full_mode
        push!(lines, "Interpretation: full mode is intended for quantitative acceptance checks.")
    else
        push!(lines, "Interpretation: quick mode is a smoke/regression run; use full mode for quantitative acceptance.")
    end
    push!(lines, "Base point: $(base_ref_key)")
    push!(lines, "")

    # -------------------------------------------------------------------------
    # 1) Potential convention check
    # -------------------------------------------------------------------------
    base_ref = nist_lj_mc_reference(base_ref_key)
    potential_cap = recommended_neighbor_cap(base_ref.rho_star)
    potential_case = run_argon_nvt_case(
        ref_key=base_ref_key,
        N=potential_N,
        dt_star=maximum(dt_grid),
        warmup_steps=potential_warmup,
        sample_steps=potential_sample,
        sample_stride=potential_stride,
        nhc_tau_star=nhc_tau_star,
        nhc_chain_length=nhc_chain_length,
        nhc_substeps=nhc_substeps,
        rcut_star=rcut_star,
        cap=potential_cap,
        compute_pair_count=true,
    )

    u_rc = 4.0 * ((1.0 / rcut_star)^12 - (1.0 / rcut_star)^6)
    pair_density = potential_case.pair_count / potential_case.N
    shift_per_particle = -pair_density * u_rc
    U_unshifted = potential_case.U_corr_mean
    U_shifted = U_unshifted + shift_per_particle
    delta_unshifted = abs(U_unshifted - base_ref.U_ref)
    delta_shifted = abs(U_shifted - base_ref.U_ref)
    closer_status = delta_unshifted <= delta_shifted ? "UNSHIFTED_CLOSER" : "SHIFTED_CLOSER"

    push!(lines, "1) Potential convention check")
    push!(lines, @sprintf("  Pair-count based shift estimate at rc*=%.3f: u(rc)=%.6e, shift_per_particle=%.6e", rcut_star, u_rc, shift_per_particle))
    push!(lines, @sprintf("  U*(unshifted+sLRC)=%.7f, |delta|=%.6e", U_unshifted, delta_unshifted))
    push!(lines, @sprintf("  U*(shifted+sLRC) =%.7f, |delta|=%.6e", U_shifted, delta_shifted))
    push!(lines, @sprintf("  Convention closer to NIST at this state point: %s", closer_status))
    push!(lines, "")

    push_metric!(rows; section="potential_convention", test=base_ref_key, metric="u_rc", value=u_rc, unit="reduced_energy", status="", note="")
    push_metric!(rows; section="potential_convention", test=base_ref_key, metric="pair_density_npairs_over_N", value=pair_density, unit="1", status="", note="")
    push_metric!(rows; section="potential_convention", test=base_ref_key, metric="shift_per_particle", value=shift_per_particle, unit="reduced_energy", status="", note="")
    push_metric!(rows; section="potential_convention", test=base_ref_key, metric="delta_unshifted", value=delta_unshifted, unit="reduced_energy", status=closer_status, note="")
    push_metric!(rows; section="potential_convention", test=base_ref_key, metric="delta_shifted", value=delta_shifted, unit="reduced_energy", status=closer_status, note="")

    # -------------------------------------------------------------------------
    # 2) Time-step convergence check
    # -------------------------------------------------------------------------
    U_dt = Float64[]
    P_dt = Float64[]
    push!(lines, "2) Time-step convergence")
    for dt in dt_grid
        case_dt = run_argon_nvt_case(
            ref_key=base_ref_key,
            N=conv_N,
            dt_star=dt,
            warmup_steps=conv_warmup,
            sample_steps=conv_sample,
            sample_stride=conv_stride,
            nhc_tau_star=nhc_tau_star,
            nhc_chain_length=nhc_chain_length,
            nhc_substeps=nhc_substeps,
            rcut_star=rcut_star,
            cap=potential_cap,
        )
        push!(U_dt, case_dt.U_corr_mean)
        push!(P_dt, case_dt.P_corr_mean)
        push!(lines, @sprintf("  dt*=%.6f -> U*=%.7f, P*=%.7f", dt, case_dt.U_corr_mean, case_dt.P_corr_mean))
        push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="U_at_dt_$(dt)", value=case_dt.U_corr_mean, unit="reduced_energy", status="", note="")
        push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="P_at_dt_$(dt)", value=case_dt.P_corr_mean, unit="reduced_pressure", status="", note="")
    end
    U0, U_slope = linear_fit(dt_grid, U_dt)
    P0, P_slope = linear_fit(dt_grid, P_dt)
    U0_delta = abs(U0 - base_ref.U_ref)
    P0_delta = abs(P0 - base_ref.P_ref)
    push!(lines, @sprintf("  Extrapolated dt*->0: U0*=%.7f (|delta|=%.6e), slope=%.6e", U0, U0_delta, U_slope))
    push!(lines, @sprintf("  Extrapolated dt*->0: P0*=%.7f (|delta|=%.6e), slope=%.6e", P0, P0_delta, P_slope))
    push!(lines, "")
    push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="U_extrapolated_dt0", value=U0, unit="reduced_energy", status="", note="")
    push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="P_extrapolated_dt0", value=P0, unit="reduced_pressure", status="", note="")
    push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="U_slope_vs_dt", value=U_slope, unit="reduced_energy_per_dt", status="", note="")
    push_metric!(rows; section="timestep_convergence", test=base_ref_key, metric="P_slope_vs_dt", value=P_slope, unit="reduced_pressure_per_dt", status="", note="")

    # -------------------------------------------------------------------------
    # 3 + 4) Canonicality and extended-H drift
    # -------------------------------------------------------------------------
    canon_case = run_argon_nvt_case(
        ref_key=base_ref_key,
        N=canon_N,
        dt_star=maximum(dt_grid),
        warmup_steps=canon_warmup,
        sample_steps=canon_sample,
        sample_stride=canon_stride,
        nhc_tau_star=nhc_tau_star,
        nhc_chain_length=nhc_chain_length,
        nhc_substeps=nhc_substeps,
        rcut_star=rcut_star,
        cap=recommended_neighbor_cap(base_ref.rho_star),
        collect_ekin=true,
        collect_extended_h=true,
    )

    dof = 3 * canon_case.N
    kshape = 0.5 * dof
    T_target = base_ref.T_star
    K_expected_mean = 0.5 * dof * T_target
    K_expected_var = 0.5 * dof * T_target * T_target
    K_sample = canon_case.K_series
    nK = length(K_sample)
    K_mean = sum(K_sample) / nK
    K_var = sum((k - K_mean)^2 for k in K_sample) / (nK - 1)
    K_mean_se = sqrt(K_expected_var / nK)
    z_mean = abs(K_mean - K_expected_mean) / K_mean_se

    # Approximate standard error for sample variance under Gamma statistics.
    gamma_kurtosis = 3.0 + 6.0 / kshape
    mu4 = gamma_kurtosis * K_expected_var * K_expected_var
    var_var = (mu4 - ((nK - 3) / (nK - 1)) * K_expected_var * K_expected_var) / nK
    z_var = abs(K_var - K_expected_var) / sqrt(max(var_var, eps(Float64)))
    canonical_status = (z_mean < 4.0 && z_var < 4.0) ? "PASS" : "WARN"

    push!(lines, "3) Canonical kinetic-energy statistics")
    push!(lines, @sprintf("  dof=%d, samples=%d", dof, nK))
    push!(lines, @sprintf("  <K> sample=%.6f, expected=%.6f, z_mean=%.3f", K_mean, K_expected_mean, z_mean))
    push!(lines, @sprintf("  Var(K) sample=%.6f, expected=%.6f, z_var=%.3f", K_var, K_expected_var, z_var))
    push!(lines, @sprintf("  Status: %s", canonical_status))
    push!(lines, "")
    push_metric!(rows; section="canonicality", test=base_ref_key, metric="z_mean_kinetic", value=z_mean, unit="sigma", status=canonical_status, note="")
    push_metric!(rows; section="canonicality", test=base_ref_key, metric="z_var_kinetic", value=z_var, unit="sigma", status=canonical_status, note="")

    H = canon_case.Hext_series
    hs = canon_case.sample_steps_series
    H_intercept, H_slope = linear_fit(Float64.(hs), Float64.(H))
    H_mean = sum(H) / length(H)
    H_min = minimum(H)
    H_max = maximum(H)
    H_rel_range = abs(H_max - H_min) / max(abs(H_mean), 1.0e-12)
    H_drift_per_million = H_slope * 1.0e6
    drift_status = H_rel_range <= drift_rel_tol ? "PASS" : "WARN"

    push!(lines, "4) Extended-Hamiltonian drift")
    push!(lines, @sprintf("  Hext fit: intercept=%.7e, slope=%.7e per step", H_intercept, H_slope))
    push!(lines, @sprintf("  Drift over 1e6 steps: %.7e", H_drift_per_million))
    push!(lines, @sprintf("  Relative range (max-min)/|mean| = %.7e (tol %.3e)", H_rel_range, drift_rel_tol))
    push!(lines, @sprintf("  Status: %s", drift_status))
    push!(lines, "")
    push_metric!(rows; section="extended_h_drift", test=base_ref_key, metric="drift_per_million_steps", value=H_drift_per_million, unit="reduced_energy", status=drift_status, note="")
    push_metric!(rows; section="extended_h_drift", test=base_ref_key, metric="relative_range", value=H_rel_range, unit="1", status=drift_status, note="")

    # -------------------------------------------------------------------------
    # 5) Multi-point LJ EOS validation
    # -------------------------------------------------------------------------
    push!(lines, @sprintf("5) Multi-point LJ EOS validation (NIST MC, rc*=3 + sLRC; abs tolerances: |ΔU*|<=%.3e, |ΔP*|<=%.3e)", eos_u_abs_tol, eos_p_abs_tol))
    eos_pass = true
    for key in multi_point_keys
        ref = nist_lj_mc_reference(key)
        cap = recommended_neighbor_cap(ref.rho_star)
        case_eos = run_argon_nvt_case(
            ref_key=key,
            N=eos_N,
            dt_star=maximum(dt_grid),
            warmup_steps=eos_warmup,
            sample_steps=eos_sample,
            sample_stride=eos_stride,
            nhc_tau_star=nhc_tau_star,
            nhc_chain_length=nhc_chain_length,
            nhc_substeps=nhc_substeps,
            rcut_star=rcut_star,
            cap=cap,
        )
        dU = case_eos.U_corr_mean - ref.U_ref
        dP = case_eos.P_corr_mean - ref.P_ref
        zU = abs(dU) / ref.U_sigma
        zP = abs(dP) / ref.P_sigma
        pass_z = (zU <= 3.0 && zP <= 3.0)
        pass_abs = (abs(dU) <= eos_u_abs_tol && abs(dP) <= eos_p_abs_tol)
        point_status = (pass_z || pass_abs) ? "PASS" : "WARN"
        eos_pass &= point_status == "PASS"

        note = pass_z ? "z-score" : (pass_abs ? "abs-tolerance" : "none")
        push!(lines, @sprintf("  %s: U*=%.7f (delta=%.3e, z=%.2f), P*=%.7f (delta=%.3e, z=%.2f), criterion=%s -> %s",
                              key, case_eos.U_corr_mean, dU, zU, case_eos.P_corr_mean, dP, zP, note, point_status))
        push_metric!(rows; section="multi_point_eos", test=key, metric="U_delta", value=dU, unit="reduced_energy", status=point_status, note="")
        push_metric!(rows; section="multi_point_eos", test=key, metric="P_delta", value=dP, unit="reduced_pressure", status=point_status, note="")
        push_metric!(rows; section="multi_point_eos", test=key, metric="U_zscore", value=zU, unit="sigma", status=point_status, note=note)
        push_metric!(rows; section="multi_point_eos", test=key, metric="P_zscore", value=zP, unit="sigma", status=point_status, note=note)
    end
    push!(lines, @sprintf("  Aggregate multi-point status: %s", eos_pass ? "PASS" : "WARN"))
    push!(lines, "")

    write_text_report(report_path, lines)
    write_metrics_csv(metrics_path, rows)

    println("NHC benchmark suite complete.")
    println("Text report: $(report_path)")
    println("Metrics CSV: $(metrics_path)")
end

main()
