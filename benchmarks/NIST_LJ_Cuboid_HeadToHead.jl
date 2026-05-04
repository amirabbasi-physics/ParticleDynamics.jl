using ParticleDynamics
using CUDA
using Printf
using Dates

CUDA.allowscalar(false)

const NIST_CUBOID_URL = "https://www.nist.gov/mml/csd/chemical-informatics-group/lennard-jones-fluid-reference-calculations-cuboid-cell"
const NIST_CUBOID_UPDATED = "2026-04-08"

const CONFIG_DIR = joinpath(@__DIR__, "lj_sample_configurations-tar")
const REPORT_PATH = joinpath(@__DIR__, "nist_cuboid_head_to_head_report.txt")
const CSV_PATH = joinpath(@__DIR__, "nist_cuboid_head_to_head.csv")

"""
    CuboidConfig

Single NIST cuboid-cell LJ configuration parsed from `lj_sample_config_periodic*.txt`.
Coordinates and box lengths are in reduced LJ units.
"""
struct CuboidConfig
    id::Int
    path::String
    box::NTuple{3,Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
end

"""
    SchemeReference

Reference values from the NIST cuboid-cell webpage table for one truncation
scheme on one configuration.
"""
struct SchemeReference
    label::String
    scheme::Symbol
    rc::Float64
    U_pair_raw::String
    W_pair_raw::String
    U_lrc_raw::String
end

"""
    ComparisonRow

One row in the head-to-head benchmark table.
"""
struct ComparisonRow
    config_id::Int
    N::Int
    rho_star::Float64
    scheme_label::String
    method::String
    rc::Float64
    U_pair_calc::Float64
    U_pair_ref::Float64
    U_pair_abs_err::Float64
    U_pair_tol::Float64
    U_pair_pass::Bool
    W_pair_calc::Float64
    W_pair_ref::Float64
    W_pair_abs_err::Float64
    W_pair_tol::Float64
    W_pair_pass::Bool
    U_lrc_calc::Float64
    U_lrc_ref::Float64
    U_lrc_abs_err::Float64
    U_lrc_tol::Float64
    U_lrc_pass::Bool
end

const NIST_CUBOID_REFS = Dict(
    1 => [
        SchemeReference("LRC_rc3", :lrc, 3.0, "-4.3515E+03", "-5.6867E+02", "-1.9849E+02"),
        SchemeReference("LRC_rc4", :lrc, 4.0, "-4.4675E+03", "-1.2639E+03", "-8.3769E+01"),
        SchemeReference("LFS_rc3", :lfs, 3.0, "-3.8709E+03",  "3.1754E+02",  "0.0000E0"),
    ],
    2 => [
        SchemeReference("LRC_rc3", :lrc, 3.0, "-6.9000E+02", "-5.6846E+02", "-2.4230E+01"),
        SchemeReference("LRC_rc4", :lrc, 4.0, "-7.0460E+02", "-6.5599E+02", "-1.0226E+01"),
        SchemeReference("LFS_rc3", :lfs, 3.0, "-6.2012E+02", "-4.4533E+02",  "0.0000E0"),
    ],
    3 => [
        SchemeReference("LRC_rc3", :lrc, 3.0, "-1.1467E+03", "-1.1649E+03", "-4.9622E+01"),
        SchemeReference("LRC_rc4", :lrc, 4.0, "-1.1754E+03", "-1.3371E+03", "-2.0942E+01"),
        SchemeReference("LFS_rc3", :lfs, 3.0, "-1.0210E+03", "-9.3578E+02",  "0.0000E0"),
    ],
    4 => [
        SchemeReference("LRC_rc3", :lrc, 3.0, "-1.6790E+01", "-4.6249E+01", "-5.4517E-01"),
        SchemeReference("LRC_rc4", :lrc, 4.0, "-1.7060E+01", "-4.7869E+01", "-2.3008E-01"),
        SchemeReference("LFS_rc3", :lfs, 3.0, "-1.5001E+01", "-4.3096E+01",  "0.0000E0"),
    ],
)

@inline function minimum_image(dx::Float64, L::Float64)
    half = 0.5 * L
    if dx > half
        return dx - L
    elseif dx < -half
        return dx + L
    end
    return dx
end

@inline function lj_value_and_derivative(r::Float64)
    invr = 1.0 / r
    invr2 = invr * invr
    invr6 = invr2 * invr2 * invr2
    invr12 = invr6 * invr6
    value = 4.0 * (invr12 - invr6)
    derivative = 24.0 * (invr^7 - 2.0 * invr^13)
    return value, derivative
end

"""
    lrc_total_energy(N, rho_star, rc)

Standard long-range correction contribution to the *total* LJ energy for
`N` particles, consistent with NIST's reduced-unit convention.
"""
function lrc_total_energy(N::Int, rho_star::Float64, rc::Float64)
    inv3 = rc^(-3)
    inv9 = rc^(-9)
    u_tail_per_particle = (8.0 * π * rho_star / 3.0) * ((inv9 / 3.0) - inv3)
    return N * u_tail_per_particle
end

"""
    parse_cuboid_config(path, id)

Parse one NIST periodic cuboid configuration text file.
"""
function parse_cuboid_config(path::String, id::Int)
    lines = readlines(path)
    length(lines) >= 3 || error("Malformed configuration file: $(path)")

    box_tokens = split(strip(lines[1]))
    length(box_tokens) == 3 || error("Malformed box line in $(path)")
    Lx = parse(Float64, box_tokens[1])
    Ly = parse(Float64, box_tokens[2])
    Lz = parse(Float64, box_tokens[3])

    N = parse(Int, strip(lines[2]))
    expected_lines = N + 2
    length(lines) >= expected_lines || error("Expected at least $(expected_lines) lines in $(path), found $(length(lines)).")

    x = Vector{Float64}(undef, N)
    y = Vector{Float64}(undef, N)
    z = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        fields = split(strip(lines[i + 2]))
        length(fields) >= 4 || error("Malformed atom line $(i + 2) in $(path)")
        x[i] = parse(Float64, fields[2])
        y[i] = parse(Float64, fields[3])
        z[i] = parse(Float64, fields[4])
    end

    return CuboidConfig(id, path, (Lx, Ly, Lz), x, y, z)
end

"""
    load_all_cuboid_configs(config_dir)

Load all `lj_sample_config_periodic*.txt` files and return them sorted by id.
"""
function load_all_cuboid_configs(config_dir::String)
    files = filter(f -> occursin(r"^lj_sample_config_periodic\d+\.txt$", f), readdir(config_dir))
    isempty(files) && error("No NIST cuboid configuration files found in $(config_dir).")

    pairs = Tuple{Int,String}[]
    for f in files
        m = match(r"periodic(\d+)\.txt$", f)
        m === nothing && continue
        push!(pairs, (parse(Int, m.captures[1]), joinpath(config_dir, f)))
    end
    sort!(pairs, by=first)

    return [parse_cuboid_config(path, id) for (id, path) in pairs]
end

"""
    build_static_state(cfg, rc)

Build a GPU simulation state (f64, all-pairs) and load the static coordinates.
"""
function build_static_state(cfg::CuboidConfig, rc::Float64)
    st = build_simulation(
        N=length(cfg.x),
        box=(cfg.box[1], cfg.box[2], cfg.box[3]),
        cutoff=rc,
        skin=0.2,
        cap=Int32(16),
        neigh_interval=20,
        use_neighborlist=false,
        epsilon=1.0,
        sigma=1.0,
        gamma=1.0,
        temperature=1.0,
        dt=1.0e-3,
        nonbonded=:lj,
        precision=:f64,
    )
    copyto!(st.rx, cfg.x)
    copyto!(st.ry, cfg.y)
    copyto!(st.rz, cfg.z)
    sync_unwrapped!(st)
    return st
end

"""
    compute_gpu_lrc_pair_quantities(cfg, rc)

Compute `U_pair` and `W_pair` on GPU for the LJ truncation-with-LRC scheme.
`U_LRC` is computed analytically from density and cutoff.
"""
function compute_gpu_lrc_pair_quantities(cfg::CuboidConfig, rc::Float64)
    st = build_static_state(cfg, rc)
    ParticleDynamics.Simulation.evaluate_forces_into_f!(st, true; freeze_spring=false)

    N = length(cfg.x)
    V = cfg.box[1] * cfg.box[2] * cfg.box[3]
    rho = N / V

    U_pair = Float64(CUDA.sum(st.Epot))
    W_pair = Float64(CUDA.sum(st.virial))
    U_lrc = lrc_total_energy(N, rho, rc)

    return (U_pair=U_pair, W_pair=W_pair, U_lrc=U_lrc)
end

"""
    compute_host_lfs_pair_quantities(cfg, rc)

Compute `U_pair` and `W_pair` for LJ with linear-force shift (LFS) at cutoff
on the host using direct pair summation and minimum-image convention.
"""
function compute_host_lfs_pair_quantities(cfg::CuboidConfig, rc::Float64)
    N = length(cfg.x)
    Lx, Ly, Lz = cfg.box
    rc2 = rc * rc
    Vc, dVc = lj_value_and_derivative(rc)

    U_pair = 0.0
    W_pair = 0.0

    @inbounds for i in 1:(N - 1)
        xi = cfg.x[i]
        yi = cfg.y[i]
        zi = cfg.z[i]
        for j in (i + 1):N
            dx = minimum_image(xi - cfg.x[j], Lx)
            dy = minimum_image(yi - cfg.y[j], Ly)
            dz = minimum_image(zi - cfg.z[j], Lz)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < rc2
                r = sqrt(r2)
                Vlj, dVlj = lj_value_and_derivative(r)

                # LFS: V = V_lj(r) - V_lj(rc) - V'_lj(rc) * (r - rc)
                V = Vlj - Vc - dVc * (r - rc)
                dVdr = dVlj - dVc

                U_pair += V
                W_pair += -r * dVdr
            end
        end
    end

    return (U_pair=U_pair, W_pair=W_pair, U_lrc=0.0)
end

@inline parse_nist_value(s::String) = parse(Float64, s)

"""
    nist_display_tolerance(s)

Return half of one least-significant displayed unit for scientific notation
value `s`, e.g. `-4.3515E+03` -> tolerance `0.05`.
"""
function nist_display_tolerance(s::String)
    m = match(r"^[+-]?\d\.(\d+)E([+-]?\d+)$", uppercase(strip(s)))
    m === nothing && error("Could not parse scientific notation token: $(s)")
    decimals = length(m.captures[1])
    exponent = parse(Int, m.captures[2])
    unit = 10.0^(exponent - decimals)
    return 0.5 * unit
end

"""
    compare_one_scheme(cfg, ref)

Compute one benchmark row (computed vs NIST reference).
"""
function compare_one_scheme(cfg::CuboidConfig, ref::SchemeReference)
    calc = if ref.scheme == :lrc
        compute_gpu_lrc_pair_quantities(cfg, ref.rc)
    elseif ref.scheme == :lfs
        compute_host_lfs_pair_quantities(cfg, ref.rc)
    else
        error("Unknown scheme $(ref.scheme)")
    end

    U_pair_ref = parse_nist_value(ref.U_pair_raw)
    W_pair_ref = parse_nist_value(ref.W_pair_raw)
    U_lrc_ref = parse_nist_value(ref.U_lrc_raw)

    U_pair_abs_err = abs(calc.U_pair - U_pair_ref)
    W_pair_abs_err = abs(calc.W_pair - W_pair_ref)
    U_lrc_abs_err = abs(calc.U_lrc - U_lrc_ref)

    U_pair_tol = nist_display_tolerance(ref.U_pair_raw)
    W_pair_tol = nist_display_tolerance(ref.W_pair_raw)
    U_lrc_tol = nist_display_tolerance(ref.U_lrc_raw)

    N = length(cfg.x)
    V = cfg.box[1] * cfg.box[2] * cfg.box[3]
    rho = N / V

    return ComparisonRow(
        cfg.id,
        N,
        rho,
        ref.label,
        ref.scheme == :lrc ? "GPU_LJ_kernel_plus_analytic_LRC" : "HOST_direct_pair_sum_LFS",
        ref.rc,
        calc.U_pair,
        U_pair_ref,
        U_pair_abs_err,
        U_pair_tol,
        U_pair_abs_err <= U_pair_tol,
        calc.W_pair,
        W_pair_ref,
        W_pair_abs_err,
        W_pair_tol,
        W_pair_abs_err <= W_pair_tol,
        calc.U_lrc,
        U_lrc_ref,
        U_lrc_abs_err,
        U_lrc_tol,
        U_lrc_abs_err <= U_lrc_tol,
    )
end

function write_csv(rows::Vector{ComparisonRow}, path::String)
    open(path, "w") do io
        println(io, "config_id,N,rho_star,scheme,method,rc,U_pair_calc,U_pair_ref,U_pair_abs_err,U_pair_tol,U_pair_pass,W_pair_calc,W_pair_ref,W_pair_abs_err,W_pair_tol,W_pair_pass,U_lrc_calc,U_lrc_ref,U_lrc_abs_err,U_lrc_tol,U_lrc_pass")
        for r in rows
            @printf(io,
                    "%d,%d,%.9f,%s,%s,%.3f,%.12e,%.12e,%.12e,%.12e,%s,%.12e,%.12e,%.12e,%.12e,%s,%.12e,%.12e,%.12e,%.12e,%s\n",
                    r.config_id, r.N, r.rho_star, r.scheme_label, r.method, r.rc,
                    r.U_pair_calc, r.U_pair_ref, r.U_pair_abs_err, r.U_pair_tol, string(r.U_pair_pass),
                    r.W_pair_calc, r.W_pair_ref, r.W_pair_abs_err, r.W_pair_tol, string(r.W_pair_pass),
                    r.U_lrc_calc, r.U_lrc_ref, r.U_lrc_abs_err, r.U_lrc_tol, string(r.U_lrc_pass))
        end
    end
end

function write_report(rows::Vector{ComparisonRow}, path::String)
    now_str = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS")

    n_u = count(r -> r.U_pair_pass, rows)
    n_w = count(r -> r.W_pair_pass, rows)
    n_l = count(r -> r.U_lrc_pass, rows)
    n_tot = length(rows)
    all_pass = all(r -> (r.U_pair_pass && r.W_pair_pass && r.U_lrc_pass), rows)

    open(path, "w") do io
        println(io, "ParticleDynamics NIST Cuboid LJ Head-to-Head Benchmark")
        println(io, "Generated at: $(now_str)")
        println(io, "Source: $(NIST_CUBOID_URL)")
        println(io, "NIST page updated: $(NIST_CUBOID_UPDATED)")
        println(io, "Configurations: lj_sample_config_periodic1..4.txt")
        println(io, "Potential convention benchmarked against NIST table:")
        println(io, "  - LRC_rc3: unshifted LJ truncated at rc*=3 plus analytic LRC")
        println(io, "  - LRC_rc4: unshifted LJ truncated at rc*=4 plus analytic LRC")
        println(io, "  - LFS_rc3: linear-force-shifted LJ at rc*=3 (no LRC)")
        println(io, "")
        println(io, @sprintf("Pass counts: U_pair %d/%d | W_pair %d/%d | U_LRC %d/%d", n_u, n_tot, n_w, n_tot, n_l, n_tot))
        println(io, @sprintf("Overall status: %s", all_pass ? "PASS" : "WARN"))
        println(io, "")
        println(io, "Columns: config scheme method rc  U_pair(calc/ref/|err|/tol)  W_pair(calc/ref/|err|/tol)  U_LRC(calc/ref/|err|/tol)")
        println(io, "")

        for r in rows
            row_status = (r.U_pair_pass && r.W_pair_pass && r.U_lrc_pass) ? "PASS" : "WARN"
            println(io, @sprintf(
                "cfg%d %-8s %-31s rc=%.1f  U=(%.6f / %.6f / %.3e / %.3e)  W=(%.6f / %.6f / %.3e / %.3e)  ULRC=(%.6f / %.6f / %.3e / %.3e)  -> %s",
                r.config_id, r.scheme_label, r.method, r.rc,
                r.U_pair_calc, r.U_pair_ref, r.U_pair_abs_err, r.U_pair_tol,
                r.W_pair_calc, r.W_pair_ref, r.W_pair_abs_err, r.W_pair_tol,
                r.U_lrc_calc, r.U_lrc_ref, r.U_lrc_abs_err, r.U_lrc_tol,
                row_status
            ))
        end
    end
end

function main()
    configs = load_all_cuboid_configs(CONFIG_DIR)
    rows = ComparisonRow[]
    for cfg in configs
        refs = get(NIST_CUBOID_REFS, cfg.id, nothing)
        refs === nothing && error("Missing NIST reference table for configuration $(cfg.id).")
        for ref in refs
            push!(rows, compare_one_scheme(cfg, ref))
        end
    end

    sort!(rows, by = r -> (r.config_id, r.scheme_label))
    write_csv(rows, CSV_PATH)
    write_report(rows, REPORT_PATH)

    println("NIST cuboid head-to-head benchmark complete.")
    println("Report: $(REPORT_PATH)")
    println("CSV:    $(CSV_PATH)")
end

main()
