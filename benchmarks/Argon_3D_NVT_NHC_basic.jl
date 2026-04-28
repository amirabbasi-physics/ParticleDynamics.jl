using NonEqSimGPU
using NonEqSimGPU: step!, collect_step_observables
using CUDA
using Printf
using Random

include(joinpath(@__DIR__, "argon_nvt_common.jl"))

CUDA.allowscalar(false)
Random.seed!(0xA36E)

# -----------------------------
# User-facing physical settings
# -----------------------------
N = parse(Int, get(ENV, "NEQSIM_ARGON_N", "2048"))
T_kelvin = parse(Float64, get(ENV, "NEQSIM_ARGON_T_K", "300.0"))
rho_kg_m3 = parse(Float64, get(ENV, "NEQSIM_ARGON_RHO_KG_M3", "1.6"))

dt_star = parse(Float64, get(ENV, "NEQSIM_ARGON_DT_STAR", "0.002"))
warmup_steps = parse(Int, get(ENV, "NEQSIM_ARGON_WARMUP_STEPS", "5000"))
production_steps = parse(Int, get(ENV, "NEQSIM_ARGON_PROD_STEPS", "20000"))
log_interval = parse(Int, get(ENV, "NEQSIM_ARGON_LOG_INTERVAL", "1000"))
write_gsd = parse_bool_env("NEQSIM_ARGON_WRITE_GSD", true)
gsd_interval = parse(Int, get(ENV, "NEQSIM_ARGON_GSD_INTERVAL", string(log_interval)))
gsd_sync_on_write = parse_bool_env("NEQSIM_ARGON_GSD_SYNC_ON_WRITE", true)

nhc_tau_star = parse(Float64, get(ENV, "NEQSIM_ARGON_NHC_TAU_STAR", "1.0"))
nhc_chain_length = parse(Int, get(ENV, "NEQSIM_ARGON_NHC_CHAIN_LENGTH", "5"))
nhc_substeps = parse(Int, get(ENV, "NEQSIM_ARGON_NHC_SUBSTEPS", "3"))

# -----------------------------
# Reduced-unit state conversion
# -----------------------------
T_star = argon_reduced_temperature(T_kelvin)
rho_star = argon_reduced_density_from_mass_density(rho_kg_m3)
L_star = argon_box_length_reduced(N, rho_star)

T = Float32
box = (T(L_star), T(L_star), T(L_star))

println("Argon NVT setup (reduced units)")
@printf("  N = %d\n", N)
@printf("  T = %.3f K  ->  T* = %.5f\n", T_kelvin, T_star)
@printf("  rho = %.5f kg/m^3  ->  rho* = %.6e\n", rho_kg_m3, rho_star)
@printf("  box L* = %.5f\n", L_star)

# -----------------------------
# Build simulation state
# -----------------------------
st = build_simulation(
    N=N,
    box=box,
    cutoff=T(2.5),
    skin=T(0.4),
    cap=Int32(64),
    neigh_interval=20,
    epsilon=T(1),
    sigma=T(1),
    mass=T(1),
    gamma=T(1),
    temperature=T(T_star),
    dt=T(dt_star),
    nonbonded=:lj,
    precision=:f32,
)

initialize_simple_cubic_lattice!(st, box; jitter_frac=T(0.15))

spec, integrator_label, _ = select_nvt_integrator(
    st;
    temperature_reduced=T_star,
    dt=dt_star,
    nhc_tau_reduced=nhc_tau_star,
    nhc_chain_length=nhc_chain_length,
    nhc_substeps=nhc_substeps,
)
println("Integrator: $(integrator_label)")

obs_path = joinpath(@__DIR__, "obs_argon_nvt_nhc_basic.csv")
gsd_path = joinpath(@__DIR__, "traj_argon_nvt_nhc_basic.gsd")
if isfile(obs_path)
    rm(obs_path; force=true)
end
if write_gsd && isfile(gsd_path)
    rm(gsd_path; force=true)
end

# -----------------------------
# Run
# -----------------------------
function run_simulation!(gsdh)
    println("Warmup phase...")
    for _ in 1:warmup_steps
        step!(st, spec, T(dt_star); compute_energy=false)
    end

    println("Production phase...")
    for i in 1:production_steps
        step!(st, spec, T(dt_star); compute_energy=true)

        if gsdh !== nothing && i % gsd_interval == 0
            write_gsd_frame!(gsdh, st; diameter=1.0, types_names=["Ar"], step=st.step, sync_on_write=gsd_sync_on_write)
        end

        if i % log_interval == 0
            obs = collect_step_observables(st, spec)
            T_inst = instantaneous_reduced_temperature(st)
            ext_h = hasproperty(obs, :extended_hamiltonian) ? obs.extended_hamiltonian : obs.Etot
            terr = hasproperty(obs, :thermostat_temperature_error) ? obs.thermostat_temperature_error : 0.0
            @printf("step=%8d  T*inst=%.5f  Etot=%.6e  ExtH=%.6e  dT*=%.3e\n",
                    st.step, T_inst, obs.Etot, ext_h, terr)
            write_observables_csv!(obs_path, st, spec)
        end
    end
end

if write_gsd
    println("Writing trajectory: $(gsd_path)")
    gsd_open(gsd_path) do gsdh
        write_gsd_frame!(gsdh, st; diameter=1.0, types_names=["Ar"], step=st.step, sync_on_write=gsd_sync_on_write)
        run_simulation!(gsdh)
    end
else
    run_simulation!(nothing)
end

println("Finished Argon NVT run.")
println("CSV observables: $(obs_path)")
if write_gsd
    println("GSD trajectory: $(gsd_path)")
end
