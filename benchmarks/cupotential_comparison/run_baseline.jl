# ==============================================================================
# Frozen v0.7.0 Baseline Runner (RTX 3090)
# Works on both `master` (v0.7.0) and `feat/cupotential-native`.
#
# Usage on RTX 3090 workstation:
#   julia --project --startup-file=no benchmarks/cupotential_comparison/run_baseline.jl
#   PD_BENCH_QUICK=1 julia --project --startup-file=no benchmarks/cupotential_comparison/run_baseline.jl
# ==============================================================================
using CUDA
using Printf
using LinearAlgebra
using ParticleDynamics
using ParticleDynamics: SimulationCore, Electrostatics

CUDA.functional(true) || error("CUDA GPU required to run baseline benchmarks.")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "bench_common.jl"))
include(joinpath(@__DIR__, "workloads.jl"))
using .BenchCommon
using .BenchWorkloads

const QUICK = get(ENV, "PD_BENCH_QUICK", "0") == "1"
const BENCH_STEPS = QUICK ? 400 : 2_000
const DRIFT_STEPS = QUICK ? 5_000 : 50_000
const OUT_DIR = joinpath(@__DIR__, "results")
mkpath(OUT_DIR)

device_name = CUDA.name(CUDA.device())
println("=== ParticleDynamics Baseline Benchmark & Drift Suite ===")
println("GPU: $(device_name) | Quick mode: $(QUICK)")

# Build high-accuracy Float64 converged reference forces/energies/virials at r_c = 1.2 nm
function converged_reference(maker, T_ref=Float64; rc_ref=1.2)
    spec_ref = electrostatics_spec(ESP(coulomb_prefactor=COULOMB_NM_KJMOL,
                                       real_cutoff=rc_ref, tolerance=1e-9), T_ref)
    st_ref, meta = maker(T_ref; cutoff=rc_ref, electrostatics_spec=spec_ref, vdw_modifier=:none)
    # Relax initial configuration gently in Float64 so NVE starts near equilibrium
    if !haskey(meta, :triples)
        minimize!(st_ref; method=FIRE(dt_initial=1e-4, dt_max=1e-3),
                  max_steps=QUICK ? 80 : 250, force_tolerance=50.0, max_displacement=0.01)
    else
        cs = csvr(st_ref; temperature=meta.kBT, tau=0.05, dt=0.001)
        for _ in 1:(QUICK ? 300 : 1_500)
            step!(st_ref, cs, 0.001; compute_energy=false)
        end
    end
    Eref, Fref, Wref = BenchCommon.evaluate_snapshot(st_ref)
    coords = (Array(st_ref.rx), Array(st_ref.ry), Array(st_ref.rz))
    vels   = (Array(st_ref.vx), Array(st_ref.vy), Array(st_ref.vz))
    st_ref = nothing; GC.gc(); CUDA.reclaim()
    return (E=Eref, F=Fref, W=Wref, coords=coords, vels=vels, meta=meta)
end

function load_snapshot!(st::SimulationState{T}, snap) where {T}
    rx, ry, rz = snap.coords
    vx, vy, vz = snap.vels
    copyto!(st.rx, T.(rx)); copyto!(st.ry, T.(ry)); copyto!(st.rz, T.(rz))
    copyto!(st.vx, T.(vx)); copyto!(st.vy, T.(vy)); copyto!(st.vz, T.(vz))
    if st.rigid_water !== nothing
        RW = ParticleDynamics.RigidWaterConstraints
        RW.constrain_velocities!(st.rigid_water.ws, st.rx, st.ry, st.rz, st.vx, st.vy, st.vz, st.box3)
    end
    SimulationCore.invalidate_forces!(st)
    return st
end

results = NamedTuple[]

suites = [
    (name="nacl_10k",  maker=(T; kw...) -> make_nacl_melt(T; ncell=11, kw...)),
    (name="spce_30k",  maker=(T; kw...) -> make_spce_water(T; nx=22, ny=22, nz=21, kw...)),
    (name="slab_15k",  maker=(T; kw...) -> make_spce_slab(T; nx=16, ny=16, nz_film=20, kw...)),
]

for suite in suites
    println("\n--- Preparing converged Float64 reference for $(suite.name) ---")
    snap = converged_reference(suite.maker)
    for (label, rc, solver_fn) in (
        ("PME_rc1.0_tol1e-4", 1.0, T -> electrostatics_spec(
            PME(coulomb_prefactor=COULOMB_NM_KJMOL, real_cutoff=1.0,
                real_space_tolerance=1e-4, mesh_spacing=0.10, order=5), T)),
        ("ESP_rc1.0_tol1e-4", 1.0, T -> electrostatics_spec(
            ESP(coulomb_prefactor=COULOMB_NM_KJMOL, real_cutoff=1.0, tolerance=1e-4), T)),
        ("ESP_rc0.7_tol1e-4_truncLJ", 0.7, T -> electrostatics_spec(
            ESP(coulomb_prefactor=COULOMB_NM_KJMOL, real_cutoff=0.7, tolerance=1e-4), T)),
    )
        for T in (Float32, Float64)
            spec_es = solver_fn(T)
            st, meta = suite.maker(T; cutoff=rc, electrostatics_spec=spec_es, vdw_modifier=:potential_shift)
            load_snapshot!(st, snap)
            E, F, W = BenchCommon.evaluate_snapshot(st)
            ferr = relative_rms_error(F, snap.F)
            werr = relative_rms_error(W, snap.W)
            # Surface tension indicator (mN/m factor optional; raw kJ/(mol·nm²) difference):
            gamma_raw = meta.box[3] * ((-W[3, 3] + 0.5 * (W[1, 1] + W[2, 2])) / (meta.box[1] * meta.box[2]))
            gamma_ref = meta.box[3] * ((-snap.W[3, 3] + 0.5 * (snap.W[1, 1] + snap.W[2, 2])) / (meta.box[1] * meta.box[2]))

            perf = measure_step_throughput(st, nve(st; dt=T(0.002)), T(0.002);
                                           warmup=QUICK ? 40 : 200, steps=BENCH_STEPS, dt_ps=0.002)
            load_snapshot!(st, snap)
            drift = measure_nve_drift(st, T(0.002);
                                      equil_steps=QUICK ? 200 : 1_000,
                                      prod_steps=DRIFT_STEPS,
                                      sample_every=QUICK ? 50 : 100,
                                      kBT=meta.kBT, time_unit_ns=1e-3)
            row = (
                workload=suite.name,
                config=label,
                precision=string(T),
                cutoff_nm=rc,
                mesh=join(Electrostatics.solver_mesh(st.electrostatics), "x"),
                ms_per_step=perf.ms_per_step,
                ns_per_day=perf.ns_per_day,
                force_rel_rms=ferr,
                virial_rel_rms=werr,
                slab_tension_err=abs(gamma_raw - gamma_ref),
                abs_drift_kBT_dof_ns=drift.abs_drift_kBT_dof_ns,
                drift_ci95_kBT_dof_ns=drift.abs_drift_ci95_kBT_dof_ns,
            )
            push!(results, row)
            @printf("  %-26s %-7s | %6.3f ms/step (%6.1f ns/d) | dF=%.2e dW=%.2e | |drift|=%.2e kBT/dof/ns\n",
                    label, string(T), row.ms_per_step, row.ns_per_day, row.force_rel_rms,
                    row.virial_rel_rms, row.abs_drift_kBT_dof_ns)
            st = nothing; GC.gc(); CUDA.reclaim()
        end
    end
end

json_path = write_json(joinpath(OUT_DIR, "baseline_v0_7_0.json"),
                       (gpu=device_name, version=string(pkgversion(ParticleDynamics)),
                        quick=QUICK, rows=results))
csv_path = write_csv_summary(joinpath(OUT_DIR, "baseline_v0_7_0.csv"), results)
println("\nSaved baseline results to:\n  $(json_path)\n  $(csv_path)")
