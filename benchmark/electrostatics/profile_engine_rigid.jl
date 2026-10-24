# Residency trace of the public rigid-water path: GROMACS-equilibrated SPC/E water
# (SHAKE/RATTLE, LJ + ESP, cluster backend) stepped with NVE through `step!`. Every
# traced step must be one graph replay (drift, SHAKE, list decide/rebuild/prune, pairs ‖
# mesh, gather, kick, RATTLE, reference save; constraint virial and observables on
# sampled steps), with no kernels, host transfers or allocations outside the graphs.
# scripts/check_device_trace.py audits it (`pair_backend = "cluster"`, `rigid_water`).
# Usage (under nsys): profile_engine_rigid.jl SYSTEMS_DIR NAME OUT.toml [steps=500] [every=25] [warmup=200]
using Printf
include(joinpath(@__DIR__, "rigid_water_states.jl"))

function main(args)
    sysdir = args[1]; name = args[2]; out = abspath(args[3])
    steps = length(args) > 3 ? parse(Int, args[4]) : 500; every = length(args) > 4 ? parse(Int, args[5]) : 25
    warmup = length(args) > 5 ? parse(Int, args[6]) : 200
    ispath(out) && error("Refusing to overwrite $out")
    CUDA.allowscalar(false)
    st, f = load_rigid_water(sysdir, name)
    N = length(st.rx); spec = SC.nve(st)
    record = Dict{String,Any}("status" => "running", "case" => "rigid_nve_$(name)", "dims" => 3, "N" => N,
        "warmup" => warmup, "steps" => steps, "energy_every" => every, "public_step" => true,
        "global_observables" => true, "neighbor_diagnostics" => false, "spatial_reorder" => false,
        "rebuild_policy" => "displacement", "forces_per_step" => 1, "pair_backend" => "cluster",
        "electrostatics" => true, "rigid_water" => true, "waters" => N ÷ 3)
    save() = open(io -> TOML.print(io, record; sorted=true), out, "w")
    try
        for i in 1:warmup
            SC.step!(st, spec, st.dt; compute_energy=i % every == 0)
        end
        e = st.engine; e isa DE.StateEngine && e.run.cluster !== nothing &&
            DE.has_feature(e.run, DE.RigidWaterFeature) || error("public step did not use the rigid-water cluster engine")
        DE.synchronize_state!(st)
        L = st.cluster.list; c0 = Array(L.counts); o0 = DE.observable_status(e.run)
        CUDA.synchronize()
        CUDA.@profile external=true begin
            for i in 1:steps
                SC.step!(st, spec, st.dt; compute_energy=(warmup + i) % every == 0)
            end
            CUDA.CUDACore.cuStreamSynchronize(CUDA.stream())
        end
        CUDA.synchronize()
        st.engine === e || error("the engine was replaced during the trace")
        DE.synchronize_state!(st)
        e.recoveries == 0 || error("a capacity recovery ran inside the trace")
        c1 = Array(L.counts)
        record["rebuilds_in_trace"] = Int(c1[CP.CL_NBUILD] - c0[CP.CL_NBUILD])
        record["prunes_in_trace"] = Int(c1[CP.CL_NPRUNE] - c0[CP.CL_NPRUNE])
        record["observable_samples_in_trace"] = DE.observable_status(e.run).samples - o0.samples
        o = DE.observable_status(e.run); record["temperature_kT"] = o.temperature
        dl, dv = constraint_errors(st)
        record["max_length_error"] = dl; record["max_velocity_projection"] = dv
        dl < 1e-4 || error("constraint lengths drifted ($dl)")
        record["status"] = "GPU checks passed; trace audit pending"; save()
        println("Engine rigid-water profile complete: N=", N, "; rebuilds=", record["rebuilds_in_trace"],
                "; prunes=", record["prunes_in_trace"], "; kT=", round(o.temperature, digits=3), "; length error ", dl)
    catch err
        record["status"] = "failed"; record["error_type"] = string(typeof(err)); save()
        rethrow()
    end
end
main(ARGS)
