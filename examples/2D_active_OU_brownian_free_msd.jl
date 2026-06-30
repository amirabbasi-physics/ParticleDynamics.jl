using ParticleDynamics

include(joinpath(@__DIR__, "_example_utils.jl"))
include(joinpath(@__DIR__, "_free_aoup_msd_utils.jl"))

function main()
    N = maybe_override_int(20_000, "SIM_NPARTICLES")
    dt = maybe_override_float(1.0e-3, "SIM_DT"; lower=1.0e-12)
    n_steps = maybe_override_int(1_200, "SIM_MAX_STEPS")
    sample_stride = maybe_override_interval(10, n_steps)

    gamma = maybe_override_float(100.0, "SIM_GAMMA"; lower=1.0e-12)
    corr_time = maybe_override_float(2.25, "SIM_CORR_TIME"; lower=0.0)
    active_impulse = maybe_override_float(2.0, "SIM_NOISE_SCALE"; lower=0.0)

    run_free_aoup_msd_check(
        label="Free 2D athermal AOUP Brownian MSD check",
        csv_name="obs2d_active_OU_brownian_free_msd.csv",
        N=N,
        dt=dt,
        n_steps=n_steps,
        sample_stride=sample_stride,
        gamma=gamma,
        thermal_temperature=0.0,
        taus=corr_time,
        scales=active_impulse,
    )
end

main()
