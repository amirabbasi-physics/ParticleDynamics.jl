using ParticleDynamics
using ParticleDynamics: Filters, csvr, hex_random_2d, collect_step_observables, reset_bath_exchange_accumulators!
using Printf

include(joinpath(@__DIR__, "_example_utils.jl"))

centerstr(s, w) = begin
    len = length(s)
    pad = max(w - len, 0)
    l = pad ÷ 2
    r = pad - l
    string(" "^l, s, " "^r)
end

function main(phi::Float64, temperature::Float64)
    n = maybe_override_int(10_000, "SIM_NPARTICLES")
    sigma = 1.0

    cfg = hex_random_2d(n, sigma, phi; T=Float64)
    box = cfg.box

    epsilon = 1e9
    rcut = sigma

    dt = 1e-5
    nsteps = maybe_override_int(10_000_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_interval(500_0, nsteps)

    warmup_enable = true
    warmup_steps = maybe_override_int(1_000_0, "SIM_WARMUP_STEPS"; lower=0)
    warmup_dt = dt * 0.1
    warmup_neigh_interval = 5

    # Match the original Langevin script's damping timescale with a CSVR coupling time.
    tau_csvr = 10 * dt

    st = build_simulation(
        N=n,
        box=(box[1], box[2]),
        cutoff=rcut,
        skin=0.55,
        cap=Int32(250),
        neigh_interval=20,
        epsilon=epsilon,
        sigma=sigma,
        mass=1.0,
        gamma=nothing,
        temperature=temperature,
        dt=dt,
        nonbonded=:soft_repulsive,
        precision=:f64,
    )

    copyto!(st.rx, Float64[p[1] for p in cfg.positions])
    copyto!(st.ry, Float64[p[2] for p in cfg.positions])

    thermostat = csvr(st; temperature=temperature, tau=tau_csvr)
    Filters.set_temperature!(thermostat, st, dt, temperature; filter=Filters.All())

    enable_collision_counting!(st; ntypes=1, bins=:all_pairs)

    if warmup_enable && warmup_steps > 0
        original_neigh = st.neigh_interval
        st.neigh_interval = warmup_neigh_interval
        for _ in 1:warmup_steps
            step!(st, thermostat, warmup_dt; compute_energy=false)
        end
        st.neigh_interval = original_neigh
    end

    fill!(st.Ekin_accum, zero(eltype(st.Ekin_accum)))
    fill!(st.Epot_accum, zero(eltype(st.Epot_accum)))
    fill!(st.virial_accum, zero(eltype(st.virial_accum)))
    collisions_reset_counts!(st)
    reset_bath_exchange_accumulators!(st, thermostat)
    st.step = 0

    println("Integrator: molecular dynamics with CSVR thermostat.")
    println(" - Thermostat: canonical stochastic velocity rescaling")
    println(" - Coupling time tau = $(tau_csvr)")

    output_dir = @__DIR__
    gsd_path = joinpath(output_dir, "traj2d_csvr_alpha_$(temperature)_fraction_$(phi).gsd")
    log_path = replace(gsd_path, ".gsd" => ".log")
    rm(gsd_path; force=true)
    rm(log_path; force=true)

    type_names = ["C"]
    gsd_open(gsd_path) do gsdh
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step, sync_on_write=true)

        open(log_path, "w") do io
            titles = [
                "Time", "E_pot", "virial", "Bath_E", "EPR",
                "Bath_E / part", "EPR / part", "collision rate",
            ]
            widths = fill(14, length(titles))
            println(io, join(map((t, w) -> centerstr(t, w), titles, widths), " | "))
        end

        start_time = time()
        max_runtime = maybe_override_runtime()

        for step in 1:nsteps
            write_output = step % log_interval == 0
            step!(st, thermostat, dt; compute_energy=true)
            ParticleDynamics.SimulationCore.accumulate_energies!(st)
            accumulate_virial!(st)

            if write_output
                obs = collect_step_observables(st, thermostat)
                e_pot = sum(Array(st.Epot_accum)) / log_interval
                virial = sum(Array(st.virial_accum)) / log_interval

                interval_time = dt * log_interval
                bath_e = obs.bath_heat_total
                epr = interval_time > 0 ? obs.bath_entropy_total / interval_time : zero(obs.bath_entropy_total)

                np = length(st.rx)
                bath_e_part = bath_e / np
                epr_part = epr / np

                counts = collisions_read_counts!(st)
                coll_rate = length(counts) >= 1 ? counts[1] / interval_time : 0.0

                open(log_path, "a") do io
                    widths = fill(14, 8)
                    vals = [
                        @sprintf("%.5e", float(st.step)),
                        @sprintf("%.5e", e_pot),
                        @sprintf("%.5e", virial),
                        @sprintf("%.5e", bath_e),
                        @sprintf("%.5e", epr),
                        @sprintf("%.5e", bath_e_part),
                        @sprintf("%.5e", epr_part),
                        @sprintf("%.5e", coll_rate),
                    ]
                    println(io, join(map((s, w) -> centerstr(s, w), vals, widths), " | "))
                end

                write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step, sync_on_write=true)

                fill!(st.Ekin_accum, zero(eltype(st.Ekin_accum)))
                fill!(st.Epot_accum, zero(eltype(st.Epot_accum)))
                fill!(st.virial_accum, zero(eltype(st.virial_accum)))
                collisions_reset_counts!(st)
                reset_bath_exchange_accumulators!(st, thermostat)

                elapsed = time() - start_time
                steps_per_sec = step / max(elapsed, 1e-6)
                remaining = nsteps - step
                eta = remaining / max(steps_per_sec, 1e-6)
                @info "progress" step=step elapsed_s=elapsed steps_per_sec=steps_per_sec eta_s=eta

                if elapsed >= max_runtime
                    @info "Reached max runtime limit" limit_s=max_runtime step=step
                    break
                end
            end
        end

        total_time = time() - start_time
        println("Total wall time ~= $(round(total_time, digits=2)) s")
    end

    println("Wrote trajectory to $(gsd_path)")
    println("Wrote log to $(log_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 2
        phi = parse(Float64, ARGS[1])
        temperature = parse(Float64, ARGS[2])
        println("Running with phi=$(phi), temperature=$(temperature)")
        main(phi, temperature)
    else
        println("Usage: julia SingleT_2D_MD_CSVR.jl <phi> <temperature>")
        println("  Example: julia SingleT_2D_MD_CSVR.jl 0.85 10000.0")
        exit(1)
    end
end
