using ParticleDynamics
using ParticleDynamics: Filters, velocityverlet, hex_random_2d
using CUDA
using Random
using Printf

"""
Randomly assign cold/hot types across the lattice without moving particles.
Ensures positions remain on the square lattice while typeid is shuffled.
"""
function random_types(N::Integer; ratio::T=0.5) where T<:AbstractFloat
    n_cold = round(Int, ratio * N)
    p = randperm(N)
    host = fill(Int32(2), N)
    @inbounds host[p[1:n_cold]] .= Int32(1)
    return host
end

function apply_types!(st, host::AbstractVector{Int32})
    @assert length(host) == length(st.rx)
    st.typeid .= CuArray(host)
    return nothing
end

function maybe_override_int(default::Int, env_name::AbstractString; lower::Int=1)
    value = get(ENV, env_name, "")
    isempty(value) && return default
    parsed = tryparse(Int, value)
    return parsed === nothing ? default : max(lower, parsed)
end

function main()
    # simulation parameters (local scope, no globals)
    n = maybe_override_int(10_000, "SIM_NPARTICLES")
    ϕ = 0.7
    sigma = 1.0
    # Random hexagonal-lattice placement at target area fraction ϕ
    cfg = hex_random_2d(n, sigma, ϕ; T=Float64)
    box = cfg.box

    epsilon = 10.0
    rcut = 2.5 * sigma

    dt = 1e-4
    tau_nhc = 100 * dt
    gamma = 1 / (2 * tau_nhc)
    #gamma = 10000.0
    nsteps = maybe_override_int(500_000, "SIM_MAX_STEPS")
    log_interval = maybe_override_int(10_000, "SIM_LOG_INTERVAL")
    log_interval = min(log_interval, max(1, nsteps))

    # Optional warmup (relaxation) configuration
    warmup_enable = true
    warmup_steps = maybe_override_int(10_000, "SIM_WARMUP_STEPS"; lower=0)
    warmup_enable = warmup_enable && warmup_steps > 0
    warmup_dt = dt * 0.1
    warmup_neigh_interval = 1
    init_steps = maybe_override_int(50_000, "SIM_INIT_STEPS"; lower=0)
    relax_steps = maybe_override_int(0, "SIM_RELAX_STEPS"; lower=0)

    # Two different bath temperatures per type (cold=type 1, hot=type 2)
    t_cold = 10000.0/500
    t_hot = 10000.0

    st = build_simulation(
        N=n,
        box=(box[1], box[2]),
        cutoff=rcut,
        skin=0.5,
        cap=Int32(100),
        neigh_interval=10,
        epsilon=epsilon,
        sigma=sigma,
        gamma=gamma,
        temperature=t_cold,
        mass=1.0,
        dt=dt,
        nonbonded=:lj,
        precision=:f64,
        unwrapped_positions=true,
    )

    # random hex-lattice positions, then assign types after initial equilibration
    copyto!(st.rx, Float64[p[1] for p in cfg.positions])
    copyto!(st.ry, Float64[p[2] for p in cfg.positions])
    sync_unwrapped!(st)

    vv = velocityverlet(st; gamma=gamma, temperature=t_cold, dt=warmup_enable ? warmup_dt : dt)

    # Warmup phase: smaller dt and/or tighter neighbor checks; no I/O
    if warmup_enable && warmup_steps > 0
        original_neigh = st.neigh_interval
        st.neigh_interval = warmup_neigh_interval
        for _ in 1:warmup_steps
            step!(st, vv, warmup_dt; compute_energy=false)
        end
        # Restore production settings
        st.neigh_interval = original_neigh
    end

    # Set production noise scale with production dt for the single-temperature pre-run
    Filters.set_temperature!(vv, st, dt, t_cold; filter=Filters.All())

    for _ in 1:init_steps
        step!(st, vv, dt; compute_energy=false)
    end

    type_host = random_types(n; ratio=0.5)
    apply_types!(st, type_host)

    # Enable GPU collision event counting with bins for (1,1), (1,2), (2,2)
    enable_collision_counting!(st; ntypes=2, bins=:all_pairs)

    cold_filter = Filters.TypeIDs(1)
    hot_filter = Filters.TypeIDs(2)

    # Per-group Langevin temperature (sets per-particle noise scale sqrt(2γTΔt))
    Filters.set_temperature!(vv, st, dt,
        cold_filter => t_cold,
        hot_filter  => t_hot)

    for _ in 1:relax_steps
        step!(st, vv, dt; compute_energy=false)
    end

    # Reset accumulators and counters; start production step count at 0
    fill!(st.Ekin_accum, zero(eltype(st.Ekin_accum)))
    fill!(st.Epot_accum, zero(eltype(st.Epot_accum)))
    fill!(st.virial_accum, zero(eltype(st.virial_accum)))
    fill!(st.dq, zero(eltype(st.dq)))
    fill!(st.dU, zero(eltype(st.dU)))
    collisions_reset_counts!(st)
    st.step = 0

    # noise scale per type is set above via Filters.set_temperature!
    # Precompute inv temperature per particle for production dt: invT_i = (2 γ_i Δt) / s_i^2
    invT = let s = Array(vv.params.noise_scale), g = Array(vv.params.gamma)
        dtT = eltype(s)(dt)
        (2 .* g .* dtT) ./ (s .^ 2)
    end

    # Integrator info (explicit)
    println("Integrator: Langevin dynamics (GJF/Velocity-Verlet).")
    println(" - Positions: r_{n+1} = r_n + b Δt v_n + (b Δt / (2m)) (Δt f_n + β_n)")
    println(" - Velocities: v_{n+1} = a v_n + (Δt/(2m)) (a f_n + f_{n+1}) + (b/m) β_n")
    println(" - Coefficients: a=(1-q)/(1+q), b=1/(1+q), q=γ Δt/(2m)")
    println(" - β ~ N(0, 2 γ kT Δt) with T set via filters")

    # Writers -----------------------------------------------------------------
    output_dir = @__DIR__
    gsd_path = joinpath(output_dir, "traj2d_filters_vv_$(t_hot)_$(dt).gsd")
    log_path = replace(gsd_path, ".gsd" => ".log")
    write_unwrapped = false

    type_names = ["C", "H"]
    gsd_open(gsd_path) do gsdh
        # First frame is the relaxed (post-warmup) configuration
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step,
                         write_unwrapped=write_unwrapped, sync_on_write=true)

    # Adopt logging format from restart_from_gsd.jl
    open(log_path, "w") do io
        # Centered, fixed-width header to align with centered values
        centerstr(s, w) = begin
            len = length(s)
            pad = max(w - len, 0)
            l = pad ÷ 2; r = pad - l
            string(" "^l, s, " "^r)
        end
        titles = [
            "Time", "E_kin", "E_pot", "E_tot", "virial", "EPR", "UPR",
            "EPR / part", "UPR / part", "EPR / part Ave",
            "cold/cold coll", "hot/cold coll", "hot/hot coll",
            "E_kin_cold", "E_kin_hot", "E_pot_cold", "E_pot_hot",
            "virial_cold", "virial_hot"
        ]
        widths = [14,14,14,14,14,14,14,14,14,16,14,14,14,14,14,14,14,14,14]
        println(io, join(map((t,w)->centerstr(t,w), titles, widths), " | "))
    end

    samples = 0
    avg_epr_per_part = 0.0

    start_time = time()
    max_runtime = let v = get(ENV, "SIM_MAX_SECONDS", "")
        isempty(v) ? Inf : parse(Float64, v)
    end

    for step in 1:nsteps
        write_output = (step % log_interval == 0)
        # Explicit integrator selection: Langevin (GJF/Velocity-Verlet)
        step!(st, vv, dt; compute_energy=true)
            ParticleDynamics.Simulation.accumulate_energies!(st)
        accumulate_virial!(st)

        if write_output
            # Compute system-wide observables (host side)
            e_kin = sum(Array(st.Ekin_accum)) / log_interval
            e_pot = sum(Array(st.Epot_accum)) / log_interval
            e_tot = e_kin + e_pot
            virial = sum(Array(st.virial_accum)) / log_interval
            # Entropy production via per-particle dq_i / T_i
            epr = sum(Array(st.dq) .* invT) / log_interval
            #epr = sum(Array(st.dq)) / log_interval

            upr = sum(Array(st.dU) .* invT) / log_interval
            #upr = sum(Array(st.dU))/log_interval

            n = length(st.rx)
            epr_part = epr / n
            upr_part = upr / n

            # Interval-averaged kinetic/potential energies per type
            e_kin_cold = Filters.sum(st.Ekin_accum, st, cold_filter) / log_interval
            e_kin_hot  = Filters.sum(st.Ekin_accum, st, hot_filter) / log_interval
            e_pot_cold = Filters.sum(st.Epot_accum, st, cold_filter) / log_interval
            e_pot_hot  = Filters.sum(st.Epot_accum, st, hot_filter) / log_interval
            virial_cold = Filters.sum(st.virial_accum, st, cold_filter) / log_interval
            virial_hot  = Filters.sum(st.virial_accum, st, hot_filter) / log_interval

            samples += 1
            avg_epr_per_part = 0.0
            # Read collision counts accumulated on GPU and convert to rates per unit time
            counts = collisions_read_counts!(st)
            # Map bins: (1,1)->bin0, (1,2)->bin1, (2,2)->bin2
            cc = length(counts) >= 1 ? counts[1] / (dt * log_interval) : 0.0
            hc = length(counts) >= 2 ? counts[2] / (dt * log_interval) : 0.0
            hh = length(counts) >= 3 ? counts[3] / (dt * log_interval) : 0.0

            open(log_path, "a") do io
                centerstr(s, w) = begin
                    len = length(s)
                    pad = max(w - len, 0)
                    l = pad ÷ 2; r = pad - l
                    string(" "^l, s, " "^r)
                end
                widths = [14,14,14,14,14,14,14,14,14,16,14,14,14,14,14,14,14,14,14]
                vals = [
                    @sprintf("%.5e", float(st.step)),
                    @sprintf("%.5e", e_kin),
                    @sprintf("%.5e", e_pot),
                    @sprintf("%.5e", e_tot),
                    @sprintf("%.5e", virial),
                    @sprintf("%.5e", epr),
                    @sprintf("%.5e", upr),
                    @sprintf("%.5e", epr_part),
                    @sprintf("%.5e", upr_part),
                    @sprintf("%.5e", avg_epr_per_part),
                    @sprintf("%.5e", cc),
                    @sprintf("%.5e", hc),
                    @sprintf("%.5e", hh),
                    @sprintf("%.5e", e_kin_cold),
                    @sprintf("%.5e", e_kin_hot),
                    @sprintf("%.5e", e_pot_cold),
                    @sprintf("%.5e", e_pot_hot),
                    @sprintf("%.5e", virial_cold),
                    @sprintf("%.5e", virial_hot)
                ]
            println(io, join(map((s,w)->centerstr(s,w), vals, widths), " | "))
        end

        # Write a GSD frame
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step,
                         write_unwrapped=write_unwrapped, sync_on_write=true)

        # Reset interval accumulators, dq/dU and collision counters after logging
        fill!(st.Ekin_accum, zero(eltype(st.Ekin_accum)))
        fill!(st.Epot_accum, zero(eltype(st.Epot_accum)))
        fill!(st.virial_accum, zero(eltype(st.virial_accum)))
        fill!(st.dq, zero(eltype(st.dq)))
        fill!(st.dU, zero(eltype(st.dU)))
            collisions_reset_counts!(st)

            # Progress / ETA
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
    println("Total wall time ≈ $(round(total_time, digits=2)) s")

    end
    println("Wrote trajectory to $(gsd_path)")
    println("Wrote log to $(log_path)")
end

# run when invoked as a script
main()
