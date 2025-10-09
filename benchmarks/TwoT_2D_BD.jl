using NonEqSimGPU
using NonEqSimGPU: Filters
using CUDA
using Random
using Printf

CUDA.allowscalar(false)

function friction_coefficient(eta::Float32, radius::Float32)
    # Stokes drag for sphere in 3D
    return Float32(6.0 * π * eta * radius)
end

function initialize_square_lattice!(st, box::Tuple{Float32,Float32})
    N = length(st.rx)
    n_side = ceil(Int, sqrt(Float64(N)))
    spacing = (box[1] / n_side, box[2] / n_side)

    rx = Vector{Float32}(undef, N)
    ry = similar(rx)

    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = linear ÷ n_side

        rx[i] = (ix + 0.5f0) * spacing[1] - box[1] / 2
        ry[i] = (iy + 0.5f0) * spacing[2] - box[2] / 2
    end

    copyto!(st.rx, rx)
    copyto!(st.ry, ry)
    return st
end

"""
Randomly assign cold/hot types across the lattice without moving particles.
Positions remain on the square lattice while typeid is shuffled.
"""
function randomize_types!(st; ratio::Float64=0.5)
    N = length(st.rx)
    n_cold = round(Int, ratio * N)
    p = randperm(N)
    host = fill(Int32(2), N)
    @inbounds host[p[1:n_cold]] .= Int32(1)
    st.typeid .= CuArray(host)
    return nothing
end

function main()
    # simulation parameters (local scope, no globals)
    n = 40_000
    box = (400.0f0, 400.0f0)
    soft_sigma = 1.0f0
    soft_epsilon = 100.0f0
    rcut = soft_sigma

    gamma = friction_coefficient(0.00234f0, Float32(3.405e-10))
    dt    = 0.00002f0
    nsteps = 10_000_000
    log_interval = 1_000_000

    t_cold = 1.0f0
    t_hot  = 100.0f0
    t_mean = 0.5f0 * (t_cold + t_hot)

    st = build_simulation(D=2,
                                     N=n,
                                     box=(box[1], box[2]),
                                     cutoff=rcut,
                                     skin=0.4f0,
                                     cap=Int32(96),
                                     neigh_interval=10,
                                     gamma=gamma,
                                     init_temperature=t_mean,
                                     nonbonded=:soft_repulsive,
                                     softrep_params=SoftRepulsiveParams{Float32}(soft_epsilon, soft_sigma))

    initialize_square_lattice!(st, box)

    # Two particle populations (random mapping)
    randomize_types!(st; ratio=0.5)
    cold_filter = Filters.TypeIDs(1)
    hot_filter  = Filters.TypeIDs(2)

    Filters.set_friction!(st, gamma; filter=Filters.All())
    Filters.set_langevin_temperature!(st, dt,
        cold_filter => t_cold,
        hot_filter  => t_hot)

    # ---------------- Integrator info (explicit) ----------------
    println("Integrator: Brownian dynamics (overdamped midpoint, predictor–corrector).")
    println(" - Midpoint: r_mid = r + 0.5 ( μ f(r) Δt + √(2D Δt) ξ )")
    println(" - Final:    Δr = μ f(r_mid) Δt + √(2D Δt) ξ;  r ← r + Δr")
    println(" - Heat increment: dq += f(r_mid) · Δr (Sekimoto)")
    println(" - Mobility μ = 1/γ, D = kT/γ with kT set per filter (cold/hot).")

    # Brownian parameters reference simulation state arrays
    bp = brownian(st)

    # Writers -----------------------------------------------------------------
    output_dir = @__DIR__
    gsd_path = joinpath(output_dir, "traj2d_twoT_BD.gsd")
    csv_path = joinpath(output_dir, "obs2d_twoT_BD.csv")

    gsdh = gsd_open(gsd_path)
    type_names = ["C", "H"]
    write_gsd_frame!(gsdh, st; diameter=soft_sigma, types_names=type_names, step=st.step)

    open(csv_path, "w") do io
        println(io, "step,dQ_cold,dQ_hot,elapsed_s,steps_per_sec,eta_s")
    end

    records = Int[]
    s_cold = Float32[]
    s_hot  = Float32[]

    start_time = time()
    max_runtime = let v = get(ENV, "NEQSIM_MAX_SECONDS", "")
        isempty(v) ? Inf : parse(Float64, v)
    end

    for step in 1:nsteps
        step!(st, bp, dt, compute_energy=false)

        if step % log_interval == 0
            push!(records, step)
            qc = Filters.sum(st.dq, st, cold_filter)
            qh = Filters.sum(st.dq, st, hot_filter)
            push!(s_cold, qc)
            push!(s_hot,  qh)

            elapsed = time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            remaining = nsteps - step
            eta = remaining / max(steps_per_sec, 1e-6)

            open(csv_path, "a") do io
                @printf(io, "%d,%.6f,%.6f,%.3f,%.3f,%.3f\n", step, qc, qh, elapsed, steps_per_sec, eta)
            end

            write_gsd_frame!(gsdh, st; diameter=soft_sigma, types_names=type_names, step=st.step)

            @info "progress (BD)" step=step elapsed_s=elapsed steps_per_sec=steps_per_sec eta_s=eta

            if elapsed >= max_runtime
                @info "Reached max runtime limit" limit_s=max_runtime step=step
                break
            end
        end
    end

    total_time = time() - start_time

    if isempty(records)
        println("No samples recorded (consider reducing log_interval)")
    else
        last_step = last(records)
        println("Recorded $(length(records)) BD samples for cold/hot groups")
        println("Final sample step: $(last_step)")
        println("  Heat (dq) cold ≈ $(last(s_cold))")
        println("  Heat (dq) hot  ≈ $(last(s_hot))")
    end

    println("Total wall time ≈ $(round(total_time, digits=2)) s")

    gsd_close(gsdh)
    println("Wrote BD trajectory to $(gsd_path)")
    println("Wrote BD observables to $(csv_path)")
end

# run when invoked as a script
main()
