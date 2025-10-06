using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions
using NonEqSimGPU.Filters
using NonEqSimGPU.Writers
using CUDA
using Random
using Printf

CUDA.allowscalar(false)

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
Ensures positions remain on the square lattice while typeid is shuffled.
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
    box = (200.0f0, 200.0f0)
    sigma = 1.0f0
    epsilon = 1000.0f0
    rcut = Float32( 2.0^(1/6)* sigma)   # WCA cutoff

    gamma = 8057.06f0
    dt    = Float32(5.0e-6)
    nsteps = 1_000_000
    log_interval = 100_000

    t_cold = 1000.0f0
    t_hot  = 1000.0f0
    t_mean = 0.5f0 * (t_cold + t_hot)

    st = Simulation.build_simulation(D=2,
                                     N=n,
                                     box=(box[1], box[2]),
                                     cutoff=rcut,
                                     skin=0.5f0,
                                     cap=Int32(200),
                                     neigh_interval=100,
                                     epsilon=epsilon,
                                     sigma=sigma,
                                     gamma=gamma,
                                     init_temperature=t_mean,
                                     nonbonded=:wca)

    # positions on a square lattice, then randomize types
    initialize_square_lattice!(st, box)
    randomize_types!(st; ratio=0.5)

    cold_filter = Filters.TypeIDs(1)
    hot_filter  = Filters.TypeIDs(2)

    # per-group Langevin temperature (sets per-particle noise scale sqrt(2γTΔt))
    Filters.set_langevin_temperature!(st, dt,
        cold_filter => t_cold,
        hot_filter  => t_hot)

    # quick check of noise scales by group
    noise_scale = Array(st.vv.noise_scale)
    cold_idx = Filters.resolve(st, cold_filter)
    hot_idx  = Filters.resolve(st, hot_filter)
    mean_ns_cold = Base.sum(noise_scale[cold_idx]) / length(cold_idx)
    mean_ns_hot  = Base.sum(noise_scale[hot_idx])  / length(hot_idx)
    @info "Noise scale" cold_first=noise_scale[cold_idx[1]] hot_first=noise_scale[hot_idx[1]] mean_cold=mean_ns_cold mean_hot=mean_ns_hot

    # Integrator info (explicit)
    println("Integrator: Langevin dynamics (GJF/Velocity-Verlet).")
    println(" - Positions: r_{n+1} = r_n + b Δt v_n + (b Δt / (2m)) (Δt f_n + β_n)")
    println(" - Velocities: v_{n+1} = a v_n + (Δt/(2m)) (a f_n + f_{n+1}) + (b/m) β_n")
    println(" - Coefficients: a=(1-q)/(1+q), b=1/(1+q), q=γ Δt/(2m)")
    println(" - β ~ N(0, 2 γ kT Δt) with T set via filters")

    # Writers -----------------------------------------------------------------
    output_dir = @__DIR__
    gsd_path = joinpath(output_dir, "traj2d_filters.gsd")
    csv_path = joinpath(output_dir, "obs2d_filters.csv")

    gsdh = Writers.gsd_open(gsd_path)
    type_names = ["C", "H"]
    Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step)

    open(csv_path, "w") do io
        println(io, "step,Ekin_cold,Ekin_hot,dQ_cold,dQ_hot,elapsed_s,steps_per_sec,eta_s")
    end

    records = Int[]
    ekin_cold = Float32[]
    ekin_hot  = Float32[]
    sdot_cold = Float32[]
    sdot_hot  = Float32[]

    start_time = time()
    max_runtime = let v = get(ENV, "NEQSIM_MAX_SECONDS", "")
        isempty(v) ? Inf : parse(Float64, v)
    end

    for step in 1:nsteps
        # Explicit integrator selection: Langevin (GJF/Velocity-Verlet)
        Simulation.step!(st, st.vv, dt; compute_energy=false)

        if step % log_interval == 0
            push!(records, step)
            kc = Filters.sum(st.Ekin, st, cold_filter) / Filters.count(st, cold_filter)
            kh = Filters.sum(st.Ekin, st, hot_filter) / Filters.count(st, hot_filter)
            qc = Filters.sum(st.dq, st, cold_filter)
            qh = Filters.sum(st.dq, st, hot_filter)
            push!(ekin_cold, kc)
            push!(ekin_hot, kh)
            push!(sdot_cold, qc / t_cold)
            push!(sdot_hot,  qh / t_hot)

            elapsed = time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            remaining = nsteps - step
            eta = remaining / max(steps_per_sec, 1e-6)

            open(csv_path, "a") do io
                @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n", step, kc, kh, qc, qh, elapsed, steps_per_sec, eta)
            end

            Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step)

            @info "progress" step=step elapsed_s=elapsed steps_per_sec=steps_per_sec eta_s=eta

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
        println("Recorded $(length(records)) samples for cold/hot groups")
        println("Final sample step: $(last_step)")
        println("  ⟨E_kin⟩ cold ≈ $(last(ekin_cold))")
        println("  ⟨E_kin⟩ hot  ≈ $(last(ekin_hot))")
        println("  Heat/T   cold ≈ $(last(sdot_cold))")
        println("  Heat/T   hot  ≈ $(last(sdot_hot))")
    end

    println("Total wall time ≈ $(round(total_time, digits=2)) s")

    Writers.gsd_close(gsdh)
    println("Wrote trajectory to $(gsd_path)")
    println("Wrote observables to $(csv_path)")
end

# run when invoked as a script
main()
