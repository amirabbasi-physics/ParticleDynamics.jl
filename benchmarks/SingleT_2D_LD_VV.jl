using NonEqSimGPU
using NonEqSimGPU: Filters
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

function main()
    # Simulation parameters
    n = 20_000
    box = (200.0f0, 200.0f0)
    sigma = 1.0f0
    epsilon = 1000.0f0
    rcut = Float32(2.0^(1/6) * sigma)   # WCA (purely repulsive)

    gamma = 615.985f0
    dt    = Float32(2.0e-5)
    nsteps = 2_000_000
    log_interval = 100_000

    # Single temperature (expect ⟨Q_tot⟩ ≈ 0)
    t_bath = 1000.0f0

    st = build_simulation(D=2,
                                     N=n,
                                     box=(box[1], box[2]),
                                     cutoff=rcut,
                                     skin=0.5f0,
                                     cap=Int32(200),
                                     neigh_interval=25,
                                     epsilon=epsilon,
                                     sigma=sigma,
                                     gamma=gamma,
                                     init_temperature=t_bath,
                                     nonbonded=:wca)

    # Lattice initialization
    initialize_square_lattice!(st, box)

    # Uniform Langevin bath at t_bath for all particles
    Filters.set_langevin_temperature!(st, dt, t_bath; filter=Filters.All())

    # Output: CSV + GSD trajectory
    output_dir = @__DIR__
    csv_path = joinpath(output_dir, "obs2d_singleT_VV.csv")
    gsd_path = joinpath(output_dir, "traj2d_singleT_VV.gsd")
    gsdh = gsd_open(gsd_path)
    type_names = ["C"]
    write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step)
    open(csv_path, "w") do io
        println(io, "step,Ekin_avg,Qtot,dQ_interval,elapsed_s,steps_per_sec,eta_s")
    end

    # Progress/time control
    start_time = time()
    max_runtime = let v = get(ENV, "NEQSIM_MAX_SECONDS", "")
        isempty(v) ? Inf : parse(Float64, v)
    end

    last_Q = 0.0f0

    println("Integrator: Langevin dynamics (GJF/Velocity-Verlet) with WCA potential.")
    println("Benchmark goal: In a single-temperature bath, total heat Qtot fluctuates around 0.")

    for step in 1:nsteps
        step!(st, vv(st), dt; compute_energy=false)

        if step % log_interval == 0
            # total kinetic energy per particle and heat exchanged with bath
            Ekin_avg = Float32(Base.sum(Array(st.Ekin)) / length(st.Ekin))
            Qtot = Float32(Base.sum(Array(st.dq)))
            dQ = Qtot - last_Q
            last_Q = Qtot

            elapsed = time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            remaining = nsteps - step
            eta = remaining / max(steps_per_sec, 1e-6)

            open(csv_path, "a") do io
                @printf(io, "%d,%.7e,%.7e,%.7e,%.3f,%.3f,%.3f\n", step, Ekin_avg, Qtot, dQ, elapsed, steps_per_sec, eta)
            end

            write_gsd_frame!(gsdh, st; diameter=sigma, types_names=type_names, step=st.step)

            @info "progress (singleT)" step=step Ekin_avg=Ekin_avg Qtot=Qtot dQ=dQ steps_per_sec=steps_per_sec eta_s=eta

            if elapsed >= max_runtime
                @info "Reached max runtime limit" limit_s=max_runtime step=step
                break
            end
        end
    end

    total_time = time() - start_time
    println("Total wall time ≈ $(round(total_time, digits=2)) s")
    gsd_close(gsdh)
    println("Wrote observables to $(csv_path)")
    println("Wrote trajectory to $(gsd_path)")
end

# Run when invoked as a script
main()
