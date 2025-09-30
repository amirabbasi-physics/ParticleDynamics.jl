using NonEqSimGPU
using NonEqSimGPU: Simulation
using NonEqSimGPU.Filters
using NonEqSimGPU.Writers
using CUDA
using Random
using Printf

CUDA.allowscalar(false)

function initialize_simple_cubic!(st, box::NTuple{3,Float32})
    N = length(st.rx)
    n_side = ceil(Int, cbrt(Float64(N)))
    spacing = (box[1] / n_side, box[2] / n_side, box[3] / n_side)

    rx = Vector{Float32}(undef, N)
    ry = similar(rx)
    rz = similar(rx)

    n_side_sq = n_side^2
    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = (linear ÷ n_side) % n_side
        iz = linear ÷ n_side_sq

        rx[i] = (ix + 0.5f0) * spacing[1] - box[1] / 2
        ry[i] = (iy + 0.5f0) * spacing[2] - box[2] / 2
        rz[i] = (iz + 0.5f0) * spacing[3] - box[3] / 2
    end

    copyto!(st.rx, rx)
    copyto!(st.ry, ry)
    copyto!(st.rz, rz)
    return st
end

"""
Randomly assign cold/hot types across the lattice without moving particles.
Ensures positions remain on the cubic lattice while typeid is shuffled.
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

# Simulation parameters
N = 40_000
BOX = (50.0f0, 50.0f0, 50.0f0)
SIGMA = 1f0
EPS   = 100f0
RCUT  = Float32(2^(1/6) * SIGMA)
GAMMA = 100f0
DT    = 0.000002f0
NSTEPS = 10_000_000
LOG_INTERVAL = 1_000_000

T_COLD = 1.0f0
T_HOT  = 100.0f0
T_mean = 0.5f0 * (T_COLD + T_HOT)

st = Simulation.build_simulation(D=3,
                                 N=N,
                                 box=BOX,
                                 cutoff=RCUT,
                                 skin=0.4f0,
                                 cap=Int32(96),
                                 neigh_interval=10,
                                 epsilon=EPS,
                                 sigma=SIGMA,
                                 gamma=GAMMA,
                                 init_temperature=T_mean)

initialize_simple_cubic!(st, BOX)

# Two particle populations distinguished by type id (randomly mapped to lattice)
randomize_types!(st; ratio=0.5)

cold_filter = Filters.TypeIDs(1)
hot_filter  = Filters.TypeIDs(2)

# Configure stochastic forcing per group (sqrt(2 γ T Δt))
Filters.set_langevin_temperature!(st, DT,
    cold_filter => T_COLD,
    hot_filter  => T_HOT)

noise_scale = Array(st.vv.noise_scale)
cold_idx = Filters.resolve(st, cold_filter)
hot_idx  = Filters.resolve(st, hot_filter)
mean_ns_cold = Base.sum(noise_scale[cold_idx]) / length(cold_idx)
mean_ns_hot  = Base.sum(noise_scale[hot_idx])  / length(hot_idx)
@info "Noise scale" cold_first=noise_scale[cold_idx[1]] hot_first=noise_scale[hot_idx[1]] mean_cold=mean_ns_cold mean_hot=mean_ns_hot

# Writers -----------------------------------------------------------------
output_dir = @__DIR__
gsd_path = joinpath(output_dir, "traj_filters.gsd")
csv_path = joinpath(output_dir, "obs_filters.csv")

gsdh = Writers.gsd_open(gsd_path)
type_names = ["C", "H"]
Writers.write_gsd_frame!(gsdh, st; diameter=SIGMA, types_names=type_names, step=st.step)

open(csv_path, "w") do io
    println(io, "step,Ekin_cold,Ekin_hot,dQ_cold,dQ_hot,elapsed_s,steps_per_sec,eta_s")
end

records = Int[]
Ekin_cold = Float32[]
Ekin_hot  = Float32[]
Sdot_cold = Float32[]
Sdot_hot  = Float32[]

start_time = time()
max_runtime = let v = get(ENV, "NEQSIM_MAX_SECONDS", "")
    isempty(v) ? Inf : parse(Float64, v)
end

for step in 1:NSTEPS
    Simulation.step!(st, DT, compute_energy=false)

    if step % LOG_INTERVAL == 0
        push!(records, step)
        kc = Filters.sum(st.Ekin, st, cold_filter) / Filters.count(st, cold_filter)
        kh = Filters.sum(st.Ekin, st, hot_filter) / Filters.count(st, hot_filter)
        qc = Filters.sum(st.dq, st, cold_filter)
        qh = Filters.sum(st.dq, st, hot_filter)
        push!(Ekin_cold, kc)
        push!(Ekin_hot, kh)
        push!(Sdot_cold, qc / T_COLD)
        push!(Sdot_hot,  qh / T_HOT)

        elapsed = time() - start_time
        steps_per_sec = step / max(elapsed, 1e-6)
        remaining = NSTEPS - step
        eta = remaining / max(steps_per_sec, 1e-6)

        open(csv_path, "a") do io
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n", step, kc, kh, qc, qh, elapsed, steps_per_sec, eta)
        end

        Writers.write_gsd_frame!(gsdh, st; diameter=SIGMA, types_names=type_names, step=st.step)

        @info "progress" step=step elapsed_s=elapsed steps_per_sec=steps_per_sec eta_s=eta

        if elapsed >= max_runtime
            @info "Reached max runtime limit" limit_s=max_runtime step=step
            break
        end
    end
end

total_time = time() - start_time

if isempty(records)
    println("No samples recorded (consider reducing LOG_INTERVAL)")
else
    last_step = last(records)
    println("Recorded $(length(records)) samples for cold/hot groups")
    println("Final sample step: $(last_step)")
    println("  ⟨E_kin⟩ cold ≈ $(last(Ekin_cold))")
    println("  ⟨E_kin⟩ hot  ≈ $(last(Ekin_hot))")
    println("  Heat/T   cold ≈ $(last(Sdot_cold))")
    println("  Heat/T   hot  ≈ $(last(Sdot_hot))")
end

println("Total wall time ≈ $(round(total_time, digits=2)) s")

Writers.gsd_close(gsdh)
println("Wrote trajectory to $(gsd_path)")
println("Wrote observables to $(csv_path)")
