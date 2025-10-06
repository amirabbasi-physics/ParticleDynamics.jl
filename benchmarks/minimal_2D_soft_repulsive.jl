using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, BrownianIntegrators, Writers
using CUDA

# Simple 2D Brownian dynamics with soft repulsive harmonic nonbonded interaction

function initialize_square_lattice!(st, box::NTuple{2,Float32})
    N = length(st.rx)
    n_side = ceil(Int, sqrt(N))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side

    rx_host = Vector{Float32}(undef, N)
    ry_host = Vector{Float32}(undef, N)

    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = linear ÷ n_side
        rx_host[i] = (ix + 0.5f0) * spacing_x - box[1] / 2
        ry_host[i] = (iy + 0.5f0) * spacing_y - box[2] / 2
    end
    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    return st
end

# ---- Params ----
N   = 10_000
box = (200.0f0, 200.0f0)

# Soft-repulsive parameters
sigma = 2.0f0
epsilon = 10000.0f0
sr = Definitions.SoftRepulsiveParams{Float32}(epsilon, sigma)

cap = Int32(96)
gamma = 10f0
temperature = 1f0
dt = 1.0f-4
N_steps = 100_000
N_log = 10_000

# Langevin noise scale required by builder (unused in BD)
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# ---- Build base simulation state (use newly wired soft-repulsive nonbonded) ----
st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=sigma, skin=0.5f0, cap=cap,
                                 neigh_interval=1,
                                 epsilon=1f0, sigma=1f0,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature,
                                 nonbonded=:soft_repulsive, softrep_params=sr)

# Initialize positions
initialize_square_lattice!(st, box)

# Brownian parameters
const BI = NonEqSimGPU.BrownianIntegrators
bp = BI.BrownianParams{Float32}(gamma, temperature)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj2d_soft_repulsive.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)

# ---- Run BD with soft repulsive forces via step!(st, bp, dt) ----
@time for s in 1:N_steps
    if s % N_log == 0
        Simulation.step!(st, bp, dt; compute_energy=true)
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs2d_soft_repulsive.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "soft-rep step" step=s Epot_sum=sum(st.Epot)
    else
        Simulation.step!(st, bp, dt; compute_energy=false)
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD: $gsd_path")
