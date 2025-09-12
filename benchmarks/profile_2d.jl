using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions
using CUDA

# Parameters (kept similar to minimal benchmark but shorter run)
N   = 10_000
box = (125.0f0, 125.0f0)

r_cut = Float32(2.5)
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
gamma = 10f0
temperature = 1f0
dt = 0.0005f0

# Langevin noise scale per particle: sqrt(2γT) (mass=1)
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# Build
st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=1,
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# Initialize positions: simple square lattice centered in box
N = length(st.rx)
n_side = ceil(Int, sqrt(N))
rx_host = Float32[]; ry_host = Float32[]
for i in 1:N
    ix = (i-1) % n_side
    iy = (i-1) ÷ n_side
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    x = (ix + 0.5f0) * spacing_x - box[1]/2
    y = (iy + 0.5f0) * spacing_y - box[2]/2
    push!(rx_host, x); push!(ry_host, y)
end
copyto!(st.rx, rx_host); copyto!(st.ry, ry_host)

# Warm-up (JIT + caches)
for s in 1:200
    Simulation.step!(st, dt)
end
CUDA.synchronize()

# Profile main loop (short)
ns = 2_000
CUDA.@profile begin
    for s in 1:ns
        Simulation.step!(st, dt)
    end
end
CUDA.synchronize()

println("Completed profiling run: steps=" * string(ns))
println("Epot_sum=", sum(st.Epot), ", Ekin_sum=", sum(st.Ekin))

