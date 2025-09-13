using NonEqSimGPU
using NonEqSimGPU: Simulation
using CUDA

# ---- Params ----
N = 10_000
box = (125.0f0, 125.0f0) # box dimensions (Lx, Ly)

# LJ parameters
r_cut = Float32(2.5) # LJ cutoff
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
gamma = 10f0
temperature = 1f0
dt = 0.001f0

# Langevin noise_scale is unused by BD, but build_simulation expects it
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# ---- Build ----
st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                neigh_interval=10,
                                epsilon=epsilon, sigma=sigma,
                                gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# ---- Initialize positions (square lattice centered in box) ----
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
    push!(rx_host, x)
    push!(ry_host, y)
end
copyto!(st.rx, rx_host)
copyto!(st.ry, ry_host)

# ---- Brownian parameters ----
const BI = NonEqSimGPU.BrownianIntegrators
bp = BI.BrownianParams{Float32}(gamma, temperature)

# ---- GSD writer ----
using NonEqSimGPU: Writers
gsd_path = joinpath(@__DIR__, "traj2d_bd.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)

# ---- Run (Brownian midpoint EM, skip energy on steady steps) ----
@time for s in 1:1_000_000
    compute_E = (s % 100_000 == 0)
    Simulation.step!(st, bp, dt; compute_energy=compute_E)
    if compute_E
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs2d_bd.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame (bd)" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD (bd): $gsd_path")
