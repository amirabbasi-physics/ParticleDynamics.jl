using NonEqSimGPU
using NonEqSimGPU: Filters
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
        rx_host[i] = (ix + 0.5) * spacing_x - box[1] / 2
        ry_host[i] = (iy + 0.5) * spacing_y - box[2] / 2
    end
    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    return st
end

# ---- Params ----
N   = 10_000
box = (200.0, 200.0)

# Soft-repulsive parameters
sigma = 2.0
epsilon = 10000.0
sr = SoftRepulsiveParams{Float64}(epsilon, sigma)

cap = Int32(96)
gamma = 10.0
temperature = 1.0
dt = 1.0e-4
N_steps = 100_000
N_log = 10_000

# ---- Build base simulation state (use newly wired soft-repulsive nonbonded) ----
st = build_simulation(D = 2, N=N, box=box, cutoff=sigma, skin=0.5, cap=cap,
                                 neigh_interval=1,
                                 epsilon=1.0, sigma=1.0,
                                 gamma=gamma, temperature=temperature, dt=dt,
                                 nonbonded=:soft_repulsive, softrep_params=sr)

# Initialize positions
initialize_square_lattice!(st, box)

# Set bath properties via filters (applies to all particles here)
Filters.set_friction!(st, gamma; filter=Filters.All())
Filters.set_langevin_temperature!(st, dt, temperature; filter=Filters.All())

# Brownian parameters
bp = brownian(st)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj2d_soft_repulsive.gsd")
gsdh = gsd_open(gsd_path)
types = ["C"]
write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)

# ---- Run BD with soft repulsive forces via step!(st, bp, dt) ----
@time for s in 1:N_steps
    if s % N_log == 0
        step!(st, bp, dt; compute_energy=true)
        write_observables_csv!(joinpath(@__DIR__, "obs2d_soft_repulsive.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "soft-rep step" step=s Epot_sum=sum(st.Epot)
    else
        step!(st, bp, dt; compute_energy=false)
    end
end

gsd_close(gsdh)
println("Done. GSD: $gsd_path")
