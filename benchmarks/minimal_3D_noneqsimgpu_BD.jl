using NonEqSimGPU
using NonEqSimGPU: Simulation
using CUDA

function initialize_simple_cubic_lattice!(st, box::NTuple{3,Float32})
    N = length(st.rx)
    n_side = ceil(Int, cbrt(Float64(N)))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    spacing_z = box[3] / n_side

    rx_host = Vector{Float32}(undef, N)
    ry_host = Vector{Float32}(undef, N)
    rz_host = Vector{Float32}(undef, N)

    n_side_sq = n_side^2
    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = (linear ÷ n_side) % n_side
        iz = linear ÷ n_side_sq

        rx_host[i] = (ix + 0.5f0) * spacing_x - box[1] / 2
        ry_host[i] = (iy + 0.5f0) * spacing_y - box[2] / 2
        rz_host[i] = (iz + 0.5f0) * spacing_z - box[3] / 2
    end

    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    copyto!(st.rz, rz_host)
    return st
end

# ---- Params ----
N = 40_000
box = (50.0f0, 50.0f0, 50.0f0)

# LJ parameters
r_cut = Float32(2^(1/6))
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
gamma = 10f0
temperature = 1f0
dt = 0.00025f0

noise_scale = CUDA.fill(sqrt(2f0 * gamma * temperature * dt), N)

# ---- Build ----
st = Simulation.build_simulation(D = 3, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                neigh_interval=10,
                                epsilon=epsilon, sigma=sigma,
                                gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# ---- Initialize positions ----
initialize_simple_cubic_lattice!(st, box)

# ---- Brownian parameters ----
const BI = NonEqSimGPU.BrownianIntegrators
bp = BI.BrownianParams{Float32}(gamma, temperature)

# ---- GSD writer ----
using NonEqSimGPU: Writers
gsd_path = joinpath(@__DIR__, "traj3d_bd.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)

# ---- Run ----
@time for s in 1:10_000_000
    compute_E = (s % 1000_000 == 0)
    Simulation.step!(st, bp, dt; compute_energy=compute_E)
    if compute_E
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs3d_bd.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame (bd)" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD (bd): $gsd_path")
