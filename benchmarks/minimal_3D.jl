using NonEqSimGPU
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
N   = 40_000
box = (250.0f0, 250.0f0, 250.0f0)

# LJ parameters
r_cut = Float32(2^(1/6))
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
gamma = 10f0
temperature = 1f0

N_steps = 10_000_000
N_log = 1_000_000
dt = 0.00005f0

# ---- Build ---- (noise scale computed internally)
st = build_simulation(D = 3, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                      neigh_interval=1,
                      epsilon=epsilon, sigma=sigma,
                      gamma=gamma, temperature=temperature, dt=dt)

# ---- Initialize positions ----
initialize_simple_cubic_lattice!(st, box)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj3d.gsd")
gsdh = gsd_open(gsd_path)
types = ["C"]
write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)

# ---- Run ----
@time for s in 1:N_steps
    if s % N_log == 0
        step!(st, dt)
        write_observables_csv!(joinpath(@__DIR__, "obs3d.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    else
        step!(st, dt, compute_energy=false)
    end
end

gsd_close(gsdh)
println("Done. GSD: $gsd_path")
