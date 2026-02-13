using NonEqSimGPU
using NonEqSimGPU: step!, eulermaruyama

# Active Ornstein–Uhlenbeck particles in 2D (overdamped limit, Fodor et al. PRL 117, 038103)
# Parameters follow the paper: N=10000, box 250×250, D=100, τ=20, μ=1 (γ=1).
# In the τ → 0 limit this reduces to overdamped Brownian dynamics at temperature T = D = 100.

N = 10000
L = 125.0
dt = 2e-7             # small enough to resolve τ and steep forces
n_steps = 5e7
write_interval = 1e5

function initialize_square_lattice!(st, box::NTuple{2,Real})
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
    copyto!(st.rx, rx_host); copyto!(st.ry, ry_host)
    return st
end

# Fodor et al. parameters (PRL 117, 038103)
gamma = 1.0f0                # mobility μ = 1/γ = 1
temperature = 100.0f0        # D = 100 => T = D when γ = 1
corr_time = 100.0           # persistence time τ

# Use the steep short-range repulsion from the paper (A = 100, a = 2). Approximate with soft_repulsive.
epsilon = 1.0e6
sigma = 1.0

st = build_simulation(
    N = N,
    box = (L, L),
    cutoff = sigma,
    skin = 0.5f0,
    cap = Int32(256),
    neigh_interval = 20,
    epsilon = epsilon,
    sigma = sigma,
    gamma = gamma,
    temperature = temperature,
    noise_corr_time = corr_time,
    dt = dt,
    nonbonded = :soft_repulsive,
    precision = :f64,
)

initialize_square_lattice!(st, (L, L))

spec = eulermaruyama(st)

gsd_path = joinpath(@__DIR__, "traj2d_active_OU_bd.gsd")
gsdh = NonEqSimGPU.Writers.gsd_open(gsd_path)
types = ["C"]
NonEqSimGPU.Writers.write_gsd_frame!(gsdh, st; diameter=1f0, types_names=types, step=st.step)

for s in 1:n_steps
    step!(st, spec, dt; compute_energy=false)
    if s % write_interval == 0
        NonEqSimGPU.Writers.write_gsd_frame!(gsdh, st; diameter=1f0, types_names=types, step=st.step)
        @info "wrote BD frame" step=s
    end
end

NonEqSimGPU.Writers.gsd_close(gsdh)
println("Finished $(n_steps) BD steps with correlated OU noise (τ=$(corr_time)). GSD: $(gsd_path)")
