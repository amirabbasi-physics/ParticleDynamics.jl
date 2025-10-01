using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, NeighborLists, BrownianIntegrators, NonBondedForces, Writers
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

# ---- Build base simulation state (for storage & neighbor list) ----
st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=sigma, skin=0.5f0, cap=cap,
                                 neigh_interval=1,
                                 epsilon=1f0, sigma=1f0,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# Initialize positions
initialize_square_lattice!(st, box)

# Brownian parameters
const BI = NonEqSimGPU.BrownianIntegrators
bp = BI.BrownianParams{Float32}(gamma, temperature)
μ = 1f0 / bp.γ
Dth = bp.kT / bp.γ
sqrt2Ddt = sqrt(2f0*Dth*dt)

# Initial forces at t=0 (soft repulsive)
NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2::Definitions.Box2, sr)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj2d_soft_repulsive.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)

# ---- Run BD with soft repulsive forces ----
@time for s in 1:N_steps
    # Neighbor list rebuild check
    if s % 20 == 0
        rebuild_needed = NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                                     skin=st.nbh.skin,
                                                     Lx=st.box2[1], Ly=st.box2[2], step=st.step)
        if rebuild_needed
            NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
        end
    end

    # Prepare midpoint positions (stores in st.vx, st.vy)
    BrownianIntegrators.bd_prepare_midpoint_2d!(st.rx, st.ry, st.fx, st.fy,
                                                st.rf_x, st.rf_y, st.vx, st.vy,
                                                μ, sqrt2Ddt, dt, st.box2::Definitions.Box2)
    # Forces at midpoint positions -> f0x,f0y (no energy to save time)
    NonBondedForces.harmonic_rep_forces_soa_noE!(st.vx, st.vy, st.f0x, st.f0y,
                                                 st.nbh, st.box2::Definitions.Box2, sr)

    # Finish step: update positions using midpoint forces
    BrownianIntegrators.bd_finish_step_2d!(st.rx, st.ry, st.f0x, st.f0y,
                                           st.rf_x, st.rf_y,
                                           μ, sqrt2Ddt, dt, st.dq, st.box2::Definitions.Box2)

    # Forces at new positions (compute energy every N_log)
    if s % N_log == 0
        NonBondedForces.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                                 st.nbh, st.box2::Definitions.Box2, sr)
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs2d_soft_repulsive.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "soft-rep step" step=s Epot_sum=sum(st.Epot)
    else
        NonBondedForces.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy,
                                                     st.nbh, st.box2::Definitions.Box2, sr)
    end

    st.step += 1
end

Writers.gsd_close(gsdh)
println("Done. GSD: $gsd_path")

