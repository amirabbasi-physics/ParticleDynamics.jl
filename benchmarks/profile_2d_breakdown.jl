using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, NeighborLists, NonBondedForces, Integrators
using CUDA
using Statistics

# Parameters
N   = 10_000
box = (125.0f0, 125.0f0)

r_cut = Float32(2.5)
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
gamma = 10f0
temperature = 1f0
dt = 0.0005f0

noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=1,
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# lattice init
N = length(st.rx); n_side = ceil(Int, sqrt(N))
rx_host = Float32[]; ry_host = Float32[]
for i in 1:N
    ix = (i-1) % n_side; iy = (i-1) ÷ n_side
    spacing_x = box[1] / n_side; spacing_y = box[2] / n_side
    x = (ix + 0.5f0) * spacing_x - box[1]/2
    y = (iy + 0.5f0) * spacing_y - box[2]/2
    push!(rx_host, x); push!(ry_host, y)
end
copyto!(st.rx, rx_host); copyto!(st.ry, ry_host)

# warm-up
for s in 1:100
    Simulation.step!(st, dt)
end
CUDA.synchronize()

ns = 1000
times = Dict{String, Vector{Float64}}(
    "nl_check" => Float64[],
    "nl_rebuild" => Float64[],
    "forces_t" => Float64[],
    "noise" => Float64[],
    "pos" => Float64[],
    "forces_tdt" => Float64[],
    "vel" => Float64[],
    "step_total" => Float64[],
)

for s in 1:ns
    t_step = time()

    # NL check
    CUDA.synchronize(); t0 = time()
    rebuild_needed = NeighborLists.update_needed!(st.nbh, st.rx, st.ry;
                                                 skin=st.nbh.skin, Lx=st.box2[1], Ly=st.box2[2], step=st.step)
    CUDA.synchronize(); push!(times["nl_check"], time()-t0)

    # NL rebuild
    if rebuild_needed
        CUDA.synchronize(); t0 = time()
        NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step=st.step)
        CUDA.synchronize(); push!(times["nl_rebuild"], time()-t0)
    else
        push!(times["nl_rebuild"], 0.0)
    end

    # forces at t -> f0*
    CUDA.synchronize(); t0 = time()
    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.f0x, st.f0y, st.Epot,
                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
    CUDA.synchronize(); push!(times["forces_t"], time()-t0)

    # noise
    CUDA.synchronize(); t0 = time()
    Integrators.vv_prepare_noise!(st.rf_x, st.rf_y, st.vv.noise_scale; beta_z=nothing)
    CUDA.synchronize(); push!(times["noise"], time()-t0)

    # positions
    CUDA.synchronize(); t0 = time()
    Integrators.vv_positions_soa!(st.rx, st.ry, st.vx, st.vy, st.f0x, st.f0y,
                                  st.rf_x, st.rf_y, st.vv, dt, st.box2::Definitions.Box2)
    CUDA.synchronize(); push!(times["pos"], time()-t0)

    # forces at t+dt -> f*
    CUDA.synchronize(); t0 = time()
    NonBondedForces.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot,
                                   st.nbh, st.box2::Definitions.Box2, st.pair_lj)
    CUDA.synchronize(); push!(times["forces_tdt"], time()-t0)

    # velocities
    CUDA.synchronize(); t0 = time()
    Integrators.vv_velocities_soa!(st.vx, st.vy, st.f0x, st.f0y, st.fx, st.fy,
                                   st.rf_x, st.rf_y, st.dq, st.Ekin, st.vv, dt)
    CUDA.synchronize(); push!(times["vel"], time()-t0)

    st.step += 1
    CUDA.synchronize(); push!(times["step_total"], time()-t_step)
end

function ms(x)
    return 1000 .* mean(x)
end

println("Breakdown over ", ns, " steps (mean ms):")
println("  nl_check:   ", ms(times["nl_check"]))
println("  nl_rebuild: ", ms(times["nl_rebuild"]))
println("  forces_t:   ", ms(times["forces_t"]))
println("  noise:      ", ms(times["noise"]))
println("  pos:        ", ms(times["pos"]))
println("  forces_tdt: ", ms(times["forces_tdt"]))
println("  vel:        ", ms(times["vel"]))
println("  step_total: ", ms(times["step_total"]))

