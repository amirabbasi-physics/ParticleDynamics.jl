using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, Writers
using CUDA


# ---- Params ----
N   = 10_000
box = (125.0f0, 125.0f0)  # box dimensions (Lx, Ly)

# LJ parameters
r_cut = Float32(2.5)  # LJ cutoff
sigma = 1f0
epsilon = 10f0

cap = Int32(100)
# Note: neigh_interval not used with new displacement-based neighbor list algorithm
gamma = 10f0
temperature = 1f0
dt = 0.0005f0

# Langevin noise scale per particle: sqrt(2γT) (mass=1)
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# ---- Build ----
st = Simulation.build_simulation(D = 2, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=1,  # Not used with displacement-based algorithm
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# ---- Initialize positions after building simulation state ----
# Square lattice positions centered in the box: [-L/2, +L/2] for each dimension
N = length(st.rx)
n_side = ceil(Int, sqrt(N))  # Number of particles per side

rx_host = Float32[]
ry_host = Float32[]

for i in 1:N
    # Convert linear index to 2D grid coordinates
    ix = (i-1) % n_side
    iy = (i-1) ÷ n_side
    
    # Calculate lattice spacing
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    
    # Position on lattice, centered around origin
    x = (ix + 0.5f0) * spacing_x - box[1]/2
    y = (iy + 0.5f0) * spacing_y - box[2]/2
    
    push!(rx_host, x)
    push!(ry_host, y)
end

# Copy to GPU arrays  
copyto!(st.rx, rx_host)
copyto!(st.ry, ry_host)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj2d.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)  # initial frame

# ---- Run ----
@time for s in 1:1000000
    Simulation.step!(st, dt)
    if s % 100000 == 0
        # Only write available observables - Ekin might not be computed yet
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs2d.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        #Writers.write_xyz!(joinpath(@__DIR__, "traj2d.xyz"); rx=st.rx, ry=st.ry, rz=nothing)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD: $gsd_path")