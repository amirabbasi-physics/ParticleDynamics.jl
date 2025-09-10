using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, Writers
using CUDA


# ---- Params ----
N   = 10_000
box = (22.75f0, 22.75f0, 22.75f0)  # box dimensions (Lx, Ly, Lz)

# LJ parameters
r_cut = Float32(2.5)  # LJ cutoff
sigma = 1f0
epsilon = 1f0


cap = Int32(100)
neigh_interval = 10
gamma = 10f0
temperature = 1f0
dt = 0.00002f0

# Langevin noise scale per particle: sqrt(2γT) (mass=1)
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# ---- Build ----
st = Simulation.build_simulation(D = 3, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=neigh_interval, epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature)

# ---- Initialize positions after building simulation state ----
# Simple cubic (SC) lattice positions centered in the box: [-L/2, +L/2] for each dimension
N = length(st.rx)
n_side = ceil(Int, cbrt(N))  # Number of particles per side in 3D (cube root)

rx_host = Float32[]
ry_host = Float32[]
rz_host = Float32[]

for i in 1:N
    # Convert linear index to 3D grid coordinates
    ix = (i-1) % n_side
    iy = ((i-1) ÷ n_side) % n_side
    iz = (i-1) ÷ (n_side^2)
    
    # Calculate lattice spacing
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    spacing_z = box[3] / n_side

    # Position on lattice, centered around origin
    x = (ix + 0.5f0) * spacing_x - box[1]/2
    y = (iy + 0.5f0) * spacing_y - box[2]/2 
    z = (iz + 0.5f0) * spacing_z - box[3]/2 

    push!(rx_host, x)
    push!(ry_host, y)
    push!(rz_host, z)
end

# Copy to GPU arrays  
copyto!(st.rx, rx_host)
copyto!(st.ry, ry_host)
copyto!(st.rz, rz_host)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj3d.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)  # initial frame

# ---- Run ----
@time for s in 1:1000000
    Simulation.step!(st, dt)
    if s % 100000 == 0
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs3d.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        #Writers.write_xyz!(joinpath(@__DIR__, "traj2d.xyz"); rx=st.rx, ry=st.ry, rz=st.rz)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame" step=s
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD: $gsd_path")