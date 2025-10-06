using NonEqSimGPU
using NonEqSimGPU: Simulation, Definitions, BondedForces, Writers
using CUDA

# Simple 3D polymer chain with bonded interactions (FENE or harmonic),
# mirroring the 2D benchmark but using 3D state (rx,ry,rz) and a 3D box.

function make_linear_chain_bonds(N::Int)
    bonds = Vector{Tuple{Int32,Int32}}()
    for i in 1:(N-1)
        push!(bonds, (Int32(i), Int32(i+1)))
    end
    return bonds
end

function init_chain_line_3d!(st, box::NTuple{3,Float32}; spacing::Float32=0.97f0)
    N = length(st.rx)
    x0 = -0.5f0*box[1] + 1.0f0
    y0 = 0.0f0
    z0 = 0.0f0
    rx_h = Vector{Float32}(undef, N)
    ry_h = Vector{Float32}(undef, N)
    rz_h = Vector{Float32}(undef, N)
    for i in 1:N
        rx_h[i] = x0 + (i-1)*spacing
        ry_h[i] = y0
        rz_h[i] = z0
    end
    copyto!(st.rx, rx_h)
    copyto!(st.ry, ry_h)
    copyto!(st.rz, rz_h)
    return st
end

# ---- Params ----
N   = 200
box = (200.0f0, 50.0f0, 50.0f0)

# LJ parameters (purely repulsive WCA)
r_cut = Float32(2^(1/6))
sigma = 1f0
epsilon = 1f0

cap = Int32(96)
gamma = 1f0
temperature = 1f0
dt = 0.0005f0
N_steps = 1_000_000
N_log = 10_000

# Langevin noise scale per particle: sqrt(2γT) (mass=1)
noise_scale = CUDA.fill(sqrt(2f0*gamma*temperature*dt), N)

# Bonds: linear chain
bonds = make_linear_chain_bonds(N)

# Choose one type of bonded interaction
use_fene = false
bond_harmonic = use_fene ? nothing : Definitions.HarmonicBondParams{Float32}(30f0, 1.0f0)
bond_fene     = use_fene ? Definitions.FENEParams{Float32}(300f0, 1.5f0) : nothing

# ---- Build ----
st = Simulation.build_simulation(D = 3, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=10,
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, noise_scale=noise_scale, init_temperature=temperature,
                                 bonds=bonds, bond_harmonic=bond_harmonic, bond_fene=bond_fene,
                                 nonbonded=:wca)

# ---- Initialize positions ----
init_chain_line_3d!(st, box)

gsd_path = joinpath(@__DIR__, "polymer3d.gsd")
gsdh = Writers.gsd_open(gsd_path)
types = ["C"]
Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)

@time for s in 1:N_steps
    if s % N_log == 0
        Simulation.step!(st, dt)
        Writers.write_observables_csv!(joinpath(@__DIR__, "obs3d_polymer.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        Writers.write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "polymer step (3d)" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    else
        Simulation.step!(st, dt, compute_energy=false)
    end
end

Writers.gsd_close(gsdh)
println("Done. GSD: $gsd_path")

