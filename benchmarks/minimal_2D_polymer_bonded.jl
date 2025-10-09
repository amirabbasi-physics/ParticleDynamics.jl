using NonEqSimGPU
using CUDA

# Simple 2D polymer chain with bonded interactions (FENE or harmonic)

function make_linear_chain_bonds(N::Int)
    bonds = Vector{Tuple{Int32,Int32}}()
    for i in 1:(N-1)
        push!(bonds, (Int32(i), Int32(i+1)))
    end
    return bonds
end

function init_chain_line!(st, box::NTuple{2,Float32}; spacing::Float32=0.97f0)
    N = length(st.rx)
    x0 = -0.5f0*box[1] + 1.0f0
    y0 = 0.0f0
    rx_h = Vector{Float32}(undef, N)
    ry_h = Vector{Float32}(undef, N)
    for i in 1:N
        rx_h[i] = x0 + (i-1)*spacing
        ry_h[i] = y0
    end
    copyto!(st.rx, rx_h)
    copyto!(st.ry, ry_h)
    return st
end

# ---- Params ----
N   = 200
box = (200.0f0, 50.0f0)

# LJ parameters (purely repulsive WCA)
r_cut = Float32(2^(1/6))
sigma = 1f0
epsilon = 1f0

cap = Int32(96)
gamma = 1f0
temperature = 1f0
dt = 0.0005f0
N_steps = 1000000
N_log = 10000

# Bonds: linear chain
bonds = make_linear_chain_bonds(N)

# Choose one type of bonded interaction
use_fene = true
bonding = use_fene ? fene_bond(300f0, 1.5f0) : harmonic_bond(30f0, 1.0f0)

# ---- Build ----
st = build_simulation(D = 2, N=N, box=box, cutoff=r_cut, skin=0.4f0, cap=cap,
                                 neigh_interval=1,
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, temperature=temperature, dt=dt,
                                 bonds=bonds, bonding=bonding,
                                 nonbonded=:wca)

# ---- Initialize positions ----
init_chain_line!(st, box)

gsd_path = joinpath(@__DIR__, "polymer2d.gsd")
gsdh = gsd_open(gsd_path)
types = ["C"]
write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)

@time for s in 1:N_steps
    if s % N_log == 0
        step!(st, dt)
        write_observables_csv!(joinpath(@__DIR__, "obs2d_polymer.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "polymer step" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    else
        step!(st, dt, compute_energy=false)
    end
end

gsd_close(gsdh)
println("Done. GSD: $gsd_path")
