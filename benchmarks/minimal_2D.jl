using NonEqSimGPU
using CUDA

function initialize_square_lattice!(st, box::NTuple{2,T}) where {T<:AbstractFloat}
    N = length(st.rx)
    n_side = ceil(Int, sqrt(N))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side

    rx_host = Vector{T}(undef, N)
    ry_host = Vector{T}(undef, N)

    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = linear ÷ n_side

        rx_host[i] = (ix + T(0.5)) * spacing_x - box[1] / 2
        ry_host[i] = (iy + T(0.5)) * spacing_y - box[2] / 2
    end

    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    return st
end


# ---- Params ----
N   = 40_000
box = (250.0, 250.0)  # box dimensions (Lx, Ly)

# LJ parameters
r_cut = 2^(1/6)  # LJ cutoff
sigma = 1
epsilon = 10

cap = Int32(100)
# Note: neigh_interval not used with new displacement-based neighbor list algorithm
gamma = 615.0
temperature = 10

N_steps = 10_000_000
N_log = 1_000_000
dt = 2.0e-6

# ---- Build ----
st = build_simulation(D = 2, N=N, box=box, dt = dt, cutoff=r_cut, skin=0.4, cap=cap,
                                 neigh_interval=100,  # Not used with displacement-based algorithm
                                 epsilon=epsilon, sigma=sigma,
                                 gamma=gamma, temperature=temperature, nonbonded=:wca,precision = :f32)

# ---- Initialize positions after building simulation state ----
initialize_square_lattice!(st, box)

# ---- GSD writer ----
gsd_path = joinpath(@__DIR__, "traj2d.gsd")
gsdh = gsd_open(gsd_path)
types = ["C"]
write_gsd_frame!(gsdh, st; diameter= sigma, types_names=types, step=st.step)  # initial frame

# ---- Run ----
@time for s in 1:N_steps
    if s % N_log == 0
        step!(st, dt)
        # Only write available observables - Ekin might not be computed yet
        write_observables_csv!(joinpath(@__DIR__, "obs2d.csv"), s; Epot=st.Epot, Ekin=st.Ekin, dq=st.dq)
        #write_xyz!(joinpath(@__DIR__, "traj2d.xyz"); rx=st.rx, ry=st.ry, rz=nothing)
        write_gsd_frame!(gsdh, st; diameter=sigma, types_names=types, step=st.step)
        @info "wrote frame" step=s Epot_sum=sum(st.Epot) Ekin_sum=sum(st.Ekin)
    else
        step!(st, dt, compute_energy=false)
    end
end

gsd_close(gsdh)
println("Done. GSD: $gsd_path")
