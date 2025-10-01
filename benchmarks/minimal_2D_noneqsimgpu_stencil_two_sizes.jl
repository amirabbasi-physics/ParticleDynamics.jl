using NonEqSimGPU
using NonEqSimGPU: Simulation
using NonEqSimGPU.Filters
using NonEqSimGPU.Writers
using NonEqSimGPU.NeighborLists
using CUDA
using Statistics
using Random
using Printf

CUDA.allowscalar(false)

function initialize_square_lattice!(st, box::Tuple{Float32,Float32})
    N = length(st.rx)
    n_side = ceil(Int, sqrt(Float64(N)))
    spacing = (box[1] / n_side, box[2] / n_side)

    rx = Vector{Float32}(undef, N)
    ry = similar(rx)

    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = linear ÷ n_side

        rx[i] = (ix + 0.5f0) * spacing[1] - box[1] / 2
        ry[i] = (iy + 0.5f0) * spacing[2] - box[2] / 2
    end

    copyto!(st.rx, rx)
    copyto!(st.ry, ry)
    return st
end

function randomize_types!(st; ratio::Float64=0.5)
    N = length(st.rx)
    n_type1 = round(Int, ratio * N)
    p = randperm(N)
    host = fill(Int32(2), N)
    @inbounds host[p[1:n_type1]] .= Int32(1)
    st.typeid .= CuArray(host)
    return nothing
end

# Simulation parameters
N = 40_000
BOX = (800.0f0, 800.0f0)
SIGMA_TYPE = Dict(1 => 5.0f0, 2 => 1.0f0)  # different particle sizes
EPS   = 1000f0
SIGMA_REF = 1.0f0
RCUT_REF  = Float32(2^(1/6) * SIGMA_REF)   # used by the LJ kernel (single-σ for now)
GAMMA = 100f0
DT    = 0.000002f0
NSTEPS = 10_000_000
LOG_INTERVAL = 1_000_000

T_TYPE1 = 1.0f0
T_TYPE2 = 200.0f0
T_mean = 0.5f0 * (T_TYPE1 + T_TYPE2)

st = Simulation.build_simulation(D=2,
                                 N=N,
                                 box=(BOX[1], BOX[2]),
                                 cutoff=RCUT_REF,
                                 skin=0.4f0,
                                 cap=Int32(128),
                                 neigh_interval=10,
                                 epsilon=EPS,
                                 sigma=SIGMA_REF,
                                 gamma=GAMMA,
                                 init_temperature=T_mean)

initialize_square_lattice!(st, BOX)

# Two particle populations distinguished by type id (randomly mapped to lattice)
randomize_types!(st; ratio=0.5)

# Configure stochastic forcing per group
Filters.set_langevin_temperature!(st, DT,
    Filters.TypeIDs(1) => T_TYPE1,
    Filters.TypeIDs(2) => T_TYPE2)

# User-specified per-pair parameters (no default mixing)
SIGMA_PAIR = Float32[5.0 3.0;
                     3.0 1.0]
EPS_PAIR   = Float32[EPS EPS;
                     EPS EPS]
RCUT_FACTOR = Float32(2^(1/6))
RCUT_PAIR = RCUT_FACTOR .* SIGMA_PAIR

st.nbh = NeighborLists.build_neighbors_stencil_by_types!(st.rx, st.ry;
                                                         box=(BOX[1], BOX[2]),
                                                         typeid=st.typeid,
                                                         rcut_pair=RCUT_PAIR,
                                                         cap=Int32(128),
                                                         skin=0.4f0)

# Writers ---------------------------------------------------------------
output_dir = @__DIR__
gsd_path = joinpath(output_dir, "traj2d_stencil_two_sizes.gsd")
csv_path = joinpath(output_dir, "obs2d_stencil_two_sizes.csv")

gsdh = Writers.gsd_open(gsd_path)
type_names = ["C", "H"]

# Per-particle diameters from type ids (use diagonal of SIGMA_PAIR)
diam_host = Vector{Float32}(undef, N)
tid_host = Array(st.typeid)
@inbounds for i in 1:N
    t = Int(tid_host[i])
    diam_host[i] = SIGMA_PAIR[t,t]
end

# Enable pairwise σ, ε, and rcut
st.sigma_pair   = CuArray(SIGMA_PAIR)
st.epsilon_pair = CuArray(EPS_PAIR)
st.rcut_pair    = CuArray(RCUT_PAIR)

Writers.write_gsd_frame!(gsdh, st; diameter=diam_host, types_names=type_names, step=st.step)

open(csv_path, "w") do io
    println(io, "step,neigh_avg_type1,neigh_avg_type2,elapsed_s,steps_per_sec")
end

records = Int[]
neigh_avg_t1 = Float32[]
neigh_avg_t2 = Float32[]

function mean_neighbors(nbh, sel::Vector{Int})
    # counts is CuArray; copy minimal data for selection
    c_host = Array(nbh.counts)
    return Base.sum(c_host[sel]) / length(sel)
end

start_time = time()
max_runtime = let v = get(ENV, "NEQSIM_MAX_SECONDS", "")
    isempty(v) ? Inf : parse(Float64, v)
end

for step in 1:NSTEPS
    Simulation.step!(st, DT, compute_energy=false)
    if step % LOG_INTERVAL == 0
        push!(records, step)
        idx_t1 = Filters.resolve(st, Filters.TypeIDs(1))
        idx_t2 = Filters.resolve(st, Filters.TypeIDs(2))
        n1 = mean_neighbors(st.nbh, idx_t1)
        n2 = mean_neighbors(st.nbh, idx_t2)
        push!(neigh_avg_t1, n1)
        push!(neigh_avg_t2, n2)
        elapsed = time() - start_time
        sps = step / max(elapsed, 1e-6)
        open(csv_path, "a") do io
            @printf(io, "%d,%.3f,%.3f,%.3f,%.3f\n", step, n1, n2, elapsed, sps)
        end
        # refresh diameters (in case types changed) without soft-scope warning
        let _tid = Array(st.typeid)
            @inbounds for i in 1:N
                t = Int(_tid[i])
                diam_host[i] = SIGMA_PAIR[t,t]
            end
        end
        Writers.write_gsd_frame!(gsdh, st; diameter=diam_host, types_names=type_names, step=st.step)
        @info "progress" step=step neigh_t1=n1 neigh_t2=n2 steps_per_sec=sps
        if (time() - start_time) >= max_runtime
            @info "Reached max runtime limit" limit_s=max_runtime step=step
            break
        end
    end
end

Writers.gsd_close(gsdh)
println("Wrote trajectory to $(gsd_path)")
println("Wrote observables to $(csv_path)")
