# V3: trajectory equivalence vs ASE velocity Verlet, 200 steps, identical
# initial conditions (from validation/md_reference.npz). Compares per-atom
# positions with minimum-image mapping between the ASE frame ([0,L)-ish,
# unwrapped) and the engine frame ([-L/2, L/2), wrapped).
#
# Run: julia --project=examples/mace examples/mace/v3_trajectory_check.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "MACEPotential.jl"))

const DT_FS = 0.098226
const MASS_SI = 28.0855

np = pyimport("numpy")
ref = np.load(joinpath(@__DIR__, "validation", "md_reference.npz"))
pos0 = pyconvert(Matrix{Float64}, ref["positions0"])
v0 = pyconvert(Matrix{Float64}, ref["velocities0"])
cellm = pyconvert(Matrix{Float64}, ref["cell"])
Z = pyconvert(Vector{Int}, ref["numbers"])
traj = pyconvert(Array{Float64,3}, ref["trajectory"])  # (nsteps, N, 3)
N = length(Z)
L = cellm[1, 1]
nsteps = size(traj, 1)

st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=MASS_SI, precision=:f64, dt=DT_FS)
copyto!(st.rx, pos0[:, 1] .- L / 2)
copyto!(st.ry, pos0[:, 2] .- L / 2)
copyto!(st.rz, pos0[:, 3] .- L / 2)
copyto!(st.vx, v0[:, 1]); copyto!(st.vy, v0[:, 2]); copyto!(st.vz, v0[:, 3])

pot = MACEPotential(Z, (L, L, L); variant=:mp, model="small", device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)
spec = SimulationCore.nve(st; dt=DT_FS)

# minimum-image deviation between engine frame (shifted by -L/2) and ASE frame
mindev(a, b) = abs(mod(a - b + L / 2, L) - L / 2)

# ASE recorded traj[1] (Julia indexing) at t=0, so the state after engine
# step i corresponds to traj[i+1]; the last comparable step is nsteps-1.
devs = Float64[]
for i in 1:(nsteps - 1)
    SimulationCore.step!(st, spec, DT_FS; compute_energy=false)
    rx = Array(st.rx) .+ L / 2
    ry = Array(st.ry) .+ L / 2
    rz = Array(st.rz) .+ L / 2
    d = 0.0
    @inbounds for j in 1:N
        d = max(d, mindev(rx[j], traj[i + 1, j, 1]),
                   mindev(ry[j], traj[i + 1, j, 2]),
                   mindev(rz[j], traj[i + 1, j, 3]))
    end
    push!(devs, d)
    if i % 50 == 0
        @printf("step %4d  max dev = %.3e Å\n", i, d)
    end
end

open(joinpath(@__DIR__, "validation", "v3_deviation.csv"), "w") do io
    println(io, "step,max_deviation_A")
    for (i, d) in enumerate(devs)
        println(io, i, ",", d)
    end
end

@printf("\nV3 RESULTS: max deviation over %d steps = %.3e Å (report; expect ≪ bond length 2.35 Å)\n",
        nsteps - 1, maximum(devs))
println(maximum(devs) < 1e-3 ? "V3: PASS" : "V3: FAIL (check integrator conventions)")
