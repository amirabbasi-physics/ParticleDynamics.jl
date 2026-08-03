# Throughput of the two MLIP drivers on the same system, same engine, same GPU.
#
# One model per process invocation (loading an Orb checkpoint sets the torch
# default dtype process-wide), each appending one line to
# validation/benzene_throughput.txt:
#
#   julia --project=examples/orb examples/orb/benzene_throughput.jl mace-off float64
#   julia --project=examples/orb examples/orb/benzene_throughput.jl mace-off float32
#   julia --project=examples/orb examples/orb/benzene_throughput.jl orb-cons float32-high
#   julia --project=examples/orb examples/orb/benzene_throughput.jl orb-direct float32-high
#   julia --project=examples/orb examples/orb/benzene_throughput.jl orb-cons float64
#
# Precision is an explicit argument because it dominates the comparison: Orb is
# a float32-native model and float64 costs ~30x on a consumer GPU (1:64 FP64
# rate), while this repo's MACE work runs float64 by default. The honest
# head-to-head is the matched-float32 row; the float64 rows show the penalty.

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
using PythonCall
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "OrbPotential.jl"))
include(joinpath(@__DIR__, "..", "mace", "MACEPotential.jl"))

const DT = 0.5 * 0.098226
const NWARM = 20
const NBENCH = 200
const MACE_OFF = expanduser("~/.cache/mace/MACE-OFF23_small.model")

key = length(ARGS) >= 1 ? ARGS[1] : "orb-cons"
prec = length(ARGS) >= 2 ? ARGS[2] : "float32-high"
system = length(ARGS) >= 3 ? ARGS[3] : "benzene_crystal.npz"

np = pyimport("numpy")
val = joinpath(@__DIR__, "validation")
d = np.load(joinpath(val, system))
pos = pyconvert(Matrix{Float64}, d["positions"])
Z = pyconvert(Vector{Int}, d["numbers"])
masses = pyconvert(Vector{Float64}, d["masses"])
L = pyconvert(Vector{Float64}, d["cell_lengths"])
N = length(Z)

st = build_simulation(; N=N, box=(L[1], L[2], L[3]), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=masses, precision=:f64, dt=DT)
copyto!(st.rx, pos[:, 1] .- L[1] / 2)
copyto!(st.ry, pos[:, 2] .- L[2] / 2)
copyto!(st.rz, pos[:, 3] .- L[3] / 2)

pot, label = if key == "mace-off"
    MACEPotential(Z, (L[1], L[2], L[3]); variant=:off, model=MACE_OFF,
                  device="cuda", dtype=prec), "MACE-OFF23-small"
elseif key == "orb-cons"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_inf_omat",
                 precision=prec), "Orb-v3-cons-inf-omat"
elseif key == "orb-direct"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_direct_inf_omat",
                 precision=prec), "Orb-v3-direct-inf-omat"
elseif key == "orb-omol"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_omol",
                 precision=prec, charge=0, spin=1), "Orb-v3-cons-omol"
else
    error("unknown model key: $key")
end
ParticleDynamics.attach_external_potential!(st, pot)

spec = SimulationCore.nve(st; dt=DT)
for _ in 1:NWARM
    SimulationCore.step!(st, spec, DT; compute_energy=false)
end
t0 = time()
for _ in 1:NBENCH
    SimulationCore.step!(st, spec, DT; compute_energy=false)
end
rate = NBENCH / (time() - t0)

line = @sprintf("%-24s %-14s N = %4d  %7.2f steps/s  %8.1f ms/step",
                label, prec, N, rate, 1000 / rate)
println(line)
open(joinpath(val, "benzene_throughput.txt"), "a") do io
    println(io, line)
end
