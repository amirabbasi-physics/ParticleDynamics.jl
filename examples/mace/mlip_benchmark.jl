# MLIP driver benchmark: ParticleDynamics.jl vs ASE driving the SAME MACE
# model on the SAME system. With an MLIP, throughput is dominated by model
# inference; this measures how much driver overhead each MD loop adds on top.
#
# Systems: Si216 (MACE-MP-0 small) and 64 H2O (MACE-OFF small), float64, CUDA.
# 300 timed steps each after 20 warm-up steps.
#
# Run: julia --project=examples/mace examples/mace/mlip_benchmark.jl

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
const NWARM = 20
const NBENCH = 300

np = pyimport("numpy")
ase = pyimport("ase")
ase_md = pyimport("ase.md.verlet")
ase_units = pyimport("ase.units")

function bench_engine(Z, pos, L, masses, dt, variant, model)
    N = length(Z)
    st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                          cap=Int32(8), neigh_interval=1,
                          use_neighborlist=false, spatial_reorder=false,
                          gamma=0.0, temperature=0.0,
                          mass=masses, precision=:f64, dt=dt)
    copyto!(st.rx, pos[:, 1] .- L / 2)
    copyto!(st.ry, pos[:, 2] .- L / 2)
    copyto!(st.rz, pos[:, 3] .- L / 2)
    pot = MACEPotential(Z, (L, L, L); variant=variant, model=model, device="cuda")
    ParticleDynamics.attach_external_potential!(st, pot)
    spec = SimulationCore.nve(st; dt=dt)
    for _ in 1:NWARM
        SimulationCore.step!(st, spec, dt; compute_energy=false)
    end
    t0 = time()
    for _ in 1:NBENCH
        SimulationCore.step!(st, spec, dt; compute_energy=false)
    end
    return NBENCH / (time() - t0)
end

function bench_ase(Z, pos, L, dt_fs, variant, model)
    atoms = ase.Atoms(; numbers=np.asarray(collect(Int, Z)),
                      positions=np.asarray(pos),
                      cell=np.asarray(L * [1.0 0 0; 0 1 0; 0 0 1]), pbc=true)
    mc = pyimport("mace.calculators")
    atoms.calc = variant === :mp ?
        mc.mace_mp(; model=model, device="cuda", default_dtype="float64") :
        mc.mace_off(; model=model, device="cuda", default_dtype="float64")
    dyn = ase_md.VelocityVerlet(atoms, pyconvert(Float64, ase_units.fs) * dt_fs)
    dyn.run(NWARM)
    t0 = time()
    dyn.run(NBENCH)
    return NBENCH / (time() - t0)
end

results = String[]

# --- Si216 / MACE-MP-0 ---
ref = np.load(joinpath(@__DIR__, "validation", "reference.npz"))
pos = pyconvert(Matrix{Float64}, ref["positions"])
Z = pyconvert(Vector{Int}, ref["numbers"])
L = pyconvert(Matrix{Float64}, ref["cell"])[1, 1]
e = bench_engine(Z, pos, L, 28.0855, DT_FS, :mp, "small")
a = bench_ase(Z, pos, L, 1.0, :mp, "small")
push!(results, @sprintf("Si216  MACE-MP-0  : engine %6.2f steps/s (%.1f ms)  |  ASE %6.2f steps/s (%.1f ms)  |  ratio %.2fx",
                        e, 1000 / e, a, 1000 / a, e / a))

# --- 64 H2O / MACE-OFF ---
ini = np.load(joinpath(@__DIR__, "validation", "water_init.npz"))
posw = pyconvert(Matrix{Float64}, ini["positions"])
Zw = pyconvert(Vector{Int}, ini["numbers"])
massesw = pyconvert(Vector{Float64}, ini["masses"])
Lw = pyconvert(Float64, ini["L"].item())
off = expanduser("~/.cache/mace/MACE-OFF23_small.model")
e = bench_engine(Zw, posw, Lw, massesw, 0.5 * DT_FS, :off, off)
a = bench_ase(Zw, posw, Lw, 0.5, :off, off)
push!(results, @sprintf("H2O192 MACE-OFF   : engine %6.2f steps/s (%.1f ms)  |  ASE %6.2f steps/s (%.1f ms)  |  ratio %.2fx",
                        e, 1000 / e, a, 1000 / a, e / a))

println("\nMLIP DRIVER BENCHMARK (RTX 3090, float64, $(NBENCH) timed steps)")
foreach(println, results)
open(joinpath(@__DIR__, "validation", "mlip_benchmark.txt"), "w") do io
    foreach(l -> println(io, l), results)
end
