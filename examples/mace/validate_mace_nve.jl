# V2: NVE energy conservation with MACE-MP-0 on Si216.
# 10,000 steps at dt = 1 fs (0.098226 t*), Float64, energy sampled every 10
# steps. Writes validation/v2_energy.csv and prints drift metrics.
#
# Run: julia --project=examples/mace examples/mace/validate_mace_nve.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Random
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "MACEPotential.jl"))

const kB = 8.617333e-5      # eV/K
const DT_FS = 0.098226      # 1 fs in engine time units (Å·√(amu/eV))
const TEMP = 300.0
const MASS_SI = 28.0855
const NSTEPS = 10_000
const SAMPLE_EVERY = 10

np = pyimport("numpy")
ref = np.load(joinpath(@__DIR__, "validation", "reference.npz"))
pos = pyconvert(Matrix{Float64}, ref["positions"])
cellm = pyconvert(Matrix{Float64}, ref["cell"])
Z = pyconvert(Vector{Int}, ref["numbers"])
N = length(Z)
L = cellm[1, 1]

st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=MASS_SI, precision=:f64, dt=DT_FS)
copyto!(st.rx, pos[:, 1] .- L / 2)
copyto!(st.ry, pos[:, 2] .- L / 2)
copyto!(st.rz, pos[:, 3] .- L / 2)

# Maxwell-Boltzmann velocities at 300 K, COM removed (units consistent: Å/t*)
rng = MersenneTwister(2026)
σv = sqrt(kB * TEMP / MASS_SI)
vx = σv .* randn(rng, N); vx .-= sum(vx) / N
vy = σv .* randn(rng, N); vy .-= sum(vy) / N
vz = σv .* randn(rng, N); vz .-= sum(vz) / N
copyto!(st.vx, vx); copyto!(st.vy, vy); copyto!(st.vz, vz)

pot = MACEPotential(Z, (L, L, L); variant=:mp, model="small", device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)
spec = SimulationCore.nve(st; dt=DT_FS)

steps = Int[]
energies = Float64[]
t_start = time()
for i in 1:NSTEPS
    sample = (i % SAMPLE_EVERY == 0) || i == 1
    SimulationCore.step!(st, spec, DT_FS; compute_energy=sample)
    if sample
        SimulationCore._refresh_kinetic_buffer!(st)
        E = sum(Array(st.Epot)) + sum(Array(st.Ekin))
        push!(steps, i); push!(energies, E)
    end
    if i % 1000 == 0
        el = time() - t_start
        @printf("step %6d  E=%.8f eV  (%.1f steps/s, %.1f min elapsed)\n",
                i, energies[end], i / el, el / 60)
        flush(stdout)
    end
end

E0 = energies[1]
drift_rel = abs(energies[end] - E0) / abs(E0)
drift_max = maximum(abs.(energies .- E0)) / abs(E0)
drift_per_step_atom = (energies[end] - E0) / NSTEPS / N

open(joinpath(@__DIR__, "validation", "v2_energy.csv"), "w") do io
    println(io, "step,total_energy_eV")
    for (s, e) in zip(steps, energies)
        println(io, s, ",", e)
    end
end

@printf("\nV2 RESULTS (Si216, MACE-MP-0 small, f64, dt=1fs, %d steps)\n", NSTEPS)
@printf("E0 = %.8f eV\n", E0)
@printf("relative drift (end)  = %.3e   (criterion < 1e-5)\n", drift_rel)
@printf("relative |dev| (max)  = %.3e\n", drift_max)
@printf("drift per step per atom = %.3e eV\n", drift_per_step_atom)
println(drift_max < 1e-5 ? "V2: PASS" : "V2: FAIL")
