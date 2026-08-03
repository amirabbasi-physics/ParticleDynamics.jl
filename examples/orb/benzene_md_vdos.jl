# Benzene crystal I at 150 K: finite-temperature head-to-head between MACE-OFF
# and Orb-v3, driven through the engine's external-potential interface.
#
# One model per process invocation, because loading an Orb checkpoint calls
# torch.set_default_dtype process-wide and MACE here runs float64:
#
#   julia --project=examples/orb examples/orb/benzene_md_vdos.jl mace-off
#   julia --project=examples/orb examples/orb/benzene_md_vdos.jl orb-cons
#   julia --project=examples/orb examples/orb/benzene_md_vdos.jl orb-direct
#   julia --project=examples/orb examples/orb/benzene_md_vdos.jl orb-omol
#
# Protocol: BAOAB Langevin equilibration at 150 K (the temperature of the
# reference crystal structure), then NVE production during which velocities are
# recorded every step. Two things come out of the production run:
#
#   * the vibrational density of states, from the mass-weighted velocity
#     autocorrelation function, compared with the experimental Raman
#     frequencies of benzene (992, 1586, 3062 cm^-1);
#   * the NVE total-energy drift, which separates Orb's `conservative` models
#     (forces = -grad E, energy conserved) from its `direct` models (forces
#     predicted independently, so no conserved energy exists).
#
# Bond-length statistics are accumulated too: the X-ray reference structure has
# C-H = 0.93 Å from foreshortening, and a good potential should relax it toward
# the neutron value of ~1.08 Å.

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using Random
using ParticleDynamics
using ParticleDynamics: SimulationCore
using PythonCall
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "OrbPotential.jl"))
include(joinpath(@__DIR__, "..", "mace", "MACEPotential.jl"))

const kB = 8.617333262e-5          # eV/K
const T_K = 150.0                  # matches the reference crystal structure
const kBT = kB * T_K
const DT = 0.5 * 0.098226          # 0.5 fs in engine time units
const N_RELAX = 1_000              # damped start from the X-ray geometry
const N_EQ = 4_000                 # 2 ps Langevin equilibration
const N_PROD = 10_000              # 5 ps NVE production (6.7 cm^-1 resolution)
const ENERGY_EVERY = 10
const MACE_OFF = expanduser("~/.cache/mace/MACE-OFF23_small.model")

key = length(ARGS) >= 1 ? ARGS[1] : "orb-cons"

np = pyimport("numpy")
val = joinpath(@__DIR__, "validation")
d = np.load(joinpath(val, "benzene_cell.npz"))
pos = pyconvert(Matrix{Float64}, d["positions"])
Z = pyconvert(Vector{Int}, d["numbers"])
masses = pyconvert(Vector{Float64}, d["masses"])
L = pyconvert(Vector{Float64}, d["cell_lengths"])
N = length(Z)
@printf("benzene crystal: N = %d atoms, box = %.4f x %.4f x %.4f Å, T = %.0f K\n",
        N, L[1], L[2], L[3], T_K)

st = build_simulation(; N=N, box=(L[1], L[2], L[3]), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=2.0, temperature=kBT,
                      mass=masses, precision=:f64, dt=DT)
copyto!(st.rx, pos[:, 1] .- L[1] / 2)
copyto!(st.ry, pos[:, 2] .- L[2] / 2)
copyto!(st.rz, pos[:, 3] .- L[3] / 2)

pot, label = if key == "mace-off"
    isfile(MACE_OFF) || error("MACE-OFF checkpoint missing at $MACE_OFF")
    MACEPotential(Z, (L[1], L[2], L[3]); variant=:off, model=MACE_OFF,
                  device="cuda"), "MACE-OFF23-small"
elseif key == "orb-cons"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_inf_omat",
                 precision="float32-high"), "Orb-v3-cons-inf-omat"
elseif key == "orb-direct"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_direct_inf_omat",
                 precision="float32-high"), "Orb-v3-direct-inf-omat"
elseif key == "orb-omol"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_omol",
                 precision="float32-high", charge=0, spin=1), "Orb-v3-cons-omol"
else
    error("unknown model key: $key")
end
ParticleDynamics.attach_external_potential!(st, pot)
println("model: ", label)

temperature_K() = begin
    SimulationCore._refresh_kinetic_buffer!(st)
    2 * sum(Array(st.Ekin)) / (3 * N) / kB
end

# --- minimum-image bond statistics ---
const CIDX = findall(==(6), Z)
const HIDX = findall(==(1), Z)
mic(dx, Lk) = dx - Lk * round(dx / Lk)
function bond_means(rx, ry, rz)
    dist(i, j) = sqrt(mic(rx[i] - rx[j], L[1])^2 +
                      mic(ry[i] - ry[j], L[2])^2 +
                      mic(rz[i] - rz[j], L[3])^2)
    cc = 0.0; ncc = 0
    for a in eachindex(CIDX), b in eachindex(CIDX)
        a < b || continue
        r = dist(CIDX[a], CIDX[b])
        if r < 1.6
            cc += r; ncc += 1
        end
    end
    ch = 0.0; nch = 0
    for a in CIDX, b in HIDX
        r = dist(a, b)
        if r < 1.35
            ch += r; nch += 1
        end
    end
    return cc / max(ncc, 1), ncc, ch / max(nch, 1), nch
end

# --- 1. damped start from the X-ray geometry (C-H is 0.93 Å there) ---
spec_relax = SimulationCore.baoab(st; gamma=15.0, temperature=0.0, dt=DT / 2)
for _ in 1:N_RELAX
    SimulationCore.step!(st, spec_relax, DT / 2; compute_energy=false)
end
cc0, ncc0, ch0, nch0 = bond_means(Array(st.rx), Array(st.ry), Array(st.rz))
@printf("relaxed (0 K): C-C = %.4f Å (%d bonds), C-H = %.4f Å (%d bonds)\n",
        cc0, ncc0, ch0, nch0)

# --- 2. seed velocities and equilibrate at 150 K ---
rng = MersenneTwister(20260729)
vx = randn(rng, N) .* sqrt.(kBT ./ masses)
vy = randn(rng, N) .* sqrt.(kBT ./ masses)
vz = randn(rng, N) .* sqrt.(kBT ./ masses)
M = sum(masses)
vx .-= sum(vx .* masses) / M
vy .-= sum(vy .* masses) / M
vz .-= sum(vz .* masses) / M
copyto!(st.vx, vx); copyto!(st.vy, vy); copyto!(st.vz, vz)

spec_eq = SimulationCore.baoab(st; gamma=2.0, temperature=kBT, dt=DT)
t0 = time()
for i in 1:N_EQ
    SimulationCore.step!(st, spec_eq, DT; compute_energy=false)
    if i % 1000 == 0
        @printf("equil %5d/%d  T = %6.1f K  (%.2f steps/s)\n",
                i, N_EQ, temperature_K(), i / (time() - t0))
        flush(stdout)
    end
end

# --- 3. NVE production: record velocities every step for the VDOS ---
spec = SimulationCore.nve(st; dt=DT)
V = Array{Float64}(undef, N_PROD, N, 3)
etimes = Float64[]; etot = Float64[]; epot = Float64[]; temps = Float64[]
ccs = Float64[]; chs = Float64[]
t0 = time()
for i in 1:N_PROD
    want_E = (i % ENERGY_EVERY == 1)
    SimulationCore.step!(st, spec, DT; compute_energy=want_E)
    V[i, :, 1] = Array(st.vx)
    V[i, :, 2] = Array(st.vy)
    V[i, :, 3] = Array(st.vz)
    if want_E
        SimulationCore._refresh_kinetic_buffer!(st)
        ep = sum(Array(st.Epot))
        ek = sum(Array(st.Ekin))
        push!(etimes, (i - 1) * DT / 0.098226 / 1000)   # ps
        push!(epot, ep); push!(etot, ep + ek)
        push!(temps, 2 * ek / (3 * N) / kB)
    end
    if i % 200 == 0
        cc, _, ch, _ = bond_means(Array(st.rx), Array(st.ry), Array(st.rz))
        push!(ccs, cc); push!(chs, ch)
    end
    if i % 2000 == 0
        @printf("prod  %5d/%d  T = %6.1f K  E = %.6f eV  (%.2f steps/s)\n",
                i, N_PROD, temps[end], etot[end], i / (time() - t0))
        flush(stdout)
    end
end
rate = N_PROD / (time() - t0)

# --- 4. NVE drift and structure summary ---
E0 = etot[1]
drift_rel = maximum(abs.(etot .- E0)) / abs(E0)
drift_per_step_atom = (etot[end] - etot[1]) / N_PROD / N
mean(x) = sum(x) / length(x)
std(x) = sqrt(sum(abs2, x .- mean(x)) / max(length(x) - 1, 1))

println()
@printf("throughput (NVE, %d atoms) : %.2f steps/s\n", N, rate)
@printf("mean T                     : %.1f ± %.1f K\n", mean(temps), std(temps))
@printf("NVE max |dE|/|E|           : %.3e\n", drift_rel)
@printf("NVE drift                  : %.3e eV/step/atom\n", drift_per_step_atom)
@printf("C-C at 150 K               : %.4f ± %.4f Å  (exp. 1.379 Å, X-ray 150 K)\n",
        mean(ccs), std(ccs))
@printf("C-H at 150 K               : %.4f ± %.4f Å  (X-ray 0.93 Å, neutron ~1.08 Å)\n",
        mean(chs), std(chs))

npz = joinpath(val, "benzene_md_$(key).npz")
np.savez(npz,
         velocities=np.asarray(V),
         masses=np.asarray(masses),
         numbers=np.asarray(collect(Int, Z)),
         dt_fs=np.asarray(0.5),
         etimes=np.asarray(etimes), etot=np.asarray(etot),
         epot=np.asarray(epot), temps=np.asarray(temps),
         cc=np.asarray(ccs), ch=np.asarray(chs),
         label=label, rate=np.asarray(rate),
         drift_rel=np.asarray(drift_rel),
         drift_per_step_atom=np.asarray(drift_per_step_atom))
println("wrote ", npz)

open(joinpath(val, "benzene_md_summary.txt"), "a") do io
    @printf(io, "%-24s  %6.2f steps/s  T = %5.1f ± %4.1f K  maxdE/E = %.3e  drift = %+.3e eV/step/atom  C-C = %.4f  C-H = %.4f\n",
            label, rate, mean(temps), std(temps), drift_rel,
            drift_per_step_atom, mean(ccs), mean(chs))
end
