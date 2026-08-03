# Heating ramp of benzene crystal I, written as a GSD trajectory plus a live
# diagnostics record, for the side-by-side MACE-OFF vs Orb movie.
#
#   julia --project=examples/orb examples/orb/benzene_melt_movie_gsd.jl mace-off
#   julia --project=examples/orb examples/orb/benzene_melt_movie_gsd.jl orb-cons
#
# Protocol (identical for every model, matched float32 precision so the live
# throughput readout in the movie is a fair comparison):
#
#   1. damped relaxation from the X-ray geometry,
#   2. CSVR equilibration at 150 K,
#   3. CSVR heating ramp 150 K -> 450 K, target updated every step.
#
# The cell is fixed: with no virial from an external potential there is no
# barostat, so this is constant-volume heating. Constant-volume runs superheat
# relative to the ambient-pressure melting point (278.7 K for benzene), so the
# disordering temperature seen here is an upper bound, not a predicted melting
# point. What it does compare, model against model, is how a foundation
# potential holds a dispersion-bound molecular crystal together as it is heated.
#
# Order parameter: mean-squared displacement of all atoms, accumulated with
# minimum-image differences between frames (no dependence on engine image
# flags). It plateaus while the crystal is solid and grows without bound once
# molecules start to diffuse.

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using Random
using ParticleDynamics
using ParticleDynamics: SimulationCore
using GSDFiles
using PythonCall
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "OrbPotential.jl"))
include(joinpath(@__DIR__, "..", "mace", "MACEPotential.jl"))

const kB = 8.617333262e-5
const DT = 0.5 * 0.098226         # 0.5 fs
const T_START = 150.0
const T_END = 800.0               # stress test: at fixed volume a crystal at its
                                  # own equilibrium density superheats strongly,
                                  # so a ramp that stops near the ambient-pressure
                                  # melting point would show no disordering at all
const N_RELAX = 600
const N_EQ = 2_000
const N_RAMP = 12_000             # 6 ps
const FRAME_EVERY = 30            # 400 frames
const RAMP_UPDATE = 20            # thermostat retarget interval
const TAU = 20 * DT               # CSVR coupling time
const MACE_OFF = expanduser("~/.cache/mace/MACE-OFF23_small.model")

key = length(ARGS) >= 1 ? ARGS[1] : "orb-cons"

np = pyimport("numpy")
val = joinpath(@__DIR__, "validation")
d = np.load(joinpath(val, "benzene_crystal.npz"))
pos = pyconvert(Matrix{Float64}, d["positions"])
Z = pyconvert(Vector{Int}, d["numbers"])
masses = pyconvert(Vector{Float64}, d["masses"])
L = pyconvert(Vector{Float64}, d["cell_lengths"])
N = length(Z)
@printf("benzene supercell: N = %d atoms, box = %.3f x %.3f x %.3f Å\n",
        N, L[1], L[2], L[3])

st = build_simulation(; N=N, box=(L[1], L[2], L[3]), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=kB * T_START,
                      mass=masses, precision=:f64, dt=DT)
copyto!(st.rx, pos[:, 1] .- L[1] / 2)
copyto!(st.ry, pos[:, 2] .- L[2] / 2)
copyto!(st.rz, pos[:, 3] .- L[3] / 2)
copyto!(st.typeid, Int32[z == 6 ? 1 : 2 for z in Z])   # 1 = C, 2 = H

pot, label = if key == "mace-off"
    MACEPotential(Z, (L[1], L[2], L[3]); variant=:off, model=MACE_OFF,
                  device="cuda", dtype="float32"), "MACE-OFF23-small"
elseif key == "orb-cons"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_inf_omat",
                 precision="float32-high"), "Orb-v3-cons-inf-omat"
elseif key == "orb-omol"
    OrbPotential(Z, (L[1], L[2], L[3]); model="orb_v3_conservative_omol",
                 precision="float32-high", charge=0, spin=1), "Orb-v3-cons-omol"
else
    error("unknown model key: $key")
end
ParticleDynamics.attach_external_potential!(st, pot)
println("model: ", label, "  (float32, matched precision)")

temperature_K() = begin
    SimulationCore._refresh_kinetic_buffer!(st)
    2 * sum(Array(st.Ekin)) / (3 * N) / kB
end

# --- 1. damped relaxation from the X-ray geometry ---
spec_relax = SimulationCore.baoab(st; gamma=15.0, temperature=0.0, dt=DT / 2)
for _ in 1:N_RELAX
    SimulationCore.step!(st, spec_relax, DT / 2; compute_energy=false)
end

# --- 2. equilibrate at 150 K ---
rng = MersenneTwister(20260729)
kBT0 = kB * T_START
vx = randn(rng, N) .* sqrt.(kBT0 ./ masses)
vy = randn(rng, N) .* sqrt.(kBT0 ./ masses)
vz = randn(rng, N) .* sqrt.(kBT0 ./ masses)
M = sum(masses)
vx .-= sum(vx .* masses) / M
vy .-= sum(vy .* masses) / M
vz .-= sum(vz .* masses) / M
copyto!(st.vx, vx); copyto!(st.vy, vy); copyto!(st.vz, vz)

# One thermostat spec for the whole run, retargeted by swapping its params so
# the spec's workspace and bath state carry over (same approach as
# examples/mace/asi_melt_quench.jl). Rebuilding the spec every step would
# discard the CSVR bath state and allocate needlessly.
make_tstat(T_K) = SimulationCore.csvr(st; temperature=kB * T_K, tau=TAU)
tstat = make_tstat(T_START)
set_tstat_target!(T_K) = (tstat.params = make_tstat(T_K).params)

for i in 1:N_EQ
    SimulationCore.step!(st, tstat, DT; compute_energy=false)
    if i % 1000 == 0
        @printf("equil %5d/%d  T = %6.1f K\n", i, N_EQ, temperature_K())
        flush(stdout)
    end
end

# --- 3. heating ramp, GSD frames + diagnostics ---
out = joinpath(val, "benzene_melt_$(key).gsd")
h = gsd_open(out)
write_gsd_frame!(h, st; types_names=["C", "H"], step=0)

mic(dx, Lk) = dx - Lk * round(dx / Lk)
prevx = Array(st.rx); prevy = Array(st.ry); prevz = Array(st.rz)
cumx = zeros(N); cumy = zeros(N); cumz = zeros(N)

times = Float64[]; temps = Float64[]; targets = Float64[]
msds = Float64[]; rates = Float64[]; epots = Float64[]
nframes = 1
t0 = time()
tlast = t0
for i in 1:N_RAMP
    T_target = T_START + (T_END - T_START) * (i - 1) / (N_RAMP - 1)
    i % RAMP_UPDATE == 1 && set_tstat_target!(T_target)
    want_E = (i % FRAME_EVERY == 0)
    SimulationCore.step!(st, tstat, DT; compute_energy=want_E)

    cx = Array(st.rx); cy = Array(st.ry); cz = Array(st.rz)
    @inbounds for a in 1:N
        cumx[a] += mic(cx[a] - prevx[a], L[1])
        cumy[a] += mic(cy[a] - prevy[a], L[2])
        cumz[a] += mic(cz[a] - prevz[a], L[3])
    end
    # `global` is required: these are top-level bindings reassigned inside a
    # top-level loop, which soft scope would otherwise shadow with locals.
    global prevx, prevy, prevz = cx, cy, cz

    if i % FRAME_EVERY == 0
        write_gsd_frame!(h, st; types_names=["C", "H"], step=i)
        global nframes += 1
        msd = sum(cumx .^ 2 .+ cumy .^ 2 .+ cumz .^ 2) / N
        now = time()
        push!(times, i * DT / 0.098226 / 1000)      # ps
        push!(temps, temperature_K())
        push!(targets, T_target)
        push!(msds, msd)
        push!(rates, FRAME_EVERY / (now - tlast))
        push!(epots, sum(Array(st.Epot)))
        global tlast = now
    end
    if i % 2000 == 0
        @printf("ramp %5d/%d  T = %6.1f K (target %6.1f)  MSD = %7.3f Å²  (%.2f steps/s)\n",
                i, N_RAMP, temps[end], targets[end], msds[end],
                i / (time() - t0))
        flush(stdout)
    end
end
gsd_close(h)
rate = N_RAMP / (time() - t0)

np.savez(joinpath(val, "benzene_melt_$(key)_diag.npz"),
         times=np.asarray(times), temps=np.asarray(temps),
         targets=np.asarray(targets), msds=np.asarray(msds),
         rates=np.asarray(rates), epots=np.asarray(epots),
         label=label, rate=np.asarray(rate),
         numbers=np.asarray(collect(Int, Z)),
         cell_lengths=np.asarray(L))
@printf("wrote %s: %d frames, %.2f steps/s overall\n", out, nframes, rate)
