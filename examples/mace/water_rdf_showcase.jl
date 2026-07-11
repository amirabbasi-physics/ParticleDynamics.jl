# Showcase: liquid water with MACE-OFF (small) in ParticleDynamics.jl.
#
# 64 H2O (192 atoms) at 0.997 g/cm^3 — a molecular liquid this engine cannot
# simulate classically (it has no angle/dihedral/electrostatics terms); the
# foundation MLIP carries all intramolecular chemistry. Protocol:
#   1. BAOAB Langevin equilibration, 300 K, 2 ps (4,000 steps, dt = 0.5 fs)
#   2. NVE production, 10 ps (20,000 steps), O positions sampled every 20
#      steps -> O-O radial distribution function
# Output: validation/water_rdf.csv, validation/water_final_state.npz,
# progress + summary on stdout.
#
# Run: julia --project=examples/mace examples/mace/water_rdf_showcase.jl

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

const kB = 8.617333e-5            # eV/K
const TEMP_K = 300.0
const kBT = kB * TEMP_K           # eV
const DT = 0.5 * 0.098226         # 0.5 fs in engine time units
const N_EQ = 4_000
const N_PROD = 20_000
const SAMPLE_EVERY = 20
const ENERGY_EVERY = 500

np = pyimport("numpy")
ini = np.load(joinpath(@__DIR__, "validation", "water_init.npz"))
pos = pyconvert(Matrix{Float64}, ini["positions"])
Z = pyconvert(Vector{Int}, ini["numbers"])
masses = pyconvert(Vector{Float64}, ini["masses"])
L = pyconvert(Float64, ini["L"].item())
N = length(Z)
o_idx = findall(==(8), Z)
NO = length(o_idx)
println("water box: N=$N atoms, $NO molecules, L=$L Å")

st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=2.0, temperature=kBT,
                      mass=masses, precision=:f64, dt=DT)
copyto!(st.rx, pos[:, 1] .- L / 2)
copyto!(st.ry, pos[:, 2] .- L / 2)
copyto!(st.rz, pos[:, 3] .- L / 2)

rng = MersenneTwister(2027)
vx = randn(rng, N) .* sqrt.(kBT ./ masses)
vy = randn(rng, N) .* sqrt.(kBT ./ masses)
vz = randn(rng, N) .* sqrt.(kBT ./ masses)
# remove COM momentum
M = sum(masses)
vx .-= sum(vx .* masses) / M
vy .-= sum(vy .* masses) / M
vz .-= sum(vz .* masses) / M
copyto!(st.vx, vx); copyto!(st.vy, vy); copyto!(st.vz, vz)

# local checkpoint path: the in-process auto-download can fail (TLS in
# embedded Python) — fetch it once with curl to ~/.cache/mace/ (see README)
off_model = expanduser("~/.cache/mace/MACE-OFF23_small.model")
isfile(off_model) || error("MACE-OFF checkpoint missing; curl it to $off_model first")
pot = MACEPotential(Z, (L, L, L); variant=:off, model=off_model, device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)

temperature_K(st, N) = begin
    SimulationCore._refresh_kinetic_buffer!(st)
    2 * sum(Array(st.Ekin)) / (3 * N) / kB
end

# --- 1a. gentle start: packed random orientations can have close contacts;
# damp them with strong friction and a 5x smaller timestep first ---
DT_SMALL = DT / 5
spec_gentle = SimulationCore.baoab(st; gamma=10.0, temperature=kBT, dt=DT_SMALL)
for i in 1:1000
    SimulationCore.step!(st, spec_gentle, DT_SMALL; compute_energy=false)
    if i % 500 == 0
        @printf("gentle %4d/1000  T = %.1f K\n", i, temperature_K(st, N))
        flush(stdout)
    end
end

# --- 1b. equilibration (BAOAB Langevin, 300 K) ---
spec_eq = SimulationCore.baoab(st; gamma=2.0, temperature=kBT, dt=DT)
t0 = time()
for i in 1:N_EQ
    SimulationCore.step!(st, spec_eq, DT; compute_energy=false)
    if i % 500 == 0
        @printf("equil %5d/%d  T = %.1f K  (%.1f steps/s)\n",
                i, N_EQ, temperature_K(st, N), i / (time() - t0))
        flush(stdout)
    end
end

# checkpoint the equilibrated state so a production failure never costs the
# equilibration again
np.savez(joinpath(@__DIR__, "validation", "water_equil_state.npz"),
         rx=np.asarray(Array(st.rx)), ry=np.asarray(Array(st.ry)),
         rz=np.asarray(Array(st.rz)), vx=np.asarray(Array(st.vx)),
         vy=np.asarray(Array(st.vy)), vz=np.asarray(Array(st.vz)),
         numbers=np.asarray(Z), L=L)

# --- 2. NVE production with O-O RDF sampling ---
spec = SimulationCore.nve(st; dt=DT)
nbins = 124
rmax = L / 2
dr = rmax / nbins
counts = zeros(Float64, nbins)
nframes = 0
energies = Float64[]

mimg(d, L) = d - L * round(d / L)

ox = zeros(NO); oy = zeros(NO); oz = zeros(NO)
t0 = time()
for i in 1:N_PROD
    sample_E = (i % ENERGY_EVERY == 0) || i == 1
    SimulationCore.step!(st, spec, DT; compute_energy=sample_E)
    if sample_E
        SimulationCore._refresh_kinetic_buffer!(st)
        push!(energies, sum(Array(st.Epot)) + sum(Array(st.Ekin)))
    end
    if i % SAMPLE_EVERY == 0
        hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
        @inbounds for (k, j) in enumerate(o_idx)
            ox[k] = hx[j]; oy[k] = hy[j]; oz[k] = hz[j]
        end
        @inbounds for a in 1:NO-1, b in a+1:NO
            dx = mimg(ox[a] - ox[b], L)
            dy = mimg(oy[a] - oy[b], L)
            dz = mimg(oz[a] - oz[b], L)
            r = sqrt(dx^2 + dy^2 + dz^2)
            bin = Int(fld(r, dr)) + 1
            if 1 <= bin <= nbins
                counts[bin] += 1
            end
        end
        global nframes += 1   # explicit: top-level soft scope would shadow otherwise
    end
    if i % 2000 == 0
        @printf("prod %6d/%d  T = %.1f K  E = %.4f eV  (%.1f steps/s, %.1f min)\n",
                i, N_PROD, temperature_K(st, N), energies[end],
                i / (time() - t0), (time() - t0) / 60)
        flush(stdout)
    end
end

# --- RDF normalization ---
V = L^3
gr = zeros(nbins)
rs = zeros(nbins)
for b in 1:nbins
    r1 = (b - 1) * dr; r2 = b * dr
    vshell = 4π / 3 * (r2^3 - r1^3)
    rs[b] = (r1 + r2) / 2
    gr[b] = 2 * counts[b] * V / (nframes * NO * (NO - 1) * vshell)
end

open(joinpath(@__DIR__, "validation", "water_rdf.csv"), "w") do io
    println(io, "r_A,g_OO")
    for b in 1:nbins
        println(io, rs[b], ",", gr[b])
    end
end

# final state for restart / inspection
np.savez(joinpath(@__DIR__, "validation", "water_final_state.npz"),
         rx=np.asarray(Array(st.rx)), ry=np.asarray(Array(st.ry)),
         rz=np.asarray(Array(st.rz)), vx=np.asarray(Array(st.vx)),
         vy=np.asarray(Array(st.vy)), vz=np.asarray(Array(st.vz)),
         numbers=np.asarray(Z), L=L)

ipk = argmax(gr)
E0, E1 = energies[1], energies[end]
@printf("\nWATER SHOWCASE RESULTS (64 H2O, MACE-OFF small, f64)\n")
@printf("frames sampled: %d   first peak: g_OO = %.2f at r = %.2f Å\n", nframes, gr[ipk], rs[ipk])
@printf("  (experimental x-ray, Skinner 2013: g_OO ≈ 2.57 at r ≈ 2.80 Å)\n")
@printf("NVE production energy drift: %.3e relative over %d steps\n",
        abs(E1 - E0) / abs(E0), N_PROD)
println("wrote water_rdf.csv + water_final_state.npz")
