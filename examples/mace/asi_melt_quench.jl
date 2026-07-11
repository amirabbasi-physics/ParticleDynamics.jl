# Showcase: crystal → liquid → amorphous silicon with MACE-MP-0 and the
# package's thermostats (CSVR by default; pass "nhc" as the first argument
# for a Nosé-Hoover chain — output files are tagged by thermostat).
#
# Protocol (Si216, float64, dt = 1 fs):
#   0. crystal reference at 300 K (3 ps; RDF sampled over last 2 ps)
#   1. melt at 3000 K (15 ps)
#   2. liquid equilibration at 1800 K (10 ps; RDF over last 5 ps)
#   3. quench 1800 K -> 300 K over 50 ps (target ramped every 100 steps)
#   4. anneal at 300 K (10 ps)
#   5. production at 300 K (NVE, 10 ps; RDF + coordination + bond angles)
#
# Amorphous-Si experimental landmarks (Laaziri et al., PRL 82, 3460 (1999)):
# first-neighbor peak ~2.35 Å, mean coordination ~3.9-4.1, tetrahedral
# bond-angle distribution centered near 109.5°.
#
# Outputs in validation/ (suffixed by thermostat tag): asi_rdf_<tag>.csv
# (crystal/liquid/amorphous g(r)), asi_thermo_<tag>.csv (T and target vs
# time), asi_angles_<tag>.csv, asi_melt_quench_<tag>.gsd (one frame / 100
# steps), asi_final_state_<tag>.npz, summary on stdout.
#
# Run: julia --project=examples/mace examples/mace/asi_melt_quench.jl

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

const kB = 8.617333e-5
const DT = 0.098226                 # 1 fs
const TAU = 50.0                    # NHC coupling ≈ 0.51 ps
const MASS_SI = 28.0855
const A0 = 5.431
const NC = 3                        # 3x3x3 conventional cells -> 216 atoms
const RAMP_UPDATE = 100             # steps between quench-target updates
const FRAME_EVERY = 100
const RDF_EVERY = 20
const NBINS = 160
const THERMO = length(ARGS) >= 1 ? Symbol(lowercase(ARGS[1])) : :csvr
const TAG = String(THERMO)

np = pyimport("numpy")

# --- crystalline start: diamond lattice, deterministic tiny rattle ---
basis = [0 0 0; 0 2 2; 2 0 2; 2 2 0; 1 1 1; 1 3 3; 3 1 3; 3 3 1] .* (A0 / 4)
L = NC * A0
pos = zeros(8 * NC^3, 3)
k = 0
for i in 0:NC-1, j in 0:NC-1, l in 0:NC-1, b in 1:8
    global k += 1
    pos[k, :] = basis[b, :] .+ A0 .* [i, j, l]
end
N = k
rng = MersenneTwister(11)
pos .+= 0.02 .* randn(rng, N, 3)
Z = fill(14, N)

st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=kB * 300.0,
                      mass=MASS_SI, precision=:f64, dt=DT)
copyto!(st.rx, pos[:, 1] .- L / 2)
copyto!(st.ry, pos[:, 2] .- L / 2)
copyto!(st.rz, pos[:, 3] .- L / 2)
σv = sqrt(kB * 300.0 / MASS_SI)
vx = σv .* randn(rng, N); vx .-= sum(vx) / N
vy = σv .* randn(rng, N); vy .-= sum(vy) / N
vz = σv .* randn(rng, N); vz .-= sum(vz) / N
copyto!(st.vx, vx); copyto!(st.vy, vy); copyto!(st.vz, vz)
copyto!(st.typeid, fill(Int32(1), N))

pot = MACEPotential(Z, (L, L, L); variant=:mp, model="small", device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)

temperature_K() = begin
    SimulationCore._refresh_kinetic_buffer!(st)
    2 * sum(Array(st.Ekin)) / (3 * N) / kB
end

# RDF accumulation on host
mimg(d) = d - L * round(d / L)
const RMAX = L / 2
const DR = RMAX / NBINS
function accumulate_rdf!(counts)
    hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
    @inbounds for a in 1:N-1, b in a+1:N
        r = sqrt(mimg(hx[a] - hx[b])^2 + mimg(hy[a] - hy[b])^2 + mimg(hz[a] - hz[b])^2)
        bin = Int(fld(r, DR)) + 1
        1 <= bin <= NBINS && (counts[bin] += 1)
    end
    return nothing
end

gsd_out = joinpath(@__DIR__, "validation", "asi_melt_quench_$(TAG).gsd")
h = gsd_open(gsd_out)
thermo = open(joinpath(@__DIR__, "validation", "asi_thermo_$(TAG).csv"), "w")
println(thermo, "step,stage,T_K,T_target_K")

make_tstat(T_K) = THERMO === :nhc ?
    SimulationCore.nosehooverchain(st; temperature=kB * T_K, tau=TAU) :
    SimulationCore.csvr(st; temperature=kB * T_K, tau=TAU)
tstat = make_tstat(300.0)
# retarget by swapping params (fresh target + consistent internal masses);
# the spec's workspace/state carries over
set_tstat_target!(T_K) = (tstat.params = make_tstat(T_K).params)

global gstep = 0
function run_stage!(name, nsteps, spec; target_K=nothing, ramp=nothing,
                    rdf_counts=nothing, rdf_from=0)
    frames = 0
    for i in 1:nsteps
        if ramp !== nothing && i % RAMP_UPDATE == 1
            T_now = ramp[1] + (ramp[2] - ramp[1]) * (i - 1) / nsteps
            set_tstat_target!(T_now)
        end
        SimulationCore.step!(st, spec, DT; compute_energy=false)
        global gstep += 1
        if gstep % FRAME_EVERY == 0
            write_gsd_frame!(h, st; types_names=["Si"], step=gstep)
        end
        if rdf_counts !== nothing && i > rdf_from && i % RDF_EVERY == 0
            accumulate_rdf!(rdf_counts)
            frames += 1
        end
        if i % 1000 == 0
            tgt = ramp === nothing ? (target_K === nothing ? NaN : target_K) :
                  ramp[1] + (ramp[2] - ramp[1]) * i / nsteps
            @printf("%-8s %6d/%d  T = %7.1f K  (target %7.1f)\n", name, i, nsteps,
                    temperature_K(), tgt)
            println(thermo, gstep, ",", name, ",", temperature_K(), ",", tgt)
            flush(stdout); flush(thermo)
        end
    end
    return frames
end

rdf_x = zeros(NBINS); rdf_l = zeros(NBINS); rdf_a = zeros(NBINS)

set_tstat_target!(300.0)
fx = run_stage!("crystal", 3_000, tstat; target_K=300.0, rdf_counts=rdf_x, rdf_from=1_000)
set_tstat_target!(3000.0)
run_stage!("melt", 15_000, tstat; target_K=3000.0)
set_tstat_target!(1800.0)
fl = run_stage!("liquid", 10_000, tstat; target_K=1800.0, rdf_counts=rdf_l, rdf_from=5_000)
run_stage!("quench", 50_000, tstat; ramp=(1800.0, 300.0))
set_tstat_target!(300.0)
run_stage!("anneal", 10_000, tstat; target_K=300.0)
nve = SimulationCore.nve(st; dt=DT)
fa = run_stage!("prod", 10_000, nve; rdf_counts=rdf_a, rdf_from=0)

gsd_close(h)
close(thermo)

# --- normalize RDFs and write ---
V = L^3
function normalize_rdf(counts, nframes)
    g = zeros(NBINS)
    for b in 1:NBINS
        r1 = (b - 1) * DR; r2 = b * DR
        vshell = 4π / 3 * (r2^3 - r1^3)
        g[b] = 2 * counts[b] * V / (nframes * N * (N - 1) * vshell)
    end
    return g
end
gx = normalize_rdf(rdf_x, fx); gl = normalize_rdf(rdf_l, fl); ga = normalize_rdf(rdf_a, fa)
open(joinpath(@__DIR__, "validation", "asi_rdf_$(TAG).csv"), "w") do io
    println(io, "r_A,g_crystal_300K,g_liquid_1800K,g_amorphous_300K")
    for b in 1:NBINS
        println(io, (b - 0.5) * DR, ",", gx[b], ",", gl[b], ",", ga[b])
    end
end

# --- coordination + bond angles from the final configuration ---
RCUT_NN = 2.85
hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
neigh = [Int[] for _ in 1:N]
for a in 1:N-1, b in a+1:N
    dx = mimg(hx[a] - hx[b]); dy = mimg(hy[a] - hy[b]); dz = mimg(hz[a] - hz[b])
    if dx^2 + dy^2 + dz^2 < RCUT_NN^2
        push!(neigh[a], b); push!(neigh[b], a)
    end
end
cn = [length(nb) for nb in neigh]
angles = Float64[]
for a in 1:N
    nb = neigh[a]
    for p in 1:length(nb)-1, q in p+1:length(nb)
        b, c = nb[p], nb[q]
        v1 = (mimg(hx[b] - hx[a]), mimg(hy[b] - hy[a]), mimg(hz[b] - hz[a]))
        v2 = (mimg(hx[c] - hx[a]), mimg(hy[c] - hy[a]), mimg(hz[c] - hz[a]))
        cosθ = (v1[1]v2[1] + v1[2]v2[2] + v1[3]v2[3]) /
               (sqrt(v1[1]^2 + v1[2]^2 + v1[3]^2) * sqrt(v2[1]^2 + v2[2]^2 + v2[3]^2))
        push!(angles, acosd(clamp(cosθ, -1, 1)))
    end
end
open(joinpath(@__DIR__, "validation", "asi_angles_$(TAG).csv"), "w") do io
    println(io, "angle_deg")
    foreach(a -> println(io, a), angles)
end

np.savez(joinpath(@__DIR__, "validation", "asi_final_state_$(TAG).npz"),
         rx=np.asarray(Array(st.rx)), ry=np.asarray(Array(st.ry)),
         rz=np.asarray(Array(st.rz)), vx=np.asarray(Array(st.vx)),
         vy=np.asarray(Array(st.vy)), vz=np.asarray(Array(st.vz)),
         numbers=np.asarray(Z), L=L)

ipk = argmax(ga)
@printf("\nAMORPHOUS SILICON RESULTS (Si%d, MACE-MP-0 small, NHC melt-quench)\n", N)
@printf("amorphous g(r) first peak: %.2f at r = %.3f Å  (exp. a-Si ~2.35 Å)\n",
        ga[ipk], (ipk - 0.5) * DR)
@printf("mean coordination (r < %.2f Å): %.2f  (exp. ~3.9-4.1)\n", RCUT_NN, sum(cn) / N)
@printf("bond-angle mean ± std: %.1f° ± %.1f°  (tetrahedral 109.5°)\n",
        sum(angles) / length(angles),
        sqrt(sum(x -> x^2, angles .- sum(angles) / length(angles)) / length(angles)))
println("wrote asi_{rdf,thermo,angles}_$(TAG).csv/gsd/npz artifacts")
