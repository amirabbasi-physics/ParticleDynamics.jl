# Kremer-Grest polymer melt showcase (classical GPU path).
#
# 100 chains x 32 beads at bead density 0.85 (LJ units): random-walk chains,
# soft-repulsive push-off, then canonical FENE (k=30, R0=1.5) + WCA production
# (WCA acts on bonded pairs too, via exclude_bonded_pairs=false) with BAOAB
# Langevin at T=1. Reports chain conformation statistics against the KG
# reference values (J. Chem. Phys. 92, 5057 (1990)):
# <l> ~ 0.97, Ree^2/Rg^2 ~ 6.3, C_inf = Ree^2/((N-1) l^2) ~ 1.7-1.8.
#
# Outputs in examples/kg_out/: kg_observables.csv, kg_bonds.csv, kg_melt.gsd
# (chains colored by type for OVITO), summary on stdout.
#
# Run: julia --project examples/3D_KG_melt_showcase.jl

using CUDA
using Random
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
CUDA.allowscalar(false)

const NCH = 100
const NB = 32
const N = NCH * NB
const RHO = 0.85
const L = cbrt(N / RHO)
const BOND = 0.97
const TEMP = 1.0
const GAMMA = 1.0
const DT_PUSH = 1.0e-4
const DT = 0.005                    # canonical KG timestep (FENE+WCA bonds)
const PUSH_STEPS = 20_000
const EQ_STEPS = 200_000
const PROD_STEPS = 500_000
const SAMPLE_EVERY = 500
const FRAME_EVERY = 5_000

outdir = joinpath(@__DIR__, "kg_out")
mkpath(outdir)
mimg(d) = d - L * round(d / L)

# --- random-walk chains, bond length BOND, wrapped into [-L/2, L/2) ---
rng = MersenneTwister(42)
rx = zeros(N); ry = zeros(N); rz = zeros(N)
idx = 0
for c in 1:NCH
    x = (rand(rng) - 0.5) * L; y = (rand(rng) - 0.5) * L; z = (rand(rng) - 0.5) * L
    for b in 1:NB
        global idx += 1
        rx[idx] = mimg(x); ry[idx] = mimg(y); rz[idx] = mimg(z)
        u = randn(rng, 3); u ./= sqrt(sum(abs2, u))
        x += BOND * u[1]; y += BOND * u[2]; z += BOND * u[3]
    end
end
bonds = Tuple{Int32,Int32}[]
for c in 0:NCH-1, b in 1:NB-1
    push!(bonds, (Int32(c * NB + b), Int32(c * NB + b + 1)))
end
# Canonical Kremer-Grest bonds: bare FENE (k=30, R0=1.5) plus the WCA pair
# potential acting on bonded pairs too — enabled by exclude_bonded_pairs=false
# below. (With the default exclusions the bare FENE has no repulsive core and
# bonds collapse.)
bond = ParticleDynamics.fene_bond(k=30.0, r0=1.5)

build(kind; dt, unwrap=false, srp=nothing) = build_simulation(;
    N=N, box=(L, L, L), cutoff=2.0^(1 / 6), skin=0.4, cap=Int32(96),
    neigh_interval=10, use_neighborlist=true, spatial_reorder=false,
    epsilon=1.0, sigma=1.0, gamma=GAMMA, temperature=TEMP,
    bonds=bonds, bonding=bond, nonbonded=kind, softrep_params=srp,
    exclude_bonded_pairs=false,
    mass=1.0, precision=:f64, dt=dt, unwrapped_positions=unwrap)

function max_bond_length()
    hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
    m = 0.0
    for (a, b) in bonds
        m = max(m, sqrt(mimg(hx[a] - hx[b])^2 + mimg(hy[a] - hy[b])^2 + mimg(hz[a] - hz[b])^2))
    end
    return m
end

# damped run at fixed dt; a fresh spec per rung keeps the BAOAB coefficients
# consistent with the timestep
function damped_steps!(nsteps, dt; gamma=2.0)
    sp = SimulationCore.baoab(st; gamma=gamma, temperature=TEMP, dt=dt)
    for _ in 1:nsteps
        SimulationCore.step!(st, sp, dt; compute_energy=false)
    end
end

function min_pair_distance()
    hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
    m = Inf
    @inbounds for a in 1:N-1, b in a+1:N
        d2 = mimg(hx[a] - hx[b])^2 + mimg(hy[a] - hy[b])^2 + mimg(hz[a] - hz[b])^2
        d2 < m && (m = d2)
    end
    return sqrt(m)
end

# --- stage 1: soft-repulsive push-off (bounded potential removes overlaps).
# Random-walk starts have deep core overlaps: ramp the soft-core strength so
# every pair is pushed beyond the WCA-safe distance before switching. ---
st = build(:softrep; dt=DT_PUSH, srp=ParticleDynamics.SoftRepulsiveParams(10.0, 1.0))
copyto!(st.rx, rx); copyto!(st.ry, ry); copyto!(st.rz, rz)
ParticleDynamics.sync_unwrapped!(st)
damped_steps!(5_000, 1.0e-5)
# ramp strength AND range: inflating the soft core (sigma up to 1.15) drives
# residual contacts beyond the WCA-safe distance before the switch
for (eps_push, sig_push, dtp) in ((10.0, 1.0, 1.0e-4), (50.0, 1.05, 1.0e-4),
                                  (200.0, 1.10, 1.0e-4), (500.0, 1.15, 5.0e-5))
    st.softrep = ParticleDynamics.SoftRepulsiveParams(eps_push, sig_push)
    damped_steps!(PUSH_STEPS, dtp)
    @printf("push-off eps=%5.0f sig=%.2f: min pair dist = %.3f, max bond = %.3f\n",
            eps_push, sig_push, min_pair_distance(), max_bond_length())
end
mb = max_bond_length()
mb < 1.4 || error("push-off left overstretched bonds (max $(mb)); aborting")
# WCA pair energy at r=0.80 (~44 eps) is still containable by FENE (~69 eps
# to R0); anything closer can snap bonds on switch-on
mpd = min_pair_distance()
mpd >= 0.80 || error("push-off left deep contacts (min pair $(mpd) < 0.80); aborting")

# --- stage 2: WCA + FENE melt ---
px = Array(st.rx); py = Array(st.ry); pz = Array(st.rz)
st = build(:wca; dt=DT, unwrap=true)
copyto!(st.rx, px); copyto!(st.ry, py); copyto!(st.rz, pz)
ParticleDynamics.sync_unwrapped!(st)
σv = sqrt(TEMP / 1.0)
copyto!(st.vx, σv .* randn(rng, N) |> v -> v .- sum(v) / N)
copyto!(st.vy, σv .* randn(rng, N) |> v -> v .- sum(v) / N)
copyto!(st.vz, σv .* randn(rng, N) |> v -> v .- sum(v) / N)
copyto!(st.typeid, Int32[mod(div(i - 1, NB), 6) + 1 for i in 1:N])

# WCA switch-on: residual near-contacts produce huge forces, so start at a
# very small timestep under strong friction and ramp both toward production
for (nsteps, dtr, g) in ((10_000, 1.0e-5, 10.0), (10_000, 1.0e-4, 5.0),
                         (10_000, 5.0e-4, 2.0), (10_000, 2.0e-3, 1.0))
    damped_steps!(nsteps, dtr; gamma=g)
end
mb = max_bond_length()
println("WCA warm-up done, max bond = ", round(mb, digits=3))
mb < 1.4 || error("WCA warm-up left overstretched bonds (max $(mb)); aborting")
ParticleDynamics.sync_unwrapped!(st)

spec = SimulationCore.baoab(st; gamma=GAMMA, temperature=TEMP, dt=DT)
t0 = time()
for i in 1:EQ_STEPS
    SimulationCore.step!(st, spec, DT; compute_energy=false)
    i % 50_000 == 0 && (@printf("equil %6d/%d (%.0f steps/s)\n", i, EQ_STEPS, i / (time() - t0)); flush(stdout))
end

# --- production with conformation + MSD sampling ---
# Conformations from chain-walk reconstruction: accumulate minimum-image bond
# vectors along each chain. Immune to box-straddling unwrap offsets.
function chain_stats(hx, hy, hz)
    ree2 = 0.0; rg2 = 0.0
    cxs = zeros(NB); cys = zeros(NB); czs = zeros(NB)
    for c in 0:NCH-1
        i1 = c * NB + 1
        cxs[1] = 0.0; cys[1] = 0.0; czs[1] = 0.0
        for b in 2:NB
            j = i1 + b - 1
            cxs[b] = cxs[b-1] + mimg(hx[j] - hx[j-1])
            cys[b] = cys[b-1] + mimg(hy[j] - hy[j-1])
            czs[b] = czs[b-1] + mimg(hz[j] - hz[j-1])
        end
        ree2 += cxs[NB]^2 + cys[NB]^2 + czs[NB]^2
        mx = sum(cxs) / NB; my = sum(cys) / NB; mz = sum(czs) / NB
        for b in 1:NB
            rg2 += ((cxs[b] - mx)^2 + (cys[b] - my)^2 + (czs[b] - mz)^2) / NB
        end
    end
    return ree2 / NCH, rg2 / NCH
end

h = gsd_open(joinpath(outdir, "kg_melt.gsd"))
# type names must not collide with element symbols ("B", "C", "F", ...):
# visualizers assign element radii to matching names. Uniform diameter=1 (LJ
# bead size) is written explicitly for the same reason.
types6 = ["bead1", "bead2", "bead3", "bead4", "bead5", "bead6"]
ux0 = Array(st.rx_unwrap); uy0 = Array(st.ry_unwrap); uz0 = Array(st.rz_unwrap)
obs = open(joinpath(outdir, "kg_observables.csv"), "w")
println(obs, "t_LJ,Ree2,Rg2,g1_bead_MSD,g3_com_MSD")
ree_acc = 0.0; rg_acc = 0.0; nsmp = 0
for i in 1:PROD_STEPS
    SimulationCore.step!(st, spec, DT; compute_energy=false)
    if i % SAMPLE_EVERY == 0
        ux = Array(st.rx_unwrap); uy = Array(st.ry_unwrap); uz = Array(st.rz_unwrap)
        ree2, rg2 = chain_stats(Array(st.rx), Array(st.ry), Array(st.rz))
        global ree_acc += ree2; global rg_acc += rg2; global nsmp += 1
        g1 = sum(@. (ux - ux0)^2 + (uy - uy0)^2 + (uz - uz0)^2) / N
        g3 = 0.0
        for c in 0:NCH-1
            r = (c * NB + 1):((c + 1) * NB)
            g3 += (sum(@view ux[r]) / NB - sum(@view ux0[r]) / NB)^2 +
                  (sum(@view uy[r]) / NB - sum(@view uy0[r]) / NB)^2 +
                  (sum(@view uz[r]) / NB - sum(@view uz0[r]) / NB)^2
        end
        println(obs, i * DT, ",", ree2, ",", rg2, ",", g1, ",", g3 / NCH)
    end
    i % FRAME_EVERY == 0 && write_gsd_frame!(h, st; types_names=types6, diameter=1.0, step=i)
    i % 100_000 == 0 && (@printf("prod %7d/%d (%.0f steps/s)\n", i, PROD_STEPS, i / (time() - t0)); flush(stdout))
end
close(obs); gsd_close(h)

# --- bond statistics from final frame ---
hx = Array(st.rx); hy = Array(st.ry); hz = Array(st.rz)
ls = Float64[]
for (a, b) in bonds
    push!(ls, sqrt(mimg(hx[a] - hx[b])^2 + mimg(hy[a] - hy[b])^2 + mimg(hz[a] - hz[b])^2))
end
open(joinpath(outdir, "kg_bonds.csv"), "w") do io
    println(io, "bond_length"); foreach(l -> println(io, l), ls)
end

ree2 = ree_acc / nsmp; rg2 = rg_acc / nsmp
lbar = sum(ls) / length(ls)
@printf("\nKREMER-GREST MELT RESULTS (%d chains x %d beads, rho=%.2f, T=1)\n", NCH, NB, RHO)
@printf("<l> = %.4f  (KG ~0.97)\n", lbar)
@printf("<Ree^2> = %.2f   <Rg^2> = %.2f   ratio = %.2f  (ideal ~6.3)\n", ree2, rg2, ree2 / rg2)
@printf("C_inf = Ree^2/((N-1)<l>^2) = %.2f  (KG ~1.7-1.8)\n", ree2 / ((NB - 1) * lbar^2))
println("wrote kg_out/{kg_observables.csv, kg_bonds.csv, kg_melt.gsd}")
