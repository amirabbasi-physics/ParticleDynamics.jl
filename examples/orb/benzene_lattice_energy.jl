# Benzene crystal I: lattice-energy head-to-head between MACE-OFF and Orb-v3,
# both driven through the same ParticleDynamics external-potential interface.
#
# The engine provides no virial while an external potential is attached, so
# there is no NPT and no cell relaxation. Instead the cell is scanned: at each
# isotropically scaled volume the atomic positions are relaxed at fixed cell by
# overdamped Langevin dynamics (BAOAB at T = 0 with strong friction), which is
# a minimisation the engine can do natively. That yields E(V) per model, and
# with it two experimentally comparable numbers:
#
#   * the equilibrium volume  V0 = argmin E(V)
#   * the lattice energy      E_latt = E_crystal/Z - E_monomer  at V0
#
# Reference values quoted in the README: diffusion Monte Carlo gives
# E_latt = -49.8 +- 0.2 kJ/mol for benzene (Della Pia et al., Phys. Rev. Lett.
# 133, 046401 (2024)); experimentally derived values scatter by several kJ/mol.
#
# Run: julia --project=examples/orb examples/orb/benzene_lattice_energy.jl

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

const EV_TO_KJMOL = 96.48533212331
const DT = 0.25 * 0.098226        # 0.25 fs in engine time units
const GAMMA = 15.0                # strong friction -> overdamped relaxation
const NRELAX = 600                # 150 fs of damped dynamics
const SCALES = [0.90, 0.93, 0.96, 0.98, 1.00, 1.02, 1.05, 1.08, 1.12]
const MACE_OFF = expanduser("~/.cache/mace/MACE-OFF23_small.model")

np = pyimport("numpy")
val = joinpath(@__DIR__, "validation")

load(f) = let d = np.load(joinpath(val, f))
    (pos = pyconvert(Matrix{Float64}, d["positions"]),
     Z = pyconvert(Vector{Int}, d["numbers"]),
     m = pyconvert(Vector{Float64}, d["masses"]),
     L = pyconvert(Vector{Float64}, d["cell_lengths"]),
     nmol = pyconvert(Int, d["nmolecules"].item()))
end

crystal = load("benzene_cell.npz")
supercell = load("benzene_crystal.npz")
monomer = load("benzene_monomer.npz")

# Retarget a provider's ASE cell without reloading the model. Both providers
# store `atoms` and `np`, so one helper covers MACE and Orb alike.
set_cell!(pot, L) =
    pot.atoms.set_cell(pot.np.diag(pot.np.asarray(collect(Float64, L))))

function make_state(sys, L)
    N = length(sys.Z)
    st = build_simulation(; N=N, box=(L[1], L[2], L[3]), cutoff=2.5, skin=0.3,
                          cap=Int32(8), neigh_interval=1,
                          use_neighborlist=false, spatial_reorder=false,
                          gamma=0.0, temperature=0.0,
                          mass=sys.m, precision=:f64, dt=DT)
    return st
end

"""Relax positions at fixed cell; return (energy_eV, max_force, final_positions)."""
function relax!(st, pot, pos, L; nsteps=NRELAX)
    copyto!(st.rx, pos[:, 1] .- L[1] / 2)
    copyto!(st.ry, pos[:, 2] .- L[2] / 2)
    copyto!(st.rz, pos[:, 3] .- L[3] / 2)
    spec = SimulationCore.baoab(st; gamma=GAMMA, temperature=0.0, dt=DT)
    for _ in 1:nsteps
        SimulationCore.step!(st, spec, DT; compute_energy=false)
    end
    SimulationCore.evaluate_forces_into_f!(st, true)
    E = sum(Array(st.Epot))
    Fmax = maximum(sqrt.(Array(st.fx) .^ 2 .+ Array(st.fy) .^ 2 .+ Array(st.fz) .^ 2))
    P = hcat(Array(st.rx) .+ L[1] / 2, Array(st.ry) .+ L[2] / 2, Array(st.rz) .+ L[3] / 2)
    return E, Fmax, P
end

geometry(P, Z) = begin
    C = findall(==(6), Z)
    H = findall(==(1), Z)
    dist(i, j) = sqrt(sum(abs2, P[i, :] .- P[j, :]))
    cc = [dist(i, j) for i in C for j in C if i < j && dist(i, j) < 1.6]
    ch = [dist(i, j) for i in C for j in H if dist(i, j) < 1.3]
    return (isempty(cc) ? NaN : sum(cc) / length(cc)),
           (isempty(ch) ? NaN : sum(ch) / length(ch))
end

struct ModelSpec
    label::String
    build::Function
end

models = ModelSpec[
    ModelSpec("MACE-OFF23-small",
              (Z, L) -> MACEPotential(Z, L; variant=:off, model=MACE_OFF,
                                      device="cuda")),
    ModelSpec("Orb-v3-cons-inf-omat",
              (Z, L) -> OrbPotential(Z, L; model="orb_v3_conservative_inf_omat",
                                     device="cuda", precision="float32-high")),
    ModelSpec("Orb-v3-cons-omol",
              (Z, L) -> OrbPotential(Z, L; model="orb_v3_conservative_omol",
                                     device="cuda", precision="float32-high",
                                     charge=0, spin=1)),
]

rows = String[]
csv = open(joinpath(val, "benzene_ev_scan.csv"), "w")
println(csv, "model,scale,volume_A3,E_crystal_eV,E_per_mol_eV,Fmax_eV_per_A,cc_A,ch_A")

summary = open(joinpath(val, "benzene_lattice_energy.txt"), "w")

for ms in models
    println("\n=== ", ms.label, " ===")

    # --- gas-phase monomer reference (relaxed, isolated in a 20 A box) ---
    pot_m = ms.build(monomer.Z, (monomer.L[1], monomer.L[2], monomer.L[3]))
    st_m = make_state(monomer, monomer.L)
    ParticleDynamics.attach_external_potential!(st_m, pot_m)
    Emono, Fm, Pm = relax!(st_m, pot_m, monomer.pos, monomer.L)
    ccm, chm = geometry(Pm, monomer.Z)
    @printf("monomer: E = %.6f eV  |Fmax| = %.2e  C-C = %.4f  C-H = %.4f A\n",
            Emono, Fm, ccm, chm)

    # --- size-consistency check: unit cell vs 2x2x2 supercell, per molecule ---
    pot_c = ms.build(crystal.Z, (crystal.L[1], crystal.L[2], crystal.L[3]))
    st_c = make_state(crystal, crystal.L)
    ParticleDynamics.attach_external_potential!(st_c, pot_c)
    Ec1, _, _ = relax!(st_c, pot_c, crystal.pos, crystal.L; nsteps=200)

    pot_s = ms.build(supercell.Z, (supercell.L[1], supercell.L[2], supercell.L[3]))
    st_s = make_state(supercell, supercell.L)
    ParticleDynamics.attach_external_potential!(st_s, pot_s)
    Es1, _, _ = relax!(st_s, pot_s, supercell.pos, supercell.L; nsteps=200)
    dsize = abs(Ec1 / crystal.nmol - Es1 / supercell.nmol) * EV_TO_KJMOL
    @printf("size consistency (unit cell vs 2x2x2, per molecule): %.4f kJ/mol\n", dsize)
    flush(stdout)
    pot_s = nothing; st_s = nothing

    # --- E(V) scan on the unit cell ---
    best = (E = Inf, V = NaN, s = NaN, cc = NaN, ch = NaN)
    for s in SCALES
        L = crystal.L .* s
        set_cell!(pot_c, L)
        st = make_state(crystal, L)
        ParticleDynamics.attach_external_potential!(st, pot_c)
        E, Fmax, P = relax!(st, pot_c, crystal.pos .* s, L)
        V = prod(L)
        Eper = E / crystal.nmol
        cc, ch = geometry(P, crystal.Z)
        @printf("  s = %.2f  V = %7.2f A^3  E/mol = %10.5f eV  |Fmax| = %.1e  C-C = %.4f  C-H = %.4f\n",
                s, V, Eper, Fmax, cc, ch)
        println(csv, join([ms.label, s, V, E, Eper, Fmax, cc, ch], ","))
        flush(csv); flush(stdout)
        if Eper < best.E
            best = (E = Eper, V = V, s = s, cc = cc, ch = ch)
        end
    end

    Elatt = (best.E - Emono) * EV_TO_KJMOL
    line = @sprintf("%-22s  V0 = %7.2f A^3 (s = %.2f)  E_latt = %8.2f kJ/mol  C-C = %.4f A  C-H = %.4f A  [size-cons. %.3f kJ/mol]",
                    ms.label, best.V, best.s, Elatt, best.cc, best.ch, dsize)
    push!(rows, line)
    println(line)
    println(summary, line)
    flush(csv); flush(summary)
end

println("\n\nBENZENE CRYSTAL I - LATTICE ENERGY HEAD-TO-HEAD")
println("experimental cell (COD 7238223, 150 K): V = ", @sprintf("%.2f", prod(crystal.L)), " A^3")
println("DMC reference: E_latt = -49.8 +- 0.2 kJ/mol (PRL 133, 046401 (2024))")
foreach(println, rows)
close(csv)
close(summary)
