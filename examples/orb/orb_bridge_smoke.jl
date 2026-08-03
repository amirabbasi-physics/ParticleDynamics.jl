# Orb bridge smoke test: build an OrbPotential, push positions through the
# engine's external-potential path, and check the resulting forces against a
# direct ASE call on the same configuration. Also proves CUDA.jl and torch
# coexist in one process (as for the MACE bridge).
#
# Run: julia --project=examples/orb examples/orb/orb_bridge_smoke.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
using PythonCall

CUDA.allowscalar(false)
x = CUDA.rand(1024)
println("CUDA.jl ok: ", sum(x) > 0)

include(joinpath(@__DIR__, "OrbPotential.jl"))

const MODEL = get(ENV, "ORB_MODEL", "orb_v3_conservative_inf_omat")
# float64 for the bridge check: it makes the model call deterministic, so any
# discrepancy is attributable to the staging path rather than to TF32
# reduction order. Production runs use float32-high (see the benchmark).
const PREC = get(ENV, "ORB_PRECISION", "float64")

# A small periodic test cell: 2x2x2 fcc-like arrangement of CH4-free carbon is
# not chemically meaningful, so use a simple periodic box of benzene-like C/H
# atoms taken from the showcase structure when present, else a random-ish but
# non-overlapping carbon lattice.
np = pyimport("numpy")
structfile = joinpath(@__DIR__, "validation", "benzene_crystal.npz")
if isfile(structfile)
    d = np.load(structfile)
    pos = pyconvert(Matrix{Float64}, d["positions"])
    Z = pyconvert(Vector{Int}, d["numbers"])
    L = pyconvert(Vector{Float64}, d["cell_lengths"])
else
    # 3x3x3 simple-cubic carbon at 2.0 Å spacing (structure only needs to be
    # a valid, non-overlapping periodic configuration for a bridge check).
    function cubic_carbon(a::Float64, n::Int)
        p = Matrix{Float64}(undef, n^3, 3)
        k = 0
        for i in 0:n-1, j in 0:n-1, m in 0:n-1
            k += 1
            p[k, :] = [i * a, j * a, m * a]
        end
        return p, fill(6, n^3), fill(a * n, 3)
    end
    global pos, Z, L = cubic_carbon(2.0, 3)
    # Break the cubic symmetry: on the perfect lattice every force vanishes by
    # symmetry, which would make the comparison below meaningless.
    using Random
    Random.seed!(20260729)
    global pos = pos .+ 0.15 .* randn(size(pos))
end
N = length(Z)
@printf("system: N = %d, box = %.4f x %.4f x %.4f Å\n", N, L[1], L[2], L[3])

masses = [z == 1 ? 1.008 : 12.011 for z in Z]

st = build_simulation(; N=N, box=(L[1], L[2], L[3]), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=masses, precision=:f64, dt=0.098226)
copyto!(st.rx, pos[:, 1] .- L[1] / 2)
copyto!(st.ry, pos[:, 2] .- L[2] / 2)
copyto!(st.rz, pos[:, 3] .- L[3] / 2)

pot = OrbPotential(Z, (L[1], L[2], L[3]); model=MODEL, device="cuda",
                   precision=PREC)
ParticleDynamics.attach_external_potential!(st, pot)

# Engine path: one force evaluation through evaluate_forces_into_f!
SimulationCore.evaluate_forces_into_f!(st, true)
Fx = Array(st.fx); Fy = Array(st.fy); Fz = Array(st.fz)
Eeng = sum(Array(st.Epot))

# Reference: independent ASE call on an identical Atoms object
ase = pyimport("ase")
pretrained = pyimport("orb_models.forcefield.pretrained")
calcmod = pyimport("orb_models.forcefield.inference.calculator")
atoms = ase.Atoms(; numbers=np.asarray(collect(Int, Z)),
                  positions=np.asarray(pos .- reshape(L, 1, 3) ./ 2),
                  cell=np.diag(np.asarray(L)), pbc=true)
loaded = pygetattr(pretrained, MODEL)(; device="cuda", precision=PREC)
atoms.calc = calcmod.ORBCalculator(loaded[0], loaded[1]; device="cuda")
Fref = pyconvert(Matrix{Float64}, np.asarray(atoms.get_forces(), dtype=np.float64))
Eref = pyconvert(Float64, atoms.get_potential_energy())

dF = max(maximum(abs.(Fx .- Fref[:, 1])),
         maximum(abs.(Fy .- Fref[:, 2])),
         maximum(abs.(Fz .- Fref[:, 3])))
dE = abs(Eeng - Eref)

println("model = ", MODEL, "  precision = ", PREC)
@printf("max|ΔF| engine vs ASE = %.3e eV/Å\n", dF)
@printf("|ΔE|    engine vs ASE = %.3e eV\n", dE)
@printf("max|F| = %.4f eV/Å,  E = %.6f eV\n", maximum(abs.(Fref)), Eref)
println("CUDA.jl still ok after torch: ", sum(CUDA.rand(1024)) > 0)

if dF < 1e-10 && dE < 1e-8
    println("ORB BRIDGE CHECK: PASS")
else
    println("ORB BRIDGE CHECK: FAIL")
    exit(1)
end
