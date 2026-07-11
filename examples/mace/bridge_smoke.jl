# Bridge smoke test: drive mace-torch from Julia via PythonCall and
# reproduce validation/reference.npz exactly. Also proves CUDA.jl and torch
# coexist in one process.
#
# Run: julia --project=examples/mace examples/mace/bridge_smoke.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using PythonCall

# CUDA.jl side first: allocate before torch initializes, to prove coexistence
CUDA.allowscalar(false)
x = CUDA.rand(1024)
println("CUDA.jl ok: ", sum(x) > 0)

np = pyimport("numpy")
ase = pyimport("ase")
mace_calc = pyimport("mace.calculators")

here = @__DIR__
ref = np.load(joinpath(here, "validation", "reference.npz"))
pos = pyconvert(Matrix{Float64}, ref["positions"])   # (N,3) Ang
cell = pyconvert(Matrix{Float64}, ref["cell"])       # 3x3
Z = pyconvert(Vector{Int}, ref["numbers"])
Eref = pyconvert(Float64, ref["energy"].item())
Fref = pyconvert(Matrix{Float64}, ref["forces"])
N = length(Z)

atoms = ase.Atoms(numbers=np.asarray(Z), positions=np.asarray(pos),
                  cell=np.asarray(cell), pbc=true)
atoms.calc = mace_calc.mace_mp(model="small", device="cuda", default_dtype="float64")

F = pyconvert(Matrix{Float64}, atoms.get_forces())
E = pyconvert(Float64, atoms.get_potential_energy())

dF = maximum(abs.(F .- Fref))
dE = abs(E - Eref)
println("N = ", N)
println("max|ΔF| vs reference = ", dF, " eV/Å   (criterion < 1e-12)")
println("|ΔE| vs reference    = ", dE, " eV")

# prove torch actually ran on GPU and CUDA.jl still works after
println("CUDA.jl still ok after torch: ", sum(CUDA.rand(1024)) > 0)

if dF < 1e-12 && dE < 1e-10
    println("BRIDGE CHECK: PASS")
else
    println("BRIDGE CHECK: FAIL")
    exit(1)
end
