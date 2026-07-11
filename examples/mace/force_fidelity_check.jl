# Force fidelity check: MACE forces through the ParticleDynamics engine path must
# match the pure-ASE reference (validation/reference.npz) to < 1e-12 eV/Å.
# This catches index-order, unit, and species-mapping bugs.
#
# Run: julia --project=examples/mace examples/mace/force_fidelity_check.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using ParticleDynamics
using ParticleDynamics: SimulationCore
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "MACEPotential.jl"))

np = pyimport("numpy")
ref = np.load(joinpath(@__DIR__, "validation", "reference.npz"))
pos = pyconvert(Matrix{Float64}, ref["positions"])   # ASE frame, ~[0, L)
cellm = pyconvert(Matrix{Float64}, ref["cell"])
Z = pyconvert(Vector{Int}, ref["numbers"])
Eref = pyconvert(Float64, ref["energy"].item())
Fref = pyconvert(Matrix{Float64}, ref["forces"])
N = length(Z)
L = cellm[1, 1]
@assert cellm ≈ L * [1 0 0; 0 1 0; 0 0 1] "orthorhombic cubic cell expected"

# Engine positions live in [-L/2, L/2): shift the ASE frame. Forces are
# translation invariant under PBC, so the reference forces still apply
# particle-by-particle.
st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=28.0855, precision=:f64, dt=0.098226)
copyto!(st.rx, pos[:, 1] .- L / 2)
copyto!(st.ry, pos[:, 2] .- L / 2)
copyto!(st.rz, pos[:, 3] .- L / 2)

pot = MACEPotential(Z, (L, L, L); variant=:mp, model="small", device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)

SimulationCore.evaluate_forces_into_f!(st, true)

dFx = maximum(abs.(Array(st.fx) .- Fref[:, 1]))
dFy = maximum(abs.(Array(st.fy) .- Fref[:, 2]))
dFz = maximum(abs.(Array(st.fz) .- Fref[:, 3]))
dE = abs(sum(Array(st.Epot)) - Eref)
println("max|ΔF| (x,y,z) = ", (dFx, dFy, dFz), " eV/Å   (criterion < 1e-12)")
println("|ΔE| = ", dE, " eV")

if max(dFx, dFy, dFz) < 1e-12 && dE < 1e-9
    println("FORCE FIDELITY: PASS")
else
    println("FORCE FIDELITY: FAIL")
    exit(1)
end
