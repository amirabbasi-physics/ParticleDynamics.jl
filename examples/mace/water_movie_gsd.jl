# Write a GSD trajectory of MACE-OFF liquid water for visualization (OVITO
# opens HOOMD-schema .gsd natively; color by particle type O/H).
#
# Continues from the equilibrated/produced state saved by
# water_rdf_showcase.jl (validation/water_final_state.npz): 4 ps of NVE at
# dt = 0.5 fs, one frame every 20 steps -> 400 frames.
#
# Run: julia --project=examples/mace examples/mace/water_movie_gsd.jl

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(ENV, "PARTICLEDYNAMICS_PYTHON",
                                  something(Sys.which("python3"), "python3"))

using CUDA
using Printf
using ParticleDynamics
using ParticleDynamics: SimulationCore
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "MACEPotential.jl"))

const DT = 0.5 * 0.098226     # 0.5 fs
const NSTEPS = 8_000          # 4 ps
const FRAME_EVERY = 20        # 400 frames

np = pyimport("numpy")
fin = np.load(joinpath(@__DIR__, "validation", "water_final_state.npz"))
Z = pyconvert(Vector{Int}, fin["numbers"])
L = pyconvert(Float64, fin["L"].item())
N = length(Z)
masses = [z == 8 ? 15.999 : 1.008 for z in Z]

st = build_simulation(; N=N, box=(L, L, L), cutoff=2.5, skin=0.3,
                      cap=Int32(8), neigh_interval=1,
                      use_neighborlist=false, spatial_reorder=false,
                      gamma=0.0, temperature=0.0,
                      mass=masses, precision=:f64, dt=DT)
copyto!(st.rx, pyconvert(Vector{Float64}, fin["rx"]))
copyto!(st.ry, pyconvert(Vector{Float64}, fin["ry"]))
copyto!(st.rz, pyconvert(Vector{Float64}, fin["rz"]))
copyto!(st.vx, pyconvert(Vector{Float64}, fin["vx"]))
copyto!(st.vy, pyconvert(Vector{Float64}, fin["vy"]))
copyto!(st.vz, pyconvert(Vector{Float64}, fin["vz"]))
# typeid: 1 = O, 2 = H (GSD types_names below maps them for OVITO)
copyto!(st.typeid, Int32[z == 8 ? 1 : 2 for z in Z])

off = expanduser("~/.cache/mace/MACE-OFF23_small.model")
pot = MACEPotential(Z, (L, L, L); variant=:off, model=off, device="cuda")
ParticleDynamics.attach_external_potential!(st, pot)
spec = SimulationCore.nve(st; dt=DT)

out = joinpath(@__DIR__, "validation", "water_mace_movie.gsd")
h = gsd_open(out)
write_gsd_frame!(h, st; types_names=["O", "H"], step=0)
t0 = time()
nframes = 1
for i in 1:NSTEPS
    SimulationCore.step!(st, spec, DT; compute_energy=false)
    if i % FRAME_EVERY == 0
        write_gsd_frame!(h, st; types_names=["O", "H"], step=i)
        global nframes += 1
    end
    if i % 2000 == 0
        @printf("movie %5d/%d  (%.1f steps/s)\n", i, NSTEPS, i / (time() - t0))
        flush(stdout)
    end
end
gsd_close(h)
@printf("wrote %s: %d frames (%.1f ps at %.1f fs/frame)\n",
        out, nframes, NSTEPS * 0.5 / 1000, FRAME_EVERY * 0.5)
