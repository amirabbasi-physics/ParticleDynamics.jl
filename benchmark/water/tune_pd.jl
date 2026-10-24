# ParticleDynamics run-time settings scan: cluster-list buffer (skin) at the matched
# electrostatics settings, NPT with the device C-rescale every 100 steps (as in the runs),
# 1000 timed steps per value after 200 warm-up steps, all in one session.
# Usage: julia --project=<PD> tune_pd.jl gro=STATE.gro out=OUT.toml method=pme pme=a,nx,ny,nz,4
#                                         [skins=0.06,0.08,0.1,0.13,0.16]
#        (method=esp esp=TOL for ESP)
include(joinpath(@__DIR__, "pd_npt.jl"))
using TOML

R, V, box = read_gro(opts["gro"])
skins = parse.(Float64, split(opt("skins", "0.06,0.08,0.1,0.13,0.16"), ","))
# prune buffers as fractions of the skin (the working list radius is r_c + prune buffer)
fracs = parse.(Float64, split(opt("prune_fracs", "0.5"), ","))
kT = KB * 298.0
rows = Dict{String,Any}[]
const CPm = ParticleDynamics.ClusterPairs
for skin in skins, frac in fracs
    st = water_state(R, V, box; skin)
    set_nonbonded_backend!(st, :cluster; prune_buffer=frac * skin)
    sp = SC.csvr(st; temperature=Float32(kT), tau=0.1f0)
    cr = SC.StochasticCellRescalingState(st; pressure=1.01325 * BAR, kT, compressibility=4.5e-5 / BAR, tau=5.0, seed=29)
    SC.attach_pressure_control!(st, cr; interval=100)
    for i in 1:200; SC.step!(st, sp, st.dt; compute_energy=false); end
    DE.synchronize_state!(st); CUDA.synchronize()
    t = @elapsed (for i in 1:1000; SC.step!(st, sp, st.dt; compute_energy=false); end; DE.synchronize_state!(st); CUDA.synchronize())
    c = Array(st.cluster.list.counts)    # device counters (builds/prunes since the list was created)
    push!(rows, Dict("skin" => skin, "prune_buffer" => frac * skin, "ms_per_step" => 1e3 * t / 1000,
                     "device_builds" => Int(c[CPm.CL_NBUILD]), "device_prunes" => Int(c[CPm.CL_NPRUNE])))
    @printf("skin %.3f prune %.3f  %.4f ms/step  device builds %d  prunes %d (1200 steps)\n", skin, frac * skin,
            1e3 * t / 1000, c[CPm.CL_NBUILD], c[CPm.CL_NPRUNE])
    flush(stdout)
    st = nothing; GC.gc(); CUDA.reclaim()
end
open(io -> TOML.print(io, Dict("state" => opts["gro"], "method" => opt("method", "pme"), "rows" => rows)), opts["out"], "w")
