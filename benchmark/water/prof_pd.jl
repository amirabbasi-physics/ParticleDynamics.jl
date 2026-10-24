# Profiling run: the pd_npt.jl run (same state, settings and device pressure control),
# with `warm` steps outside and `steps` steps inside a cudaProfilerApi range for
# `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`.
# Usage: julia --project=<PD> prof_pd.jl gro=STATE.gro pme_tuned=... [warm=300 steps=200]
include(joinpath(@__DIR__, "pd_npt.jl"))
R, V, box = read_gro(opts["gro"])
warm = parse(Int, opt("warm", "300")); steps = parse(Int, opt("steps", "200"))
kT = KB * parse(Float64, opt("temperature", "298"))
st = water_state(R, V, box; skin=parse(Float64, opt("skin", "0.2")))
set_nonbonded_backend!(st, :cluster; prune_buffer=parse(Float64, opt("prune", "0.1")))
sp = SC.csvr(st; temperature=Float32(kT), tau=Float32(parse(Float64, opt("tau_t", "0.1"))))
cr = SC.StochasticCellRescalingState(st; pressure=1.01325 * BAR, kT, compressibility=4.5e-5 / BAR, tau=5.0, seed=29)
SC.attach_pressure_control!(st, cr; interval=parse(Int, opt("nstpcouple", "100")))
SC.evaluate_forces_into_f!(st, true)
for i in 1:warm; SC.step!(st, sp, st.dt; compute_energy=false); end
DE.synchronize_state!(st); CUDA.synchronize()
t = time()
CUDA.@profile external=true begin
    for i in 1:steps; SC.step!(st, sp, st.dt; compute_energy=false); end
    DE.synchronize_state!(st); CUDA.synchronize()
end
dt = time() - t
c = Array(st.cluster.list.counts)
const CPm = ParticleDynamics.ClusterPairs
@printf("profiled %d steps in %.3f s (%.4f ms/step, with profiler); device builds %d prunes %d since creation\n",
        steps, dt, 1e3 * dt / steps, c[CPm.CL_NBUILD], c[CPm.CL_NPRUNE])
s = ParticleDynamics.ClusterPairs.statistics(st.cluster)
println("cluster list: ", s)
