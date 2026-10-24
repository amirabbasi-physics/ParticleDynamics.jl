# Residency trace of the public cluster-pair path: a charged NaCl melt (rock salt,
# LJ table + SPME, cluster backend) stepped with CSVR through `SimulationCore.step!`.
# Every traced step must be one graph replay (device decide → [rebuild] → [pruning
# pass] → pair pass → scatter ‖ mesh → gather, kick, observables), with no kernels,
# host transfers or allocations outside the graphs. scripts/check_device_trace.py
# audits the trace (`pair_backend = "cluster"`).
# Usage (under nsys, user terminal): profile_engine_cluster.jl N DENSITY OUT.toml [steps=500] [every=25] [warmup=200]
#   N ≈ ions (rounded to an even rock-salt edge), DENSITY = low (molten, 32.07 ions/nm³) or high (44.6, crystal)
# PD_CHAINS=1: every lattice row along z is a charged chain (harmonic bonds, LJ
# excluded for bonded pairs); Coulomb exceptions: every other bond (scale 0) and all 1–3
# pairs (scale 0.5), so all bond-exclusion corrections run.
# PD_MIXED_SIGMA=1: per-particle σ (σ+ = 0.33284, σ− = 0.440104, rc = 2.3σij) instead of the type table.
# PD_COLLISIONS=1: collision counting (ELL contact rows in the control block, event kernels each step).
using CUDA, ParticleDynamics, TOML, Random, Statistics
const SC = ParticleDynamics.SimulationCore
const DE = ParticleDynamics.DeviceEngine
const CP = ParticleDynamics.ClusterPairs
const ESX = ParticleDynamics.Electrostatics
const KB = 0.0083144626

function melt_state(n::Int, rho::Float64; seed=11)
    T = Float32; a = (1 / rho)^(1 / 3); L = ntuple(_ -> n * a, 3); rng = MersenneTwister(seed)
    sites = [(i, j, k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]; N = length(sites)
    R = [(s[d] + 0.5) * a + 0.01 * randn(rng) for d in 1:3, s in sites]
    q = [iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty = Int32[iseven(sum(s)) ? 1 : 2 for s in sites]
    V = randn(rng, 3, N) .* sqrt(KB * 1100 / 29)
    sig = [0.33284 0.38647; 0.38647 0.440104]; e12 = sqrt(0.0115897 * 0.4184); eps = [0.0115897 e12; e12 0.4184]
    chains = get(ENV, "PD_CHAINS", "0") == "1"
    bonds = Tuple{Int32,Int32}[]; ex = Tuple{Int,Int,Float64}[]
    if chains   # k is fastest: consecutive indices along a z row are lattice neighbours
        for r in 0:N÷n-1, k in 1:n-1
            i = r * n + k
            push!(bonds, (Int32(i), Int32(i + 1)))
            isodd(k) && push!(ex, (i, i + 1, 0.0))
            k <= n - 2 && push!(ex, (i, i + 2, 0.5))
        end
    end
    st = SC.build_simulation(; N, box=L, nonbonded=:lj, cutoff=T(1.0), skin=T(0.187), cap=Int32(600),
        epsilon=T(0.4184), sigma=T(0.440104), mass=T(29), dt=T(0.002), precision=:f32, charges=q,
        spatial_reorder=false, neigh_interval=10_000,
        bonds=chains ? bonds : nothing, bonding=chains ? ParticleDynamics.harmonic_bond(k=2.0e5, r0=a) : nothing)   # stiff (C–C like): bonded ± pairs keep full Coulomb without LJ
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    copyto!(st.vx, T.(V[1, :])); copyto!(st.vy, T.(V[2, :])); copyto!(st.vz, T.(V[3, :]))
    copyto!(st.typeid, ty)
    if get(ENV, "PD_MIXED_SIGMA", "0") == "1"
        st.sigma_particle = CuArray(T[iseven(sum(s)) ? 0.33284 : 0.440104 for s in sites]); st.rcut_factor = T(2.3)
    else
        st.sigma_pair = CuArray(T.(sig)); st.epsilon_pair = CuArray(T.(eps)); st.rcut_pair = CuArray(fill(T(1.0), 2, 2))
    end
    spec = ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458, real_cutoff=1.0, split=ESX.GaussianSplit(3.2),
        window=ESX.BSplineWindow(6), mesh=ntuple(d -> 8 * cld(ceil(Int, L[d] / 0.12), 8), 3), influence=:spme)
    get(ENV, "PD_E6_NO_MESH", "0") == "1" || attach_electrostatics!(st, spec; exceptions=ex)   # diagnostics: LJ only
    set_nonbonded_backend!(st, :cluster)
    get(ENV, "PD_COLLISIONS", "0") == "1" && ParticleDynamics.enable_collision_counting!(st; ntypes=2)
    return st
end

contact_builds(e) = ParticleDynamics.NeighborLists.device_neighbor_status(e.run.neighbors).builds

function main(args)
    Nreq = parse(Int, args[1]); density = args[2]; out = abspath(args[3])
    steps = length(args) > 3 ? parse(Int, args[4]) : 500; every = length(args) > 4 ? parse(Int, args[5]) : 25
    warmup = length(args) > 5 ? parse(Int, args[6]) : 200
    ispath(out) && error("Refusing to overwrite $out")
    rho = density == "low" ? 32.07 : density == "high" ? 44.6 : error("density must be low or high")
    n = 2 * round(Int, cbrt(Nreq) / 2)
    CUDA.allowscalar(false)
    DE.MESH_PARALLEL[] = get(ENV, "PD_E6_SERIAL_MESH", "0") != "1"            # diagnostics: mesh in sequence
    st = melt_state(n, rho); N = length(st.rx)
    spec = SC.csvr(st; temperature=Float32(KB * 1100), tau=0.1f0)
    record = Dict{String,Any}("status" => "running", "case" => "cluster_csvr_$(density)", "dims" => 3, "N" => N,
        "warmup" => warmup, "steps" => steps, "energy_every" => every, "public_step" => true,
        "global_observables" => true, "neighbor_diagnostics" => false, "spatial_reorder" => false,
        "rebuild_policy" => "displacement", "forces_per_step" => 1, "pair_backend" => "cluster",
        "electrostatics" => get(ENV, "PD_E6_NO_MESH", "0") != "1", "mesh_parallel" => DE.MESH_PARALLEL[],
        "density_ions_per_nm3" => rho, "bond_exclusions" => get(ENV, "PD_CHAINS", "0") == "1")
    record["bond_exclusions"] && (record["case"] = "cluster_chains_csvr_$(density)")
    record["mixed_sigma"] = get(ENV, "PD_MIXED_SIGMA", "0") == "1"
    record["collisions"] = get(ENV, "PD_COLLISIONS", "0") == "1"
    save() = open(io -> TOML.print(io, record; sorted=true), out, "w")
    try
        for i in 1:warmup
            SC.step!(st, spec, st.dt; compute_energy=i % every == 0)
        end
        e = st.engine; e isa DE.StateEngine && e.run.cluster !== nothing || error("public step did not use the cluster engine")
        DE.synchronize_state!(st)
        L = st.cluster.list; c0 = Array(L.counts); o0 = DE.observable_status(e.run); nb0 = contact_builds(e)
        record["thermostat"] = string(nameof(typeof(e.run.thermostat)))
        record["outer_capacity"] = L.necap
        # Which bond-exclusion corrections the engine launches: the LJ restore when
        # unbonded Coulomb exceptions exist; the Coulomb shift only with a shifted real space.
        cc = st.cluster.corrections; coul = st.cluster.coul
        record["lj_restore"] = cc !== nothing && cc.lj !== nothing
        record["coulomb_shift"] = cc !== nothing && cc.coul_shift !== nothing && st.cluster.coulomb &&
            hasproperty(coul, :eshift) && coul.eshift != 0
        CUDA.synchronize()
        CUDA.@profile external=true begin
            for i in 1:steps
                SC.step!(st, spec, st.dt; compute_energy=(warmup + i) % every == 0)
            end
            CUDA.CUDACore.cuStreamSynchronize(CUDA.stream())
        end
        CUDA.synchronize()
        st.engine === e || error("the engine was replaced during the trace")
        DE.synchronize_state!(st)
        e.recoveries == 0 || error("a capacity recovery ran inside the trace")
        c1 = Array(L.counts)
        record["rebuilds_in_trace"] = Int(c1[CP.CL_NBUILD] - c0[CP.CL_NBUILD])
        record["contact_rebuilds_in_trace"] = contact_builds(e) - nb0
        record["prunes_in_trace"] = Int(c1[CP.CL_NPRUNE] - c0[CP.CL_NPRUNE])
        record["observable_samples_in_trace"] = DE.observable_status(e.run).samples - o0.samples
        o = DE.observable_status(e.run); record["temperature_kT"] = o.temperature
        # independent check: a staged evaluation of the same coordinates on a fresh twin
        twin = melt_state(n, rho)
        for (a, b) in ((twin.rx, st.rx), (twin.ry, st.ry), (twin.rz, st.rz)); copyto!(a, b); end
        SC.invalidate_forces!(twin); SC.evaluate_forces_into_f!(twin, false)
        F(x) = vcat(Float64.(Array(x.fx))', Float64.(Array(x.fy))', Float64.(Array(x.fz))')
        rel = sqrt(sum(abs2, F(st) .- F(twin)) / sum(abs2, F(twin)))
        record["force_vs_staged"] = rel
        rel < 1e-4 || error("engine forces differ from a staged evaluation ($rel)")
        record["status"] = "GPU checks passed; trace audit pending"; save()
        println("Engine cluster profile complete: N=", N, " ", density, "; rebuilds=", record["rebuilds_in_trace"],
                "; prunes=", record["prunes_in_trace"], "; kT=", round(o.temperature, digits=3), "; F vs staged ", rel)
    catch err
        record["status"] = "failed"; record["error_type"] = string(typeof(err)); save()
        rethrow()
    end
end
main(ARGS)
