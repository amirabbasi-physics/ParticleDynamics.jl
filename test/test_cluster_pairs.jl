isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
using LinearAlgebra
using Random

const CPX = ParticleDynamics.ClusterPairs
const ESX = ParticleDynamics.Electrostatics
include(joinpath(@__DIR__, "engine_golden.jl")); using .EngineGolden

# Molecules of three: two partners 0.1 apart from the first atom, charges
# (−0.8, 0.4, 0.4); exceptions exclude the two bonds and scale the third pair.
function _cp_system(nmol, L; seed=1)
    rng = MersenneTwister(seed)
    N = 3nmol
    R = zeros(3, N)
    for m in 0:(nmol - 1)
        o = 3m + 1
        R[:, o] .= rand(rng, 3) .* collect(L)
        for k in 1:2
            v = randn(rng, 3); v ./= norm(v)
            R[:, o + k] .= R[:, o] .+ 0.1 .* v
        end
    end
    q = repeat([-0.8, 0.4, 0.4], nmol)
    exc = Tuple{Int,Int,Float64}[]
    for m in 0:(nmol - 1)
        o = 3m + 1
        push!(exc, (o, o + 1, 0.0), (o, o + 2, 0.0), (o + 1, o + 2, 0.5))
    end
    types = repeat(Int32[1, 2, 3], nmol)
    return R, q, exc, types
end

function _cp_state(R, L; T=Float64, nonbonded=:lj, cutoff=1.0, skin=0.1, q=nothing, cap=Int32(600), kwargs...)
    N = size(R, 2)
    st = build_simulation(; N=N, box=L, nonbonded=nonbonded, cutoff=cutoff, skin=skin, cap=cap,
                          precision=T === Float64 ? :f64 : :f32, charges=q, spatial_reorder=false,
                          neigh_interval=10_000, kwargs...)
    copyto!(st.rx, T.(R[1, :])); copyto!(st.ry, T.(R[2, :])); copyto!(st.rz, T.(R[3, :]))
    return st
end

function _cp_eval(st)
    SimulationCore.invalidate_forces!(st)
    SimulationCore.evaluate_forces_into_f!(st, true)
    F = vcat(Float64.(Array(st.fx))', Float64.(Array(st.fy))', Float64.(Array(st.fz))')
    W = virial_tensor(st)
    return sum(Float64, Array(st.Epot)), F, [W.xx, W.yy, W.zz, W.xy, W.xz, W.yz]
end

_relF(a, b) = norm(a .- b) / norm(b)

# ELL reference, then the same state on the cluster backend. Charged states run on
# the cluster backend only: their ELL reference is a golden record of the staged
# engine (`golden` names it).
function _cp_compare(st; prune_buffer=nothing, golden=nothing)
    if golden === nothing
        set_nonbonded_backend!(st, :ell)
        E1, F1, W1 = _cp_eval(st)
    else
        ref = staged_reference(golden) do
            set_nonbonded_backend!(st, :ell)
            E, F, Wv = _cp_eval(st)
            return Dict{String,Any}("step" => st.step), (E=E, F=F, W=Wv)
        end
        E1 = extra(ref, :E); F1 = reshape(extra(ref, :F), 3, :); W1 = extra(ref, :W)
    end
    set_nonbonded_backend!(st, :cluster; prune_buffer=prune_buffer)
    E2, F2, W2 = _cp_eval(st)
    return (dE=abs(E2 - E1) / abs(E1), dF=_relF(F2, F1), dW=norm(W2 .- W1) / norm(W1))
end

@testset "ClusterPairs: Lennard-Jones" begin
    L = (5.3, 5.6, 6.1)
    rng = MersenneTwister(11)
    N = 2_501                                   # not a multiple of 8
    R = rand(rng, 3, N) .* collect(L)
    # uniform LJ, Float64 and Float32
    for (T, tol) in ((Float64, 1e-12), (Float32, 2e-5))
        st = _cp_state(R, L; T=T, epsilon=0.8, sigma=0.32, cutoff=1.0)
        d = _cp_compare(st)
        @test d.dF < tol
        @test d.dE < tol
        @test d.dW < tol
        @test nonbonded_backend(st) === :cluster
        @test st.nbh isa ParticleDynamics.NeighborLists.AllPairsNeighborMatrix
    end
    # non-separable 3-type table with per-pair cutoffs
    types = Int32.(rand(rng, 1:3, N))
    sig = [0.30 0.31 0.33; 0.31 0.35 0.36; 0.33 0.36 0.28]
    eps = [0.6 0.2 0.9; 0.2 0.4 0.0; 0.9 0.0 1.1]
    rcm = [1.0 0.9 1.0; 0.9 0.8 0.0; 1.0 0.0 1.0]
    st = _cp_state(R, L; cutoff=1.0)
    copyto!(st.typeid, types)
    st.sigma_pair = CuArray(sig); st.epsilon_pair = CuArray(eps); st.rcut_pair = CuArray(rcm)
    d = _cp_compare(st)
    @test d.dF < 1e-12
    @test d.dE < 1e-12
    @test d.dW < 1e-12
    @test st.cluster.ljk == CPX.LJ_TABLE_SHARED
end

@golden_testset :engine "ClusterPairs: Coulomb with exclusions and scaled pairs" begin
    L = (4.6, 4.8, 5.1)
    R, q, exc, types = _cp_system(1_000, L; seed=3)
    rc = 1.0
    specs = (
        ("SPME", ESX.ElectrostaticsSpec(coulomb_prefactor=138.9, real_cutoff=rc, split=ESX.GaussianSplit(3.2),
                                        window=ESX.BSplineWindow(6), mesh=(48, 48, 52), influence=:spme)),
        ("P3M-ik", ESX.ElectrostaticsSpec(coulomb_prefactor=138.9, real_cutoff=rc, split=ESX.GaussianSplit(3.2),
                                          window=ESX.BSplineWindow(5), mesh=(40, 40, 44), influence=:p3m_ik)),
        ("ESP", ESX.ElectrostaticsSpec(coulomb_prefactor=138.9, real_cutoff=rc,
                                       split=ESX.ProlateSplit(rc, ESX.prolate_shape_for_tolerance(1e-6)),
                                       window=ESX.ProlateWindow(8, ESX.prolate_shape_for_tolerance(5e-7)),
                                       mesh=(32, 32, 36), influence=:esp)),
        ("SPME shifted", ESX.ElectrostaticsSpec(coulomb_prefactor=138.9, real_cutoff=rc, split=ESX.GaussianSplit(3.2),
                                                window=ESX.BSplineWindow(6), mesh=(48, 48, 52), influence=:spme,
                                                real_space_modifier=:potential_shift)))
    sig = [0.32 0.0 0.30; 0.0 0.0 0.0; 0.30 0.0 0.28]
    eps = [0.65 0.0 0.40; 0.0 0.0 0.0; 0.40 0.0 0.30]
    rcm = [1.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 1.0]
    for (name, spec) in specs, (T, tol) in ((Float64, 1e-11), (Float32, 3e-5))
        @testset "$name $T" begin
            st = _cp_state(R, L; T=T, q=q, cutoff=rc)
            copyto!(st.typeid, types)
            st.sigma_pair = CuArray(T.(sig)); st.epsilon_pair = CuArray(T.(eps)); st.rcut_pair = CuArray(T.(rcm))
            attach_electrostatics!(st, spec; exceptions=exc)
            d = _cp_compare(st; golden="cluster_pairs_coulomb_$(replace(name, ' ' => '_'))_$T")
            @test d.dF < tol
            @test d.dE < tol
            @test d.dW < tol
            @test st.cluster.list.npairs == 1_000      # every 0.5-scaled pair goes to the pair list
        end
    end
    # Coulomb only (no vdW term)
    st = _cp_state(R, L; q=q, nonbonded=:none, cutoff=rc)
    attach_electrostatics!(st, specs[1][2]; exceptions=exc)
    d = _cp_compare(st; golden="cluster_pairs_coulomb_only")
    @test d.dF < 1e-11
    @test d.dE < 1e-11
    # Coulomb exceptions are corrected on the cluster backend only
    capturing() || @test_throws ArgumentError set_nonbonded_backend!(st, :ell)
end

@golden_testset :engine "ClusterPairs: dynamics, rebuilds and switching" begin
    # ±1 ions on a jittered lattice with LJ repulsion (r_c 2.5) and SPME with a
    # longer real-space cutoff (3.0), reduced units
    rng = MersenneTwister(5)
    n = 13
    a0 = 1.25
    L = (n * a0, n * a0, n * a0)
    R = zeros(3, n^3)
    k = 0
    for i in 0:(n - 1), j in 0:(n - 1), l in 0:(n - 1)
        k += 1
        R[:, k] .= ([i, j, l] .+ 0.5) .* a0 .+ 0.05 .* randn(rng, 3)
    end
    q = [isodd(i + j + l) ? 1.0 : -1.0 for i in 0:(n - 1) for j in 0:(n - 1) for l in 0:(n - 1)]
    q .-= sum(q) / length(q)
    spec = ESX.ElectrostaticsSpec(coulomb_prefactor=1.0, real_cutoff=3.0, split=ESX.GaussianSplit(1.0),
                                  window=ESX.BSplineWindow(5), mesh=(24, 24, 24), influence=:spme)
    V = randn(rng, 3, n^3)
    V .-= sum(V; dims=2) ./ n^3
    function run(backend)
        st = _cp_state(R, L; q=q, cutoff=2.5, skin=0.3, epsilon=1.0, sigma=1.0, mass=1.0, temperature=1.0)
        copyto!(st.vx, V[1, :]); copyto!(st.vy, V[2, :]); copyto!(st.vz, V[3, :])
        attach_electrostatics!(st, spec)
        backend === :cluster && set_nonbonded_backend!(st, :cluster; prune_buffer=0.15)
        integ = nve(st; dt=0.002)
        for _ in 1:400
            step!(st, integ, 0.002; compute_energy=false)
        end
        return st
    end
    a = staged_reference(() -> run(:ell), "cluster_pairs_dynamics_ell"); b = run(:cluster)
    Ra = vcat((field(a, f)' for f in (:rx, :ry, :rz))...); Rb = vcat(Array(b.rx)', Array(b.ry)', Array(b.rz)')
    D = Ra .- Rb
    D .-= collect(L) .* round.(D ./ collect(L))      # wrapping may differ at the box edge
    @test maximum(abs, D) < 1e-7
    s = CPX.statistics(b.cluster)
    @test s.builds >= 2
    @test s.prunes > s.builds
    # charged states without Coulomb exceptions may return to the dense list (15 Oct 2026)
    capturing() || (set_nonbonded_backend!(b, :ell); @test b.nonbonded_backend === :ell)
end

@testset "ClusterPairs: pair physics and limits" begin
    # reaction field: e = C qq (1/r + k r² − c), f/r = C qq (1/r³ − 2k), zero energy at r_c
    C, rc, er, erf = 138.9, 1.1, 15.0, Inf
    cp = CPX.coulomb_reaction_field(C, rc; eps_r=er, eps_rf=erf, T=Float64)
    k = 1 / (2rc^3); c = 1 / rc + k * rc^2
    for r in (0.3, 0.7, 1.05)
        f, e = CPX.pair_force(r^2, 1 / r, 0.5, 0.0, 0.0, 0.0, 0.0, false, cp, Val(0), Val(false), Val(true), Val(true))
        @test e ≈ C / er * 0.5 * (1 / r + k * r^2 - c) rtol = 1e-12
        @test f ≈ C / er * 0.5 * (1 / r^3 - 2k) rtol = 1e-12
        fx, ex = CPX.pair_force(r^2, 1 / r, 0.5, 0.0, 0.0, 0.0, 0.0, true, cp, Val(0), Val(false), Val(true), Val(true))
        @test ex ≈ C / er * 0.5 * (k * r^2 - c) rtol = 1e-12
        @test fx ≈ -C / er * 0.5 * 2k rtol = 1e-12
    end
    _, e = CPX.pair_force(rc^2 * (1 - 1e-12), 1 / (rc * (1 - 5e-13)), 1.0, 0.0, 0.0, 0.0, 0.0, false, cp,
                          Val(0), Val(false), Val(true), Val(true))
    @test abs(e) < 1e-9
    # Ewald polynomial against erfc
    sp = ESX.GaussianSplit(3.0)
    cg = CPX.coulomb_polynomial(sp, 1.0; T=Float64, cutoff=1.2)
    for r in (0.1, 0.5, 1.1)
        f, e = CPX.pair_force(r^2, 1 / r, 1.0, 0.0, 0.0, 0.0, 0.0, false, cg, Val(0), Val(false), Val(true), Val(true))
        # 1/r + P(r²): absolute accuracy relative to the bare Coulomb term
        @test abs(e - ESX._host_erfc(3.0r) / r) < 1e-12 / r
        @test abs(f - (ESX._host_erfc(3.0r) / r + 2 * 3.0 / sqrt(pi) * exp(-9.0r^2)) / r^2) < 1e-12 / r^3
    end
    # unsupported configurations
    st = _cp_state(rand(MersenneTwister(1), 3, 500) .* 4.0, (4.0, 4.0, 4.0); nonbonded=:wca, cutoff=1.0)
    set_nonbonded_backend!(st, :cluster)                       # WCA runs on the cluster backend
    @test nonbonded_backend(st) === :cluster
    @test_throws ArgumentError set_nonbonded_backend!(st, :fast)
    st2 = build_simulation(N=400, box=(10.0, 10.0), cutoff=2.5, skin=0.3)
    @test_throws ArgumentError set_nonbonded_backend!(st2, :cluster)
end

@testset "ClusterPairs: workflow ClusterPairList" begin
    rng = MersenneTwister(41)
    n = 10; a = 1.35; Lb = n * a
    pos = [(a * (i - 0.5) - Lb / 2, a * (j - 0.5) - Lb / 2, a * (k - 0.5) - Lb / 2) .+ 0.03 .* (randn(rng), randn(rng), randn(rng))
           for i in 1:n for j in 1:n for k in 1:n]
    typeids = Int32[isodd(i + j + k) ? 1 : 2 for i in 1:n for j in 1:n for k in 1:n]
    system = ParticleSystem(pos; box=(Lb, Lb, Lb), types=[:Na, :Cl], typeids=typeids,
                            masses=Dict(:Na => 1.0, :Cl => 1.0), charges=Dict(:Na => 1.0, :Cl => -1.0),
                            topology=Topology(exclusions=[(1, 2), (3, 4)]))
    allg = Group(:all, AllSelection())
    thermo = ThermodynamicObservable(allg; name=:all)
    function run(nl)
        sim = Simulation(system; groups=Groups(allg), observables=[thermo],
            integrator=Integrator(dt=0.002,
                forces=[LennardJones(epsilon=1.0, sigma=1.0, cutoff=2.5, pairs=:neighborlist, neighborlist=nl),
                        PME(coulomb_prefactor=2.0, real_cutoff=3.4, alpha=1.0, mesh=(30, 30, 30), order=5,
                            scaled_pairs=[(5, 6, 0.5)])],
                methods=[ConstantVolume(allg)]),
            precision=Float64, seed=3)
        run!(sim, Stage(:nve, steps=20; progress=false))
        st = state(sim)
        SimulationCore.evaluate_forces_into_f!(st, true)
        e = ParticleDynamics.Workflow.sample_observable(sim, thermo; fields=[:potential_energy]).potential_energy
        return st, e
    end
    sa, ea = run(CellList(capacity=300))
    sb, eb = run(ClusterPairList(buffer=0.3))
    @test nonbonded_backend(sb) === :cluster
    Ra = vcat(Array(sa.rx)', Array(sa.ry)', Array(sa.rz)'); Rb = vcat(Array(sb.rx)', Array(sb.ry)', Array(sb.rz)')
    D = Ra .- Rb
    D .-= Lb .* round.(D ./ Lb)
    @test maximum(abs, D) < 1e-9
    @test eb ≈ ea rtol = 1e-10
end

@testset "ClusterPairs: box changes" begin
    rng = MersenneTwister(9)
    L = (5.0, 5.2, 5.4)
    N = 1500
    R = rand(rng, 3, N) .* collect(L)
    q = randn(rng, N); q .-= sum(q) / N
    spec = ESX.ElectrostaticsSpec(coulomb_prefactor=1.0, real_cutoff=1.2, split=ESX.GaussianSplit(2.6),
                                  window=ESX.BSplineWindow(5), mesh=(40, 40, 40), influence=:spme)
    s = 1.03
    L2 = L .* s
    function forces_after_box_change(backend)
        st = _cp_state(R, L; q=q, cutoff=1.0, epsilon=0.3, sigma=0.25)
        attach_electrostatics!(st, spec)
        backend === :cluster && set_nonbonded_backend!(st, :cluster)
        _cp_eval(st)
        eng = st.cluster
        copyto!(st.rx, R[1, :] .* s); copyto!(st.ry, R[2, :] .* s); copyto!(st.rz, R[3, :] .* s)
        set_box!(st, L2)
        out = _cp_eval(st)
        backend === :cluster && @test st.cluster === eng            # same engine, list rebuilt
        return out
    end
    E1, F1, W1 = forces_after_box_change(:ell)
    E2, F2, W2 = forces_after_box_change(:cluster)
    @test _relF(F2, F1) < 1e-11
    @test abs(E2 - E1) < 1e-11 * abs(E1)
end
