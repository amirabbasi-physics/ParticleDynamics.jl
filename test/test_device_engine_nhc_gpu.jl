module DeviceEngineNHCGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Nosé–Hoover chains through the public `step!`
# (device engine; staged chain kernels launched inside the graph, per-bath block
# reduction). NHC is deterministic, so the device run is compared step by step
# with staged golden records (`engine_golden.jl`) for all three propagators;
# plus extended-Hamiltonian
# conservation with two baths while sorting permutes the bath ids, and
# equilibrium statistics against the staged equilibrated goldens.
using Test, CUDA, ParticleDynamics, TOML, SHA, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_equilibrium_reference.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

const NHC_WORKSPACE=(:xi,:eta,:kinetic_total_per_bath,:cumulative_energy_exchange_per_bath,:thermostat_kinetic_per_bath,
    :thermostat_potential_per_bath,:last_velocity_scale_per_bath,:dof_per_bath)
@testset "NHC on the device engine" begin
@golden_testset :engine "Device vs staged golden step by step ($prop, $(D)D, $T)" for prop in (:gromacs,:legacy,:lammps), D in (2,3), T in (Float32,Float64)
    N=257; pos,_=lattice(D,N,1.12,T); vel=velocities(D,N,0.8,T)
    A,B=map(1:2) do _; st=build(D,T,:lj,N;a=1.12); place!(st,D,pos,vel); end
    mk(st)=SC.nosehooverchain(st;temperature=T(1),tau=T(0.1),chain_length=3,substeps=3,propagator=prop)
    sa=mk(A)
    @test DE.engine_supported(A,sa)
    for _ in 1:50; SC.step!(A,sa,A.dt;compute_energy=true); end
    ref=staged_reference("nhc_$(prop)_$(D)d_$(T)") do
        sb=mk(B)
        for _ in 1:50; SC.run_integrator_step!(B,sb,B.dt;compute_energy=true); end
        rec=golden_snapshot(B); wb=sb.workspace
        rec,(ws=NamedTuple{NHC_WORKSPACE}(map(f->Array(getfield(wb,f)),NHC_WORKSPACE)),obs=SC.collect_step_observables(B,sb))
    end
    @test A.engine isa DE.StateEngine && A.engine.run.thermostat isa DE.NHCData
    # Float64: K summation order differs (block-reduced Float64 vs staged atomics).
    # Float32: staged also sums K in Float32 and drifts positions in Float32.
    tol=T===Float32 ? 2e-3 : 1e-10
    for n in (:rx,:vx,:fx,:Epot,:Ekin)
        @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
    end
    wa=sa.workspace; wb=extra(ref,:ws)
    for f in NHC_WORKSPACE[1:end-1]
        @test relerr(vec(Array(getfield(wa,f))),wb[String(f)])<=tol
    end
    @test Array(wa.dof_per_bath)==wb["dof_per_bath"]
    oa=SC.collect_step_observables(A,sa); ob=extra(ref,:obs)
    @test isapprox(oa.extended_hamiltonian,ob["extended_hamiltonian"];rtol=T===Float32 ? 1e-3 : 1e-9)
    @test A.last_integrator==SC.INTEGRATOR_ID_NHC
end

@testset "Extended Hamiltonian conserved; two baths follow sorting (3D liquid)" begin
    T=Float64; N=1728; a=equilibrated_spacing(3,N); pos,L=lattice(3,N,a,T)
    st=SC.build_simulation(;N,box=ntuple(_->T(L),3),cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),mass=T(1),precision=:f64,
        spatial_reorder=true,reorder_interval=1)
    place!(st,3,pos,[T.(randn(Xoshiro(5),N)) for _ in 1:3])
    spec=SC.nosehooverchain(st;temperatures=T[0.8,1.2],taus=T[0.2,0.2])
    SC.ensure_integrator_workspace!(spec,st)
    bath=Int32[isodd(i) ? 1 : 2 for i in 1:N]; copyto!(spec.workspace.particle_bath_id,bath)
    for _ in 1:2000; SC.step!(st,spec,st.dt;compute_energy=false); end
    SC.step!(st,spec,st.dt); H0=SC.collect_step_observables(st,spec).extended_hamiltonian
    H=Float64[]
    for _ in 1:20
        for i in 1:200; SC.step!(st,spec,st.dt;compute_energy=i==200); end
        push!(H,SC.collect_step_observables(st,spec).extended_hamiltonian)
    end
    @test st.engine isa DE.StateEngine && DE.ordering_status(st.engine.run).count>=2
    @test Array(spec.workspace.particle_bath_id)==bath[Array(st.tag)]
    U=abs(SC.collect_step_observables(st,spec).Epot_total)
    # LJ is cut at 2.5σ without an energy shift: pairs crossing the cutoff make
    # energy jumps, so H_ext fluctuates independently of dt. Measured on this
    # protocol ($TMPDIR/nhc_drift.jl, max |ΔH|/|U| over 20 samples): device
    # 3.8e-4 / 2.3e-4 and staged 2.0e-4 / 2.8e-4 at dt = 0.002 / 0.001.
    @test maximum(abs.(H.-H0))/U<1e-3
    println("NHC 3D liquid, 4000 steps: max |ΔH_ext|/|U| = ",maximum(abs.(H.-H0))/U)
    K=Array(spec.workspace.kinetic_total_per_bath); dof=Array(spec.workspace.dof_per_bath)
    @test dof==[3*(N÷2),3*(N÷2)]
end

@golden_testset (D == 2 ? :equilibrium_2d : :equilibrium_3d) "Equilibrium statistics match the staged equilibrated goldens ($(D)D NHC)" for D in (2,3)
    compare_to_golden(D,:nhc)
end
end # testset
end
