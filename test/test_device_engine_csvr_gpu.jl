module DeviceEngineCSVRGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# CSVR (Bussi) thermostat through the public `step!`
# (device engine; per-bath reduction, Philox Bussi draw, scaling). Checks:
# τ→∞ (scale 1) against staged golden records (`engine_golden.jl`) including
# per-bath K/dof and Ekin; the exact canonical kinetic-energy distribution of an
# ideal gas with two
# baths while sorting permutes the bath ids; equilibrium statistics against the
# staged equilibrated goldens.
using Test, CUDA, ParticleDynamics, TOML, SHA, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_equilibrium_reference.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

@testset "CSVR on the device engine" begin
@golden_testset :engine "τ→∞: device vs staged golden ($(D)D, $T)" for D in (2,3), T in (Float32,Float64)
    N=257; pos,_=lattice(D,N,1.12,T); vel=velocities(D,N,0.5,T)
    A,B=map(1:2) do _; st=build(D,T,:lj,N;a=1.12); place!(st,D,pos,vel); end
    sa=SC.csvr(A;temperature=T(1),tau=T(1e30))
    @test DE.engine_supported(A,sa)
    for _ in 1:50; SC.step!(A,sa,A.dt;compute_energy=true); end
    ref=staged_reference("csvr_tauinf_$(D)d_$(T)") do
        sb=SC.csvr(B;temperature=T(1),tau=T(1e30))
        for _ in 1:50; SC.run_integrator_step!(B,sb,B.dt;compute_energy=true); end
        rec=golden_snapshot(B); wb=sb.workspace
        rec,(dof=Array(wb.dof_per_bath),K=Array(wb.kinetic_total_per_bath),obs=SC.collect_step_observables(B,sb))
    end
    @test A.engine isa DE.StateEngine && A.engine.run.thermostat isa DE.CSVRData
    tol=T===Float32 ? 2e-3 : 1e-11 # Float32: staged Float32 drift rounding, amplified
    for n in (:rx,:vx,:fx,:Epot,:Ekin)
        @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
    end
    wa=sa.workspace
    @test Array(wa.dof_per_bath)==extra(ref,:dof)==[D*N]
    @test relerr(Array(wa.kinetic_total_per_bath),extra(ref,:K))<=(T===Float32 ? 2e-3 : 1e-11)
    @test Array(wa.last_velocity_scale_per_bath)==[1] && Array(wa.cumulative_energy_exchange_per_bath)==[0]
    oa=SC.collect_step_observables(A,sa); ob=extra(ref,:obs)
    for k in (:Ekin_total,:Epot_total,:extended_hamiltonian)
        @test isapprox(getfield(oa,k),ob[String(k)];rtol=T===Float32 ? 1e-3 : 1e-10)
    end
    @test A.last_integrator==SC.INTEGRATOR_ID_CSVR
end

# Ideal gas (soft repulsion, ε=1e-20, dilute lattice): kick and drift leave K
# unchanged, so K of each bath samples the canonical Gamma(f/2, T) distribution
# exactly: mean f·T/2, variance f·T²/2.
@testset "Canonical kinetic-energy distribution, two baths, sorting on ($(D)D)" for D in (2,3)
    T=Float32; N=512; m=ceil(Int,N^(1/D)); L=T(3m)
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(0.005),mass=T(1),
        precision=:f32,spatial_reorder=true,reorder_interval=1)
    pos=[T[3*(mod((i-1)÷m^(d-1),m))-L/2+T(1.5) for i in 1:N] for d in 1:D]
    place!(st,D,pos,velocities(D,N,1.0,T))
    spec=SC.csvr(st;temperatures=T[0.5,2.0],taus=T[0.05,0.05])
    SC.ensure_integrator_workspace!(spec,st)
    bath=Int32[isodd(i) ? 1 : 2 for i in 1:N]                 # by build id
    copyto!(spec.workspace.particle_bath_id,bath)
    for _ in 1:400; SC.step!(st,spec,st.dt;compute_energy=false); end
    K=[Float64[] for _ in 1:2]
    for _ in 1:3000
        for _ in 1:20; SC.step!(st,spec,st.dt;compute_energy=false); end   # corr e⁻² apart
        k=Array(spec.workspace.kinetic_total_per_bath); push!(K[1],k[1]); push!(K[2],k[2])
    end
    @test DE.ordering_status(st.engine.run).count>=2                       # sorting happened
    @test Array(spec.workspace.particle_bath_id)==bath[Array(st.tag)]      # ids followed their particles
    ek=Float64.(Array(st.Ekin))
    @test isapprox(sum(ek[Array(spec.workspace.particle_bath_id).==1]),K[1][end];rtol=1e-4)
    for (b,Tb) in ((1,0.5),(2,2.0))
        f=D*N/2; μ=f*Tb/2; σ2=f*Tb^2/2; n=length(K[b])
        @test abs(mean(K[b])-μ)<5*sqrt(σ2/n)*2                             # ×2: residual correlation
        @test abs(var(K[b])/σ2-1)<5*sqrt((2+6/(f/2))/n)*2
    end
end

@golden_testset (D == 2 ? :equilibrium_2d : :equilibrium_3d) "Equilibrium statistics match the staged equilibrated goldens ($(D)D CSVR)" for D in (2,3)
    compare_to_golden(D,:csvr)
end
end # testset
end
