module DeviceEngineLangevinGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# BAOAB and GSM through the public `step!` (device
# engine, Philox noise keyed on particle tags). Checks: T=0 (noise-free) against
# staged golden records (`engine_golden.jl`) including heat/work bookkeeping;
# noise invariance under a storage permutation; force-free Maxwell–Boltzmann
# velocities (one and two temperatures); equilibrium statistics against the
# staged equilibrated goldens.
using Test, CUDA, ParticleDynamics, TOML, SHA, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_equilibrium_reference.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
langevin(st,kind;gamma=1,temperature=1)=kind===:gsm ? SC.gsm(st;gamma,temperature) :
    kind===:baoa ? SC.baoa(st;gamma,temperature) : kind===:vv ? SC.velocityverlet(st;gamma,temperature) :
    SC.baoab(st;gamma,temperature)

@testset "Langevin family (BAOAB, GSM, BAOA, VV/GJF) on the device engine" begin

@golden_testset :engine "T=0 Langevin: device vs staged golden, including dq/dU/Ekin ($kind, $(D)D, $T)" for kind in (:baoab,:gsm,:baoa,:vv), D in (2,3), T in (Float32,Float64)
    N=257; pos,_=lattice(D,N,1.12,T); vel=velocities(D,N,0.5,T)
    A,B=map(1:2) do _
        st=build(D,T,:lj,N;a=1.12)
        place!(st,D,pos,vel)
    end
    sa=langevin(A,kind;gamma=T(0.7),temperature=T(0))
    @test DE.engine_supported(A,sa)
    for _ in 1:50; SC.step!(A,sa,A.dt;compute_energy=true); end
    ref=staged_reference("langevin_t0_$(kind)_$(D)d_$(T)") do
        sb=langevin(B,kind;gamma=T(0.7),temperature=T(0))
        for _ in 1:50; SC.run_integrator_step!(B,sb,B.dt;compute_energy=true); end
        rec=golden_snapshot(B)
        rec,(obs=SC.collect_step_observables(B,sb),)
    end
    @test A.engine isa DE.StateEngine && A.engine.run.thermostat isa DE.LangevinData
    # Float64 is the decisive check (measured ≤1e-12 over 50 steps). In Float32
    # the staged drift rounds positions in Float32 (two floor-wraps) while the
    # device uses Float64 intermediates; the 2e-7 step-1 position difference
    # grows through the steep compact-lattice forces to ≤6.1e-4 at step 50.
    tol=T===Float32 ? 2e-3 : 1e-11
    names=D==3 ? (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot,:Ekin,:dq,:dU) : (:rx,:ry,:vx,:vy,:fx,:fy,:Epot,:Ekin,:dq,:dU)
    for n in names
        @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
    end
    oa=SC.collect_step_observables(A,sa); ob=extra(ref,:obs)
    for k in (:Ekin_total,:Epot_total,:dq_total,:dU_total,:bath_heat_total)
        @test isapprox(getfield(oa,k),ob[String(k)];rtol=T===Float32 ? 1e-3 : 1e-10,atol=1e-8)
    end
    @test A.last_integrator==SC.INTEGRATOR_ID_LANGEVIN
end

# Dilute lattice (spacing 3 > cutoff 1) with soft repulsion: forces are exactly
# zero while particles stay apart (short runs). With ε=1e-20 the bounded
# repulsion (|f| ≤ ε/σ) keeps long runs an ideal gas to ~1e-20.
function free_state(D,::Type{T},N;reorder=false,eps=1) where {T}
    m=ceil(Int,N^(1/D)); L=T(3m)
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(eps),sigma=T(1),temperature=T(0),dt=T(0.005),mass=T(1),
        precision=T===Float64 ? :f64 : :f32,
        spatial_reorder=reorder,reorder_interval=1_000_000)
    pos=[T[3*(mod((i-1)÷m^(d-1),m))-L/2+T(1.5) for i in 1:N] for d in 1:D]
    place!(st,D,pos,[zeros(T,N) for _ in 1:D])
end

@testset "Noise is keyed on particle tags, not storage slots ($(D)D)" for D in (2,3)
    T=Float32; N=1000
    A=free_state(D,T,N;reorder=true); B=free_state(D,T,N)
    perm=randperm(Xoshiro(3),N)                                  # scramble A's storage
    for x in (A.rx,A.ry,A.rz,A.vx,A.vy,A.vz)
        x===nothing || copyto!(x,Array(x)[perm])
    end
    copyto!(A.tag,Int32.(perm)); ParticleDynamics.invalidate_forces!(A)
    sa=SC.baoab(A;gamma=T(1),temperature=T(1)); sb=SC.baoab(B;gamma=T(1),temperature=T(1))
    Random.seed!(11); SC.step!(A,sa,A.dt)
    Random.seed!(11); SC.step!(B,sb,B.dt)
    for _ in 1:19; SC.step!(A,sa,A.dt); SC.step!(B,sb,B.dt); end
    @test A.engine isa DE.StateEngine && B.engine isa DE.StateEngine
    @test maximum(abs,Array(A.fx))==0                          # still force-free
    for (a,b) in zip((A.rx,A.ry,A.vx,A.vy,A.dq),(B.rx,B.ry,B.vx,B.vy,B.dq))
        @test ordered(A,a)==ordered(B,b)                          # bitwise per particle
    end
    Random.seed!(12); C=free_state(D,T,N); sc=SC.baoab(C;gamma=T(1),temperature=T(1))
    for _ in 1:20; SC.step!(C,sc,C.dt); end
    @test Array(C.vx)!=ordered(B,B.vx)                          # another seed, other noise
end

# Exact for BAOAB/GSM/BAOA (the O step is exact and forces vanish); not for the
# GJF on-site velocity, whose variance depends on γdt, so VV is checked by the
# T=0 comparison and the equilibrium goldens.
@testset "Force-free Maxwell–Boltzmann velocities, one and two temperatures ($kind, $(D)D)" for kind in (:baoab,:gsm,:baoa), D in (2,3)
    T=Float32; N=4096
    st=free_state(D,T,N;eps=1e-20)
    temps=T[i<=N÷2 ? 0.5 : 2.0 for i in 1:N]
    spec=langevin(st,kind;gamma=T(2),temperature=temps)
    for _ in 1:400; SC.step!(st,spec,st.dt;compute_energy=false); end       # ≈ 8 relaxation times
    v2=[Float64[] for _ in 1:2]; v4=[Float64[] for _ in 1:2]
    for _ in 1:20
        for _ in 1:200; SC.step!(st,spec,st.dt;compute_energy=false); end   # v² correlation e⁻⁴ apart
        for x in (st.vx,st.vy,st.vz)[1:D]
            h=Float64.(Array(x))
            push!(v2[1],mean(abs2,h[1:N÷2])); push!(v2[2],mean(abs2,h[N÷2+1:end]))
            push!(v4[1],mean(z->z^4,h[1:N÷2])); push!(v4[2],mean(z->z^4,h[N÷2+1:end]))
        end
    end
    @test maximum(abs,Array(st.fx))<=1e-19                    # ideal gas
    for (g,Tg) in ((1,0.5),(2,2.0))
        n=20*D*(N÷2)                                                         # ≈ independent draws
        @test abs(mean(v2[g])-Tg)<5*Tg*sqrt(2/n)
        @test abs(mean(v4[g])/mean(v2[g])^2-3)<5*sqrt(96/n)
    end
end

@golden_testset (D == 2 ? :equilibrium_2d : :equilibrium_3d) "Equilibrium statistics match the staged equilibrated goldens ($(D)D $integ)" for D in (2,3), integ in (:baoab,:gsm,:baoa,:vv)
    compare_to_golden(D,integ)
end
end # testset
end
