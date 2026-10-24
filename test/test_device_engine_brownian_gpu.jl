module DeviceEngineBrownianGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Euler–Maruyama and Euler–Heun (midpoint) through
# the public `step!` (device engine; Philox noise keyed on tags, redrawn for the
# Heun final step; two neighbor checks per Heun step). Checks: T=0 against staged
# golden records (`engine_golden.jl`) including dq/dU and the midpoint slots;
# noise invariance under a storage permutation; exact free diffusion with two
# temperatures while sorting permutes γ/noise; equilibrium statistics against the
# staged goldens.
using Test, CUDA, ParticleDynamics, TOML, SHA, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_equilibrium_reference.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
bd(st,kind;gamma=1,temperature=1)=kind===:em ? SC.eulermaruyama(st;gamma,temperature) : SC.eulerheun(st;gamma,temperature)

@testset "Brownian dynamics on the device engine" begin
@golden_testset :engine "T=0 device vs staged golden, including dq/dU and midpoint slots ($kind, $(D)D, $T)" for kind in (:em,:heun), D in (2,3), T in (Float32,Float64)
    N=257; pos,_=lattice(D,N,1.12,T)
    A,B=map(1:2) do _; st=build(D,T,:lj,N;a=1.12,dt=1e-3); place!(st,D,pos,[zeros(T,N) for _ in 1:D]); end
    sa=bd(A,kind;gamma=T(2),temperature=T(0))
    @test DE.engine_supported(A,sa)
    for _ in 1:50; SC.step!(A,sa,A.dt;compute_energy=true); end
    ref=staged_reference("brownian_t0_$(kind)_$(D)d_$(T)") do
        sb=bd(B,kind;gamma=T(2),temperature=T(0))
        for _ in 1:50; SC.run_integrator_step!(B,sb,B.dt;compute_energy=true); end
        rec=golden_snapshot(B)
        rec,(obs=SC.collect_step_observables(B,sb),)
    end
    @test A.engine isa DE.StateEngine && A.engine.run.thermostat isa DE.BrownianData
    tol=T===Float32 ? 2e-3 : 1e-11
    names=(:rx,:ry,:fx,:fy,:Epot,:dq,:dU)
    kind===:heun && (names=(names...,:vx,:vy))                      # x_m left in the velocity slots
    for n in names
        @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
    end
    oa=SC.collect_step_observables(A,sa); ob=extra(ref,:obs)
    for k in (:Epot_total,:dq_total,:dU_total,:virial_total)
        @test isapprox(getfield(oa,k),ob[String(k)];rtol=T===Float32 ? 1e-3 : 1e-9,atol=1e-8)
    end
    @test A.last_integrator==SC.INTEGRATOR_ID_BROWNIAN
end

function ideal_state(D,::Type{T},N;reorder=false,dt=1e-3) where {T}
    m=ceil(Int,N^(1/D)); L=T(3m)
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(dt),mass=T(1),
        precision=T===Float64 ? :f64 : :f32,spatial_reorder=reorder,reorder_interval=1)
    pos=[T[3*(mod((i-1)÷m^(d-1),m))-L/2+T(1.5) for i in 1:N] for d in 1:D]
    return place!(st,D,pos,[zeros(T,N) for _ in 1:D]),pos,L
end

@testset "Noise is keyed on particle tags ($kind, $(D)D)" for kind in (:em,:heun), D in (2,3)
    T=Float32; N=1000
    A,_,_=ideal_state(D,T,N;reorder=true); B,_,_=ideal_state(D,T,N)
    perm=randperm(Xoshiro(3),N)
    for x in (A.rx,A.ry,A.rz)
        x===nothing || copyto!(x,Array(x)[perm])
    end
    copyto!(A.tag,Int32.(perm)); ParticleDynamics.invalidate_forces!(A)
    sa=bd(A,kind); sb=bd(B,kind)
    Random.seed!(21); SC.step!(A,sa,A.dt)
    Random.seed!(21); SC.step!(B,sb,B.dt)
    for _ in 1:19; SC.step!(A,sa,A.dt); SC.step!(B,sb,B.dt); end
    for (a,b) in zip((A.rx,A.ry,A.dq),(B.rx,B.ry,B.dq))
        @test ordered(A,a)==ordered(B,b)
    end
end

# Free particles: Δx per component is Gaussian with variance 2·T_i·t/γ for both
# schemes, exactly. Two temperatures by build id; sorting permutes γ and noise.
@testset "Exact free diffusion, two temperatures, sorting on ($kind, $(D)D)" for kind in (:em,:heun), D in (2,3)
    T=Float32; N=4096; γ=2.0
    st,pos,L=ideal_state(D,T,N;reorder=true)
    temps=T[isodd(i) ? 0.5 : 2.0 for i in 1:N]
    spec=bd(st,kind;gamma=T(γ),temperature=temps)
    n=2000; for _ in 1:n; SC.step!(st,spec,st.dt;compute_energy=false); end
    t=n*Float64(st.dt)
    @test DE.ordering_status(st.engine.run).count>=2
    @test maximum(abs,Array(st.fx))<=1e-19
    for (g,Tg) in ((1,0.5),(2,2.0))
        dx=Float64[]
        for d in 1:D
            x=ordered(st,(st.rx,st.ry,st.rz)[d]); x0=Float64.(pos[d])
            Δ=@. x-x0; Δ.-=L.*round.(Δ./L)
            append!(dx,Δ[g:2:end])
        end
        σ2=2Tg*t/γ; m=length(dx)
        @test abs(mean(abs2,dx)/σ2-1)<5*sqrt(2/m)
        @test abs(mean(z->z^4,dx)/mean(abs2,dx)^2-3)<5*sqrt(96/m)
    end
end

@golden_testset (D == 2 ? :equilibrium_2d : :equilibrium_3d) "Equilibrium statistics match the staged equilibrated goldens ($(D)D $integ)" for D in (2,3), integ in (:eulermaruyama,:eulerheun)
    compare_to_golden(D,integ)
end
end # testset
end
