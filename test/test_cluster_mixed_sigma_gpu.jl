module ClusterMixedSigmaGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# per-particle σ on the cluster-pair backend
# (σij = (σi + σj)/2, rc = factor·σij, one ε). Checks: neutral mixed-σ LJ on the cluster
# backend equals the ELL engine (plain and potential-shifted); charged mixed-σ chains:
# forces/energies/virials with bond exclusions minus those without equal minus the host
# mixed-σ LJ of the bonded pairs (the bonded-pair restore path with per-particle σ); graph steps
# equal staged cluster steps.
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ESX=ParticleDynamics.Electrostatics
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
F(st)=vcat((Array(x)' for x in (st.fx,st.fy,st.fz))...)
pos(st)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz))...)
function evaluate(st)
    SC.evaluate_configuration!(st,SC.EvaluationRequest(energy=true,virial=true))
    W=virial_tensor(st)
    return sum(Array(st.Epot)),F(st),[W.xx,W.yy,W.zz,W.xy,W.xz,W.yz]
end
const n=12; const a=1.25; const FACTOR=2.5; const EPS=1.0
σof(i)=1+0.18*sin(1.7i)
function system(;seed=4)
    rng=MersenneTwister(seed)
    sites=[(i,j,k) for k in 0:n-1 for j in 0:n-1 for i in 0:n-1]
    X=hcat([collect((s.+0.5).*a).+0.03 .*randn(rng,3) for s in sites]...); N=length(sites)
    q=[iseven(sum(s)) ? 0.6 : -0.6 for s in sites]
    bonds=Tuple{Int32,Int32}[]; ex=Tuple{Int,Int,Float64}[]
    for c in 0:N÷4-1
        b=4c; append!(bonds,[(Int32(b+1),Int32(b+2)),(Int32(b+2),Int32(b+3)),(Int32(b+3),Int32(b+4))])
        push!(ex,(b+1,b+2,0.0),(b+1,b+3,0.5),(b+2,b+4,0.5))
    end
    return X,q,bonds,ex
end
function state(X;q=nothing,bonds=nothing,ex=nothing,exclude=true,backend=:cluster,modifier=:none)
    N=size(X,2); σp=[σof(i) for i in 1:N]
    st=SC.build_simulation(;N,box=ntuple(_->n*a,3),nonbonded=:lj,cutoff=FACTOR*maximum(σp),skin=0.3,cap=Int32(200),
        epsilon=EPS,sigma=1.0,mass=1.0,dt=0.001,precision=:f64,charges=q,spatial_reorder=false,
        bonds,bonding=bonds===nothing ? nothing : ParticleDynamics.harmonic_bond(k=500.0,r0=a),
        exclude_bonded_pairs=exclude,vdw_modifier=modifier)
    for (d,arr) in enumerate((st.rx,st.ry,st.rz)); copyto!(arr,X[d,:]); end
    for arr in (st.vx,st.vy,st.vz); fill!(arr,0.0); end
    st.sigma_particle=CuArray(σp); st.rcut_factor=FACTOR
    if q!==nothing
        attach_electrostatics!(st,ESX.ElectrostaticsSpec(coulomb_prefactor=1.0,real_cutoff=4.0,split=ESX.GaussianSplit(0.8),
            window=ESX.BSplineWindow(6),mesh=(48,48,48),influence=:spme);exceptions=ex)
    end
    backend===:cluster && nonbonded_backend(st)!==:cluster && set_nonbonded_backend!(st,:cluster)
    SC.invalidate_forces!(st)
    return st
end
function bonded_lj(X,bonds)
    N=size(X,2); L=n*a; E=0.0; Fh=zeros(3,N); W=zeros(6)
    for (i,j) in bonds
        d=X[:,i].-X[:,j]; d.-=L.*round.(d./L); r2=sum(abs2,d)
        s=(σof(i)+σof(j))/2; r2<(FACTOR*s)^2 || continue
        s6=(s^2/r2)^3; E+=4EPS*(s6^2-s6); f=24EPS*(2s6^2-s6)/r2
        Fh[:,i].+=f.*d; Fh[:,j].-=f.*d
        W.+=f.*[d[1]^2,d[2]^2,d[3]^2,d[1]*d[2],d[1]*d[3],d[2]*d[3]]
    end
    return E,Fh,W
end

@testset "per-particle σ on the cluster backend" begin
    X,q,bonds,ex=system()
    @testset "neutral mixed σ: cluster = ELL engine (modifier $m)" for m in (:none,:potential_shift)
        A=state(X;modifier=m); B=state(X;backend=:ell,modifier=m)
        @test nonbonded_backend(A)===:cluster && nonbonded_backend(B)===:ell
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing && B.engine isa DE.StateEngine
        @test isapprox(EA,EB;rtol=1e-11) && relerr(FA,FB)<=1e-11 && relerr(WA,WB)<=1e-11
    end
    @testset "charged mixed-σ chains: bond exclusions remove exactly the bonded LJ" begin
        A=state(X;q,bonds,ex); B=state(X;q,bonds,ex,exclude=false)
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test nonbonded_backend(A)===:cluster && A.engine isa DE.StateEngine
        Eb,Fb,Wb=bonded_lj(X,bonds)
        @test isapprox(EB-EA,Eb;rtol=1e-9) && relerr(FB.-FA,Fb)<=1e-9 && relerr(WB.-WA,Wb)<=1e-8
    end
    @golden_testset :engine "graph steps = staged cluster steps (charged chains, NVE, 40 steps)" begin
        A=state(X;q,bonds,ex); C=state(X;q,bonds,ex)
        sa=SC.nve(A); sc=SC.nve(C)
        for i in 1:40; SC.step!(A,sa,A.dt;compute_energy=i%10==0); end
        DE.synchronize_state!(A)
        ref=staged_reference("cluster_mixed_sigma_nve") do
            for i in 1:40; SC.run_integrator_step!(C,sc,C.dt;compute_energy=i%10==0); end; C
        end
        rp=vcat((field(ref,f)' for f in (:rx,:ry,:rz))...); rf=vcat((field(ref,f)' for f in (:fx,:fy,:fz))...)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        @test relerr(pos(A),rp)<=1e-10 && relerr(F(A),rf)<=1e-8
    end
end
end
