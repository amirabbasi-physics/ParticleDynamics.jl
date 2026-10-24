module ClusterWCASmallBoxGPUChecks
# WCA on the cluster-pair backend and charged
# states in small boxes.
# - WCA (uniform, pair matrix, per-particle σ) maps onto the cluster LJ table with the WCA
#   cutoff and the +ε shift: forces, energies and virials equal the ELL engine's, and
#   short trajectories agree.
# - A charged box with 2·r_c < L < 2·(r_c + skin), formerly kept on the cell-list path,
#   moves to the cluster backend with the skin capped at L/2 − r_c; forces and energy
#   equal the Ewald oracle.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ES=ParticleDynamics.Electrostatics
include(joinpath(@__DIR__,"ewald_reference.jl")); using .EwaldReference
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
F(st)=vcat((Array(x)' for x in (st.fx,st.fy,st.fz))...)
pos(st)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz))...)
function evaluate(st)
    SC.evaluate_configuration!(st,SC.EvaluationRequest(energy=true,virial=true))
    W=virial_tensor(st)
    return sum(Array(st.Epot)),F(st),[W.xx,W.yy,W.zz,W.xy,W.xz,W.yz]
end

function wca_state(style,X,backend)
    N=size(X,2); L=12*1.12
    st=SC.build_simulation(;N,box=(L,L,L),nonbonded=:wca,cutoff=2^(1/6),skin=0.3,cap=Int32(128),epsilon=1.0,sigma=1.0,
        temperature=0.0,dt=0.002,precision=:f64,spatial_reorder=false,neighbor_cutoff=style===:uniform ? nothing : 1.6)
    for (d,x) in enumerate((st.rx,st.ry,st.rz)); copyto!(x,X[d,:]); end
    if style===:mixed
        st.sigma_particle=CuArray([1+0.12*sin(1.7i) for i in 1:N]); st.rcut_factor=2^(1/6)
    elseif style===:matrix
        st.typeid=CuArray(Int32[1+(i%2) for i in 1:N])
        σm=[1.0 1.1; 1.1 1.2]; st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray([1.0 0.8; 0.8 1.2])
        st.rcut_pair=CuArray(2^(1/6) .*σm)
    end
    backend===:cluster && set_nonbonded_backend!(st,:cluster)
    SC.invalidate_forces!(st)
    return st
end

@testset "WCA on the cluster backend; small charged boxes" begin
    rng=MersenneTwister(5); n=12; a=1.12; L=n*a
    X=hcat([collect(((i,j,k).+0.5).*a.-L/2).+0.04 .*randn(rng,3) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]...)
    @testset "WCA ($style): cluster = cell-list engine" for style in (:uniform,:matrix,:mixed)
        A=wca_state(style,X,:cluster); B=wca_state(style,X,:ell)
        @test nonbonded_backend(A)===:cluster && nonbonded_backend(B)===:ell
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing && B.engine isa DE.StateEngine
        @test isapprox(EA,EB;rtol=1e-11) && relerr(FA,FB)<=1e-11 && relerr(WA,WB)<=1e-11
        V=0.5 .*randn(MersenneTwister(6),3,size(X,2))
        for st in (A,B), (d,x) in enumerate((st.vx,st.vy,st.vz)); copyto!(x,V[d,:]); end
        for i in 1:30; SC.step!(A,SC.nve(A),A.dt;compute_energy=i%10==0); SC.step!(B,SC.nve(B),B.dt;compute_energy=i%10==0); end
        DE.synchronize_state!(A); DE.synchronize_state!(B)
        @test relerr(pos(A),pos(B))<=1e-9
    end
    @testset "Charged box below twice the list radius: cluster backend, Ewald oracle" begin
        C=1.7; rc=4.0; Lb=(8.3,8.4,8.5); N=60
        rng=MersenneTwister(9); R=rand(rng,3,N).*collect(Lb).-collect(Lb)./2; q=randn(rng,N); q.-=sum(q)/N
        Eref,Fref,_=ewald_reference(R,q,Matrix(Diagonal(collect(Lb)));C=C)
        st=SC.build_simulation(;N,box=Lb,nonbonded=:none,cutoff=rc,skin=0.3,cap=Int32(N),precision=:f64,charges=q,
            spatial_reorder=false,use_neighborlist=false)
        for (d,x) in enumerate((st.rx,st.ry,st.rz)); copyto!(x,R[d,:]); end
        spec=ES.ElectrostaticsSpec(coulomb_prefactor=C,real_cutoff=rc,split=ES.GaussianSplit(0.9),
            window=ES.BSplineWindow(8),mesh=(48,48,48),influence=:spme)
        attach_electrostatics!(st,spec)
        @test nonbonded_backend(st)===:cluster && st.cluster_config.skin<=minimum(Lb)/2-rc
        E,Fd,_=evaluate(st)
        @test st.engine isa DE.StateEngine && st.engine.run.cluster!==nothing
        @test sqrt(sum(abs2,Fd.-Fref)/sum(abs2,Fref))<=1e-5 && isapprox(E,Eref;rtol=1e-6)
        for i in 1:20; SC.step!(st,SC.nve(st),1e-3;compute_energy=false); end
        DE.synchronize_state!(st); @test all(isfinite,Array(st.fx))
    end
end
end
