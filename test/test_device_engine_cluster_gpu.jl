module DeviceEngineClusterGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# cluster-pair states (Lennard-Jones + real-space Coulomb
# in 8×8 tiles, ESP/SPME mesh) through the public `step!`. Checks: the engine owns the
# state and maintains the cluster list on the device; trajectories against staged
# golden records for an ionic melt (Float64/Float32, NVE and thermostats) and a
# molecular system with exclusions and scaled 1–4-type pairs; device observables
# against host values (the mesh virial included); outer-list overflow recovered at the
# boundary with an unchanged trajectory; a box change (barostat-type scaling) mid-run.
using Test, CUDA, ParticleDynamics, Random, Statistics, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const CP=ParticleDynamics.ClusterPairs
const ESX=ParticleDynamics.Electrostatics
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

const KB=0.0083144626   # kJ/(mol K)
spme(L)=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.2),
    window=ESX.BSplineWindow(6),mesh=ntuple(d->8*cld(ceil(Int,L[d]/0.12),8),3),influence=:spme)

"""Rock-salt NaCl at the molten density (32.07 ions/nm³), jittered, 1100 K velocities."""
function ionic_states(::Type{T};n=12,seed=7) where {T}
    a=(1/32.07)^(1/3); L=ntuple(_->n*a,3); rng=MersenneTwister(seed)
    sites=[(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]; N=length(sites)
    R=[ (s[d]+0.5)*a+0.01*randn(rng) for d in 1:3, s in sites]
    q=[iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty=Int32[iseven(sum(s)) ? 1 : 2 for s in sites]
    V=randn(rng,3,N).*sqrt(KB*1100/29)
    sig=[0.33284 0.38647; 0.38647 0.440104]; eps=[0.0115897 sqrt(0.0115897*0.4184); sqrt(0.0115897*0.4184) 0.4184]
    states=map(1:2) do _
        st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.187),cap=Int32(600),epsilon=T(0.4184),
            sigma=T(0.440104),mass=T(29),dt=T(0.002),precision=T===Float64 ? :f64 : :f32,charges=q,
            spatial_reorder=false,neigh_interval=10_000)
        place!(st,R,V,ty,T); st.sigma_pair=CuArray(T.(sig)); st.epsilon_pair=CuArray(T.(eps))
        st.rcut_pair=CuArray(fill(T(1.0),2,2))
        attach_electrostatics!(st,spme(L)); set_nonbonded_backend!(st,:cluster)
        st
    end
    return states[1],states[2]
end

"""Three-site molecules (charges −0.8/0.4/0.4, LJ on the first site) on a lattice;
the two bonds excluded, the third pair scaled by ½ (in-list scaled pairs)."""
function molecular_states(::Type{T};n=12,seed=3) where {T}
    a=0.31; L=ntuple(_->n*a,3); rng=MersenneTwister(seed); nm=n^3; N=3nm
    R=zeros(3,N); ex=Tuple{Int,Int,Float64}[]
    for (m,(i,j,k)) in enumerate((i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1)
        o=3m-2; R[:,o].=((i,j,k).+0.5).*a
        u=normalize(randn(rng,3)); w=normalize(cross(u,randn(rng,3)))
        R[:,o+1].=R[:,o].+0.1 .*u; R[:,o+2].=R[:,o].+0.1 .*(cosd(109.47).*u.+sind(109.47).*w)
        push!(ex,(o,o+1,0.0),(o,o+2,0.0),(o+1,o+2,0.5))
    end
    q=repeat([-0.8,0.4,0.4],nm); ty=repeat(Int32[1,2,2],nm)
    V=randn(rng,3,N).*sqrt(KB*300/6)
    states=map(1:2) do _
        st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.173),cap=Int32(800),epsilon=T(0.65),
            sigma=T(0.3166),mass=T(6),dt=T(0.0005),precision=T===Float64 ? :f64 : :f32,charges=q,
            spatial_reorder=false,neigh_interval=10_000)
        place!(st,R,V,ty,T)
        st.sigma_pair=CuArray(T.([0.3166 0.0; 0.0 0.0])); st.epsilon_pair=CuArray(T.([0.65 0.0; 0.0 0.0]))
        st.rcut_pair=CuArray(T.([1.0 0.0; 0.0 0.0]))
        attach_electrostatics!(st,spme(L);exceptions=ex); set_nonbonded_backend!(st,:cluster)
        st
    end
    return states[1],states[2]
end

function place!(st,R,V,ty,::Type{T}) where {T}
    copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
    copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
    copyto!(st.typeid,ty)
    return st
end

function spec0(kind,st,T)
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.7),temperature=T(0))
    kind===:csvr && return SC.csvr(st;temperature=T(1100),tau=T(1e30))
    return SC.nosehooverchain(st;temperature=T(1100),tau=T(0.2))
end

const COMPARED=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot)
compare(A,ref::AbstractDict;tol)=for name in COMPARED
    @test relerr(Array(getfield(A,name)),field(ref,name))<=tol
end
compare(A,B;tol)=for name in COMPARED
    @test relerr(Array(getfield(A,name)),Array(getfield(B,name)))<=tol
end
advance!(A,sa,n;every=10)=for i in 1:n; SC.step!(A,sa,A.dt;compute_energy=i%every==0); end
staged_advance!(B,sb,n;every=10)=for i in 1:n; SC.run_integrator_step!(B,sb,B.dt;compute_energy=i%every==0); end
device_counts(st)=Array(st.cluster.list.counts)

@testset "cluster-pair states on the device engine" begin
@golden_testset :engine "Ionic melt: device vs staged golden ($kind, $T)" for (kind,T,tol) in
        ((:nve,Float64,1e-8),(:nve,Float32,2e-3),(:baoab,Float64,1e-8),(:csvr,Float64,1e-8),(:nhc,Float64,1e-8))
    A,B=ionic_states(T); sa=spec0(kind,A,T)
    @test DE.engine_supported(A,sa)
    advance!(A,sa,60)
    DE.synchronize_state!(A)
    @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
    c=device_counts(A)
    @test c[CP.CL_NBUILD]>=1 && c[CP.CL_NPRUNE]>=c[CP.CL_NBUILD] && c[CP.CL_ERR]==0
    ref=staged_reference("cluster_ionic_$(kind)_$(T)") do
        sb=spec0(kind,B,T); staged_advance!(B,sb,60); B
    end
    compare(A,ref;tol)
end

@golden_testset :engine "Ionic melt: Euler–Heun (two cluster evaluations per step) vs staged golden" begin
    T=Float64; A,B=ionic_states(T); dt=1e-3
    sa=SC.eulerheun(A;gamma=T(5000),temperature=T(0),dt)
    @test DE.engine_supported(A,sa)
    for i in 1:60; SC.step!(A,sa,dt;compute_energy=i%10==0); end
    DE.synchronize_state!(A)
    @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
    c=device_counts(A)
    @test c[CP.CL_ERR]==0 && c[CP.CL_NBUILD]>=1
    ref=staged_reference("cluster_ionic_heun") do
        sb=SC.eulerheun(B;gamma=T(5000),temperature=T(0),dt)
        for i in 1:60; SC.run_integrator_step!(B,sb,dt;compute_energy=i%10==0); end
        B
    end
    for name in (:rx,:ry,:rz,:fx,:fy,:fz,:Epot)
        @test relerr(Array(getfield(A,name)),field(ref,name))<=1e-8
    end
end

@golden_testset :engine "Molecular: exclusions and scaled pairs, device vs staged golden" begin
    T=Float64; A,B=molecular_states(T); sa=SC.nve(A)
    advance!(A,sa,40)
    @test A.engine isa DE.StateEngine && A.cluster.list.npairs>0       # scaled pairs in use
    ref=staged_reference("cluster_molecular_nve") do
        staged_advance!(B,SC.nve(B),40); B
    end
    compare(A,ref;tol=1e-8)
end

@testset "Device observables include the mesh virial" begin
    T=Float64; A,_=ionic_states(T); sa=SC.nve(A)
    advance!(A,sa,10)                                  # step 10 is sampled
    o=DE.observable_status(A.engine.run)
    DE.synchronize_state!(A)
    U=sum(Float64,Array(A.Epot)); W=virial_tensor(A)
    @test isapprox(o.potential,U;rtol=1e-10)
    @test isapprox(o.virial[1:3],[W.xx,W.yy,W.zz];rtol=1e-9)
    @test isapprox(o.virial[4:6],[W.xy,W.xz,W.yz];rtol=1e-8,atol=1e-6*abs(W.xx))
    @test any(!iszero,Array(A.virial_global))
end

@testset "Outer-list overflow: recovered and replayed" begin
    T=Float64; A,B=ionic_states(T); sa=SC.nve(A); sb=SC.nve(B)
    advance!(A,sa,5); advance!(B,sb,5)
    DE.tighten_cluster_capacity!(A)
    n0=A.cluster.list.necap
    advance!(A,sa,40); advance!(B,sb,40)
    DE.synchronize_state!(A); DE.synchronize_state!(B)
    @test A.engine.recoveries>=1 && A.cluster.list.necap>n0
    compare(A,B;tol=1e-10)
end

@testset "Non-finite coordinates stop the run with an error" begin
    T=Float64; A,_=ionic_states(T); sa=SC.nve(A)
    advance!(A,sa,3); DE.synchronize_state!(A)
    CUDA.@allowscalar A.rx[17]=T(NaN)
    advance!(A,sa,2)
    @test_throws ArgumentError DE.synchronize_state!(A)
end

@golden_testset :engine "Box change mid-run (barostat-type scaling)" begin
    T=Float64; A,B=ionic_states(T); sa=SC.nve(A)
    advance!(A,sa,10)
    ref=staged_reference("cluster_ionic_boxchange") do
        sb=SC.nve(B); staged_advance!(B,sb,10)
        s=1.004; SC.set_box!(B,B.box3.*s)
        for x in (B.rx,B.ry,B.rz); x.*=T(s); end
        SC.invalidate_forces!(B); staged_advance!(B,sb,20); B
    end
    s=1.004; SC.set_box!(A,A.box3.*s)
    for x in (A.rx,A.ry,A.rz); x.*=T(s); end
    SC.invalidate_forces!(A); advance!(A,sa,20)
    @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
    compare(A,ref;tol=1e-8)
end
end
end
