module DeviceEngineRigidGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# rigid SPC/E water (SHAKE/RATTLE inside the step graph)
# with Lennard-Jones and SPME through the public `step!` (charged, so the cluster
# backend: user rule, 13 Oct). Checks: the engine owns the state; trajectories against
# a staged golden record (NVE); constraint lengths and orthogonality after the run;
# total energy against the staged twin; the device observables' temperature uses the constrained dof.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra, StaticArrays
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const RW=ParticleDynamics.RigidWaterConstraints
const ESX=ParticleDynamics.Electrostatics
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
const KB=0.0083144626

"""SPC/E waters on a cubic lattice (spacing 0.31 nm), random orientations, 300 K."""
function water_states(::Type{T};n=12,seed=5) where {T}
    a=0.31; L=ntuple(_->n*a,3); rng=MersenneTwister(seed); nw=n^3; N=3nw
    oh=0.1; hh=0.16330; θ=2asin(hh/2/oh)
    R=zeros(3,N); triples=NTuple{3,Int}[]; ex=Tuple{Int,Int,Float64}[]
    for (w,(i,j,k)) in enumerate((i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1)
        o=3w-2; R[:,o].=((i,j,k).+0.5).*a
        u=normalize(randn(rng,3)); p=normalize(cross(u,randn(rng,3)))
        R[:,o+1].=R[:,o].+oh.*u; R[:,o+2].=R[:,o].+oh.*(cos(θ).*u.+sin(θ).*p)
        push!(triples,(o,o+1,o+2)); push!(ex,(o,o+1,0.0),(o,o+2,0.0),(o+1,o+2,0.0))
    end
    m=repeat([15.9994,1.008,1.008],nw); q=repeat([-0.8476,0.4238,0.4238],nw); ty=repeat(Int32[1,2,2],nw)
    V=randn(rng,3,N).*sqrt.(KB*300 ./ m')
    spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.12),
        window=ESX.BSplineWindow(6),mesh=(32,32,32),influence=:spme)
    states=map(1:2) do _
        st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.1),cap=Int32(800),
            epsilon=T(0.650629),sigma=T(0.316557),mass=T.(m),dt=T(0.002),precision=T===Float64 ? :f64 : :f32,
            charges=q,spatial_reorder=false,neigh_interval=10_000)
        copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
        copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
        copyto!(st.typeid,ty)
        st.sigma_pair=CuArray(T.([0.316557 0.0; 0.0 0.0])); st.epsilon_pair=CuArray(T.([0.650629 0.0; 0.0 0.0]))
        st.rcut_pair=CuArray(T.([1.0 0.0; 0.0 0.0]))
        attach_electrostatics!(st,spec;exceptions=ex)
        attach_rigid_water!(st,triples;oh,hh)
        st
    end
    return states[1],states[2],(;oh,hh,triples,m)
end

function geometry_errors(st,g)
    r=map(Array,(st.rx,st.ry,st.rz)); v=map(Array,(st.vx,st.vy,st.vz)); box=st.box3
    dl=0.0; dv=0.0
    for t in g.triples
        raw=ntuple(k->SVector(ntuple(d->Float64(r[d][t[k]]),3)),3)
        x=RW.local_water(raw,raw[1],Float64.(box))
        vv=ntuple(k->SVector(ntuple(d->Float64(v[d][t[k]]),3)),3)
        dl=max(dl,RW.distance_error(x,(g.oh^2,g.oh^2,g.hh^2)))
        dv=max(dv,maximum(abs(dot(e,w)) for (e,w) in zip(RW._edges(x),RW._edges(vv))))
    end
    return dl,dv
end

@testset "rigid SPC/E water on the device engine" begin
@golden_testset :engine "NVE vs staged golden, constraints, energy" begin
    T=Float64; A,B,g=water_states(T); sa=SC.nve(A)
    @test nonbonded_backend(A)===:cluster && DE.engine_supported(A,sa)
    for i in 1:50; SC.step!(A,sa,A.dt;compute_energy=i%10==0); end
    DE.synchronize_state!(A)
    @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing && DE.has_feature(A.engine.run,DE.RigidWaterFeature)
    ref=staged_reference("rigid_spce_nve") do
        sb=SC.nve(B); for i in 1:50; SC.run_integrator_step!(B,sb,B.dt;compute_energy=i%10==0); end; B
    end
    for name in (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz)
        @test relerr(Array(getfield(A,name)),field(ref,name))<=1e-8
    end
    dl,dv=geometry_errors(A,g)
    @test dl<1e-9 && dv<1e-6
    # total energy against the staged twin's at step 50 (a freshly built lattice relaxes
    # violently, so conservation from step 0 is not a meaningful check here)
    U=sum(Float64,Array(A.Epot)); K=0.5*sum(g.m .* (Array(A.vx).^2 .+ Array(A.vy).^2 .+ Array(A.vz).^2))
    Ur=sum(field(ref,:Epot)); Kr=0.5*sum(g.m .* (field(ref,:vx).^2 .+ field(ref,:vy).^2 .+ field(ref,:vz).^2))
    @test abs((U+K)-(Ur+Kr))<1e-8*abs(Ur+Kr)
    o=DE.observable_status(A.engine.run)
    dof=3*length(A.rx)-3*length(g.triples)
    @test isapprox(o.temperature,2*o.kinetic/dof;rtol=1e-12)                 # constrained dof
end

@golden_testset :engine "Constrained CSVR: τ → ∞ reproduces constrained NVE" begin
    T=Float64; A,_,g=water_states(T)
    sa=SC.csvr(A;temperature=T(KB*300),tau=T(1e30))      # c1 = 1, c2 = 0: the rescale factor is exactly 1
    @test DE.engine_supported(A,sa)
    for i in 1:50; SC.step!(A,sa,A.dt;compute_energy=i%10==0); end
    DE.synchronize_state!(A)
    @test A.engine isa DE.StateEngine && A.engine.run.thermostat isa DE.CSVRData
    @test !DE.csvr_lazy_report(A.engine.run)
    ref=staged_reference("rigid_spce_nve") do; error("recorded by the NVE testset"); end
    for name in (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz)
        @test relerr(Array(getfield(A,name)),field(ref,name))<=1e-8
    end
end

@testset "Constrained CSVR: canonical temperature and fluctuations" begin
    T=Float64; A,_,g=water_states(T); kT=KB*300
    sa=SC.csvr(A;temperature=T(kT),tau=T(0.1))
    dof=3*length(A.rx)-3*length(g.triples)
    Ts=Float64[]; Ks=Float64[]
    # The lattice start relaxes for ~2000 steps (U settles); then 400 samples every 10 steps.
    for i in 1:8000
        SC.step!(A,sa,A.dt;compute_energy=i%10==0)
        if i%10==0 && i>4000
            o=DE.observable_status(A.engine.run); push!(Ts,o.temperature); push!(Ks,o.kinetic)
        end
    end
    DE.synchronize_state!(A)
    @test isapprox(sum(Ts)/length(Ts),kT;rtol=0.01)                         # target with constrained dof
    rel=sqrt(sum(abs2,Ks.-sum(Ks)/length(Ks))/length(Ks))/(sum(Ks)/length(Ks))
    @test 0.8*sqrt(2/dof)<rel<1.2*sqrt(2/dof)                              # canonical K fluctuations
    dl,dv=geometry_errors(A,g)
    @test dl<1e-9 && dv<1e-6
    # constraint dof per bath; a rigid water split across baths is rejected at capture
    bath=ones(Int32,length(A.rx))
    @test DE._constraint_dof_per_bath(A,CuArray(bath),1)==[3.0*length(g.triples)]
    bath[2]=Int32(2)
    @test_throws ArgumentError DE._constraint_dof_per_bath(A,CuArray(bath),2)
end
end
end
