module DeviceEngineStencilGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# states with type-dependent cutoff lists
# (`StencilNeighborMatrix`, the two-size systems of examples/*_stencil_two_sizes*). The
# engine builds one list of the largest pair radius; the pair-matrix model applies each
# pair's cutoff inside the evaluation. Checks: the engine owns stencil states (2D/3D,
# LJ and WCA pair tables); trajectories equal the staged stencil twin; forces, energies
# and virials equal an independent Float64 host reference.
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const NL=ParticleDynamics.NeighborLists
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
pos(st,D)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz)[1:D])...)
F(st,D)=vcat((Array(x)' for x in (st.fx,st.fy,st.fz)[1:D])...)

# Two sizes (σ = 1 and 2, mixed 1.5), type 2 every fourth particle, on a jittered lattice.
const σm=[1.0 1.5; 1.5 2.0]; const εm=[1.0 1.0; 1.0 1.0]
function twosize(D,pot;n=D==3 ? 12 : 30,a=1.6,seed=2)
    rng=MersenneTwister(seed); L=n*a; N=n^D
    sites=D==3 ? [(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1] : [(i,j) for i in 0:n-1 for j in 0:n-1]
    X=hcat([collect((s.+0.5).*a.-L/2).+0.05 .*randn(rng,D) for s in sites]...)
    V=0.5 .*randn(rng,D,N); V.-=sum(V;dims=2)./N
    ty=Int32[i%4==0 ? 2 : 1 for i in 1:N]
    rcm=pot===:wca ? 2^(1/6) .*σm : 2.5 .*σm
    states=map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=maximum(rcm),skin=0.3,cap=Int32(256),epsilon=1.0,
            sigma=maximum(σm),temperature=0.0,dt=0.001,precision=:f64,nonbonded=pot,spatial_reorder=false)
        for (d,x) in enumerate((st.rx,st.ry,st.rz)[1:D]); copyto!(x,X[d,:]); end
        for (d,x) in enumerate((st.vx,st.vy,st.vz)[1:D]); copyto!(x,V[d,:]); end
        copyto!(st.typeid,ty)
        st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(rcm)
        r=(st.rx,st.ry,st.rz)[1:D]
        st.nbh=NL.build_neighbors_stencil_by_types!(r...;box=ntuple(_->L,D),typeid=st.typeid,rcut_pair=rcm,
            cap=Int32(256),skin=0.3)
        SC.invalidate_forces!(st); st
    end
    return states...,L,ty,rcm
end
function host(X,L,ty,rcm,pot)
    D,N=size(X); Fh=zeros(D,N); U=0.0; W=0.0
    for i in 1:N-1, j in i+1:N
        a,b=ty[i],ty[j]; d=X[:,i].-X[:,j]; d.-=L.*round.(d./L); r2=sum(abs2,d)
        r2<rcm[a,b]^2 || continue
        s6=(σm[a,b]^2/r2)^3; e=εm[a,b]; f=24e*s6*(2s6-1)/r2
        U+=4e*s6*(s6-1)+(pot===:wca ? e : 0.0); W+=f*r2
        Fh[:,i].+=f.*d; Fh[:,j].-=f.*d
    end
    return Fh,U,W
end

@testset "typed stencil-list states on the device engine" begin
    @golden_testset :engine "stencil state = staged stencil twin ($(D)D, $pot)" for D in (3,2), pot in (:lj,:wca)
        A,B,L,ty,rcm=twosize(D,pot)
        sa=SC.nve(A); sb=SC.nve(B)
        @test A.nbh isa NL.StencilNeighborMatrix && DE.engine_supported(A,sa)
        for n in 1:200; SC.step!(A,sa,A.dt;compute_energy=n%20==0); end
        DE.synchronize_state!(A)
        ref=staged_reference("stencil_$(D)d_$(pot)") do
            for n in 1:200; SC.run_integrator_step!(B,sb,B.dt;compute_energy=n%20==0); end; B
        end
        rp=vcat((field(ref,f)' for f in (:rx,:ry,:rz)[1:D])...); rf=vcat((field(ref,f)' for f in (:fx,:fy,:fz)[1:D])...)
        @test A.engine isa DE.StateEngine
        @test relerr(pos(A,D),rp)<=1e-9 && relerr(F(A,D),rf)<=1e-8
        @test isapprox(sum(Array(A.Epot)),sum(field(ref,:Epot));rtol=1e-9)
        Fh,U,W=host(pos(A,D),L,ty,rcm,pot)
        @test relerr(F(A,D),Fh)<=1e-11
        @test isapprox(sum(Array(A.Epot)),U;rtol=1e-11)
        @test isapprox(total_virial_trace(A),W;rtol=1e-10)
    end
end
end
