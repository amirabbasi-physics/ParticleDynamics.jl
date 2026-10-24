module DeviceEngineSmallBoxGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# small boxes (fewer than 3 list radii per side). The device
# list takes its rows from all pairs (one minimum image; list radius ≤ L/2), so every box
# the staged all-pairs path runs is an engine state. Checks: the engine owns all-pairs
# states (3D and 2D, LJ and WCA); trajectories equal the staged all-pairs twin; forces,
# energies and virials equal an independent Float64 host reference; NVE conserves
# energy; a cell-list state whose cells would only fit a tiny skin switches to all pairs.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
pos(st,D)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz)[1:D])...)
F(st,D)=vcat((Array(x)' for x in (st.fx,st.fy,st.fz)[1:D])...)
refpos(ref,D)=vcat((field(ref,f)' for f in (:rx,:ry,:rz)[1:D])...)
refF(ref,D)=vcat((field(ref,f)' for f in (:fx,:fy,:fz)[1:D])...)

function smallbox(D,pot;n=6,a=D==3 ? 1.08 : 1.12,seed=3,neighborlist=false,skin=0.3)
    rng=MersenneTwister(seed); L=n*a; N=n^D
    sites=D==3 ? [(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1] : [(i,j) for i in 0:n-1 for j in 0:n-1]
    X=hcat([collect((s.+0.5).*a.-L/2).+0.05 .*randn(rng,D) for s in sites]...)
    V=0.6 .*randn(rng,D,N); V.-=sum(V;dims=2)./N
    map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=pot===:wca ? 2^(1/6) : 2.5,skin,cap=Int32(256),
            epsilon=1.0,sigma=1.0,temperature=0.0,dt=0.002,precision=:f64,nonbonded=pot,
            use_neighborlist=neighborlist,spatial_reorder=false)
        for (d,x) in enumerate((st.rx,st.ry,st.rz)[1:D]); copyto!(x,X[d,:]); end
        for (d,x) in enumerate((st.vx,st.vy,st.vz)[1:D]); copyto!(x,V[d,:]); end
        SC.invalidate_forces!(st); st
    end
end
"""Float64 host reference over all pairs (minimum image): forces, energy, virial trace."""
function host(X,L,pot)
    D,N=size(X); Fh=zeros(D,N); U=0.0; W=0.0; rc2=pot===:wca ? 2^(1/3) : 6.25
    for i in 1:N-1, j in i+1:N
        d=X[:,i].-X[:,j]; d.-=L.*round.(d./L); r2=sum(abs2,d)
        r2<rc2 || continue
        s6=1/r2^3; f=24s6*(2s6-1)/r2
        U+=4s6*(s6-1)+(pot===:wca ? 1.0 : 0.0); W+=f*r2
        Fh[:,i].+=f.*d; Fh[:,j].-=f.*d
    end
    return Fh,U,W
end

@testset "small boxes on the device engine" begin
    @golden_testset :engine "all-pairs state = staged twin ($(D)D, $pot, $kind)" for D in (3,2), pot in (:lj,:wca), kind in (:nve,:baoab)
        # WCA (rc = 1.12) needs a smaller box than LJ before cells stop fitting
        A,B=smallbox(D,pot;n=pot===:wca ? 3 : 6); L=Float64(A.rz===nothing ? A.box2[1] : A.box3[1])
        mk(st)=kind===:nve ? SC.nve(st) : SC.baoab(st;gamma=0.7,temperature=0.0)
        sa=mk(A); sb=mk(B)
        @test A.nbh isa ParticleDynamics.NeighborLists.AllPairsNeighborMatrix && DE.engine_supported(A,sa)
        for n in 1:200; SC.step!(A,sa,A.dt;compute_energy=n%20==0); end
        DE.synchronize_state!(A)
        ref=staged_reference("smallbox_$(D)d_$(pot)_$(kind)") do
            for n in 1:200; SC.run_integrator_step!(B,sb,B.dt;compute_energy=n%20==0); end; B
        end
        @test A.engine isa DE.StateEngine && A.engine.run.neighbors.allpairs
        @test relerr(pos(A,D),refpos(ref,D))<=1e-9 && relerr(F(A,D),refF(ref,D))<=1e-8
        @test isapprox(sum(Array(A.Epot)),sum(field(ref,:Epot));rtol=1e-9)
        Fh,U,W=host(pos(A,D),L,pot)
        @test relerr(F(A,D),Fh)<=1e-11
        @test isapprox(sum(Array(A.Epot)),U;rtol=1e-11)
        @test isapprox(total_virial_trace(A),W;rtol=1e-10)
    end
    @testset "NVE energy conservation (3D LJ, potential shift, 3000 steps)" begin
        A,_=smallbox(3,:lj); set_vdw_modifier!(A,:potential_shift); sa=SC.nve(A)   # continuous energy at rc
        E(st)=total_potential_energy(st)+total_kinetic_energy(st)
        SC.step!(A,sa,A.dt); E0=E(A)
        drift=0.0
        for n in 1:3000
            SC.step!(A,sa,A.dt;compute_energy=n%100==0)
            n%100==0 && (drift=max(drift,abs(E(A)-E0)))
        end
        @test A.engine isa DE.StateEngine && A.engine.run.neighbors.allpairs
        @test drift<=2e-3*abs(E0)
    end
    @golden_testset :engine "cell list with a too-small skin switches to all pairs" begin
        # L = 7.65: 3 cells fit rc = 2.5 only with a skin under 0.05 (< a quarter of the
        # engine's skin) → all pairs, with the full skin (≤ L/2 − rc)
        A,B=smallbox(3,:lj;n=7,a=7.65/7,neighborlist=true,skin=0.04)
        @test A.nbh isa ParticleDynamics.NeighborLists.NeighborMatrix
        sa=SC.nve(A); sb=SC.nve(B)
        for n in 1:100; SC.step!(A,sa,A.dt;compute_energy=n%20==0); end
        DE.synchronize_state!(A)
        ref=staged_reference("smallbox_cells_to_allpairs") do
            for n in 1:100; SC.run_integrator_step!(B,sb,B.dt;compute_energy=n%20==0); end; B
        end
        @test A.engine.run.neighbors.allpairs
        @test relerr(pos(A,3),refpos(ref,3))<=1e-9
    end
end
end
