module DeviceEngineNPTChargedGPUChecks
# Electrostatics with cell changes on the device engine:
# - a changed cell gives the forces, energies and virials of a solver and list built
#   fresh at the new cell (mesh influence, reciprocal virial, background, exceptions,
#   cluster list all follow the box);
# - a rejected Monte Carlo trial restores the committed state exactly, and the next
#   evaluation again matches a fresh build;
# - an NPT run (BAOAB + MC volume moves) of a charged melt stays on the engine and
#   accepts and rejects moves.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ESX=ParticleDynamics.Electrostatics
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
const KB=0.0083144626

function melt(::Type{T},R,V,L;seed=7) where {T}
    N=size(R,2); n=round(Int,cbrt(N))
    q=[isodd(i) ? 1.0 : -1.0 for i in 1:N]; ty=Int32[isodd(i) ? 1 : 2 for i in 1:N]
    sig=[0.33284 0.38647; 0.38647 0.440104]; e12=sqrt(0.0115897*0.4184); eps=[0.0115897 e12; e12 0.4184]
    st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.187),cap=Int32(600),epsilon=T(0.4184),
        sigma=T(0.440104),mass=T(29),dt=T(0.002),precision=T===Float64 ? :f64 : :f32,charges=q,
        spatial_reorder=false,neigh_interval=10_000)
    copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
    copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(T.(sig)); st.epsilon_pair=CuArray(T.(eps)); st.rcut_pair=CuArray(fill(T(1.0),2,2))
    spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.2),
        window=ESX.BSplineWindow(6),mesh=(40,40,40),influence=:spme)
    attach_electrostatics!(st,spec)
    return st
end
function lattice(n;seed=7)
    a=(1/32.07)^(1/3); rng=MersenneTwister(seed)
    R=hcat([[(i+0.5)*a,(j+0.5)*a,(k+0.5)*a].+0.01 .*randn(rng,3) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]...)
    # alternate charges on the rock-salt sublattices: order sites so that odd/even = ±
    sites=[(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]
    perm=sortperm([iseven(sum(s)) ? 0 : 1 for s in sites]; alg=MergeSort)
    plus=perm[1:end÷2]; minus=perm[end÷2+1:end]
    order=vec(permutedims(hcat(plus,minus)))           # +,−,+,−,…
    R=R[:,order]
    return R,randn(rng,3,size(R,2)).*sqrt(KB*1100/29),ntuple(_->n*a,3)
end
pos(st)=vcat((Float64.(Array(x))' for x in (st.rx,st.ry,st.rz))...)
F(st)=vcat((Float64.(Array(x))' for x in (st.fx,st.fy,st.fz))...)
function evaluate(st)
    SC.evaluate_configuration!(st,SC.EvaluationRequest(energy=true,virial=true))
    W=virial_tensor(st)
    return sum(Float64,Array(st.Epot)),F(st),[W.xx,W.yy,W.zz,W.xy,W.xz,W.yz]
end

@testset "electrostatics with cell changes on the device engine" begin
    T=Float64; R,V,L=lattice(12)
    @testset "changed cell = fresh build (s = $s)" for s in (1.012,0.991)
        A=melt(T,R,V,L); sa=SC.nve(A)
        for _ in 1:20; SC.step!(A,sa,A.dt;compute_energy=false); end      # engine owns the state
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        set_cell_and_positions!(A,A.box3.*s)
        E1,F1,W1=evaluate(A)
        @test A.engine isa DE.StateEngine                                  # still on the engine
        B=melt(T,pos(A),zeros(3,size(R,2)),A.box3)                          # solver and list built at the new cell
        E2,F2,W2=evaluate(B)
        @test abs(E1-E2)<=1e-10*abs(E2)
        @test relerr(F1,F2)<=1e-9
        @test relerr(W1,W2)<=1e-9
        # stepping continues on the engine in the new cell
        for _ in 1:10; SC.step!(A,sa,A.dt;compute_energy=false); end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && all(isfinite,Array(A.fx))
    end
    @testset "rejected trial restores the committed state" begin
        A=melt(T,R,V,L); sa=SC.baoab(A;gamma=T(1),temperature=T(KB*1100))
        for _ in 1:20; SC.step!(A,sa,A.dt;compute_energy=false); end
        mc=SC.MonteCarloVolumeMove(A;pressure=1e9,kT=1e-9,log_volume_step=0.05,seed=0xE8)   # every expansion rejected
        E0,F0,W0=evaluate(A); x0=pos(A); b0=A.box3
        rejected=0
        for k in 1:6
            acc=SC.monte_carlo_volume_move!(A,mc)
            if !acc
                rejected+=1
                @test A.box3==b0 && pos(A)==x0
                E,Fk,Wk=evaluate(A)
                @test E==E0 && Fk==F0 && Wk==W0                                # exact restore
            else
                b0=A.box3; x0=pos(A); E0,F0,W0=evaluate(A)
            end
        end
        @test rejected>=1
        B=melt(T,pos(A),zeros(3,size(R,2)),A.box3)                         # still equal to a fresh build
        E2,F2,_=evaluate(B); E1,F1,_=evaluate(A)
        @test abs(E1-E2)<=1e-10*abs(E2) && relerr(F1,F2)<=1e-9
    end
    @testset "NPT (BAOAB + MC every 25 steps) on the engine" begin
        T=Float32; A=melt(T,R,V,L); kT=KB*1100
        sa=SC.baoab(A;gamma=T(1),temperature=T(kT))
        mc=SC.MonteCarloVolumeMove(A;pressure=0.06022,kT=kT,log_volume_step=0.002,seed=0xE9)   # 1 bar in kJ/mol/nm³
        for i in 1:500
            SC.step!(A,sa,A.dt;compute_energy=i%25==0)
            i%25==0 && SC.monte_carlo_volume_move!(A,mc)
        end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        @test mc.attempts==20 && 0<mc.accepts<20
        @test all(isfinite,Array(A.fx))
    end
end
end
