module VdwModifierGPUChecks
# Lennard-Jones potential-shift modifier (`vdw_modifier = :potential_shift`, the GROMACS
# convention): u(r) − u(r_c) for every pair inside its cutoff. Forces, virials and
# trajectories equal plain truncation's (to the 1e-16 spread of two independent list
# builds, whose row order is not deterministic); energies differ by u(r_c) per pair
# inside the cutoff (host pair count). Covered: device engine ELL (uniform LJ in the
# generic and the single-lane fast kernel, mixed σ, pair matrix) and the cluster backend
# (charged melt, per-pair cutoffs). The staged neighbor-row path rejects the modifier.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ESX=ParticleDynamics.Electrostatics

uc(σ,ε,rc)=4ε*((σ/rc)^12-(σ/rc)^6)
pos(st)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz))...)
F(st)=vcat((Array(x)' for x in (st.fx,st.fy,st.fz))...)
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
same(a,b;tol=1e-13)=relerr(a,b)<=tol
function evaluate(st)
    SC.evaluate_configuration!(st,SC.EvaluationRequest(energy=true,virial=true))
    W=virial_tensor(st)
    return Array(st.Epot),F(st),[W.xx,W.yy,W.zz,W.xy,W.xz,W.yz]
end
"""Σ over pairs inside their cutoff of `shift(i,j)` (host, minimum image)."""
function shift_sum(X,L,rc,shift)
    N=size(X,2); s=0.0
    for i in 1:N-1, j in i+1:N
        d=X[:,i].-X[:,j]; d.-=L.*round.(d./L)
        r=sqrt(sum(abs2,d)); r<rc(i,j) && (s+=shift(i,j))
    end
    return s
end
function jittered(n,a;seed=3)
    rng=MersenneTwister(seed)
    X=hcat([[(i+0.5)*a,(j+0.5)*a,(k+0.5)*a].+0.08a.*randn(rng,3) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]...)
    L=fill(n*a,3); X.-=L.*round.(X./L)
    return X,L
end
function place!(st,X,V)
    for (d,a) in enumerate((st.rx,st.ry,st.rz)); copyto!(a,X[d,:]); end
    for (d,a) in enumerate((st.vx,st.vy,st.vz)); copyto!(a,V[d,:]); end
    st
end
function pair(modifier,style,X,V,L)
    N=size(X,2)
    st=SC.build_simulation(;N,box=Tuple(L),cutoff=2.5,skin=0.3,cap=Int32(200),neigh_interval=5,epsilon=1.0,
        sigma=1.0,temperature=0.0,dt=0.002,mass=1.0,precision=:f64,spatial_reorder=false,
        neighbor_cutoff=style===:uniform ? nothing : 2.5*1.2)
    if style===:mixed
        st.sigma_particle=CuArray([1+0.18*sin(1.7i) for i in 1:N]); st.rcut_factor=2.5
    elseif style===:matrix
        st.typeid=CuArray(Int32[1+(i%2) for i in 1:N])
        st.sigma_pair=CuArray([1.0 1.1; 1.1 1.2]); st.epsilon_pair=CuArray([1.0 0.8; 0.8 1.2])
        st.rcut_pair=CuArray([2.5 2.3; 2.3 2.8])
    end
    set_vdw_modifier!(st,modifier)
    return place!(st,X,V)
end

@testset "vdW potential-shift modifier" begin
    @testset "ELL engine: $style" for style in (:uniform,:mixed,:matrix)
        X,L=jittered(9,1.15); N=size(X,2); V=0.3 .*randn(MersenneTwister(4),3,N)
        A=pair(:none,style,X,V,L); B=pair(:potential_shift,style,X,V,L)
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test A.engine isa DE.StateEngine && B.engine isa DE.StateEngine
        @test same(FA,FB) && same(WA,WB)                                   # forces and virials unchanged
        σp=1 .+0.18 .*sin.(1.7 .*(1:N)); ty=[1+(i%2) for i in 1:N]
        σm=[1.0 1.1; 1.1 1.2]; εm=[1.0 0.8; 0.8 1.2]; rcm=[2.5 2.3; 2.3 2.8]
        rc,sh=style===:uniform ? ((i,j)->2.5,(i,j)->uc(1.0,1.0,2.5)) :
              style===:mixed ? ((i,j)->2.5*(σp[i]+σp[j])/2,(i,j)->uc(1.0,1.0,2.5)) :
              ((i,j)->rcm[ty[i],ty[j]],(i,j)->uc(σm[ty[i],ty[j]],εm[ty[i],ty[j]],rcm[ty[i],ty[j]]))
        expected=shift_sum(X,L,rc,sh)
        @test isapprox(sum(EA)-sum(EB),expected;rtol=1e-10,atol=1e-10)
        # stepping: identical trajectories, the energy offset follows the pair count
        sa=SC.nve(A); sb=SC.nve(B)
        for i in 1:30
            SC.step!(A,sa,A.dt;compute_energy=i%10==0); SC.step!(B,sb,B.dt;compute_energy=i%10==0)
        end
        DE.synchronize_state!(A); DE.synchronize_state!(B)
        @test same(pos(A),pos(B);tol=1e-10) && same(F(A),F(B);tol=1e-9)
        expected=shift_sum(pos(B),L,rc,sh)   # positions agree to ~1e-10: energies to ~1e-8 of the offset
        @test isapprox(sum(Array(A.Epot))-sum(Array(B.Epot)),expected;rtol=1e-7)
    end
    @testset "ELL engine: single-lane fast kernel (N = 314k, low density, evaluation)" begin
        n=68; X,L=jittered(n,2.2;seed=8); N=size(X,2); V=zeros(3,N)
        A=pair(:none,:uniform,X,V,L); B=pair(:potential_shift,:uniform,X,V,L)
        # evaluations (and unfused steps) at this size take the single-lane kernel
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test B.engine.settings.force_lanes==1 && B.engine.run.model isa DE.UniformLJ && !DE._generic_pairs(B.engine.run)
        @test same(FA,FB) && same(WA,WB)
        # per particle: E_none − E_shift = ½ u(r_c) × (neighbors inside r_c), an integer multiple
        k=(EA.-EB)./(0.5*uc(1.0,1.0,2.5))
        @test maximum(abs.(k.-round.(k)))<1e-6 && all(round.(k).>=0) && sum(round.(k))>0
    end
    @testset "cluster backend: charged melt, per-pair cutoffs" begin
        # rock salt (charges on the two sublattices), small jitter: a stable start
        n=12; a=(1/32.07)^(1/3); rng=MersenneTwister(7)
        sites=[(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]
        X=hcat([collect((s.+0.5).*a).+0.01 .*randn(rng,3) for s in sites]...); L=fill(n*a,3); N=size(X,2)
        q=[iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty=[iseven(sum(s)) ? 1 : 2 for s in sites]
        σm=[0.33284 0.38647; 0.38647 0.440104]; e12=sqrt(0.0115897*0.4184); εm=[0.0115897 e12; e12 0.4184]
        rcm=[0.9 0.95; 0.95 1.0]
        states=map((:none,:potential_shift)) do m
            st=SC.build_simulation(;N,box=Tuple(L),nonbonded=:lj,cutoff=1.0,skin=0.187,cap=Int32(600),epsilon=0.4184,
                sigma=0.440104,mass=29.0,dt=0.002,precision=:f64,charges=q,spatial_reorder=false,
                neigh_interval=10_000,vdw_modifier=m)
            place!(st,X,zeros(3,N)); copyto!(st.typeid,Int32.(ty))
            st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(rcm)
            attach_electrostatics!(st,ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,
                split=ESX.GaussianSplit(3.2),window=ESX.BSplineWindow(6),mesh=(32,32,32),influence=:spme))
            st
        end
        A,B=states
        @test nonbonded_backend(B)===:cluster && vdw_modifier(B)===:potential_shift
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test same(FA,FB) && same(WA,WB)
        expected=shift_sum(X,L,(i,j)->rcm[ty[i],ty[j]],(i,j)->uc(σm[ty[i],ty[j]],εm[ty[i],ty[j]],rcm[ty[i],ty[j]]))
        @test isapprox(sum(EA)-sum(EB),expected;rtol=1e-9)
        sa=SC.nve(A); sb=SC.nve(B)
        for i in 1:20
            SC.step!(A,sa,A.dt;compute_energy=i%10==0); SC.step!(B,sb,B.dt;compute_energy=i%10==0)
        end
        DE.synchronize_state!(A); DE.synchronize_state!(B)
        @test B.engine isa DE.StateEngine && B.engine.run.cluster!==nothing
        @test same(pos(A),pos(B);tol=1e-10)
        expected=shift_sum(pos(B),L,(i,j)->rcm[ty[i],ty[j]],(i,j)->uc(σm[ty[i],ty[j]],εm[ty[i],ty[j]],rcm[ty[i],ty[j]]))
        @test isapprox(sum(Array(A.Epot))-sum(Array(B.Epot)),expected;rtol=1e-7)
        # switching the modifier on a live state takes effect at the next evaluation
        EB1=sum(Array(B.Epot))
        set_vdw_modifier!(B,:none)
        EB2,_,_=evaluate(B)
        @test isapprox(sum(EB2)-EB1,expected;rtol=1e-9)
    end
    @testset "arguments" begin
        st=SC.build_simulation(;N=8,box=(6.0,6.0,6.0),cutoff=2^(1/6),skin=0.3,cap=Int32(32),epsilon=1.0,sigma=1.0,
            nonbonded=:wca,precision=:f64,spatial_reorder=false)
        @test vdw_modifier(st)===:none
        @test_throws ArgumentError set_vdw_modifier!(st,:potential_shift)    # WCA is shifted already
        @test_throws ArgumentError set_vdw_modifier!(st,:force_switch)
        @test_throws ArgumentError SC.build_simulation(;N=8,box=(6.0,6.0,6.0),cutoff=2.5,skin=0.3,cap=Int32(32),
            epsilon=1.0,sigma=1.0,nonbonded=:wca,precision=:f64,vdw_modifier=:potential_shift)
    end
end
end
