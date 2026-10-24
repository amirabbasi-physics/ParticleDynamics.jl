module DeviceEngineELLMeshMethodsGPUChecks
# Charged states on the dense (ELL) list with a CSVR thermostat, for each mesh influence function:
# the forces a step leaves at the final coordinates equal a fresh evaluation there. The fused step's
# gather-kick differentiates the potential grid (16-lane AD gather), so P3M-ik must run unfused
# (22 Oct 2026: fused P3M-ik read the charge grid on steps without energies).
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ESX=ParticleDynamics.Electrostatics
const KB=0.0083144626

function melt(influence,P)
    T=Float32; n=12; a=(1/32.07)^(1/3); L=n*a; N=n^3; rng=MersenneTwister(11)
    sites=[(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]
    R=hcat([[(s[1]+0.5)*a,(s[2]+0.5)*a,(s[3]+0.5)*a].+(0.01 .*randn(rng,3) .- L/2) for s in sites]...)   # centered box
    q=[iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty=Int32[iseven(sum(s)) ? 1 : 2 for s in sites]
    sig=[0.33284 0.38647; 0.38647 0.440104]; e12=sqrt(0.0115897*0.4184); eps=[0.0115897 e12; e12 0.4184]
    st=SC.build_simulation(;N,box=(L,L,L),nonbonded=:lj,cutoff=T(1.0),skin=T(0.187),cap=Int32(600),epsilon=T(0.4184),
        sigma=T(0.440104),mass=T(29),dt=T(0.002),precision=:f32,charges=q,spatial_reorder=false,neigh_interval=10_000)
    copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
    V=randn(rng,3,N).*sqrt(KB*1100/29)
    copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(T.(sig)); st.epsilon_pair=CuArray(T.(eps)); st.rcut_pair=CuArray(fill(T(1.0),2,2))
    spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.2),
        window=ESX.BSplineWindow(P),mesh=(32,32,32),influence=influence)
    attach_electrostatics!(st,spec)
    set_nonbonded_backend!(st,:ell)
    return st
end
F(st)=vcat((Float64.(Array(x))' for x in (st.fx,st.fy,st.fz))...)

@testset "ELL charged CSVR steps: step forces = fresh evaluation, per influence function" begin
    @testset "$influence" for (influence,P) in ((:spme,4),(:p3m_ad,5),(:p3m_ik,5))
        st=melt(influence,P)
        sp=SC.csvr(st;temperature=Float32(KB*1100),tau=Float32(0.1),interval=1)
        for _ in 1:12; SC.step!(st,sp,st.dt;compute_energy=false); end
        DE.synchronize_state!(st)
        @test st.engine isa DE.StateEngine && st.engine.run.cluster===nothing
        @test (st.engine.run.fusion===nothing)==(influence===:p3m_ik)
        Fs=F(st)
        SC.evaluate_forces_into_f!(st,false)
        Fe=F(st)
        @test all(isfinite,Fs)
        @test maximum(abs.(Fs.-Fe))<=1e-4*maximum(abs.(Fe))
    end
end
end
