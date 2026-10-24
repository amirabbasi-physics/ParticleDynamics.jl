module ClusterBondExclusionsGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Lennard-Jones bond exclusions on the cluster-pair backend
# (charged molecules). The tile mask is shared with the Coulomb exceptions, so the checks
# cover every combination: bonded pairs excluded from Coulomb (scale 0), bonded pairs with
# full Coulomb (restored with scale 1 plus the potential shift), and scaled Coulomb
# exceptions that are not bonded (their LJ restored).
# - Charged chains: forces, energies and virials with exclusions minus those without equal
#   minus the host LJ of the bonded pairs (Coulomb unchanged, shift included).
# - Neutral bonded LJ: the cluster backend equals the ELL engine's bond-exclusion path.
# - Stepping on the graph engine equals the staged cluster step.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
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

# Rock-salt lattice, x fastest: consecutive indices along a row are nearest neighbours with
# opposite charges; chains of 4 along x. Exceptions: the first bond of each chain (scale 0)
# and the 1–3 pairs (scale 0.5, not bonded); the other bonds keep full Coulomb.
const n=12; const a=(1/32.07)^(1/3)
function system(;seed=7)
    rng=MersenneTwister(seed)
    sites=[(i,j,k) for k in 0:n-1 for j in 0:n-1 for i in 0:n-1]
    X=hcat([collect((s.+0.5).*a).+0.01 .*randn(rng,3) for s in sites]...)
    q=[iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty=Int32[iseven(sum(s)) ? 1 : 2 for s in sites]
    N=length(sites); bonds=Tuple{Int32,Int32}[]; ex=Tuple{Int,Int,Float64}[]
    for c in 0:N÷4-1
        b=4c
        append!(bonds,[(Int32(b+1),Int32(b+2)),(Int32(b+2),Int32(b+3)),(Int32(b+3),Int32(b+4))])
        push!(ex,(b+1,b+2,0.0),(b+1,b+3,0.5),(b+2,b+4,0.5))
    end
    return X,q,ty,bonds,ex
end
const σm=[0.33284 0.38647; 0.38647 0.440104]
const εm=let e12=sqrt(0.0115897*0.4184); [0.0115897 e12; e12 0.4184] end
function state(X,q,ty,bonds,ex;exclude=true,charged=true,backend=:cluster,modifier=:none)
    N=size(X,2); T=Float64
    st=SC.build_simulation(;N,box=ntuple(_->n*a,3),nonbonded=:lj,cutoff=1.0,skin=0.187,cap=Int32(600),
        epsilon=0.4184,sigma=0.440104,mass=29.0,dt=0.002,precision=:f64,charges=charged ? q : nothing,
        spatial_reorder=false,neigh_interval=10_000,bonds,bonding=ParticleDynamics.harmonic_bond(k=2000.0,r0=a),
        exclude_bonded_pairs=exclude,vdw_modifier=modifier)
    for (d,arr) in enumerate((st.rx,st.ry,st.rz)); copyto!(arr,X[d,:]); end
    for arr in (st.vx,st.vy,st.vz); fill!(arr,0.0); end
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(fill(1.0,2,2))
    if charged
        spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.2),
            window=ESX.BSplineWindow(6),mesh=(40,40,40),influence=:spme)
        attach_electrostatics!(st,spec;exceptions=ex)
    end
    backend===:cluster && nonbonded_backend(st)!==:cluster && set_nonbonded_backend!(st,:cluster)
    return st
end
"""Host LJ of the bonded pairs: energy, forces (3×N), virial (xx yy zz xy xz yz)."""
function bonded_lj(X,ty,bonds;shift=false)
    N=size(X,2); L=n*a; E=0.0; Fh=zeros(3,N); W=zeros(6)
    for (i,j) in bonds
        d=X[:,i].-X[:,j]; d.-=L.*round.(d./L); r2=sum(abs2,d)
        r2<1.0 || continue
        s=σm[ty[i],ty[j]]; e=εm[ty[i],ty[j]]; s6=(s^2/r2)^3
        E+=4e*(s6^2-s6)-(shift ? 4e*(s^12-s^6) : 0.0)
        f=24e*(2s6^2-s6)/r2
        Fh[:,i].+=f.*d; Fh[:,j].-=f.*d
        W.+=f.*[d[1]^2,d[2]^2,d[3]^2,d[1]*d[2],d[1]*d[3],d[2]*d[3]]
    end
    return E,Fh,W
end

@testset "LJ bond exclusions on the cluster backend" begin
    X,q,ty,bonds,ex=system()
    @testset "charged chains: exclusions remove exactly the bonded LJ (modifier $m)" for m in (:none,:potential_shift)
        A=state(X,q,ty,bonds,ex;modifier=m); B=state(X,q,ty,bonds,ex;exclude=false,modifier=m)
        @test nonbonded_backend(A)===:cluster
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        Eb,Fb,Wb=bonded_lj(X,ty,bonds;shift=m===:potential_shift)
        # the harmonic bond terms are identical in both states and cancel
        @test isapprox(EB-EA,Eb;rtol=1e-9)
        @test relerr(FB.-FA,Fb)<=1e-9
        @test relerr(WB.-WA,Wb)<=1e-8
    end
    @testset "neutral bonded LJ: cluster backend = ELL engine" begin
        A=state(X,q,ty,bonds,ex;charged=false,backend=:cluster)
        B=state(X,q,ty,bonds,ex;charged=false,backend=:ell)
        @test nonbonded_backend(A)===:cluster && nonbonded_backend(B)===:ell
        EA,FA,WA=evaluate(A); EB,FB,WB=evaluate(B)
        @test A.engine isa DE.StateEngine && B.engine isa DE.StateEngine
        @test isapprox(EA,EB;rtol=1e-11) && relerr(FA,FB)<=1e-11 && relerr(WA,WB)<=1e-11
    end
    @golden_testset :engine "graph steps = staged cluster steps (NVE, 40 steps)" begin
        A=state(X,q,ty,bonds,ex); C=state(X,q,ty,bonds,ex)
        sa=SC.nve(A); sc=SC.nve(C)
        for i in 1:40; SC.step!(A,sa,A.dt;compute_energy=i%10==0); end
        DE.synchronize_state!(A)
        ref=staged_reference("cluster_bond_exclusions_nve") do
            for i in 1:40; SC.run_integrator_step!(C,sc,C.dt;compute_energy=i%10==0); end; C
        end
        rp=vcat((field(ref,f)' for f in (:rx,:ry,:rz))...); rf=vcat((field(ref,f)' for f in (:fx,:fy,:fz))...)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        @test relerr(pos(A),rp)<=1e-10 && relerr(F(A),rf)<=1e-8
        @test isapprox(sum(Array(A.Epot)),sum(field(ref,:Epot));rtol=1e-10)
    end
end
end
