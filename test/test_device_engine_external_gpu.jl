module DeviceEngineExternalGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# external force providers through the public
# `step!` with a split graph per step (graph segments around each host force call).
# Checks: device vs staged golden records (`engine_golden.jl`) step by step
# for every integrator family (one provider, same calls); provider call pattern
# and energy flags (Euler–Heun:
# midpoint without energy, then final); an analytic harmonic-trap oscillator;
# an independent all-pairs LJ provider against the engine's internal LJ; features
# that combine with providers (masses, unwrapped coordinates, freeze hold,
# collisions); all-pairs states as in the MLIP examples; the freeze tether acts on
# top of a provider and typed bonded terms cannot combine with one;
# a provider may not release the engine mid-step; a throwing provider leaves a state
# that recaptures; device observables.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const BF=ParticleDynamics.BondedForces
const F=ParticleDynamics.Filters
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
mic(x,L)=x-L*round(x/L)

# Harmonic trap to per-particle centers, minimum image (independent physics).
# Records the energy flag of every call.
struct TrapProvider{T,D} <: SC.AbstractExternalPotential
    k::T
    c::NTuple{D,CuArray{T,1}}
    box::NTuple{D,T}
    flags::Vector{Bool}
end
coords(st)=st.rz===nothing ? (st.rx,st.ry) : (st.rx,st.ry,st.rz)
forces(st)=st.rz===nothing ? (st.fx,st.fy) : (st.fx,st.fy,st.fz)
function SC.external_forces!(p::TrapProvider{T,D},st,energy::Bool) where {T,D}
    push!(p.flags,energy)
    for (r,f,c,L) in zip(coords(st),forces(st),p.c,p.box)
        f .= .-p.k .* (r .- c .- L .* round.((r .- c) ./ L))
    end
    if energy
        E=sum(sum(abs2,r .- c .- L .* round.((r .- c) ./ L)) for (r,c,L) in zip(coords(st),p.c,p.box))
        fill!(st.Epot,T(0.5)*p.k*E/length(st.rx)); fill!(st.virial_nonbonded,zero(T))
    end
    return nothing
end

# Independent all-pairs LJ (unshifted, rc) provider: plain minimum image, no lists.
struct LJProvider{T} <: SC.AbstractExternalPotential
    rc2::T
    box::NTuple{3,T}
    e::CuArray{T,1}
end
function _lj_all_kernel!(rx,ry,rz,fx,fy,fz,e,L::NTuple{3,T},rc2::T) where {T}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    N=length(rx); i>N && return nothing
    ax=zero(T); ay=zero(T); az=zero(T); u=zero(T)
    @inbounds for j in 1:N
        j==i && continue
        dx=rx[i]-rx[j]; dx-=L[1]*round(dx/L[1])
        dy=ry[i]-ry[j]; dy-=L[2]*round(dy/L[2])
        dz=rz[i]-rz[j]; dz-=L[3]*round(dz/L[3])
        r2=dx*dx+dy*dy+dz*dz
        if r2<rc2
            s6=one(T)/(r2*r2*r2); fr=T(24)*s6*(T(2)*s6-one(T))/r2
            ax+=fr*dx; ay+=fr*dy; az+=fr*dz; u+=T(2)*s6*(s6-one(T))
        end
    end
    @inbounds begin fx[i]=ax; fy[i]=ay; fz[i]=az; e[i]=u end
    return nothing
end
function SC.external_forces!(p::LJProvider{T},st,energy::Bool) where {T}
    N=length(st.rx)
    @cuda threads=128 blocks=cld(N,128) _lj_all_kernel!(st.rx,st.ry,st.rz,st.fx,st.fy,st.fz,p.e,p.box,p.rc2)
    if energy
        fill!(st.Epot,sum(p.e)/N); fill!(st.virial_nonbonded,zero(T))
    end
    return nothing
end

function spec0(kind,st,T;dt=st.dt)   # deterministic / noise-free variants (stochastic specs are built for dt)
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.7),temperature=T(0))
    kind===:baoa && return SC.baoa(st;gamma=T(0.7),temperature=T(0))
    kind===:vv && return SC.velocityverlet(st;gamma=T(0.7),temperature=T(0))
    kind===:csvr && return SC.csvr(st;temperature=T(1),tau=T(1e30))
    kind===:nhc && return SC.nosehooverchain(st;temperature=T(1),tau=T(0.2))
    kind===:em && return SC.eulermaruyama(st;gamma=T(2),temperature=T(0),dt)
    return SC.eulerheun(st;gamma=T(2),temperature=T(0),dt)
end

"""A (device engine) and B (staged), each with its own trap provider (same centers)."""
function trap_states(D,::Type{T};N=125,k=20.0,use_neighborlist=true,masses=false,unwrapped=false,
                     collisions=false,dt=0.002) where {T}
    pos,L=lattice(D,N,1.12,T); L=max(L,T(12)); vel=velocities(D,N,0.5,T)
    m=masses ? T[0.5+1.5*(0.5+0.5*sin(1.3i)) for i in 1:N] : T(1)
    map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(dt),mass=m,precision=T===Float64 ? :f64 : :f32,
            spatial_reorder=false,use_neighborlist,unwrapped_positions=unwrapped)
        place!(st,D,pos,vel)
        if collisions
            copyto!(st.typeid,Int32[1+(i%2) for i in 1:N]); ParticleDynamics.enable_collision_counting!(st;ntypes=2)
        end
        SC.attach_external_potential!(st,TrapProvider(T(k),map(copy,coords(st)),ntuple(_->T(L),D),Bool[]))
        st
    end
end

const COMPARED=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot,:dq,:dU,:rx_unwrap,:ry_unwrap)
compare(A,B;tol)=for name in COMPARED
    a=getfield(A,name); a===nothing && continue
    @test relerr(Array(a),Array(getfield(B,name)))<=tol
end
compare(A,ref::AbstractDict;tol)=for name in COMPARED
    a=getfield(A,name); a===nothing && continue
    @test relerr(Array(a),field(ref,name))<=tol
end
advance!(A,sa,n;every=10)=for i in 1:n; SC.step!(A,sa,A.dt;compute_energy=i%every==0); end
staged_advance!(B,sb,n;every=10)=for i in 1:n; SC.run_integrator_step!(B,sb,B.dt;compute_energy=i%every==0); end

@testset "external force providers on the device engine" begin
@golden_testset :engine "Provider: device vs staged golden ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc,:em,:heun), D in (2,3)
    T=Float64; dt=kind in (:em,:heun) ? 1e-3 : 0.002
    A,B=trap_states(D,T;dt); sa=spec0(kind,A,T;dt)
    @test DE.engine_supported(A,sa)
    SC.step!(A,sa,dt;compute_energy=false)
    @test A.engine isa DE.StateEngine && DE.has_feature(A.engine.run,DE.ExternalPotentialFeature)
    @test A.engine.plain isa DE.SplitStep && length(A.engine.plain.segments)==(kind===:heun ? 3 : 2)
    p=A.external_potential; n0=length(p.flags)
    advance!(A,sa,99)
    ref=staged_reference("external_provider_$(kind)_$(D)d") do
        sb=spec0(kind,B,T;dt)
        SC.run_integrator_step!(B,sb,dt;compute_energy=false); staged_advance!(B,sb,99)
        B
    end
    per=kind===:heun ? 2 : 1                                        # Euler–Heun: midpoint and final
    @test length(p.flags)-n0==99*per
    want=vcat(([fill(false,per-1);i%10==0] for i in 1:99)...)        # energy only at the final evaluation
    @test p.flags[n0+1:end]==want
    compare(A,ref;tol=1e-12)
end

@golden_testset :engine "All-pairs provider states (MLIP examples' layout): device vs staged golden ($kind)" for kind in (:nve,:baoab,:heun)
    T=Float64; dt=kind===:heun ? 1e-3 : 0.002
    A,B=trap_states(3,T;use_neighborlist=false,dt); sa=spec0(kind,A,T;dt)
    @test A.nbh isa ParticleDynamics.NeighborLists.AllPairsNeighborMatrix && DE.engine_supported(A,sa)
    advance!(A,sa,100)
    ref=staged_reference("external_allpairs_$(kind)") do
        sb=spec0(kind,B,T;dt); staged_advance!(B,sb,100); B
    end
    @test A.engine isa DE.StateEngine && !DE._list_needed(A.engine.run)   # no list maintenance captured
    compare(A,ref;tol=1e-12)
end

@testset "Analytic harmonic trap (NVE, $(D)D)" for D in (2,3)
    T=Float64; k=20.0; ω=sqrt(k); dt=0.002
    A,_=trap_states(D,T;k,dt); spec=SC.nve(A)
    c=[Array(x) for x in A.external_potential.c]; v0=[Array(x) for x in (A.vx,A.vy,A.vz)[1:D]]; L=Float64(A.external_potential.box[1])
    E=Float64[]
    for n in 1:300
        SC.step!(A,spec,dt;compute_energy=n%50==0)
        n%50==0 && push!(E,SC.collect_step_observables(A,spec).Etot)
        if n in (100,250)                                            # r = c + (v0/ω) sin ωt per component
            r=[Array(x) for x in coords(A)]
            @test maximum(abs(mic(r[d][i]-c[d][i],L)-v0[d][i]/ω*sin(ω*n*dt)) for d in 1:D, i in 1:length(A.rx))<=
                1e-3*maximum(abs,reduce(vcat,v0))/ω
        end
    end
    @test maximum(abs.(E.-E[1]))<=1e-4*abs(E[1])
    o=DE.observable_status(A.engine.run)                               # device observables: provider energy
    @test isapprox(o.potential,sum(Array(A.Epot));rtol=1e-12)
end

@testset "Independent all-pairs LJ provider = engine's internal LJ (NVE)" begin
    T=Float64; N=216; pos,L=lattice(3,N,1.12,T); L=max(L,T(12)); vel=velocities(3,N,0.5,T)
    mk()=place!(SC.build_simulation(;N,box=ntuple(_->T(L),3),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),precision=:f64,spatial_reorder=false),3,pos,vel)
    A=mk(); B=mk()
    SC.attach_external_potential!(A,LJProvider(T(6.25),ntuple(_->T(L),3),CUDA.zeros(T,N)))
    sa=SC.nve(A); sb=SC.nve(B)
    for i in 1:50; SC.step!(A,sa,A.dt;compute_energy=i%10==0); SC.step!(B,sb,B.dt;compute_energy=i%10==0); end
    @test A.engine isa DE.StateEngine && B.engine isa DE.StateEngine && A.engine.plain isa DE.SplitStep
    for name in (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz); @test relerr(Array(getfield(A,name)),Array(getfield(B,name)))<=1e-10; end
    @test isapprox(sum(Array(A.Epot)),sum(Array(B.Epot));rtol=1e-11)
end

@golden_testset :engine "Features with a provider: device vs staged golden ($label)" for (key,label,kw,kind) in (
        (:masses_baoab,"per-particle masses, BAOAB",(;masses=true),:baoab),
        (:unwrapped_nve,"unwrapped coordinates, NVE",(;unwrapped=true),:nve),
        (:collisions_nve,"collision counting, NVE",(;collisions=true),:nve),
        (:collisions_baoab,"collision counting, BAOAB (split A1 sample)",(;collisions=true),:baoab),
        (:hold_nve,"freeze hold, NVE",(;),:nve))
    T=Float64; A,B=trap_states(3,T;kw...)
    if key===:hold_nve
        for st in (A,B); F.freeze_particles!(st;filter=F.Indices(1:7:length(st.rx)),mode=:hold); end
    end
    sa=spec0(kind,A,T)
    @test DE.engine_supported(A,sa)
    advance!(A,sa,100)
    ref=staged_reference("external_features_$(key)") do
        sb=spec0(kind,B,T); staged_advance!(B,sb,100)
        B,(coll_counts=get(kw,:collisions,false) ? Array(B.coll_counts) : Float64[],)
    end
    @test A.engine isa DE.StateEngine
    compare(A,ref;tol=1e-12)
    if get(kw,:collisions,false)
        @test DE._list_needed(A.engine.run)
        @test Float64.(vec(Array(A.coll_counts)))==extra(ref,:coll_counts)
    end
end

# User decision (11 Oct): restraints survive on top of a provider; force-field terms
# (bonded) do not combine with one.
@golden_testset :engine "Freeze tether on top of a provider: device vs staged ($kind)" for kind in (:nve,:heun)
    T=Float64; dt=kind===:heun ? 1e-3 : 0.002
    A,B=trap_states(3,T;dt); N=length(A.rx); frozen=1:7:N
    for st in (A,B); F.freeze_particles!(st;filter=F.Indices(frozen),mode=:spring,k=5.0); end
    anchors=[Array(x) for x in coords(A)]
    sa=spec0(kind,A,T;dt)
    @test DE.engine_supported(A,sa)
    advance!(A,sa,100)
    ref=staged_reference("external_tether_$(kind)") do
        sb=spec0(kind,B,T;dt); staged_advance!(B,sb,100); B
    end
    @test A.engine isa DE.StateEngine && DE.has_feature(A.engine.run,DE.FreezeFeature)
    compare(A,ref;tol=1e-12)
    if kind===:nve                                         # forces = trap + tether at the final positions
        p=A.external_potential; L=Float64(p.box[1]); r=[Array(x) for x in coords(A)]
        c=[Array(x) for x in p.c]; f=[Array(x) for x in forces(A)]
        tether=[i in frozen ? 5.0 : 0.0 for i in 1:N]
        want=[[-p.k*mic(r[d][i]-c[d][i],L)-tether[i]*mic(r[d][i]-anchors[d][i],L) for i in 1:N] for d in 1:3]
        @test maximum(maximum(abs.(f[d].-want[d])) for d in 1:3)<=1e-10*max(1.0,maximum(maximum(abs,w) for w in want))
    end
end

@testset "Typed bonded terms and a provider do not combine (attach errors)" begin
    T=Float64; A,_=trap_states(3,T); N=length(A.rx)
    terms()=BF.BondedTerms((BF.BondedFamily(BF.HarmonicAngleKind(),BF.build_interaction_table(N,[(Int32(1),Int32(2),Int32(3))],[1]),
        (CuArray(T[10.0]),CuArray(T[2.0])),T,3;name=:angle),),3)
    @test_throws ErrorException SC.attach_bonded_terms!(A,terms())                 # provider first
    p=A.external_potential; SC.detach_external_potential!(A)
    SC.attach_bonded_terms!(A,terms())
    @test_throws ErrorException SC.attach_external_potential!(A,p)                 # bonded terms first
end

mutable struct ReentrantProvider <: SC.AbstractExternalPotential
    armed::Bool
    inner::Bool
end
function SC.external_forces!(p::ReentrantProvider,st,energy::Bool)
    if p.armed && !p.inner
        p.inner=true
        try SC.evaluate_forces_into_f!(st,false) finally p.inner=false end
    end
    foreach(f->fill!(f,0),forces(st)); energy && (fill!(st.Epot,0); fill!(st.virial_nonbonded,0))
    return nothing
end
mutable struct FlakyProvider <: SC.AbstractExternalPotential
    inner::TrapProvider{Float64,3}
    fail_at::Int
    calls::Int
end
function SC.external_forces!(p::FlakyProvider,st,energy::Bool)
    p.calls+=1
    p.calls==p.fail_at && error("provider failure")
    return SC.external_forces!(p.inner,st,energy)
end

@testset "A provider may not release the engine; a throwing provider recaptures" begin
    T=Float64; A,_=trap_states(3,T); spec=SC.nve(A)
    p=ReentrantProvider(false,false); SC.attach_external_potential!(A,p)
    SC.step!(A,spec,A.dt)
    @test A.engine isa DE.StateEngine
    p.armed=true
    err=try SC.step!(A,spec,A.dt); nothing catch e; e end
    @test err isa ErrorException && occursin("must not evaluate forces",err.msg)
    @test A.engine isa DE.StateEngine && !A.force_valid
    p.armed=false; SC.step!(A,spec,A.dt); @test A.force_valid                    # recaptured
    B,_=trap_states(3,T); spec=SC.nve(B)
    q=FlakyProvider(B.external_potential,5,0); SC.attach_external_potential!(B,q)
    for _ in 1:3; SC.step!(B,spec,B.dt); end                                     # capture call + 3 steps
    @test_throws ErrorException SC.step!(B,spec,B.dt)                           # 5th call fails
    @test !B.force_valid
    for _ in 1:5; SC.step!(B,spec,B.dt); end                                     # recaptures and continues
    @test B.force_valid && all(isfinite,Array(B.rx))
    DE.synchronize_state!(B)                                                     # device step count consistent
end
end # testset
end
