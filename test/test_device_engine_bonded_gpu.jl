module DeviceEngineBondedGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# legacy harmonic/FENE bonds, typed bonded families
# and bonded exclusions through the public `step!` (bonded states are unsorted).
# Checks: device vs staged golden records (`engine_golden.jl`) step by step
# for every integrator family, pair model and combination of terms; an
# independent Float64 host reference (all
# pairs minus excluded bonded pairs, plus the bond terms) for forces, energy and
# virials, including every force-lane variant of the cooperative pair kernel;
# NVE energy and momentum conservation; device global observables include the
# bonded virial; FENE-domain and typed-geometry errors surface at the host
# boundary with the staged messages, the device state stopping where the staged
# path throws. States are built with spatial sorting on, so the engine sorts them
# (topology keeps particle ids, mapped through `tag`/`slot_of`); records compare
# in particle-id order.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const BF=ParticleDynamics.BondedForces
const NL=ParticleDynamics.NeighborLists
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

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

# Four-atom molecules (bonds 1-2, 2-3, 3-4; angles 1-2-3, 2-3-4; torsion 1-2-3-4)
# on a lattice of spacing 3.2σ. No triple is collinear, so angles and torsions are
# well defined; 2D uses the planar projection. Ids are storage slots (unsorted).
const SHAPE=((0.0,0.0,0.0),(1.0,0.05,0.0),(1.45,0.92,0.1),(2.4,1.05,0.65))
function molecules(D,::Type{T};m=D==2 ? 6 : 4,spacing=3.2) where {T}
    nmol=m^D; N=4nmol; L=m*spacing
    pos=[zeros(T,N) for _ in 1:D]
    for c in 0:nmol-1, a in 1:4, d in 1:D
        i=4c+a; cell=mod(c÷m^(d-1),m)
        pos[d][i]=T((cell+0.15)*spacing-L/2+SHAPE[a][d]+0.03*sin(1.7i+d))
    end
    bonds=Tuple{Int32,Int32}[(Int32(4c+a),Int32(4c+a+1)) for c in 0:nmol-1 for a in 1:3]
    return pos,T(L),N,bonds
end

function typed_terms(D,::Type{T},N) where {T}
    nmol=N÷4; fams=Any[]
    table(K,starts,ptype)=BF.build_interaction_table(N,[ntuple(a->Int32(s+a-1),K) for s in starts],ptype)
    bs=[4c+a for c in 0:nmol-1 for a in 1:3]; as=[4c+a for c in 0:nmol-1 for a in 1:2]; ts=[4c+1 for c in 0:nmol-1]
    push!(fams,BF.BondedFamily(BF.HarmonicBondKind(),table(2,bs,[isodd(s) ? 1 : 2 for s in bs]),
        (CuArray(T[60.0,80.0]),CuArray(T[1.0,1.05])),T,D;name=:typed_bond))
    push!(fams,BF.BondedFamily(BF.HarmonicAngleKind(),table(3,as,ones(Int,length(as))),
        (CuArray(T[30.0]),CuArray(T[2.0])),T,D;name=:angle))
    push!(fams,BF.BondedFamily(BF.CosineSquaredAngleKind(),table(3,as,ones(Int,length(as))),
        (CuArray(T[10.0]),CuArray(T[cos(1.9)])),T,D;name=:cos2_angle))
    if D==3
        push!(fams,BF.BondedFamily(BF.FourierTorsionKind(),table(4,ts,ones(Int,length(ts))),
            (CuArray(Int32[0]),CuArray(Int32[2]),CuArray(T[1.3,-0.4]),CuArray(T[1.0,3.0]),CuArray(T[0.25,1.0])),T,D;
            name=:torsion))
        push!(fams,BF.BondedFamily(BF.HarmonicImproperKind(),table(4,ts,ones(Int,length(ts))),
            (CuArray(T[5.0]),CuArray(T[0.4])),T,D;name=:improper))
    end
    return BF.BondedTerms(Tuple(fams),D)
end

# A: device engine (public step!), B: staged. `style` selects uniform, mixed-σ or
# pair-matrix coefficients; `legacy`/`typed` select the bonded terms.
function bonded_states(D,::Type{T};pot=:lj,bonding=harmonic_bond(k=100.0,r0=1.0),exclude=true,legacy=true,
                       typed=false,style=:uniform,kick=nothing) where {T}
    pos,L,N,bonds=molecules(D,T); vel=velocities(D,N,0.5,T)
    kick===nothing || kick(vel)
    factor=T(pot===:wca ? 2^(1/6) : 2.5); σp=T[1+0.18*sin(1.7i) for i in 1:N]
    σm=T[1.0 1.1; 1.1 1.2]; εm=T[1.0 0.8; 0.8 1.2]; rcm=factor.*σm
    reach=style===:mixed ? factor*maximum(σp) : style===:matrix ? maximum(rcm) : nothing
    map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.001),precision=T===Float64 ? :f64 : :f32,nonbonded=pot,
            bonds=legacy ? bonds : nothing,bonding=legacy ? bonding : nothing,exclude_bonded_pairs=exclude,
            neighbor_cutoff=reach===nothing ? nothing : T(reach))
        if style===:mixed
            st.sigma_particle=CuArray(σp); st.rcut_factor=factor
        elseif style===:matrix
            st.typeid=CuArray(Int32[1+(i%2) for i in 1:N])
            st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(rcm)
        end
        typed && SC.attach_bonded_terms!(st,typed_terms(D,T,N))
        place!(st,D,pos,vel)
    end
end

const COMPARED=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot,:virial,:virial_bonded,:virial_tensor,:dq,:dU)
compare(A,ref::AbstractDict;tol)=for name in COMPARED
    a=getfield(A,name); a===nothing && continue
    @test relerr(ordered(A,a),field(ref,name))<=tol
end
features(st)=Set(nameof(typeof(f)) for f in st.engine.run.features)

# Independent Float64 host reference: LJ (unshifted) or WCA (shifted by ε) over
# all pairs with minimum image, skipping bonded pairs when excluded, plus harmonic
# (½k(r−r0)²) or FENE (−½kR0² ln(1−r²/R0²)) bonds. Returns forces (D×N), total
# energy and the virial traces Σ r_ij·F_ij of the nonbonded and bonded terms.
function host_reference(pos,L,bonds;pot,bond,exclude,rc=2.5)
    D=length(pos); N=length(pos[1]); F=zeros(D,N); U=0.0; Wnb=0.0; Wb=0.0
    bonded=Set((min(Int(i),Int(j)),max(Int(i),Int(j))) for (i,j) in bonds)
    rc2=pot===:wca ? 2^(1/3) : rc^2
    mi(x)=x-L*round(x/L)
    for i in 1:N-1, j in i+1:N
        exclude && (i,j) in bonded && continue
        d=[mi(Float64(pos[k][i])-Float64(pos[k][j])) for k in 1:D]; r2=sum(abs2,d)
        r2<rc2 || continue
        s6=1/r2^3; fr=24*s6*(2s6-1)/r2
        U+=4*s6*(s6-1)+(pot===:wca ? 1.0 : 0.0); Wnb+=fr*r2
        F[:,i].+=fr.*d; F[:,j].-=fr.*d
    end
    bond===nothing && return F,U,Wnb,Wb
    kind,k,r0=bond
    for (i,j) in bonds
        d=[mi(Float64(pos[q][i])-Float64(pos[q][j])) for q in 1:D]; r2=sum(abs2,d); r=sqrt(r2)
        if kind===:harmonic
            fr=-k*(r-r0)/r; U+=0.5k*(r-r0)^2
        else
            fr=-k/(1-r2/r0^2); U+=-0.5k*r0^2*log(1-r2/r0^2)
        end
        Wb+=fr*r2; F[:,i].+=fr.*d; F[:,j].-=fr.*d
    end
    return F,U,Wnb,Wb
end

@testset "bonded terms and exclusions on the device engine" begin
@golden_testset :engine "All terms: device vs staged golden ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc,:em,:heun), D in (2,3)
    T=Float64; A,B=bonded_states(D,T;typed=true)
    sa=spec0(kind,A,T)
    @test DE.engine_supported(A,sa)
    for n in 1:100; SC.step!(A,sa,A.dt;compute_energy=n%10==0); end
    ref=staged_reference("bonded_all_$(kind)_$(D)d") do
        sb=spec0(kind,B,T)
        for n in 1:100; SC.run_integrator_step!(B,sb,B.dt;compute_energy=n%10==0); end
        B
    end
    @test A.engine isa DE.StateEngine
    @test issubset((:BondExclusionFeature,:BondFeature,:BondedTermsFeature),features(A))
    compare(A,ref;tol=1e-9)
end

@golden_testset :engine "Terms and pair models: device vs staged golden (NVE, $(D)D, $label)" for D in (2,3), (key,label,kw) in (
        (:lj_harmonic_excl,"LJ + harmonic, excluded",(;)),
        (:lj_harmonic,"LJ + harmonic, not excluded",(;exclude=false)),
        (:wca_fene,"WCA + FENE (Kremer–Grest)",(;pot=:wca,bonding=fene_bond(k=30.0,r0=1.5),exclude=false)),
        (:wca_fene_excl_typed,"WCA + FENE excluded + typed",(;pot=:wca,bonding=fene_bond(k=30.0,r0=1.5),typed=true)),
        (:lj_typed,"LJ + typed only",(;legacy=false,typed=true)),
        (:softrep_harmonic_excl,"soft repulsive + harmonic, excluded",(;pot=:soft_repulsive)),
        (:mixed_lj_harmonic_excl,"mixed-σ LJ + harmonic, excluded",(;style=:mixed)),
        (:matrix_wca_harmonic_excl,"pair-matrix WCA + harmonic, excluded",(;pot=:wca,style=:matrix)))
    T=Float64; A,B=bonded_states(D,T;kw...)
    sa=SC.nve(A)
    for n in 1:100; SC.step!(A,sa,A.dt;compute_energy=n%10==0); end
    ref=staged_reference("bonded_terms_$(key)_$(D)d") do
        sb=SC.nve(B)
        for n in 1:100; SC.run_integrator_step!(B,sb,B.dt;compute_energy=n%10==0); end
        B
    end
    @test A.engine isa DE.StateEngine
    compare(A,ref;tol=1e-9)
end

@testset "Independent host reference: forces, energy, virials ($(D)D, $pot)" for D in (2,3), pot in (:lj,:wca)
    T=Float64
    bonding,bond,exclude=pot===:lj ? (harmonic_bond(k=100.0,r0=1.0),(:harmonic,100.0,1.0),true) :
                                     (fene_bond(k=30.0,r0=1.5),(:fene,30.0,1.5),false)
    A,_=bonded_states(D,T;pot,bonding,exclude)
    spec=SC.nve(A)
    for n in 1:30; SC.step!(A,spec,A.dt;compute_energy=n==30); end
    @test A.engine isa DE.StateEngine
    # particle-id order (the engine sorts these states; the bond list holds ids)
    pos=[ordered(A,x) for x in (A.rx,A.ry,A.rz)[1:D]]; L=Float64(A.box2===nothing ? A.box3[1] : A.box2[1])
    _,_,_,bonds=molecules(D,T)
    F,U,Wnb,Wb=host_reference(pos,L,bonds;pot,bond,exclude)
    Fd=vcat((ordered(A,x)' for x in (A.fx,A.fy,A.fz)[1:D])...)
    @test maximum(abs.(Fd.-F))<=1e-10*maximum(abs.(F))
    @test isapprox(sum(Array(A.Epot)),U;rtol=1e-11)
    @test isapprox(sum(Array(A.virial_bonded)[:,1:D]),Wb;rtol=1e-11)
    @test isapprox(sum(Array(A.virial)),Wnb+Wb;rtol=1e-11)
end

@testset "Exclusions in every cooperative force-lane variant ($(D)D, G=$G)" for D in (2,3), G in (1,2,4,8)
    T=Float64; pos,L,N,bonds=molecules(D,T); bl=BF.build_bondlist(N,bonds)
    mk(features)=DE.Run(Tuple(CuArray.(pos)),ntuple(_->CUDA.zeros(T,N),D),ntuple(_->L,D);cutoff=2.5,skin=0.3,
        cap=96,dt=0.001,force_lanes=G,features)
    run=mk((DE.BondExclusionFeature(bl.index,bl.flat,bl.counts),))
    @test DE._generic_pairs(run)
    F,_,_,_=host_reference(pos,L,bonds;pot=:lj,bond=nothing,exclude=true)
    Fd=vcat((Array(x)' for x in run.f)...)
    @test maximum(abs.(Fd.-F))<=1e-10*maximum(abs.(F))
    plain=mk(())                                                  # the exclusions matter here
    @test maximum(abs.(vcat((Array(x)' for x in plain.f)...).-F))>1.0
end

# WCA: continuous at its cutoff. Truncated, unshifted LJ jumps by U(rc) at every
# cutoff crossing (measured 11 Oct: drift/<K> 1.3e-2 in 2D, 3.1e-3 in 3D, identical
# on both paths); WCA gives 1.0e-4 / 2.5e-5.
@testset "NVE conservation with all terms (engine, WCA, $(D)D)" for D in (2,3)
    T=Float64; A,_=bonded_states(D,T;pot=:wca,typed=true); spec=SC.nve(A)
    P0=[sum(Array(v)) for v in (A.vx,A.vy,A.vz)[1:D]]; E=Float64[]; K=Float64[]
    for n in 1:2000
        SC.step!(A,spec,A.dt;compute_energy=n%100==0)
        if n%100==0
            o=SC.collect_step_observables(A,spec); push!(E,o.Etot); push!(K,o.Ekin_total)
        end
    end
    @test A.engine isa DE.StateEngine
    @test maximum(abs.(E.-E[1]))<=5e-4*mean(K)                                       # no drift
    P=[sum(Array(v)) for v in (A.vx,A.vy,A.vz)[1:D]]
    @test maximum(abs.(P.-P0))<=1e-9*length(A.rx)                                    # Newton's third law
end

@testset "Device global observables include the bonded virial ($(D)D, legacy=$legacy)" for D in (2,3), legacy in (true,false)
    T=Float64; A,_=bonded_states(D,T;legacy,typed=true); spec=SC.nve(A)
    for n in 1:20; SC.step!(A,spec,A.dt;compute_energy=n==20); end
    o=DE.observable_status(A.engine.run)
    vt=sum(Float64.(Array(A.virial_tensor));dims=1)
    want=D==3 ? vec(vt) : [vt[1],vt[2],0.0,vt[3],0.0,0.0]
    @test maximum(abs.(o.virial.-want))<=1e-11*maximum(abs.(want))
    @test isapprox(o.potential,sum(Array(A.Epot));rtol=1e-12)
    @test maximum(abs.(Array(A.virial_bonded)))>0                                   # bonded rows present
end

@golden_testset :engine "FENE domain errors surface at the boundary with the staged message ($(D)D)" for D in (2,3)
    T=Float64; fene=fene_bond(k=30.0,r0=1.5)
    # at capture: bond 1–2 stretched beyond R0
    A,B=bonded_states(D,T;pot=:wca,bonding=fene,exclude=false)
    for st in (A,B); x=Array(st.rx); x[2]=x[1]+1.6; copyto!(st.rx,x); SC.invalidate_forces!(st); end
    ea=try SC.step!(A,SC.nve(A),A.dt); nothing catch err; err end
    ref=staged_reference("bonded_fene_capture_$(D)d") do
        eb=try SC.run_integrator_step!(B,SC.nve(B),B.dt); nothing catch err; err end
        B,(error_type=string(typeof(eb)),message=sprint(showerror,eb))
    end
    @test ea isa DomainError && extra(ref,:error_type)=="DomainError" && sprint(showerror,ea)==extra(ref,:message)
    # during the run: atoms 1 and 2 pulled apart; the device stops at the staged failure
    pull(v)=(v[1][1]=-40.0; v[1][2]=40.0; v)
    A,B=bonded_states(D,T;pot=:wca,bonding=fene,exclude=false,kick=pull)
    sa=SC.nve(A)
    ref=staged_reference("bonded_fene_run_$(D)d") do
        sb=SC.nve(B); failed=0; eb=nothing
        for n in 1:200
            try
                SC.run_integrator_step!(B,sb,B.dt;compute_energy=false)
            catch err
                failed=n; eb=err; break
            end
        end
        B,(failed=failed,error_type=string(typeof(eb)),message=eb===nothing ? "" : sprint(showerror,eb))
    end
    failed=Int(extra(ref,:failed))
    @test failed>1 && extra(ref,:error_type)=="DomainError"
    for _ in 1:failed+5; SC.step!(A,sa,A.dt;compute_energy=false); end              # no host boundary in between
    ea=try DE.synchronize_state!(A); nothing catch err; err end
    @test ea isa DomainError && sprint(showerror,ea)==extra(ref,:message)
    for name in (:rx,:ry,:vx,:vy,:fx,:fy)
        @test relerr(ordered(A,getfield(A,name)),field(ref,name))<=1e-12               # state at the failing step
    end
    @test_throws ErrorException NL.device_neighbor_status(A.engine.run.neighbors)  # generic latch message
end

@golden_testset :engine "Typed-geometry errors surface with the staged message ($(D)D)" for D in (2,3)
    T=Float64; A,B=bonded_states(D,T;typed=true)
    for st in (A,B)                                                                 # atoms 1–2–3 collinear
        cs=[Array(x) for x in (st.rx,st.ry,st.rz)[1:D]]
        for a in 2:3; cs[1][a]=cs[1][1]+(a-1); for d in 2:D; cs[d][a]=cs[d][1]; end; end
        for (x,c) in zip((st.rx,st.ry,st.rz)[1:D],cs); copyto!(x,c); end
        SC.invalidate_forces!(st)
    end
    ea=try SC.step!(A,SC.nve(A),A.dt); nothing catch err; err end
    ref=staged_reference("bonded_typed_geometry_$(D)d") do
        eb=try SC.run_integrator_step!(B,SC.nve(B),B.dt); nothing catch err; err end
        B,(error_type=string(typeof(eb)),message=sprint(showerror,eb))
    end
    @test ea isa DomainError && extra(ref,:error_type)=="DomainError" && sprint(showerror,ea)==extra(ref,:message)
end

@testset "sorted bonded states = unsorted, in particle-id order ($(D)D)" for D in (2,3)
    # A sorts (spatial_reorder on), U does not; same molecules, legacy bonds, typed
    # families and exclusions. Topology tables keep particle ids in both.
    T=Float64; pos,L,N,bonds=molecules(D,T;m=D==2 ? 12 : 7); vel=velocities(D,N,0.5,T)
    A,U=map((true,false)) do sorted
        st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.001),precision=:f64,
            bonds,bonding=harmonic_bond(k=100.0,r0=1.0),exclude_bonded_pairs=true,spatial_reorder=sorted)
        SC.attach_bonded_terms!(st,typed_terms(D,T,N))
        place!(st,D,pos,vel)
    end
    @test A.tag!==nothing && U.tag===nothing && DE.engine_supported(A,SC.nve(A))
    for n in 1:300
        SC.step!(A,SC.nve(A),A.dt;compute_energy=n%50==0); SC.step!(U,SC.nve(U),U.dt;compute_energy=n%50==0)
    end
    DE.synchronize_state!(A); DE.synchronize_state!(U)
    @test DE.ordering_status(A.engine.run).count>0 && Array(A.tag)!=1:N                 # the engine did sort
    @test Array(A.slot_of)[Array(A.tag)]==1:N                                          # slot_of inverts tag
    for name in (:rx,:ry,:vx,:vy,:fx,:fy,:Epot,:virial_bonded)
        @test relerr(ordered(A,getfield(A,name)),ordered(U,getfield(U,name)))<=1e-9
    end
    @test isapprox(total_potential_energy(A),total_potential_energy(U);rtol=1e-11)
    # a fresh evaluation of the sorted state (engine; topology mapped through tag/slot_of)
    FA=ordered(A,A.fx); SC.evaluate_forces_into_f!(A,true)
    @test relerr(ordered(A,A.fx),FA)<=1e-12
end
end # testset
end
