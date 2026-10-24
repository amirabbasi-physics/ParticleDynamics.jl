module DeviceEngineMassesFreezeGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# per-particle masses and frozen particles
# (hold and spring) through the public `step!`, with spatial sorting on.
# Checks: noise-free/deterministic runs against staged golden records (
# `engine_golden.jl`) for every integrator family; mass-weighted equipartition
# (force-free BAOAB); frozen
# particles stay at their anchors (hold); tether force and energy (spring);
# freeze expiry; thermostat dof exclude frozen particles.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const F=ParticleDynamics.Filters
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

function pair_states(D,::Type{T};masses=false,N=257) where {T}
    pos,L=lattice(D,N,1.12,T); L=max(L,T(12)); vel=velocities(D,N,0.5,T)
    m=masses ? T[0.5+1.5*(0.5+0.5*sin(1.3i)) for i in 1:N] : T(1)
    map(1:2) do k
        st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),mass=m,precision=T===Float64 ? :f64 : :f32,
            spatial_reorder=k==1,reorder_interval=1,unwrapped_positions=true)
        place!(st,D,pos,vel)
    end
end

compare(A,ref::AbstractDict,D;tol=1e-9)=for n in (:rx,:ry,:vx,:vy,:fx,:fy,:Epot,:Ekin,:dq,:dU,:rx_unwrap)
    @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
end

@testset "masses and freezing on the device engine" begin
@golden_testset :engine "Per-particle masses: device vs staged golden ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc), D in (2,3)
    T=Float64; A,B=pair_states(D,T;masses=true)
    sa=spec0(kind,A,T)
    @test A.mass_particle!==nothing && DE.engine_supported(A,sa)
    for _ in 1:100; SC.step!(A,sa,A.dt;compute_energy=true); end
    ref=staged_reference("masses_$(kind)_$(D)d") do
        sb=spec0(kind,B,T)
        for _ in 1:100; SC.run_integrator_step!(B,sb,B.dt;compute_energy=true); end
        rec=golden_snapshot(B)                                                               # B is unsorted
        rec,(mass=Array(B.mass_particle),obs=SC.collect_step_observables(B,sb))
    end
    @test A.engine isa DE.StateEngine && DE.has_feature(A.engine.run,DE.MassFeature)
    @test DE.ordering_status(A.engine.run).count>=1
    compare(A,ref,D)
    @test relerr(ordered(A,A.mass_particle),extra(ref,:mass))==0                     # masses followed sorting
    oa=SC.collect_step_observables(A,sa)
    @test isapprox(oa.Ekin_total,extra(ref,:obs)["Ekin_total"];rtol=1e-9)
end

@testset "Mass-weighted equipartition, force-free BAOAB ($(D)D)" for D in (2,3)
    T=Float32; m0=D==2 ? 64 : 16; N=m0^D; L=T(3m0)
    ms=T[isodd(i) ? 0.5 : 4.0 for i in 1:N]
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(5e-3),mass=ms,precision=:f32)
    place!(st,D,[T[3*(mod((i-1)÷m0^(d-1),m0))-L/2+T(1.5) for i in 1:N] for d in 1:D],[zeros(T,N) for _ in 1:D])
    spec=SC.baoab(st;gamma=T(2),temperature=T(1.5))
    for _ in 1:1000; SC.step!(st,spec,st.dt;compute_energy=false); end      # γt = 10 (light), 1.25 per m for heavy
    for _ in 1:1500; SC.step!(st,spec,st.dt;compute_energy=false); end
    for (g,m) in ((1:2:N,0.5),(2:2:N,4.0))
        mv2=Float64[]
        for x in (st.vx,st.vy,st.vz)[1:D]; append!(mv2,m.*abs2.(ordered(st,x)[g])); end
        @test abs(mean(mv2)/1.5-1)<5*sqrt(2/length(mv2))
    end
end

@golden_testset :engine "Freeze $mode: device vs staged golden ($kind, $(D)D)" for mode in (:hold,:spring), kind in (:nve,:baoab,:csvr,:nhc,:em,:heun), D in (2,3)
    T=Float64; A,B=pair_states(D,T)
    N=length(A.rx); frozen=collect(1:5:N)
    for st in (A,B); F.freeze_particles!(st;filter=F.Indices(frozen),mode,k=mode===:spring ? 20.0 : 0,steps=60); end
    anchors=[Array(x)[frozen] for x in (A.rx,A.ry,A.rz)[1:D]]
    dt=kind in (:em,:heun) ? 1e-3 : 0.002
    sa=spec0(kind,A,T;dt)
    @test DE.engine_supported(A,sa)
    for n in 1:100
        SC.step!(A,sa,dt;compute_energy=true)
        if n==40
            @test DE.has_feature(A.engine.run,DE.FreezeFeature)
            if mode===:hold                                                         # frozen particles at anchors
                for d in 1:D; @test ordered(A,(A.rx,A.ry,A.rz)[d])[frozen]==anchors[d]; end
            end
            if kind in (:csvr,:nhc)
                @test Array(sa.workspace.dof_per_bath)==[D*(N-length(frozen))]
            end
        end
    end
    ref=staged_reference("freeze_$(mode)_$(kind)_$(D)d") do
        sb=spec0(kind,B,T;dt); dof=Float64[]
        for n in 1:100
            SC.run_integrator_step!(B,sb,dt;compute_energy=true)
            n==40 && kind in (:csvr,:nhc) && (dof=Array(sb.workspace.dof_per_bath))
        end
        B,(freeze_mode=Int(B.freeze_mode),dof=dof)
    end
    kind in (:csvr,:nhc) && @test extra(ref,:dof)==[D*(N-length(frozen))]              # staged dof while frozen
    @test A.freeze_mode==SC.FREEZE_NONE && extra(ref,:freeze_mode)==Int(SC.FREEZE_NONE)  # expired after 60 steps
    @test !DE.has_feature(A.engine.run,DE.FreezeFeature)                           # recaptured without an active freeze
    compare(A,ref,D;tol=1e-8)
end

# Regression (11 Oct 2026): the tether used r − anchor of wrapped positions, so a
# tethered particle crossing a periodic face felt ≈ k·L (blow-up at N = 1M), and
# hold reset unwrapped coordinates to the wrapped anchors (lost image counts).
# Non-interacting particles start δ inside a face, moving out along one axis.
function boundary_states(D,::Type{T};delta,v=1.0) where {T}
    L=T(12); g=T[-4.5,-1.5,1.5,4.5]
    pos=[T[] for _ in 1:D]; vel=[T[] for _ in 1:D]
    for d in 1:D, s in (1,-1), a in (s==1 ? g[[1,3]] : g[[2,4]]), b in (D==3 ? g : T[0])
        o=filter(!=(d),1:D); p=zeros(T,D); q=zeros(T,D)
        p[d]=s*(L/2-T(delta)); p[o[1]]=a; D==3 && (p[o[2]]=b); q[d]=s*T(v)
        for k in 1:D; push!(pos[k],p[k]); push!(vel[k],q[k]); end
    end
    N=length(pos[1])
    map(1:2) do k
        st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=5,
            nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(0.002),mass=T(1),
            precision=T===Float64 ? :f64 : :f32,spatial_reorder=k==1,reorder_interval=1,unwrapped_positions=true)
        place!(st,D,pos,vel)
    end
end
wrapped(st,D)=[ordered(st,x) for x in (st.rx,st.ry,st.rz)[1:D]]
unwrapped(st,D)=[ordered(st,x) for x in (st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]]
mic(x,L)=x-L*round(x/L)

@golden_testset :engine "Freeze across periodic faces: $mode ($(D)D)" for mode in (:spring,:hold), D in (2,3)
    T=Float64; L=12.0; k=20.0; ω=sqrt(k); v=1.0; dt=0.002
    A,B=boundary_states(D,T;delta=mode===:spring ? 0.05 : 0.001,v); N=length(A.rx)
    side=[sign(ordered(A,(A.vx,A.vy,A.vz)[d])[i]) for i in 1:N, d in 1:D]           # ±1 on the moving axis
    for st in (A,B), (d,u) in enumerate((st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D])
        u.+=CuArray(T[L*(mod(i+d,3)-1) for i in 1:N])                               # earlier face crossings
    end
    anchors=wrapped(A,D); u0=unwrapped(A,D)
    for st in (A,B); F.freeze_particles!(st;filter=F.All(),mode,k=mode===:spring ? k : 0); end
    sa=SC.nve(A); E=Float64[]; crossed=false
    ref=staged_reference("freeze_faces_$(mode)_$(D)d") do
        sb=SC.nve(B); EB=Float64[]
        for n in 1:400
            SC.run_integrator_step!(B,sb,dt;compute_energy=n%50==0)
            n%50==0 && push!(EB,SC.collect_step_observables(B,sb).Etot)   # also fills st.Ekin
        end
        B,(EB=EB,)
    end
    EB=extra(ref,:EB)
    for n in 1:400
        SC.step!(A,sa,dt;compute_energy=n%50==0)
        n%50==0 && push!(E,SC.collect_step_observables(A,sa).Etot)        # as on the staged twin: fills st.Ekin
        if mode===:spring && n in (25,100,175,250)                                  # harmonic: d = s(v/ω)sin(ωt)
            r=wrapped(A,D)
            dev=maximum(abs(mic(r[d][i]-anchors[d][i],L)-(side[i,d]!=0)*side[i,d]*v/ω*sin(ω*n*dt)) for i in 1:N, d in 1:D)
            @test dev<1e-3*v/ω
            n==175 && (crossed=all(any(side[i,d]!=0 && sign(r[d][i])!=side[i,d] for d in 1:D) for i in 1:N))
        end
    end
    mode===:spring && @test crossed                                                  # every tethered particle crossed its face
    mode===:spring && @test maximum(abs.(E.-E[1]))<1e-4*abs(E[1])                     # VV oscillation ≤ (ωdt)²/4 = 2e-5
    mode===:spring && @test maximum(abs.(EB.-EB[1]))<1e-4*abs(EB[1])
    @test maximum(abs.(E.-EB))<=1e-9*maximum(abs.(EB))
    if mode===:hold                                                                  # held particles at anchors, no unwrapped jump
        @test wrapped(A,D)==anchors
        @test maximum(maximum(abs.(a.-b)) for (a,b) in zip(unwrapped(A,D),u0))<1e-10
    end
    uw=[(unwrapped(A,D),wrapped(A,D)),
        ([field(ref,n) for n in (:rx_unwrap,:ry_unwrap,:rz_unwrap)[1:D]],[field(ref,n) for n in (:rx,:ry,:rz)[1:D]])]
    for (us,rs) in uw, (u,r) in zip(us,rs)                                            # unwrapped − wrapped ∈ L·ℤ
        @test maximum(abs.((u.-r)./L.-round.((u.-r)./L)))<1e-9
    end
    compare(A,ref,D;tol=1e-9)
end
end # testset
end
