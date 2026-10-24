module DeviceEngineIdentityGPUChecks
# Particle identity vs storage slot (user requirement, 10 Oct 2026). Spatial
# sorting permutes storage; every per-particle array must follow its particle.
# 1. Guard: on a state with all engine-supported optional fields, every
#    per-particle array of the state and of the integrator spec is permuted by
#    the engine's sorting, or is listed below as staged-only or scratch.
# 2. Invariance: sorting at every rebuild vs no sorting gives the same
#    per-particle results (identity order) for every integrator family with
#    unwrapped coordinates, per-particle masses, a mid-run freeze selected by
#    `Filters.Indices` (particle ids) and OU noise on an id-selected subset.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const F=ParticleDynamics.Filters
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

# Per-particle arrays the engine does not permute, and why. A new per-particle
# field must be added to the engine's sorting or listed here with a reason.
const STAGED_ONLY=(:charge,) # the engine declines these states (coll_prev is per neighbor entry, not per particle)
const SCRATCH=(:rf_x,:rf_y,:rf_z)  # staged noise buffers, redrawn every staged step

function spec_for(kind,st,T;temperature=T(0.5))
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.7),temperature)
    kind===:baoa && return SC.baoa(st;gamma=T(0.7),temperature)
    kind===:vv && return SC.velocityverlet(st;gamma=T(0.7),temperature)
    kind===:csvr && return SC.csvr(st;temperature,tau=T(0.3))
    kind===:nhc && return SC.nosehooverchain(st;temperature,tau=T(0.3))
    kind===:em && return SC.eulermaruyama(st;gamma=T(2),temperature,dt=T(1e-3))
    return SC.eulerheun(st;gamma=T(2),temperature,dt=T(1e-3))
end
stochastic(kind)=kind in (:baoab,:baoa,:vv,:em,:heun)

function make_state(D,::Type{T},kind;sort) where {T}
    N=257; pos,L=lattice(D,N,1.12,T); L=max(L,T(12))
    st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(kind in (:em,:heun) ? 1e-3 : 0.002),
        mass=T[0.6+0.8*(0.5+0.5*sin(1.3i)) for i in 1:N],precision=T===Float64 ? :f64 : :f32,
        spatial_reorder=sort,reorder_interval=1,unwrapped_positions=true)
    return place!(st,D,pos,velocities(D,N,0.6,T))
end

per_particle(x,N)=x isa CuArray && size(x,1)==N

@testset "Identity: per-particle arrays follow their particles" begin
@testset "Guard: every per-particle array is permuted or declared ($kind, $(D)D)" for kind in (:baoab,:vv,:csvr,:nhc,:em), D in (2,3)
    T=Float32; st=make_state(D,T,kind;sort=true); N=length(st.rx)
    F.freeze_particles!(st;filter=F.Indices(1:7:N),mode=:spring,k=5.0)
    ParticleDynamics.enable_collision_counting!(st;ntypes=1)       # snapshot arrays must be sorted too
    spec=spec_for(kind,st,T)
    stochastic(kind) && F.set_ou_spectrum!(spec,st,T[0.1],T[0.01];filter=F.Indices(2:3:N))
    SC.step!(st,spec,st.dt)
    run=st.engine.run; ow=run.ordering
    @test ow!==nothing
    permuted=Set(pointer(a) for a in (run.r...,run.v...,run.f...,run.energy,run.virial,ow.tags,
        ow.extra_real...,ow.extra_int...,ow.extra_mat...))
    unhandled=String[]
    for name in fieldnames(typeof(st))
        x=getfield(st,name)
        per_particle(x,N) || continue
        pointer(x) in permuted || name in STAGED_ONLY || push!(unhandled,"state.$name")
    end
    for (label,obj) in (("params",spec.params),("workspace",hasproperty(spec,:workspace) ? spec.workspace : nothing))
        obj===nothing && continue
        for name in fieldnames(typeof(obj))
            x=getfield(obj,name)
            per_particle(x,N) || continue
            pointer(x) in permuted || name in SCRATCH || push!(unhandled,"spec.$label.$name")
        end
    end
    isempty(unhandled) || println("unhandled per-particle arrays: ",unhandled)
    @test isempty(unhandled)
end

@testset "Sorting on vs off, same per-particle results ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc,:em,:heun), D in (2,3)
    T=Float64; A=make_state(D,T,kind;sort=true); B=make_state(D,T,kind;sort=false)
    N=length(A.rx)
    sa=spec_for(kind,A,T); sb=spec_for(kind,B,T)
    if stochastic(kind)
        F.set_ou_spectrum!(sa,A,T[0.1],T[0.01];filter=F.Indices(2:3:N))
        F.set_ou_spectrum!(sb,B,T[0.1],T[0.01];filter=F.Indices(2:3:N))
        # Initial OU states are drawn with CUDA's RNG (not `Random.seed!`): give both the same.
        SC.ensure_integrator_workspace!(sa,A); SC.ensure_integrator_workspace!(sb,B)
        for f in (:ou_x,:ou_y,:ou_z)
            x=getfield(sa.workspace,f); x===nothing || copyto!(x,getfield(sb.workspace,f))
        end
    end
    Random.seed!(41); SC.step!(A,sa,A.dt)
    Random.seed!(41); SC.step!(B,sb,B.dt)
    for _ in 1:39; SC.step!(A,sa,A.dt); SC.step!(B,sb,B.dt); end
    @test DE.ordering_status(A.engine.run).count>=1 && A.tag!=CuArray(Int32.(1:N))   # A is sorted
    frozen=collect(5:11:N)                                                           # particle ids
    F.freeze_particles!(A;filter=F.Indices(frozen),mode=:hold)
    F.freeze_particles!(B;filter=F.Indices(frozen),mode=:hold)
    anchors=[ordered(A,x)[frozen] for x in (A.rx,A.ry,A.rz)[1:D]]
    Random.seed!(42); SC.step!(A,sa,A.dt)                                            # freeze recaptures (new seed)
    Random.seed!(42); SC.step!(B,sb,B.dt)
    for _ in 1:39; SC.step!(A,sa,A.dt); SC.step!(B,sb,B.dt); end
    for d in 1:D; @test ordered(A,(A.rx,A.ry,A.rz)[d])[frozen]==anchors[d]; end      # the right particles held
    for (a,b) in zip((A.rx,A.ry,A.vx,A.vy,A.fx,A.Epot,A.dq,A.rx_unwrap),(B.rx,B.ry,B.vx,B.vy,B.fx,B.Epot,B.dq,B.rx_unwrap))
        @test relerr(ordered(A,a),ordered(B,b))<=1e-9
    end
end
end
end
