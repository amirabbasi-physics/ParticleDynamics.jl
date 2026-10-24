module DeviceEngineCollisionsGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# collision counting as an engine feature. The
# engine (sorting at every rebuild) and the staged path (which never sorts with
# collisions; golden records, `engine_golden.jl`) must count the same contact
# entries per type-pair bin for every integrator family, with uniform and per-type
# contact cutoffs, including the two samples per step of BAOAB/BAOA; handoffs
# between the two paths (live, removed with the staged engine).
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

function spec0(kind,st,T;dt)   # deterministic / noise-free variants
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.3),temperature=T(0),dt)
    kind===:baoa && return SC.baoa(st;gamma=T(0.3),temperature=T(0),dt)
    kind===:gsm && return SC.gsm(st;gamma=T(0.3),temperature=T(0),dt)
    kind===:vv && return SC.velocityverlet(st;gamma=T(0.3),temperature=T(0),dt)
    kind===:csvr && return SC.csvr(st;temperature=T(1),tau=T(1e30))
    kind===:nhc && return SC.nosehooverchain(st;temperature=T(1),tau=T(0.5))
    kind===:em && return SC.eulermaruyama(st;gamma=T(1),temperature=T(0),dt)
    return SC.eulerheun(st;gamma=T(1),temperature=T(0),dt)
end

function pair_states(D,::Type{T},kind;pair_cutoffs=false) where {T}
    N=257; pos,L=lattice(D,N,1.12,T); L=max(L,T(12)); vel=velocities(D,N,kind in (:em,:heun) ? 0.0 : 1.2,T)
    dt=T(kind in (:em,:heun) ? 1e-3 : 0.002)
    map(1:2) do k
        st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt,mass=T(1),precision=T===Float64 ? :f64 : :f32,
            spatial_reorder=true,reorder_interval=1)
        copyto!(st.typeid,Int32[1+(i%2) for i in 1:N])
        place!(st,D,pos,vel)
        ParticleDynamics.enable_collision_counting!(st;ntypes=2)
        pair_cutoffs && ParticleDynamics.set_collision_pair_cutoffs!(st,[1.2 1.5; 1.5 1.8])
        st
    end
end

counts(st)=ParticleDynamics.collisions_read_counts!(st)

@testset "collision counting on the device engine" begin
@golden_testset :engine "Counts: device vs staged golden ($kind, $(D)D, pair cutoffs=$pc)" for kind in (:nve,:baoab,:baoa,:gsm,:vv,:csvr,:nhc,:em,:heun),
                                                                        D in (2,3), pc in (false,true)
    T=Float64; A,B=pair_states(D,T,kind;pair_cutoffs=pc)
    dt=A.dt; sa=spec0(kind,A,T;dt)
    @test DE.engine_supported(A,sa)
    for _ in 1:200; SC.step!(A,sa,dt;compute_energy=false); end
    ref=staged_reference("collisions_$(kind)_$(D)d_pc$(Int(pc))") do
        sb=spec0(kind,B,T;dt)
        for _ in 1:200; SC.run_integrator_step!(B,sb,dt;compute_energy=false); end
        B,(counts=counts(B),)
    end
    @test A.engine isa DE.StateEngine && DE.has_feature(A.engine.run,DE.CollisionFeature)
    @test DE.ordering_status(A.engine.run).count>=1                        # sorted while counting
    ca=counts(A); cb=extra(ref,:counts)
    @test sum(cb)>0
    @test Float64.(vec(ca))==cb
    @test relerr(ordered(A,A.rx),field(ref,:rx))<=1e-9
end

end # testset
end
