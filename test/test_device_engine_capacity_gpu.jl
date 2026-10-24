module DeviceEngineCapacityGPUChecks
# Device-engine neighbor-capacity policy: the user's
# `cap` is a floor, and a capacity overflow is recovered exactly at the next host
# boundary (grow the list, recapture, redo the failed rebuild from its section,
# replay the skipped steps with their energy flags). Checks: a run whose list is
# tightened to its current maximum row (so it overflows and recovers) matches a
# twin with ample capacity (same seeds, same extra rebuild) to 1e-9 for every
# integrator family in 2D and 3D, the fused NVE path, the BAOAB family sampling
# after A1 (two sections), Euler–Heun (two sections) and split host-force steps
# with a kept list; the user's `cap` is honored as a floor. Not bitwise: the cell
# scatter is atomic, so row order (summation order) differs between any two runs;
# a missed, doubled or mis-seeded step would differ at order one.
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const NL=ParticleDynamics.NeighborLists
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))

function spec_t(kind,st,T;dt)   # with temperature: the noise path is replayed too
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.7),temperature=T(1))
    kind===:baoa && return SC.baoa(st;gamma=T(0.7),temperature=T(1))
    kind===:vv && return SC.velocityverlet(st;gamma=T(0.7),temperature=T(1))
    kind===:csvr && return SC.csvr(st;temperature=T(1),tau=T(0.5))
    kind===:nhc && return SC.nosehooverchain(st;temperature=T(1),tau=T(0.2))
    kind===:em && return SC.eulermaruyama(st;gamma=T(2),temperature=T(1),dt)
    return SC.eulerheun(st;gamma=T(2),temperature=T(1),dt)
end

"""Two identical sorted LJ states (a melting lattice, so rows keep changing)."""
function twin(D,::Type{T};N=D==2 ? 400 : 512,dt=0.002,collisions=false,reorder=true) where {T}
    pos,L=lattice(D,N,1.05,T); L=max(L,T(12)); vel=velocities(D,N,1.5,T)
    map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(dt),precision=T===Float64 ? :f64 : :f32,
            spatial_reorder=reorder,reorder_interval=7)
        place!(st,D,pos,vel)
        if collisions
            copyto!(st.typeid,Int32[1+(i%2) for i in 1:N]); ParticleDynamics.enable_collision_counting!(st;ntypes=2)
        end
        st
    end
end

const COMPARED=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot,:dq,:dU)
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
close(A,B;tol=1e-9)=all(getfield(A,n)===nothing || relerr(ordered(A,getfield(A,n)),ordered(B,getfield(B,n)))<=tol for n in COMPARED)

"""Capture both with equal seeds, tighten A (B gets the same extra rebuild), run with
host boundaries every 25 steps; A must recover and stay equal to B (1e-9)."""
function overflow_run!(A,B,sa,sb;steps=300,every=10,boundary=25)
    dt=A.dt
    Random.seed!(0xCA9); SC.step!(A,sa,dt;compute_energy=false)
    Random.seed!(0xCA9); SC.step!(B,sb,dt;compute_energy=false)
    DE.tighten_capacity!(A); DE.force_rebuild!(B)
    for n in 1:steps
        SC.step!(A,sa,dt;compute_energy=n%every==0); SC.step!(B,sb,dt;compute_energy=n%every==0)
        if n%boundary==0
            SC.collect_step_observables(A,sa); SC.collect_step_observables(B,sb)
        end
    end
    DE.synchronize_state!(A); DE.synchronize_state!(B)
    return A.engine.recoveries
end

# Trap to per-particle anchors (minimum image), for split steps that keep a list.
struct Trap{T,D} <: SC.AbstractExternalPotential
    k::T
    c::NTuple{D,CuArray{T,1}}
    L::T
end
function SC.external_forces!(p::Trap{T,D},st,energy::Bool) where {T,D}
    for (r,f,c) in zip((st.rx,st.ry,st.rz)[1:D],(st.fx,st.fy,st.fz)[1:D],p.c)
        f .= .-p.k .* (r .- c .- p.L .* round.((r .- c) ./ p.L))
    end
    energy && (fill!(st.Epot,zero(T)); fill!(st.virial_nonbonded,zero(T)))
    return nothing
end

@testset "Device engine: neighbor-capacity floor and exact overflow recovery" begin
@testset "Overflow recovery is exact ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc,:em,:heun), D in (2,3)
    # Overdamped runs need dt = 1e-4 on this dense lattice (1e-3 is unstable: the twins
    # diverged without any recovery, and EM 2D blew up; 11 Oct). They move less per
    # step, so they run longer to reach rebuilds that overflow.
    brownian=kind in (:em,:heun); T=Float64; dt=brownian ? 1e-4 : 0.002
    A,B=twin(D,T;dt); sa=spec_t(kind,A,T;dt); sb=spec_t(kind,B,T;dt)
    @test overflow_run!(A,B,sa,sb;steps=brownian ? 1500 : 300,boundary=brownian ? 75 : 25)>=1
    @test close(A,B)
    oa=DE.observable_status(A.engine.run); ob=DE.observable_status(B.engine.run)
    @test oa.step==ob.step && isapprox(oa.total,ob.total;rtol=1e-9)
    kind===:nve && D==3 && @test A.engine.settings.fused                   # the fused path recovered too
end

@testset "Two-section steps: BAOAB sampling after A1 (collisions)" begin
    T=Float64; A,B=twin(3,T;collisions=true); sa=spec_t(:baoab,A,T;dt=0.002); sb=spec_t(:baoab,B,T;dt=0.002)
    @test overflow_run!(A,B,sa,sb)>=1
    @test close(A,B) && Array(A.coll_counts)==Array(B.coll_counts)
end

@testset "Split host-force steps with a kept list ($kind)" for kind in (:nve,:heun)
    T=Float64; dt=kind===:heun ? 1e-3 : 0.002
    A,B=twin(3,T;dt,collisions=true,reorder=false)                         # providers index by slot
    for st in (A,B); SC.attach_external_potential!(st,Trap(T(2),map(copy,(st.rx,st.ry,st.rz)),T(st.box3[1]))); end
    sa=spec_t(kind,A,T;dt); sb=spec_t(kind,B,T;dt)
    @test DE.engine_supported(A,sa)
    @test overflow_run!(A,B,sa,sb)>=1
    @test A.engine.plain isa DE.SplitStep && DE._list_needed(A.engine.run)
    @test close(A,B) && Array(A.coll_counts)==Array(B.coll_counts)
end

@testset "The user's cap is a floor (scaled to the engine's list radius)" begin
    T=Float32; N=1000; pos,L=lattice(3,N,1.2,T); L=max(L,T(12)); vel=velocities(3,N,0.5,T)
    st=place!(SC.build_simulation(;N,box=ntuple(_->T(L),3),cutoff=T(2.5),skin=T(0.3),cap=Int32(400),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),precision=:f32),3,pos,vel)
    SC.step!(st,SC.nve(st),st.dt)
    reach=2.5+st.engine.settings.skin
    @test Int(st.engine.run.neighbors.list.cap)>=ceil(Int,400*(reach/2.8)^3)
end
end # capacity testset
end
