module DeviceEngineEvaluationGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Forces and energies at the
# current configuration from the engine that owns the state, outside `step!`
# (`evaluate_configuration!`, `ensure_energies!`, `total_potential_energy`, …).
# Checks: the evaluation runs on the engine (no release, no recapture, storage
# order unchanged) and matches the staged driver (golden records of a twin that
# stepped on the engine and was then evaluated by the staged driver;
# `engine_golden.jl`) for every pair model in 2D and 3D, with bonded terms and
# exclusions, the freeze tether and an external provider; the trajectory continues
# as if no evaluation had happened; a rigid translation beyond skin/2 rebuilds the
# list without sorting and leaves the energy unchanged; a capacity overflow in that
# rebuild is repaired; invalidated forces are refreshed and untethered evaluations
# use an evaluation engine without changing the state's tether.
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const NL=ParticleDynamics.NeighborLists
const F=ParticleDynamics.Filters
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
const FIELDS=(:fx,:fy,:fz,:Epot,:virial,:virial_nonbonded,:virial_bonded,:virial_tensor)
function same(A,ref;tol)
    for n in FIELDS
        a=getfield(A,n); (a===nothing || !haskey(ref,String(n))) && continue
        @test relerr(ordered(A,a),field(ref,n))<=tol
    end
end

function twins(D,::Type{T},pot;N=257,k=2,kw...) where {T}
    a=pot===:soft_repulsive ? 0.9 : 1.12
    pos,_=lattice(D,N,a,T); vel=velocities(D,N,0.5,T)
    map(1:k) do _
        st=build(D,T,pot,N;a,kw...)
        place!(st,D,pos,vel)
    end
end
advance!(st,spec,n)=for _ in 1:n; SC.step!(st,spec,st.dt;compute_energy=false); end
# The reference: the staged driver at the same configuration (releases the engine).
staged!(st;energy=true,spring=st.force_freeze_spring)=SC.evaluate_forces_into_f!(st,energy;freeze_spring=spring)

# Harmonic trap to per-particle anchors (minimum image); counts provider calls.
struct Trap{T,D} <: SC.AbstractExternalPotential
    k::T
    c::NTuple{D,CuArray{T,1}}
    L::T
    calls::Base.RefValue{Int}
end
function SC.external_forces!(p::Trap{T,D},st,energy::Bool) where {T,D}
    p.calls[]+=1
    for (r,f,c) in zip((st.rx,st.ry,st.rz)[1:D],(st.fx,st.fy,st.fz)[1:D],p.c)
        f .= .-p.k .* (r .- c .- p.L .* round.((r .- c) ./ p.L))
    end
    if energy
        fill!(st.Epot,zero(T))
        for (r,c) in zip((st.rx,st.ry,st.rz)[1:D],p.c)
            st.Epot .+= T(0.5) .* p.k .* abs2.(r .- c .- p.L .* round.((r .- c) ./ p.L))
        end
        fill!(st.virial_nonbonded,zero(T))
    end
    return nothing
end

@testset "engine evaluation outside step!" begin

@golden_testset :engine "Engine evaluation = staged driver ($pot, $(D)D, $T)" for pot in (:lj,:wca,:soft_repulsive), D in (2,3), T in (Float64,Float32)
    A,B=twins(D,T,pot); sa=SC.nve(A)
    advance!(A,sa,30)
    e=A.engine; g=e.plain; tags=Array(A.tag)
    @test e isa DE.StateEngine && !A.energy_valid
    U=total_potential_energy(A)                                 # engine
    @test A.engine===e && e.plain===g && A.energy_valid          # kept: no release, no recapture
    @test Array(A.tag)==tags                                     # storage order unchanged
    ref=staged_reference("eval_$(pot)_$(D)d_$(T)") do
        advance!(B,SC.nve(B),30); staged!(B); @assert B.engine===nothing
        B,(U=total_potential_energy(B),W=total_virial_trace(B))
    end
    # Staged 2D WCA/soft-repulsive kernels subtract, then wrap, in Float32: their
    # forces are less accurate than the device's (test_device_engine_2d_gpu.jl
    # measures both against Float64; same 5e-5 allowance there).
    tol=T===Float64 ? 1e-12 : (D==2 && pot in (:wca,:soft_repulsive) ? 5e-5 : 1e-5)
    same(A,ref;tol)
    @test isapprox(U,extra(ref,:U);rtol=tol)
    @test isapprox(total_virial_trace(A),extra(ref,:W);rtol=tol)
end

@testset "The trajectory continues as if no evaluation happened ($(D)D)" for D in (2,3)
    T=Float64; A,C=twins(D,T,:lj); sa=SC.nve(A); sc=SC.nve(C)
    advance!(A,sa,5); advance!(C,sc,5)
    e=A.engine; g=e.plain
    for n in 1:60
        n%7==0 && total_potential_energy(A)
        n%11==0 && SC.evaluate_configuration!(A,SC.EvaluationRequest(virial=true))
        SC.step!(A,sa,A.dt;compute_energy=false); SC.step!(C,sc,C.dt;compute_energy=false)
    end
    @test A.engine===e && e.plain===g                            # never recaptured
    for (a,c) in zip((A.rx,A.ry,A.rz,A.vx,A.vy,A.vz),(C.rx,C.ry,C.rz,C.vx,C.vy,C.vz))
        a===nothing && continue
        @test relerr(ordered(A,a),ordered(C,c))<=1e-10
    end
end

@golden_testset :engine "Bonded terms and exclusions ($(D)D)" for D in (2,3)
    T=Float64; N=D==2 ? 196 : 343; m=round(Int,N^(1/D))
    # Bonds inside lattice rows (x-index pairs 0–1, 2–3, 4–5), so no bond spans a row end.
    bonds=Tuple{Int32,Int32}[(Int32(i),Int32(i+1)) for i in 1:N-1 if iseven(mod(i-1,m)) && mod(i-1,m)<m-1]
    A,B=map(1:2) do _
        pos,L=lattice(D,N,1.12,T); L=max(L,T(12))
        st=SC.build_simulation(;N,box=ntuple(_->T(L),D),cutoff=T(2.5),skin=T(0.3),cap=Int32(160),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.001),precision=:f64,
            bonds,bonding=harmonic_bond(k=100.0,r0=1.0),exclude_bonded_pairs=true)
        place!(st,D,pos,velocities(D,N,0.5,T))
    end
    advance!(A,SC.nve(A),20)
    e=A.engine; @test e isa DE.StateEngine
    total_potential_energy(A); @test A.engine===e
    ref=staged_reference("eval_bonded_$(D)d") do
        advance!(B,SC.nve(B),20); staged!(B); B
    end
    same(A,ref;tol=1e-12)
end

@golden_testset :engine "Freeze tether: engine evaluation, also without the tether" begin
    T=Float64; A,B=twins(3,T,:lj); N=length(A.rx)
    for st in (A,B); F.freeze_particles!(st;filter=F.Indices(collect(1:4:N)),mode=:spring,k=20.0); end
    advance!(A,SC.nve(A),25)
    e=A.engine; @test DE.has_feature(e.run,DE.FreezeFeature) && A.force_freeze_spring
    SC.ensure_energies!(A); @test A.engine===e
    ref=staged_reference("eval_tether") do
        advance!(B,SC.nve(B),25); staged!(B;spring=true); B
    end
    same(A,ref;tol=1e-12)
    Ft=vcat((Array(x)' for x in (A.fx,A.fy,A.fz))...); frozen=Array(A.freeze_mask).!=0
    SC.evaluate_configuration!(A,SC.EvaluationRequest(energy=true);freeze_spring=false)
    @test A.engine isa DE.StateEngine && !A.force_freeze_spring  # untethered, on an evaluation engine
    Fu=vcat((Array(x)' for x in (A.fx,A.fy,A.fz))...)
    @test maximum(abs.(Fu[:,.!frozen].-Ft[:,.!frozen]))<=1e-12*maximum(abs.(Ft))   # only the tethered particles change
    @test maximum(abs.(Fu[:,frozen].-Ft[:,frozen]))>0
    advance!(A,SC.nve(A),5); @test A.engine isa DE.StateEngine && A.force_freeze_spring   # tethered steps recapture
end

@golden_testset :engine "External provider: one host call, engine kept" begin
    T=Float64; N=125; pos,L=lattice(3,N,1.12,T); L=max(L,T(12))
    A,B=map(1:2) do _                                          # unsorted: providers index by slot
        st=SC.build_simulation(;N,box=ntuple(_->T(L),3),cutoff=T(2.5),skin=T(0.3),cap=Int32(96),neigh_interval=5,
            epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),precision=:f64,spatial_reorder=false)
        place!(st,3,pos,velocities(3,N,0.5,T))
        SC.attach_external_potential!(st,Trap(T(2),map(copy,(st.rx,st.ry,st.rz)),T(st.box3[1]),Ref(0)))
        st
    end
    advance!(A,SC.nve(A),15)
    e=A.engine; @test e isa DE.StateEngine && e.plain isa DE.SplitStep
    n0=A.external_potential.calls[]
    SC.ensure_energies!(A)
    @test A.engine===e && A.external_potential.calls[]==n0+1 && A.energy_valid
    ref=staged_reference("eval_provider") do
        advance!(B,SC.nve(B),15); staged!(B); B
    end
    same(A,ref;tol=1e-12)
end

@testset "Rigid translation beyond skin/2: rebuild without sorting, energy unchanged" begin
    T=Float64; A,=twins(3,T,:lj;k=1); s=SC.nve(A); advance!(A,s,20)
    e=A.engine; ws=e.run.neighbors
    U0=total_potential_energy(A); tags=Array(A.tag); builds=NL.device_neighbor_status(ws).builds
    sorts=DE.ordering_status(e.run).count
    for (x,d,L) in zip((A.rx,A.ry,A.rz),(0.5,0.4,0.3),A.box3)
        x .+= T(d); x .-= L .* floor.((x .+ L/2) ./ L)             # wrapped to [-L/2, L/2)
    end
    A.energy_valid=false                                         # edited in place; forces kept valid on purpose
    U1=total_potential_energy(A)
    @test A.engine===e
    @test NL.device_neighbor_status(ws).builds==builds+1         # the displacement check rebuilt the list
    @test DE.ordering_status(e.run).count==sorts && Array(A.tag)==tags   # without sorting
    @test isapprox(U1,U0;rtol=1e-12)
    advance!(A,s,10); @test A.engine===e                         # stepping continues on the same graphs
end

@golden_testset :engine "A capacity overflow in an evaluation rebuild is repaired" begin
    T=Float64; A,B=twins(3,T,:lj); sa=SC.nve(A); advance!(A,sa,20)
    DE.tighten_capacity!(A); cap=Int(A.engine.run.neighbors.list.cap)
    for x in (A.rx,A.ry,A.rz); x .*= T(0.9); end                  # denser rows than the tight list holds
    A.energy_valid=false
    UA=total_potential_energy(A)
    @test A.engine isa DE.StateEngine && Int(A.engine.run.neighbors.list.cap)>cap
    ref=staged_reference("eval_overflow") do
        advance!(B,SC.nve(B),20)
        for x in (B.rx,B.ry,B.rz); x .*= T(0.9); end
        SC.invalidate_forces!(B); UB=total_potential_energy(B); @assert B.engine===nothing
        B,(U=UB,)
    end
    @test isapprox(UA,extra(ref,:U);rtol=1e-12); same(A,ref;tol=1e-12)
    advance!(A,sa,10); @test A.engine isa DE.StateEngine          # the recaptured graphs run
end

@testset "Invalidated forces: an evaluation engine takes the evaluation (the next step recaptures)" begin
    A,=twins(3,Float64,:lj;k=1); s=SC.nve(A); advance!(A,s,5); e=A.engine
    SC.invalidate_forces!(A); total_potential_energy(A)
    @test A.engine isa DE.StateEngine && A.engine!==e && A.engine.spec===nothing && A.energy_valid
    advance!(A,s,5); @test A.engine isa DE.StateEngine && A.engine.spec!==nothing
end
end # testset
end
