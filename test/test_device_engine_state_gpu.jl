module DeviceEngineStateGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# public `step!` for 3D uniform-LJ NVE runs the device
# engine over the SimulationState's own arrays. Compared against the staged-engine
# golden records and independent CUDA references.
using Test, CUDA, ParticleDynamics, TOML, SHA, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const NL=ParticleDynamics.NeighborLists
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"..","benchmark","gromacs","GPUForceReference.jl"))
const GOLDEN=golden_directory(:deterministic)

function golden(name)
    file=joinpath(GOLDEN,name*".toml")
    sums=Dict(reverse(split(l))[1]=>split(l)[1] for l in eachline(joinpath(GOLDEN,"SHA256SUMS")))
    @test bytes2hex(sha256(read(file)))==sums[name*".toml"] # golden record unchanged
    return TOML.parsefile(file)
end

relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

@golden_testset :deterministic "Public NVE step! matches the staged golden records (3D LJ, F32/F64)" begin
    for T in (Float32,Float64)
        g=golden("det_3d_$(T)_lj_nve")
        rec=deterministic_case(3,T,:lj,:nve)                 # step! now routes to the device engine
        tol=T===Float32 ? 1e-5 : 1e-12
        for stage in ("step1","step50"), k in ("rx","ry","rz","vx","vy","vz","fx","fy","fz","Epot","virial")
            @test relerr(rec[stage][k],g[stage][k])<=tol
        end
        @test rec["step50"]["step"]==50
        for (k,v) in g["step50_observables"]
            haskey(rec["step50_observables"],k) && @test isapprox(rec["step50_observables"][k],v;rtol=T===Float32 ? 1e-4 : 1e-10,atol=1e-10)
        end
    end
end

function lj_state(::Type{T};N=257) where {T}
    pos,_=lattice(3,N,1.12,T); vel=velocities(3,N,0.5,T)
    st=build(3,T,:lj,N;a=1.12)
    place!(st,3,pos,vel)
    return st
end
function forces_vs_reference(st)
    N=length(st.rx); ids=CuArray(Int32.(1:N))
    ref=GPUForceReference.sampled_lj_oracle_gpu(st.rx,st.ry,st.rz,st.box3,ids,st.pair_lj.σ,st.pair_lj.ϵ,st.pair_lj.rcut)
    return norm(permutedims(hcat(st.fx,st.fy,st.fz))-ref.F)/max(norm(ref.F),1)
end

@testset "Engine ownership: capture, sorting, boundaries, handoff and recapture" begin
    for T in (Float32,Float64)
        tol=T===Float32 ? 2e-5 : 3e-12
        st=lj_state(T); spec=SC.nve(st)
        @test DE.engine_supported(st,spec)
        addresses=map(a->a===nothing ? nothing : pointer(a),DE._particle_arrays(st))
        for _ in 1:600; SC.step!(st,spec,st.dt;compute_energy=false); end  # crosses the 500-step sort interval
        e=st.engine
        @test e isa DE.StateEngine && e.launched==600 && st.step==600
        @test map(a->a===nothing ? nothing : pointer(a),DE._particle_arrays(st))==addresses # sorted in place
        @test DE.ordering_status(e.run).count>=2
        @test sort(Array(st.tag))==collect(1:length(st.rx))    # still a permutation
        @test forces_vs_reference(st)<=tol
        obs=SC.collect_step_observables(st,spec)                # boundary check passes
        # Another integrator recaptures on the engine (no staged fallback).
        lang=SC.nosehooverchain(st;temperature=T(1),tau=T(0.5))
        SC.step!(st,lang,st.dt)
        @test st.engine isa DE.StateEngine && st.engine!==e && st.step==601
        @test forces_vs_reference(st)<=tol
        e=st.engine; SC.step!(st,spec,st.dt)
        @test st.engine isa DE.StateEngine && st.engine!==e      # back to NVE: recaptured
        # invalidate_forces! (manual edits) recaptures: no stale first kick.
        e1=st.engine; ParticleDynamics.invalidate_forces!(st); SC.step!(st,spec,st.dt)
        @test st.engine!==e1 && forces_vs_reference(st)<=tol
        # Fingerprint: a different dt recaptures.
        e2=st.engine; SC.step!(st,spec,st.dt/2)
        @test st.engine!==e2 && st.engine.launched==1
        # A latched device error surfaces at the next boundary.
        st.rx[1:1].=T(NaN)
        SC.step!(st,spec,st.dt/2)
        @test_throws ArgumentError SC.collect_step_observables(st,spec)
    end
end

@testset "Charges without a solver run on the engine; unsupported states throw" begin
    # Charges travel with the particles when sorting; long-range electrostatics run on the
    # cluster backend, so an explicit cell-list choice for them is an error (no staged fallback).
    pos,_=lattice(3,257,1.12,Float32)
    stc=SC.build_simulation(;N=257,box=ntuple(_->12f0,3),cutoff=2.5f0,skin=0.3f0,cap=Int32(160),neigh_interval=5,
        epsilon=1f0,sigma=1f0,temperature=0f0,dt=0.002f0,precision=:f32,spatial_reorder=false,
        charges=fill(0.1f0,257))
    place!(stc,3,pos,velocities(3,257,0.5,Float32))
    specc=SC.nve(stc)
    @test DE.engine_supported(stc,specc)
    q0=Array(stc.charge)
    for _ in 1:600; SC.step!(stc,specc,stc.dt;compute_energy=false); end
    @test stc.engine isa DE.StateEngine && stc.tag===nothing   # unsorted here (spatial_reorder=false)
    std=SC.build_simulation(;N=257,box=ntuple(_->12f0,3),cutoff=2.5f0,skin=0.3f0,cap=Int32(160),neigh_interval=5,
        epsilon=1f0,sigma=1f0,temperature=0f0,dt=0.002f0,precision=:f32,charges=Float32[isodd(i) ? 0.1 : -0.1 for i in 1:257])
    place!(std,3,pos,velocities(3,257,0.5,Float32)); qd=Array(std.charge)
    for _ in 1:600; SC.step!(std,SC.nve(std),std.dt;compute_energy=false); end
    @test DE.ordering_status(std.engine.run).count>=1 && Array(std.charge)==qd[Array(std.tag)]   # charges follow the sort
    @test_throws ArgumentError SC.step!(stc,SC.baoab(stc;gamma=0f0,temperature=1f0),stc.dt)   # zero friction: error
    @test DE.engine_settings(10_000,0.3)==(build_lanes=8,force_lanes=8,skin=0.70,fused=true,cell_sub=1)
    @test DE.engine_settings(100_000,0.3).fused==false
end
end
