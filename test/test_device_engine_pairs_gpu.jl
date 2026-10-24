module DeviceEnginePairsGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# WCA, soft repulsion, mixed σ and pair-matrix LJ/WCA
# through the public `step!` (generic cooperative pair kernel). Checked against
# staged golden records (`engine_golden.jl` for mixed σ and pair
# matrices) and an independent Float64 CUDA all-pairs reference.
using Test, CUDA, ParticleDynamics, TOML, SHA, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"pair_reference_gpu.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
const GOLDEN=golden_directory(:deterministic)

relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
function golden(name)
    file=joinpath(GOLDEN,name*".toml")
    sums=Dict(split(l)[2]=>split(l)[1] for l in eachline(joinpath(GOLDEN,"SHA256SUMS")))
    @test bytes2hex(sha256(read(file)))==sums[name*".toml"]
    return TOML.parsefile(file)
end

@golden_testset :deterministic "Public NVE step! matches staged golden records: WCA and soft repulsion (3D)" begin
    for T in (Float32,Float64), pot in (:wca,:soft_repulsive)
        g=golden("det_3d_$(T)_$(pot)_nve")
        rec=deterministic_case(3,T,pot,:nve)
        tol=T===Float32 ? 1e-5 : 1e-12
        for stage in ("step1","step50"), k in ("rx","ry","rz","vx","vy","vz","fx","fy","fz","Epot","virial")
            @test relerr(rec[stage][k],g[stage][k])<=tol
        end
    end
end

# Two identical states: A advanced by public step! (device engine), B by the staged step.
function mixed_pair(::Type{T},pot,style;N=257) where {T}
    pos,_=lattice(3,N,1.12,T); vel=velocities(3,N,0.4,T)
    ids=1:N
    σp=T[1+0.18*sin(1.7i) for i in ids]
    factor=pot===:wca ? T(2)^(one(T)/6) : T(2.5)
    σm=T[1.0 1.1; 1.1 1.2]; εm=T[1.0 0.8; 0.8 1.2]; rcm=factor.*σm
    reach=Float64(style===:mixed ? factor*maximum(σp) : maximum(rcm))
    states=map(1:2) do _
        st=SC.build_simulation(;N,box=ntuple(_->T(12),3),cutoff=T(pot===:wca ? 2^(1/6) : 2.5),skin=T(0.3),
            cap=Int32(200),neigh_interval=5,epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),mass=T(1),
            nonbonded=pot,precision=T===Float64 ? :f64 : :f32,neighbor_cutoff=T(reach))
        if style===:mixed
            st.sigma_particle=CuArray(σp); st.rcut_factor=factor
        else
            st.typeid=CuArray(Int32[1+(i%2) for i in ids])
            st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(rcm)
        end
        place!(st,3,pos,vel)
    end
    return states...,(;σp,factor,σm,εm,rcm)
end

function reference_error(st,pot,style,p)
    N=length(st.rx); ids=CuArray(Int32.(1:N)); kind=pot===:wca ? 2 : 1
    ref=style===:mixed ?
        PairReferenceGPU.reference_forces((st.rx,st.ry,st.rz),st.box3,ids;kind,style=2,epsilon=1.0,
            sigma_particle=st.sigma_particle,factor=p.factor) :
        PairReferenceGPU.reference_forces((st.rx,st.ry,st.rz),st.box3,ids;kind,style=3,typeid=st.typeid,
            sigma_pair=st.sigma_pair,epsilon_pair=st.epsilon_pair,rcut_pair=st.rcut_pair)
    return norm(permutedims(hcat(st.fx,st.fy,st.fz))-ref.F)/max(norm(ref.F),1),
        abs(sum(Float64.(st.Epot))-sum(ref.energy))/max(abs(sum(ref.energy)),1)
end

@golden_testset :engine "Mixed σ and pair-matrix LJ/WCA: device vs staged golden vs independent reference" begin
    for T in (Float32,Float64), pot in (:lj,:wca), style in (:mixed,:matrix)
        A,B,p=mixed_pair(T,pot,style)
        spec=SC.nve(A)
        @test DE.engine_supported(A,spec)
        for _ in 1:50; SC.step!(A,spec,A.dt;compute_energy=true); end
        ref=staged_reference("pairs_$(pot)_$(style)_$(T)") do
            for _ in 1:50; SC.run_integrator_step!(B,SC.nve(B),B.dt;compute_energy=true); end
            B
        end
        @test A.engine isa DE.StateEngine
        tol=T===Float32 ? 1e-4 : 1e-11
        for n in (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Epot)
            @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
        end
        ferr,eerr=reference_error(A,pot,style,p)
        @test ferr<=(T===Float32 ? 2e-5 : 1e-11)
        @test eerr<=(T===Float32 ? 1e-5 : 1e-11)
    end
end

@testset "Neighbor-based dispatch and fast path" begin
    @test DE.engine_settings(10_000,0.3)==(build_lanes=8,force_lanes=8,skin=0.70,fused=true,cell_sub=1)
    # many neighbours per atom (molten NaCl 10.6k: 101): 16 builder lanes on cells of half the list radius
    s=DE.engine_settings(10_648,24.05;cutoff=1.0)
    @test (s.build_lanes,s.force_lanes,s.fused,s.cell_sub)==(16,8,true,2) && s.skin≈0.14
    @test DE.engine_settings(97_336,24.05;cutoff=1.0).cell_sub==1
    s=DE.engine_settings(10_000,0.3;cutoff=2^(1/6)) # WCA at LJ-medium density: few neighbours
    @test s.skin≈1.00*2^(1/6)/2.5 && s.force_lanes==8
    A,_,_=mixed_pair(Float32,:lj,:matrix)
    spec=SC.nve(A); SC.step!(A,spec,A.dt)
    e=A.engine
    for _ in 1:20; SC.step!(A,spec,A.dt;compute_energy=false); end
    @test A.engine===e && e.launched==21 # no recapture on the fast path
    A.coll_enabled=true # a feature the engine does not support changes the signature
    @test DE._fingerprint(A,A.dt)!=e.fingerprint
    A.coll_enabled=false
end
end
