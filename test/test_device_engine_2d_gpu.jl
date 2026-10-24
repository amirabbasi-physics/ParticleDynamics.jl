module DeviceEngine2DGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# 2D NVE through the public `step!` (serial row
# builder, generic cooperative pair kernel, unfused steps). Checked against the
# staged golden records (`engine_golden.jl` for mixed σ) and the
# independent reference.
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

# Staged 2D WCA and soft-repulsive kernels compute periodic differences as
# mic_fast(xi − xj) (subtract, then wrap). In Float32 their forces deviate more
# from the Float64 reference than the device's compensated arithmetic (measured
# at the golden initial configuration: WCA 2.6e-6 vs 6.5e-7, soft repulsion
# 1.0e-6 vs 3.7e-7). For these two Float32 cases the golden comparison
# therefore allows 5e-5 (the staged error amplified over 50 chaotic steps), and
# the device must first be shown at least as accurate as staged.
function device_at_least_as_accurate(g,pot)
    T=Float32; r=(CuArray(T.(g["forces"]["rx"])),CuArray(T.(g["forces"]["ry"]))); box=(T(g["box"][1]),T(g["box"][2]))
    model=pot===:wca ? DE.UniformWCA(T(1),T(1)) : DE.UniformSoftRepulsive(T(1),T(1))
    run=DE.Run(r,(CUDA.zeros(T,length(r[1])),CUDA.zeros(T,length(r[1]))),box;cutoff=T(pot===:wca ? 2^(1/6) : 1),
        skin=0.3,cap=128,dt=0.002,force_lanes=2,model)
    ref=PairReferenceGPU.reference_forces(r,box,CuArray(Int32.(1:length(r[1])));kind=pot===:wca ? 2 : 3).F
    staged=permutedims(hcat(CuArray(Float64.(g["forces"]["fx"])),CuArray(Float64.(g["forces"]["fy"]))))
    device=Float64.(permutedims(hcat(run.f...)))
    return norm(device-ref)<=norm(staged-ref)
end

@golden_testset :deterministic "2D public NVE step! matches staged golden records (LJ, WCA, soft repulsion)" begin
    for T in (Float32,Float64), pot in (:lj,:wca,:soft_repulsive)
        g=golden("det_2d_$(T)_$(pot)_nve")
        rec=deterministic_case(2,T,pot,:nve)
        staged_limited=T===Float32 && pot in (:wca,:soft_repulsive)
        staged_limited && @test device_at_least_as_accurate(g,pot)
        tol=T===Float32 ? (staged_limited ? 5e-5 : 1e-5) : 1e-12
        for stage in ("step1","step50"), k in ("rx","ry","vx","vy","fx","fy","Epot","virial")
            @test relerr(rec[stage][k],g[stage][k])<=tol
        end
        for (k,v) in g["step50_observables"]
            haskey(rec["step50_observables"],k) && @test isapprox(rec["step50_observables"][k],v;rtol=T===Float32 ? 1e-4 : 1e-10,atol=1e-10)
        end
    end
end

@golden_testset :engine "2D engine ownership and a mixed-σ comparison" begin
    for T in (Float32,Float64)
        N=257; pos,_=lattice(2,N,1.12,T); vel=velocities(2,N,0.4,T)
        σp=T[1+0.18*sin(1.7i) for i in 1:N]
        states=map(1:2) do _
            st=SC.build_simulation(;N,box=(T(24),T(24)),cutoff=T(2.5),skin=T(0.3),cap=Int32(120),neigh_interval=5,
                epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),mass=T(1),nonbonded=:lj,
                precision=T===Float64 ? :f64 : :f32,neighbor_cutoff=T(2.5*maximum(σp)))
            st.sigma_particle=CuArray(σp); st.rcut_factor=T(2.5)
            place!(st,2,pos,vel)
        end
        A,B=states; spec=SC.nve(A)
        @test DE.engine_supported(A,spec)
        for _ in 1:50; SC.step!(A,spec,A.dt;compute_energy=true); end
        ref=staged_reference("pairs2d_mixed_lj_$(T)") do
            for _ in 1:50; SC.run_integrator_step!(B,SC.nve(B),B.dt;compute_energy=true); end
            B
        end
        e=A.engine
        @test e isa DE.StateEngine && e.settings.build_lanes==1 && e.settings.fused==false
        tol=T===Float32 ? 1e-4 : 1e-11
        for n in (:rx,:ry,:vx,:vy,:fx,:fy,:Epot)
            @test relerr(ordered(A,getfield(A,n)),field(ref,n))<=tol
        end
        ids=CuArray(Int32.(1:N))
        ref=PairReferenceGPU.reference_forces((A.rx,A.ry),A.box2,ids;kind=1,style=2,epsilon=1.0,
            sigma_particle=A.sigma_particle,factor=2.5)
        @test norm(permutedims(hcat(A.fx,A.fy))-ref.F)/max(norm(ref.F),1)<=(T===Float32 ? 2e-5 : 1e-11)
        obs=SC.collect_step_observables(A,spec) # boundary check passes
        @test A.step==50
    end
    @test_throws ArgumentError DE.Run((CUDA.zeros(Float32,4),CUDA.zeros(Float32,4)),(CUDA.zeros(Float32,4),CUDA.zeros(Float32,4)),
        (12f0,12f0);cutoff=2.5,skin=0.3,cap=8,dt=0.001,build_lanes=8)
end
end
