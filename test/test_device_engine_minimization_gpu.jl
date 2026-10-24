module DeviceEngineMinimizationGPUChecks
# energy minimization on the device engine. The
# minimizer evaluates through an evaluation-only engine (no graphs, no sorting)
# created for a state without one, or through the stepping engine that already
# owns the state; evaluations never reorder storage, so its slot-indexed
# workspace stays valid. Checks against independent references: LJ dimers relax
# to r = 2^(1/6)σ with U = −ε each; a perturbed LJ lattice converges to the
# requested tolerance as measured by the independent Float64 CUDA all-pairs forces;
# pinned particles stay bit-for-bit; storage order unchanged; the first `step!`
# after minimizing a fresh state captures a stepping engine; a stepping engine
# that owns the state is kept and keeps its graphs. FIRE and steepest descent,
# Float64 and Float32.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"pair_reference_gpu.jl"))

const METHODS=(("SD",SteepestDescent(initial_step=0.05)),("FIRE",FIRE(dt_initial=0.005,dt_max=0.05,fictitious_mass=1.0)))

# M LJ dimers on a coarse grid (6σ apart, beyond cutoff + skin), each stretched to 1.3σ.
function dimers(::Type{T};m=4) where {T}
    M=m^3; N=2M; L=T(6m); pos=[zeros(T,N) for _ in 1:3]
    rng=Xoshiro(7)
    for k in 0:M-1
        c=[6*(mod(k÷m^(d-1),m))-L/2+3 for d in 1:3]; u=normalize(randn(rng,3))
        for d in 1:3; pos[d][2k+1]=T(c[d]); pos[d][2k+2]=T(c[d]+1.3*u[d]); end
    end
    st=SC.build_simulation(;N,box=(L,L,L),cutoff=T(2.5),skin=T(0.3),cap=Int32(32),neigh_interval=5,
        epsilon=T(1),sigma=T(1),temperature=T(0),dt=T(0.002),precision=T===Float64 ? :f64 : :f32)
    return place!(st,3,pos,[zeros(T,N) for _ in 1:3]),M
end
pair_distance(st,i,j)=norm([mod(Float64(Array(x)[i])-Float64(Array(x)[j])+L/2,L)-L/2 for (x,L) in zip((st.rx,st.ry,st.rz),Float64.(st.box3))])

@testset "minimization on the device engine" begin

@testset "LJ dimers relax to 2^(1/6)σ ($name, $T)" for (name,method) in METHODS, T in (Float64,Float32)
    st,M=dimers(T); tags=Array(st.tag)
    # Float32 steepest descent stalls once energy decreases fall below Float32
    # resolution of the total (Armijo test): on the staged path too it stops with
    # :step_too_small at max|F| = 1.6e-3, U = -64.0 exactly (12 Oct).
    tol=T===Float64 ? 1e-8 : (method isa SteepestDescent ? 3e-3 : 2e-4)
    res=minimize!(st;method,force_tolerance=tol,max_displacement=0.1,max_iterations=50_000,max_evaluations=100_000)
    @test res.converged && res.reason===:force_tolerance
    e=st.engine
    @test e isa DE.StateEngine && e.plain===nothing                  # evaluation-only engine
    @test Array(st.tag)==tags                                        # never sorted
    r=[pair_distance(st,2k+1,2k+2) for k in 0:M-1]
    @test maximum(abs.(r.-2^(1/6)))<=(T===Float64 ? 1e-7 : 2e-3)
    @test isapprox(res.final_energy,-M;rtol=T===Float64 ? 1e-10 : 1e-5)
    @test st.force_valid && st.energy_valid && all(iszero,Array(st.vx))
    SC.step!(st,SC.nve(st),st.dt)                                    # first step: stepping engine
    @test st.engine isa DE.StateEngine && st.engine!==e && st.engine.plain!==nothing
end

# Steepest descent needs an energy that is continuous at the cutoff (Armijo line
# search): truncated, unshifted LJ stops it with :step_too_small at |F| ≈ 1.5 on the
# staged path too (12 Oct). SD therefore relaxes WCA; FIRE relaxes LJ.
@testset "Perturbed lattice: tolerance met by independent forces; pinned particles fixed ($name, $pot)" for
        (name,method,pot,kind) in (("SD",METHODS[1][2],:wca,2),("FIRE",METHODS[2][2],:lj,1))
    T=Float64; N=343; pos,_=lattice(3,N,1.12,T)
    st=place!(build(3,T,pot,N;a=1.12),3,pos,[zeros(T,N) for _ in 1:3])
    pinned=collect(1:7:N); mobile=setdiff(1:N,pinned)
    before=[Array(x)[pinned] for x in (st.rx,st.ry,st.rz)]
    ftol=1e-5
    res=minimize!(st;method,force_tolerance=ftol,max_displacement=0.05,mobile,max_iterations=100_000,max_evaluations=200_000)
    @test res.converged && res.final_energy<res.initial_energy
    @test st.engine isa DE.StateEngine && st.engine.plain===nothing
    ref=PairReferenceGPU.reference_forces((st.rx,st.ry,st.rz),st.box3,CuArray(Int32.(1:N));kind)
    F=Array(ref.F)
    @test maximum(norm(F[:,i]) for i in mobile)<=ftol*(1+1e-6)+1e-9
    @test isapprox(res.final_energy,sum(Array(ref.energy));rtol=1e-10)   # reference: ½U per particle and pair
    for (x,b) in zip((st.rx,st.ry,st.rz),before); @test Array(x)[pinned]==b; end   # bit for bit
end

@testset "A stepping engine that owns the state is kept (FIRE)" begin
    T=Float64; N=343; pos,_=lattice(3,N,1.12,T)
    st=place!(build(3,T,:lj,N;a=1.12),3,pos,velocities(3,N,0.3,T))
    spec=SC.nve(st); for _ in 1:20; SC.step!(st,spec,st.dt;compute_energy=false); end
    e=st.engine; g=e.plain; tags=Array(st.tag)
    @test e isa DE.StateEngine && g!==nothing
    res=minimize!(st;method=FIRE(dt_initial=0.005,dt_max=0.05,fictitious_mass=1.0),force_tolerance=1e-5,
        max_displacement=0.05,max_iterations=100_000,max_evaluations=200_000)
    @test res.converged
    @test st.engine===e && e.plain===g && Array(st.tag)==tags        # evaluated in place, never sorted
    ref=PairReferenceGPU.reference_forces((st.rx,st.ry,st.rz),st.box3,CuArray(Int32.(1:N));kind=1)
    @test maximum(norm(Array(ref.F)[:,i]) for i in 1:N)<=1e-5*(1+1e-6)+1e-9
    SC.step!(st,spec,st.dt)
    @test st.engine===e && e.plain===g                               # the graphs continue
end
end # testset
end
