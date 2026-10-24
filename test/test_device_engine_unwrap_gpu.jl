module DeviceEngineUnwrapGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# unwrapped coordinates through the public `step!`
# for every integrator family. Checks: noise-free runs against the staged
# path's golden records (unwrapped and wrapped coordinates; recorded from the
# staged engine, `engine_golden.jl`); exact free diffusion measured from
# the unwrapped coordinates in a small box (many boundary crossings) while
# sorting permutes the arrays; unwrapped − wrapped is a whole number of boxes.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)

function spec0(kind,st,T)   # noise-free variants
    kind===:nve && return SC.nve(st)
    kind===:baoab && return SC.baoab(st;gamma=T(0.7),temperature=T(0))
    kind===:baoa && return SC.baoa(st;gamma=T(0.7),temperature=T(0))
    kind===:vv && return SC.velocityverlet(st;gamma=T(0.7),temperature=T(0))
    kind===:csvr && return SC.csvr(st;temperature=T(1),tau=T(1e30))
    kind===:nhc && return SC.nosehooverchain(st;temperature=T(1),tau=T(10)) # weak coupling keeps the x drift
    kind===:em && return SC.eulermaruyama(st;gamma=T(2),temperature=T(0))
    return SC.eulerheun(st;gamma=T(2),temperature=T(0))
end

@testset "unwrapped coordinates on the device engine" begin
@golden_testset :engine "Noise-free device vs staged golden with unwrapped coordinates ($kind, $(D)D)" for kind in (:nve,:baoab,:baoa,:vv,:csvr,:nhc,:em,:heun), D in (2,3)
    T=Float64; N=257; pos,_=lattice(D,N,1.12,T); vel=velocities(D,N,kind in (:em,:heun) ? 0.0 : 0.5,T)
    kind in (:em,:heun) || (vel[1].+=10)                       # uniform drift along x: particles cross the box
    dt=kind in (:em,:heun) ? 1e-3 : 0.002
    A,B=map(1:2) do _; st=build(D,T,:lj,N;a=1.12,dt,unwrapped=true); place!(st,D,pos,vel); end
    sa=spec0(kind,A,T)
    @test DE.engine_supported(A,sa)
    for _ in 1:200; SC.step!(A,sa,A.dt;compute_energy=false); end
    ref=staged_reference("unwrap_$(kind)_$(D)d") do
        sb=spec0(kind,B,T)
        for _ in 1:200; SC.run_integrator_step!(B,sb,B.dt;compute_energy=false); end
        B
    end
    @test A.engine isa DE.StateEngine
    ua=(A.rx_unwrap,A.ry_unwrap,A.rz_unwrap)[1:D]
    for (a,f) in zip((ua...,A.rx,A.ry),((:rx_unwrap,:ry_unwrap,:rz_unwrap)[1:D]...,:rx,:ry))
        @test relerr(ordered(A,a),field(ref,f))<=1e-9
    end
    # Per axis, unwrapped − wrapped is a whole number of box lengths (a corner
    # exit changes several axes at once; repeated crossings give larger multiples).
    for d in 1:D
        k=(ordered(A,ua[d]).-ordered(A,(A.rx,A.ry,A.rz)[d]))./Float64(A.rz===nothing ? A.box2[d] : A.box3[d])
        @test maximum(abs.(k.-round.(k)))<1e-9
        d==1 && !(kind in (:em,:heun)) && @test maximum(abs.(round.(k)))>=1   # crossings along x happened
    end
end

@testset "Free diffusion from unwrapped coordinates, small box, sorting ($kind, $(D)D)" for kind in (:em,:heun), D in (2,3)
    T=Float32; m=D==2 ? 16 : 10; a=T(1.5); N=m^D; L=a*m  # box 24/15σ; particles diffuse ≈ 9σ per component
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(1e-2),mass=T(1),
        precision=:f32,spatial_reorder=true,reorder_interval=1,unwrapped_positions=true)
    pos=[T[a*(mod((i-1)÷m^(d-1),m))-L/2+a/2 for i in 1:N] for d in 1:D]
    place!(st,D,pos,[zeros(T,N) for _ in 1:D])
    spec=kind===:em ? SC.eulermaruyama(st;gamma=T(1),temperature=T(1)) : SC.eulerheun(st;gamma=T(1),temperature=T(1))
    n=4000; for _ in 1:n; SC.step!(st,spec,st.dt;compute_energy=false); end
    @test DE.ordering_status(st.engine.run).count>=2
    t=n*Float64(st.dt); u=(st.rx_unwrap,st.ry_unwrap,st.rz_unwrap)[1:D]; r=(st.rx,st.ry,st.rz)[1:D]
    Δ2=Float64[]; img=Float64[]; crossings=0
    for d in 1:D
        ud=ordered(st,u[d]); rd=ordered(st,r[d])
        append!(Δ2,abs2.(ud.-Float64.(pos[d])))
        k=(ud.-rd)./Float64(L); append!(img,abs.(k.-round.(k)))
        crossings+=sum(abs.(round.(Int,k)))
    end
    @test maximum(img)<1e-3              # per axis, unwrapped − wrapped is a whole number of boxes
    @test crossings>N÷4                  # many boundary crossings (2D: 129 of 256 particles measured)
    @test abs(mean(Δ2)/(2*t)-1)<5*sqrt(2/length(Δ2))        # ⟨Δx²⟩ = 2Tt/γ per component
end
end # testset
end
