module DeviceEngineOUGPUChecks
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Ornstein–Uhlenbeck (coloured) noise on top of the
# white noise for BAOAB, BAOA, VV/GJF, Euler–Maruyama and Euler–Heun through the
# public `step!`. Force-free gas, a two-mode OU spectrum on the odd particles
# only: device (sorting at every rebuild) vs staged statistics (no sorting; golden
# records, `engine_golden.jl`) per-group mean squared displacement and velocity
# variance; OU state variance = scale²;
# `active_idx` written back to the current storage slots.
using Test, CUDA, ParticleDynamics, Random, Statistics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const F=ParticleDynamics.Filters
include(joinpath(@__DIR__,"..","scripts","capture_staged_golden.jl"))
include(joinpath(@__DIR__,"engine_golden.jl")); using .EngineGolden

function make(kind,st,T)
    kind===:baoab && return SC.baoab(st;gamma=T(1),temperature=T(1))
    kind===:baoa && return SC.baoa(st;gamma=T(1),temperature=T(1))
    kind===:vv && return SC.velocityverlet(st;gamma=T(1),temperature=T(1))
    kind===:em && return SC.eulermaruyama(st;gamma=T(1),temperature=T(1))
    return SC.eulerheun(st;gamma=T(1),temperature=T(1))
end

function gas(D,::Type{T},N,dt;reorder) where {T}
    m=ceil(Int,N^(1/D)); L=T(3m)
    st=SC.build_simulation(;N,box=ntuple(_->L,D),cutoff=T(1),skin=T(0.3),cap=Int32(32),neigh_interval=20,
        nonbonded=:soft_repulsive,epsilon=T(1e-20),sigma=T(1),temperature=T(0),dt=T(dt),mass=T(1),
        precision=:f32,spatial_reorder=reorder,reorder_interval=1)
    pos=[T[3*(mod((i-1)÷m^(d-1),m))-L/2+T(1.5) for i in 1:N] for d in 1:D]
    return place!(st,D,pos,[zeros(T,N) for _ in 1:D]),pos,L
end

const TAUS=[0.05,0.5]
const SCALES=Dict(true=>[0.004,0.008],false=>[0.02,0.04])   # Brownian (dt 1e-3) / Langevin (dt 5e-3)

@testset "OU coloured noise on the device engine" begin
@golden_testset :engine "OU on odd particles: device vs staged golden statistics ($kind, $(D)D)" for (kind,D,n) in
        ((:baoab,2,2000),(:baoa,2,2000),(:vv,2,2000),(:em,2,2000),(:heun,2,2000),(:baoab,3,600),(:heun,3,600))
    T=Float32; N=4096; brownian=kind in (:em,:heun); dt=brownian ? 1e-3 : 5e-3
    odd=collect(1:2:N)
    statistics(device)=begin
        st,pos,L=gas(D,T,N,dt;reorder=device)
        spec=make(kind,st,T)
        F.set_ou_spectrum!(spec,st,T.(TAUS),T.(SCALES[brownian]);filter=F.Indices(odd))
        stp!()=device ? SC.step!(st,spec,st.dt;compute_energy=false) : SC.run_integrator_step!(st,spec,st.dt;compute_energy=false)
        for _ in 1:n; stp!(); end
        if device
            @test st.engine isa DE.StateEngine && DE.has_feature(st.engine.run,DE.OUFeature)
            @test DE.ordering_status(st.engine.run).count>=2
            SC._engine_boundary!(st.engine,st)                                    # writes active_idx back
            @test sort(Array(st.tag)[Array(spec.params.ou.active_idx)])==odd      # columns follow particles
        else
            @test st.engine===nothing
        end
        groups=map((1:2:N,2:2:N)) do g
            Δ2=Float64[]; v2=Float64[]
            for d in 1:D
                x=ordered(st,(st.rx,st.ry,st.rz)[d]); Δ=x.-Float64.(pos[d]); Δ.-=L.*round.(Δ./L)
                append!(Δ2,abs2.(Δ[g]))
                brownian || append!(v2,abs2.(ordered(st,(st.vx,st.vy,st.vz)[d])[g]))
            end
            (Δ2,v2)
        end
        ws=spec.workspace
        sv=[var(Float64.(vec(Array(x)[k,:]))) for x in (ws.ou_x,ws.ou_y) for k in 1:2]
        (groups,sv,st)
    end
    gd,svd,_=statistics(true)
    ref=staged_reference("ou_$(kind)_$(D)d") do
        g,sv,st=statistics(false)
        st,(msd1=g[1][1],v21=g[1][2],msd2=g[2][1],v22=g[2][2],sv=sv)
    end
    gs=((extra(ref,:msd1),extra(ref,:v21)),(extra(ref,:msd2),extra(ref,:v22)))
    for g in 1:2, q in 1:(brownian ? 1 : 2)
        a=gd[g][q]; b=gs[g][q]
        se=sqrt(var(a)/length(a)+var(b)/length(b))
        @test abs(mean(a)-mean(b))<5se
        println(kind," ",D,"D group ",g==1 ? "OU " : "white"," ",q==1 ? "MSD" : "v²",": device ",round(mean(a),sigdigits=5),
            " staged ",round(mean(b),sigdigits=5)," (",round((mean(a)-mean(b))/se,digits=2)," SE)")
    end
    @test mean(gd[1][1])>mean(gd[2][1])                                         # OU particles move further
    sc=SCALES[brownian]; K=length(odd)
    for (v,s) in zip(svd,(sc[1],sc[2],sc[1],sc[2]))
        @test abs(v/s^2-1)<5*sqrt(2/K)                                         # stationary OU variance
    end
end
end # testset
end
