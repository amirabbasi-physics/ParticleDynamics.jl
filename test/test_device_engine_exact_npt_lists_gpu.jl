module DeviceEngineExactNPTListsGPUChecks
# Exact (displacement-triggered) lists stay complete across device
# C-rescale contractions. Two LJ particles start just outside the list radius across the
# periodic face, move 97% of half the skin (no rebuild by displacement), then one event
# contracts the cell so that the pair is inside the cutoff. The engine's force must equal the
# analytic pair force. Before the fix the cluster list kept its stale rows and returned zero.
using Test, CUDA, ParticleDynamics
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine

lj(r,ϵ,σ)=24ϵ*(2*(σ/r)^12-(σ/r)^6)/r

@testset "Exact lists after a device C-rescale contraction ($backend)" for backend in (:cluster,:ell)
    T=Float64; L=10.0; rc=1.0; σ=0.3; ϵ=1.0; dt=0.001; iv=2
    skin=backend===:cluster ? 0.2 : DE.engine_settings(2,2/L^3;cutoff=rc).skin
    rl=rc+skin; sep0=rl+0.002; m=0.97*skin/2
    x0=(L-sep0)/2; x1=x0+m; sepm=sep0-2m
    s=0.995*rc/sepm                       # contraction that brings the pair inside the cutoff
    @test s>=0.98 && abs(s*x1-x0)<skin/2  # within the cell allowance; displacement below half the skin
    st=SC.build_simulation(;N=2,box=(L,L,L),nonbonded=:lj,cutoff=T(rc),skin=T(0.2),cap=Int32(16),
        epsilon=T(ϵ),sigma=T(σ),temperature=T(0),dt=T(dt),precision=:f64,spatial_reorder=false)
    copyto!(st.rx,T[-x0,x0]); for a in (st.ry,st.rz,st.vx,st.vy,st.vz); fill!(a,0); end
    backend===:cluster && set_nonbonded_backend!(st,:cluster;prune_buffer=0.1)
    # kT → 0 and no interacting pair: the event is deterministic, s = (1 − hκP0/2τ)^(2/3)
    P0=2*(1-s^1.5)/(iv*dt)
    cr=SC.StochasticCellRescalingState(st;pressure=P0,kT=1e-12,compressibility=1.0,tau=1.0,seed=7)
    SC.attach_pressure_control!(st,cr;interval=iv)
    sp=SC.nve(st)
    SC.step!(st,sp,st.dt)
    run=st.engine.run; geom=run.neighbors.geom
    @test Array(run.f[1])==[0.0,0.0]       # outside the cutoff
    copyto!(run.r[1],T[-x1,x1])
    changed=false
    for _ in 1:3
        g0=Array(geom)[1]
        SC.step!(st,sp,st.dt)
        g=Array(geom)[1]; x=Array(run.r[1]); f=Array(run.f[1])
        d=g-(x[2]-x[1])
        @test isapprox(f[1],d<rc ? lj(d,ϵ,σ) : 0.0;atol=1e-12,rtol=1e-9)
        if g!=g0
            changed=true
            @test g≈s*L && d<rc
            break
        end
    end
    @test changed
end
end
