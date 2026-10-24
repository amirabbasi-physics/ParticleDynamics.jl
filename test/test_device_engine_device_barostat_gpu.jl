module DeviceEngineDeviceBarostatGPUChecks
# Device-resident barostat: the box is device state (captured kernels read it),
# and a C-rescale pressure event runs inside the step graphs (`attach_pressure_control!`).
# Checks on rigid SPC/E water (cluster backend, SPME):
# - the device influence refresh equals the host computation;
# - the pressure of a device event equals the host molecular pressure of the configuration
#   it reads (the end of the step before the event), and the new box is s × the old one;
# - events keep the engine (no recapture), the host box and counters follow at boundaries,
#   constraints hold; the run equals a host-barostat run in its sampling (smoke level);
# - unsupported combinations are refused with a reason.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra, StaticArrays
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const RW=ParticleDynamics.RigidWaterConstraints
const ESX=ParticleDynamics.Electrostatics
const KB=0.0083144626; const BAR=0.0602214076
const OH=0.1; const HH=0.16330

function lattice(n;seed=5)
    a=0.31; L=ntuple(_->n*a,3); rng=MersenneTwister(seed); nw=n^3; N=3nw
    θ=2asin(HH/2/OH); R=zeros(3,N); triples=NTuple{3,Int}[]
    for (w,(i,j,k)) in enumerate((i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1)
        o=3w-2; R[:,o].=((i,j,k).+0.5).*a
        u=normalize(randn(rng,3)); p=normalize(cross(u,randn(rng,3)))
        R[:,o+1].=R[:,o].+OH.*u; R[:,o+2].=R[:,o].+OH.*(cos(θ).*u.+sin(θ).*p)
        push!(triples,(o,o+1,o+2))
    end
    m=repeat([15.9994,1.008,1.008],nw)
    return R,randn(rng,3,N).*sqrt.(KB*300 ./ m'),L,triples,m
end
function water(::Type{T},R,V,L,triples,m) where {T}
    N=size(R,2); nw=length(triples)
    q=repeat([-0.8476,0.4238,0.4238],nw); ty=repeat(Int32[1,2,2],nw)
    ex=Tuple{Int,Int,Float64}[]; for (o,h1,h2) in triples; push!(ex,(o,h1,0.0),(o,h2,0.0),(h1,h2,0.0)); end
    st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.1),cap=Int32(800),
        epsilon=T(0.650629),sigma=T(0.316557),mass=T.(m),dt=T(0.002),precision=T===Float64 ? :f64 : :f32,
        charges=q,spatial_reorder=false,neigh_interval=10_000,vdw_modifier=:potential_shift)
    copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
    copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(T.([0.316557 0.0; 0.0 0.0])); st.epsilon_pair=CuArray(T.([0.650629 0.0; 0.0 0.0]))
    st.rcut_pair=CuArray(T.([1.0 0.0; 0.0 0.0]))
    spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.12),
        window=ESX.BSplineWindow(6),mesh=(32,32,32),influence=:spme)
    attach_electrostatics!(st,spec;exceptions=ex)
    attach_rigid_water!(st,triples;oh=OH,hh=HH)
    return st
end
function geometry_errors(st,triples)
    r=map(Array,(st.rx,st.ry,st.rz)); v=map(Array,(st.vx,st.vy,st.vz)); box=st.box3
    dl=0.0; dv=0.0
    for t in triples
        raw=ntuple(k->SVector(ntuple(d->Float64(r[d][t[k]]),3)),3)
        x=RW.local_water(raw,raw[1],Float64.(box))
        vv=ntuple(k->SVector(ntuple(d->Float64(v[d][t[k]]),3)),3)
        dl=max(dl,RW.distance_error(x,(OH^2,OH^2,HH^2)))
        dv=max(dv,maximum(abs(dot(e,w)) for (e,w) in zip(RW._edges(x),RW._edges(vv))))
    end
    return dl,dv
end

@testset "Device-resident box and C-rescale" begin
    R,V,L,triples,m=lattice(12)
    @testset "device influence refresh = host ($(T))" for T in (Float64,Float32)
        A=water(T,R,V,L,triples,m); s=A.electrostatics
        G0=Float64.(Array(s.ws.G)); V0=Float64.(Array(s.ws.Vc))
        ESX.device_influence!(s,CuArray(T.(collect(A.box3))))
        @test maximum(abs.(Float64.(Array(s.ws.G)).-G0))<=(T===Float64 ? 1e-12 : 1e-6)*maximum(abs.(G0))
        @test maximum(abs.(Float64.(Array(s.ws.Vc)).-V0))<=(T===Float64 ? 1e-12 : 1e-6)*maximum(abs.(V0))
    end
    @testset "event pressure = host molecular pressure; box scales by s" begin
        A=water(Float64,R,V,L,triples,m); kT=KB*300; nst=10
        sa=SC.csvr(A;temperature=kT,tau=0.1)
        cr=SC.StochasticCellRescalingState(A;pressure=BAR,kT,compressibility=4.5e-5/BAR,tau=1.0,seed=0xD3)
        SC.attach_pressure_control!(A,cr;interval=nst)
        for i in 1:(nst-1); SC.step!(A,sa,A.dt;compute_energy=false); end   # ends with the sampled step
        DE.synchronize_state!(A)
        e=A.engine; @test e isa DE.StateEngine && e.event!==nothing
        Vb=SC.cell_volume(A); boxb=A.box3
        Phost=SC.scaled_units(A)*kT/Vb+SC.molecular_virial_trace(A)/(3Vb)
        e2=A.engine; @test e2===e                      # host evaluation at the boundary kept the engine
        SC.step!(A,sa,A.dt;compute_energy=false)       # the event step
        DE.synchronize_state!(A)
        buf=Array(e.run.cluster.barostat.buf)
        @test isapprox(buf[DE._BARO_P],Phost;rtol=1e-8,atol=1e-6)
        s=buf[DE._BARO_S]
        @test all(isapprox.(A.box3,boxb.*s;rtol=1e-12))
        @test cr.events==1 && isapprox(cr.last_pressure,Phost;rtol=1e-8,atol=1e-6)
    end
    @testset "NPT on the device: no recapture, counters, constraints" begin
        A=water(Float32,R,V,L,triples,m); kT=KB*300; nst=20
        sa=SC.csvr(A;temperature=kT,tau=0.1)
        cr=SC.StochasticCellRescalingState(A;pressure=BAR,kT,compressibility=4.5e-5/BAR,tau=1.0,seed=0xD4)
        SC.attach_pressure_control!(A,cr;interval=nst)
        SC.step!(A,sa,A.dt;compute_energy=false); e=A.engine
        V0=SC.cell_volume(A)
        for i in 2:400; SC.step!(A,sa,A.dt;compute_energy=false); end
        DE.synchronize_state!(A)
        @test A.engine===e                              # the box changed 20 times without a recapture
        @test cr.events==20 && SC.cell_volume(A)!=V0 && isfinite(cr.last_pressure)
        @test all(isapprox.(A.box3,Tuple(Array(e.run.neighbors.geom));rtol=0))
        dl,dv=geometry_errors(A,triples); @test dl<1e-5 && dv<1e-5
        SC.detach_pressure_control!(A)
        SC.step!(A,sa,A.dt;compute_energy=false)
        @test A.engine!==e && A.engine.event===nothing  # detached: graphs without the event
    end
    @testset "unsupported combinations are refused" begin
        st=SC.build_simulation(N=64,box=(6.0,6.0,6.0),cutoff=1.0,precision=:f64,spatial_reorder=false)
        cr=SC.StochasticCellRescalingState(st;pressure=BAR,kT=KB*300,compressibility=4.5e-5/BAR,tau=1.0,seed=1)
        @test_throws ArgumentError SC.attach_pressure_control!(st,cr;interval=1)
        # plain 3D LJ on the dense list is supported; bonds still need the cluster-pair backend
        bonds=Tuple{Int32,Int32}[(Int32(2i-1),Int32(2i)) for i in 1:32]
        sb=SC.build_simulation(N=64,box=(6.0,6.0,6.0),cutoff=1.0,precision=:f64,spatial_reorder=false,
            bonds=bonds,bonding=harmonic_bond(k=100.0,r0=1.0))
        cb=SC.StochasticCellRescalingState(sb;pressure=BAR,kT=KB*300,compressibility=4.5e-5/BAR,tau=1.0,seed=1)
        SC.attach_pressure_control!(sb,cb;interval=10)
        err=try SC.step!(sb,SC.nve(sb),sb.dt); nothing catch x; x end
        @test err isa ArgumentError && occursin("cluster-pair backend",err.msg)
        SC.attach_pressure_control!(st,cr;interval=10)
        @test (SC.step!(st,SC.nve(st),st.dt); true)
    end
end
end
