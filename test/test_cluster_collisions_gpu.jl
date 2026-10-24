module ClusterCollisionsGPUChecks
# collision counting on the cluster-pair backend.
# The engine keeps ELL contact rows (vdW cutoff + cluster skin) in its control block,
# checked after every physical move (and after A1 for the split BAOAB family), and runs
# the staged event kernels on them. Checks: charged melt (NVE, CSVR): the device bins
# equal a host recount from the positions sampled every step (entries i<j whose
# separation crossed below the per-type contact cutoff); split BAOAB on a neutral LJ
# fluid: cluster backend counts equal the ELL engine's; a small box uses all-pairs
# contact rows.
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const ESX=ParticleDynamics.Electrostatics
const Coll=ParticleDynamics.Collisions
pos(st)=vcat((Array(x)' for x in (st.rx,st.ry,st.rz))...)
const KB=0.0083144626

function melt(;n=12,seed=7)
    a=(1/32.07)^(1/3); rng=MersenneTwister(seed)
    sites=[(i,j,k) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]; N=length(sites); L=n*a
    X=hcat([collect((s.+0.5).*a).+0.01 .*randn(rng,3) for s in sites]...)
    q=[iseven(sum(s)) ? 1.0 : -1.0 for s in sites]; ty=Int32[iseven(sum(s)) ? 1 : 2 for s in sites]
    V=randn(rng,3,N).*sqrt(KB*1100/29)
    σm=[0.33284 0.38647; 0.38647 0.440104]; e12=sqrt(0.0115897*0.4184); εm=[0.0115897 e12; e12 0.4184]
    rcm=[0.9 0.95; 0.95 1.0]
    st=SC.build_simulation(;N,box=ntuple(_->L,3),nonbonded=:lj,cutoff=1.0,skin=0.187,cap=Int32(600),epsilon=0.4184,
        sigma=0.440104,mass=29.0,dt=0.002,precision=:f64,charges=q,spatial_reorder=false,neigh_interval=10_000,
        use_neighborlist=n>=12)   # small boxes: no staged cell grid (the cluster backend takes over)
    for (d,x) in enumerate((st.rx,st.ry,st.rz)); copyto!(x,X[d,:]); end
    for (d,x) in enumerate((st.vx,st.vy,st.vz)); copyto!(x,V[d,:]); end
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(σm); st.epsilon_pair=CuArray(εm); st.rcut_pair=CuArray(rcm)
    attach_electrostatics!(st,ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,
        split=ESX.GaussianSplit(3.2),window=ESX.BSplineWindow(6),mesh=(40,40,40),influence=:spme))
    Coll.enable_collision_counting!(st)
    return st,L,ty,rcm
end
"""Host recount: entries (i<j) crossing below the contact cutoff between consecutive samples."""
function host_events(Xs,L,ty,rcm,lut)
    nb=maximum(lut)+1; counts=zeros(Int,nb); N=size(Xs[1],2)
    for s in 2:length(Xs)
        A=Xs[s-1]; B=Xs[s]
        for i in 1:N-1, j in i+1:N
            rc2=rcm[ty[i],ty[j]]^2
            d=B[:,i].-B[:,j]; d.-=L.*round.(d./L); sum(abs2,d)<rc2 || continue
            e=A[:,i].-A[:,j]; e.-=L.*round.(e./L); sum(abs2,e)>=rc2 || continue
            counts[lut[ty[i],ty[j]]+1]+=1
        end
    end
    return counts
end

@testset "collision counting on the cluster backend" begin
    @testset "charged melt: device bins = host recount ($kind)" for kind in (:nve,:csvr)
        st,L,ty,rcm=melt()
        @test nonbonded_backend(st)===:cluster
        sp=kind===:nve ? SC.nve(st) : SC.csvr(st;temperature=KB*1100,tau=0.1)
        @test DE.engine_supported(st,sp)
        Xs=[pos(st)]
        for i in 1:60
            SC.step!(st,sp,st.dt;compute_energy=i%20==0)
            DE.synchronize_state!(st); push!(Xs,pos(st))
        end
        @test st.engine isa DE.StateEngine && st.engine.run.cluster!==nothing &&
              any(f->f isa DE.CollisionFeature,st.engine.run.features)
        @test !DE.csvr_lazy_report(st.engine.run)
        dev=Array(st.coll_counts); lut=Array(st.coll_bins)
        host=host_events(Xs,L,ty,rcm,lut)
        @test sum(host)>0 && dev==host
    end
    @testset "split BAOAB: cluster backend counts = ELL engine counts (neutral LJ)" begin
        n=10; a=1.12; L=n*a; N=n^3; rng=MersenneTwister(3)
        X=hcat([collect(((i,j,k).+0.5).*a.-L/2).+0.05 .*randn(rng,3) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]...)
        V=0.8 .*randn(rng,3,N); V.-=sum(V;dims=2)./N
        A,B=map((:cluster,:ell)) do backend
            st=SC.build_simulation(;N,box=ntuple(_->L,3),cutoff=2.5,skin=0.3,cap=Int32(200),epsilon=1.0,sigma=1.0,
                temperature=0.0,dt=0.002,precision=:f64,spatial_reorder=false)
            for (d,x) in enumerate((st.rx,st.ry,st.rz)); copyto!(x,X[d,:]); end
            for (d,x) in enumerate((st.vx,st.vy,st.vz)); copyto!(x,V[d,:]); end
            if backend===:cluster
                set_nonbonded_backend!(st,:cluster)
            else
                ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh,st.rx,st.ry,st.rz;box=st.box3,step=st.step)
            end
            Coll.enable_collision_counting!(st); SC.invalidate_forces!(st); st
        end
        sa=SC.baoab(A;gamma=0.5,temperature=0.0); sb=SC.baoab(B;gamma=0.5,temperature=0.0)
        for i in 1:80
            SC.step!(A,sa,A.dt;compute_energy=i%20==0); SC.step!(B,sb,B.dt;compute_energy=i%20==0)
        end
        DE.synchronize_state!(A); DE.synchronize_state!(B)
        @test A.engine.run.cluster!==nothing && B.engine.run.cluster===nothing
        @test sum(Array(B.coll_counts))>0 && Array(A.coll_counts)==Array(B.coll_counts)
    end
    @testset "small box: all-pairs contact rows" begin
        st,L,ty,rcm=melt(n=8)          # L ≈ 2.52: fewer than 3 contact cells of 1.19
        sp=SC.nve(st); Xs=[pos(st)]
        for i in 1:40
            SC.step!(st,sp,st.dt;compute_energy=false); DE.synchronize_state!(st); push!(Xs,pos(st))
        end
        @test st.engine.run.neighbors.allpairs
        @test Array(st.coll_counts)==host_events(Xs,L,ty,rcm,Array(st.coll_bins))
    end
end
end
