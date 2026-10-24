module DeviceEngineNPTRigidGPUChecks
# Molecular volume moves for rigid water on the device engine:
# - `set_cell_and_molecules!` scales each water's centre of mass with the cell and
#   translates the molecule rigidly (lengths, angles, velocities unchanged);
# - a changed cell gives the forces, energies and virials of a state built fresh at
#   the new cell;
# - Monte Carlo volume moves (molecular, N counts waters once) restore rejected trials
#   exactly and keep the run on the engine with the constraints satisfied;
# - C-rescale scales molecules with the molecular virial (-dU/ds check);
#   atom-wise scaling stays rejected for rigid water.
using Test, CUDA, ParticleDynamics, Random, LinearAlgebra, StaticArrays
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine
const RW=ParticleDynamics.RigidWaterConstraints
const ESX=ParticleDynamics.Electrostatics
relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
const KB=0.0083144626
const OH=0.1; const HH=0.16330

"""SPC/E waters on a cubic lattice (spacing 0.31 nm), random orientations, 300 K."""
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
function water(::Type{T},R,V,L,triples,m;real_space_modifier=:none) where {T}
    N=size(R,2); nw=length(triples)
    q=repeat([-0.8476,0.4238,0.4238],nw); ty=repeat(Int32[1,2,2],nw)
    ex=Tuple{Int,Int,Float64}[]; for (o,h1,h2) in triples; push!(ex,(o,h1,0.0),(o,h2,0.0),(h1,h2,0.0)); end
    st=SC.build_simulation(;N,box=L,nonbonded=:lj,cutoff=T(1.0),skin=T(0.1),cap=Int32(800),
        epsilon=T(0.650629),sigma=T(0.316557),mass=T.(m),dt=T(0.002),precision=T===Float64 ? :f64 : :f32,
        charges=q,spatial_reorder=false,neigh_interval=10_000)
    copyto!(st.rx,T.(R[1,:])); copyto!(st.ry,T.(R[2,:])); copyto!(st.rz,T.(R[3,:]))
    copyto!(st.vx,T.(V[1,:])); copyto!(st.vy,T.(V[2,:])); copyto!(st.vz,T.(V[3,:]))
    copyto!(st.typeid,ty)
    st.sigma_pair=CuArray(T.([0.316557 0.0; 0.0 0.0])); st.epsilon_pair=CuArray(T.([0.650629 0.0; 0.0 0.0]))
    st.rcut_pair=CuArray(T.([1.0 0.0; 0.0 0.0]))
    spec=ESX.ElectrostaticsSpec(coulomb_prefactor=138.935458,real_cutoff=1.0,split=ESX.GaussianSplit(3.12),
        window=ESX.BSplineWindow(6),mesh=(32,32,32),influence=:spme,real_space_modifier=real_space_modifier)
    attach_electrostatics!(st,spec;exceptions=ex)
    attach_rigid_water!(st,triples;oh=OH,hh=HH)
    return st
end
pos(st)=vcat((Float64.(Array(x))' for x in (st.rx,st.ry,st.rz))...)
vel(st)=vcat((Float64.(Array(x))' for x in (st.vx,st.vy,st.vz))...)
F(st)=vcat((Float64.(Array(x))' for x in (st.fx,st.fy,st.fz))...)
function evaluate(st)
    SC.evaluate_configuration!(st,SC.EvaluationRequest(energy=true,virial=true))
    W=virial_tensor(st)
    return sum(Float64,Array(st.Epot)),F(st),[W.xx,W.yy,W.zz,W.xy,W.xz,W.yz]
end
image(d,L)=d.-round.(d./L).*L
"""Molecule-whole coordinates and centres of mass (host, Float64)."""
function molecules(X,L,triples,m)
    map(triples) do t
        o=X[:,t[1]]; x=[o.+image(X[:,j].-o,collect(L)) for j in t]
        x,sum(m[t[k]].*x[k] for k in 1:3)./sum(m[collect(t)])
    end
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

@testset "molecular volume moves for rigid water on the device engine" begin
    R,V,L,triples,m=lattice(12)
    @testset "molecular scaling is rigid (s = $s)" for s in (1.013,0.987)
        A=water(Float64,R,V,L,triples,m)
        X0=pos(A); V0=vel(A); b0=A.box3
        set_cell_and_molecules!(A,b0.*s)
        @test all(A.box3.≈b0.*s)
        X1=pos(A)
        @test vel(A)==V0                                                   # velocities unchanged
        M0=molecules(X0,b0,triples,m); M1=molecules(X1,A.box3,triples,m)
        dcom=maximum(maximum(abs.(image(c1.-s.*c0,collect(A.box3)))) for ((_,c0),(_,c1)) in zip(M0,M1))
        @test dcom<1e-12                                                   # centres of mass follow the cell
        dshape=maximum(maximum(abs.((x1[k].-x1[1]).-(x0[k].-x0[1]))) for ((x0,_),(x1,_)) in zip(M0,M1) for k in 2:3)
        @test dshape<1e-12                                                 # each molecule translated rigidly
        @test all(abs.(X1).<=collect(A.box3)./2 .+1e-12)                  # wrapped into the new cell
        dl,_=geometry_errors(A,triples); @test dl<1e-9
    end
    @testset "changed cell = fresh build (s = $s)" for s in (1.01,0.992)
        A=water(Float64,R,V,L,triples,m); sa=SC.nve(A)
        for _ in 1:20; SC.step!(A,sa,A.dt;compute_energy=false); end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && DE.has_feature(A.engine.run,DE.RigidWaterFeature)
        set_cell_and_molecules!(A,A.box3.*s)
        E1,F1,W1=evaluate(A)
        @test A.engine isa DE.StateEngine                                  # still on the engine
        B=water(Float64,pos(A),vel(A),A.box3,triples,m)
        E2,F2,W2=evaluate(B)
        @test abs(E1-E2)<=1e-10*abs(E2)
        @test relerr(F1,F2)<=1e-9
        @test relerr(W1,W2)<=1e-8
        for _ in 1:10; SC.step!(A,sa,A.dt;compute_energy=false); end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && all(isfinite,Array(A.fx))
        dl,dv=geometry_errors(A,triples); @test dl<1e-9 && dv<1e-6
    end
    @testset "rejected trial restores the committed state" begin
        A=water(Float64,R,V,L,triples,m); sa=SC.csvr(A;temperature=KB*300,tau=0.1)
        for _ in 1:20; SC.step!(A,sa,A.dt;compute_energy=false); end
        mc=SC.MonteCarloVolumeMove(A;pressure=1e9,kT=1e-9,log_volume_step=0.05,seed=0xE8)   # every expansion rejected
        E0,F0,W0=evaluate(A); x0=pos(A); b0=A.box3
        rejected=0
        for k in 1:6
            acc=SC.monte_carlo_volume_move!(A,mc)
            if !acc
                rejected+=1
                @test A.box3==b0 && pos(A)==x0
                E,Fk,Wk=evaluate(A)
                @test E==E0 && Fk==F0 && Wk==W0                                # exact restore
            else
                b0=A.box3; x0=pos(A); E0,F0,W0=evaluate(A)
            end
        end
        @test rejected>=1
        for _ in 1:10; SC.step!(A,sa,A.dt;compute_energy=false); end      # the restored reference drives SHAKE
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine
        dl,dv=geometry_errors(A,triples); @test dl<1e-9 && dv<1e-6
    end
    @testset "NPT (CSVR + MC every 25 steps) on the engine" begin
        A=water(Float64,R,V,L,triples,m); kT=KB*300
        sa=SC.csvr(A;temperature=kT,tau=0.1)
        mc=SC.MonteCarloVolumeMove(A;pressure=0.06022,kT=kT,log_volume_step=0.002,seed=0xE9)   # 1 bar
        @test SC.scaled_units(A)==length(triples)
        for i in 1:500
            SC.step!(A,sa,A.dt;compute_energy=i%25==0)
            i%25==0 && SC.monte_carlo_volume_move!(A,mc)
        end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        @test mc.attempts==20 && 0<mc.accepts<20
        dl,dv=geometry_errors(A,triples); @test dl<1e-9 && dv<1e-6
    end
    @testset "atom-wise cell changes stay rejected" begin
        A=water(Float64,R,V,L,triples,m)
        @test_throws ArgumentError set_cell_and_positions!(A,A.box3.*1.01)
    end
    # C-rescale with rigid water. The molecular virial is -dU/ds under molecular
    # scaling (centres of mass and cell by s); with shifted LJ and real-space Coulomb no
    # pair crossing the cutoff makes U jump, so a central difference resolves it.
    @testset "molecular virial = -dU/ds under molecular scaling" begin
        shifted(;kw...)=begin
            A=water(Float64,R,V,L,triples,m;real_space_modifier=:potential_shift)
            set_vdw_modifier!(A,:potential_shift); A
        end
        A=shifted(); W=SC.molecular_virial_trace(A)
        ε=1e-5; U=map((1-ε,1+ε)) do s
            B=shifted(); SC.set_cell_and_molecules!(B,B.box3.*s)
            SC.evaluate_configuration!(B,SC.EvaluationRequest(energy=true)); sum(Float64,Array(B.Epot))
        end
        @test isapprox(W,-(U[2]-U[1])/(2ε);rtol=2e-5)
        # the atomic configurational virial differs by the intramolecular term
        @test !isapprox(W,SC.total_virial_trace(A)-SC.constraint_virial_trace(A);rtol=1e-3)
    end
    @testset "NPT (CSVR + C-rescale every 25 steps) on the engine" begin
        A=water(Float64,R,V,L,triples,m); kT=KB*300
        sa=SC.csvr(A;temperature=kT,tau=0.1)
        cr=SC.StochasticCellRescalingState(A;pressure=0.06022,kT=kT,compressibility=4.5e-5/0.06022,
            tau=1.0,seed=0xC5)
        V0=SC.cell_volume(A)
        for i in 1:500
            SC.step!(A,sa,A.dt;compute_energy=i%25==0)
            i%25==0 && SC.stochastic_cell_rescale!(A,cr,25*A.dt;compute_energy=false)
        end
        DE.synchronize_state!(A)
        @test A.engine isa DE.StateEngine && A.engine.run.cluster!==nothing
        @test cr.events==20 && SC.cell_volume(A)!=V0 && isfinite(cr.last_pressure)
        dl,dv=geometry_errors(A,triples); @test dl<1e-9 && dv<1e-6
    end
end
end
