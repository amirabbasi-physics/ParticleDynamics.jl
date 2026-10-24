module DeviceArgonFusionGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists

lattice(::Type{T},N,a) where {T}=ntuple(d->CuArray(T[a*(mod((i-1)÷7^(d-1),7)-3)+0.02*sin(i*(0.5+0.1d)) for i in 1:N]),3)
# Perturbed cubic lattice filling a box of side m*a (generated on the GPU).
function scattered(::Type{T},N,a) where {T}
    m=ceil(Int,cbrt(N)); ids=CuArray(Float64.(0:N-1))
    return ntuple(d->T.(a.*(mod.(div.(ids,m^(d-1)),m).-m/2 .+0.5) .+0.1a.*sin.(ids.*(1.1+0.37d))),3)
end
boxside(N,a)=ceil(Int,cbrt(N))*a

refs(run)=(run.neighbors.list.rref_x,run.neighbors.list.rref_y,run.neighbors.list.rref_z)

# Make B's physical/list/control state an exact copy of A's.
function copy_state!(B,A)
    for (b,a) in zip((B.r...,B.v...,B.f...,refs(B)...,B.energy,B.virial,B.neighbors.state,
                      B.neighbors.list.neighbors_flat,B.neighbors.list.counts),
                     (A.r...,A.v...,A.f...,refs(A)...,A.energy,A.virial,A.neighbors.state,
                      A.neighbors.list.neighbors_flat,A.neighbors.list.counts))
        copyto!(b,a)
    end
    return nothing
end

pair(T,N,a,G;kw...)=(DA.Run(scattered(T,N,a),ntuple(d->CUDA.fill(T(0.7)*(-1)^d,N),3),ntuple(_->T(max(boxside(N,a),12)),3);
        cutoff=2.5,skin=0.3,cap=256,dt=0.004,force_lanes=G,rebuild_policy=:displacement,kw...),
    DA.Run(scattered(T,N,a),ntuple(d->CUDA.fill(T(0.7)*(-1)^d,N),3),ntuple(_->T(max(boxside(N,a),12)),3);
        cutoff=2.5,skin=0.3,cap=256,dt=0.004,force_lanes=G,rebuild_policy=:displacement,fused=true,kw...))

@testset "Fused prefix is bitwise identical to drift + displacement + decision" begin
    for T in (Float32,Float64), N in (1,255,256,257,65537)
        A,B=pair(T,N,1.25,1)
        for scenario in (:keep,:displacement,:invalid,:nonfinite,:latched), rep in 1:3
            copy_state!(B,A)
            if scenario===:displacement
                refs(A)[1] .+= T(0.4); copy_state!(B,A)       # every atom moved > skin/2
            elseif scenario===:invalid
                A.neighbors.state[NL.DN_VALID:NL.DN_VALID].=0; copy_state!(B,A)
            elseif scenario===:nonfinite
                A.r[2][end:end].=T(NaN); copy_state!(B,A)
            elseif scenario===:latched
                A.neighbors.state[NL.DN_ERROR:NL.DN_ERROR].=NL.DN_CAPACITY_ERROR; copy_state!(B,A)
            end
            DA.enqueue_drift!(A); DA.decide_policy!(A.neighbors,A.r,Val(:displacement))
            DA.enqueue_fused_prefix!(B,UInt64(0),Val(false))
            @test all(all(a .=== b) for (a,b) in zip((A.r...,A.v...),(B.r...,B.v...)))
            @test Array(A.neighbors.state)==Array(B.neighbors.state)
            @test Array(B.fusion.ticket)==Int32[0] && Array(B.fusion.maxbits)==[zero(eltype(B.fusion.maxbits))]
            # restore a valid, error-free reference state for the next scenario
            A,B=scenario===:keep ? (A,B) : pair(T,N,1.25,1)
        end
    end
end

@testset "Fused force + kick + step counter is bitwise identical to the separate kernels" begin
    for T in (Float32,Float64), N in (17,257,4099), G in (1,2,8), energy in (false,true)
        A,B=pair(T,N,1.25,G)
        copy_state!(B,A)
        DA.enqueue_finish!(A;energy)
        DA.fused_suffix!(B;energy)
        @test all(all(a .=== b) for (a,b) in zip((A.f...,A.v...),(B.f...,B.v...)))
        @test Array(A.neighbors.state)==Array(B.neighbors.state)
        if energy
            @test all(A.energy .=== B.energy) && all(A.virial .=== B.virial)
            @test all(A.observables.values .=== B.observables.values)
            @test Array(A.observables.stamp)==Array(B.observables.stamp)
        end
    end
end

@testset "Fused graph replay: every-step all-atom forces and agreement with the unfused control" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true), (Gf,Gb) in ((1,1),(8,8),(2,1))
        N=257; box=ntuple(_->T(20),3); ids=CuArray(Int32.(1:N))
        v=ntuple(d->T(1.2).*sin.(T.(ids).*T(0.3+0.1d)),3)
        args=(;cutoff=2.5,skin=0.12,cap=128,dt=0.002,spatial_reorder=sorted,reorder_interval=7,
            rebuild_policy=:displacement,force_lanes=Gf,build_lanes=Gb)
        control=DA.Run(lattice(T,N,1.3),v,box;args...)
        run=DA.Run(lattice(T,N,1.3),v,box;args...,fused=true)
        graphs=DA.capture_step(run;conditional),DA.capture_step(run;conditional,energy=true)
        controls=DA.capture_step(control;conditional),DA.capture_step(control;conditional,energy=true)
        addresses=map(pointer,(run.r...,run.v...,run.f...,run.fusion.maxbits,run.fusion.ticket))
        worst=0.0
        for step in 1:300
            CUDA.launch(graphs[step%25==0 ? 2 : 1]); CUDA.launch(controls[step%25==0 ? 2 : 1])
            ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,box,ids,1,1,2.5)
            worst=max(worst,norm(permutedims(hcat(run.f...))-ref.F)/max(norm(ref.F),1))
        end
        @test worst<=(T===Float32 ? 2e-5 : 3e-12)
        s=NL.device_neighbor_status(run.neighbors); c=NL.device_neighbor_status(control.neighbors)
        @test s.step==c.step==300 && s.checks==c.checks
        d=NL.device_neighbor_diagnostics(run.neighbors)
        @test d.builds>3 && d.age_builds==0 && d.builds==d.invalid_builds+d.displacement_builds
        @test Array(run.fusion.ticket)==Int32[0]
        @test addresses==map(pointer,(run.r...,run.v...,run.f...,run.fusion.maxbits,run.fusion.ticket))
        tags=sorted ? run.ordering.tags : ids; ctags=sorted ? control.ordering.tags : ids
        perm=sortperm(tags); cperm=sortperm(ctags)
        tol=T===Float32 ? 1e-4 : 3e-11
        for (a,b) in zip((run.r...,run.v...,run.f...),(control.r...,control.v...,control.f...))
            @test norm(a[perm]-b[cperm])<=tol*max(norm(b),1)
        end
        @test DA.observable_status(run;expected_step=300).samples==DA.observable_status(control;expected_step=300).samples
        @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-11
    end
end

@testset "Fused graphs stop consumers on a latched capacity failure" begin
    for conditional in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3));v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,
            rebuild_policy=:displacement,fused=true,spatial_reorder=true,reorder_interval=1)
        graph=DA.capture_step(run;conditional,energy=true)
        DA.run_steps!(run,graph,1)
        forces=map(copy,run.f); stamp=copy(run.observables.stamp)
        copyto!(run.r[1],Float32[-1,0,1])
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        @test all(all(a .== b) for (a,b) in zip(run.f,forces))
        @test all(run.observables.stamp .== stamp)
        step=NL.device_neighbor_status(run.neighbors;check=false).step
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test all(run.observables.stamp .== stamp)
        @test NL.device_neighbor_status(run.neighbors;check=false).step==step
    end
    @test_throws ArgumentError DA.Run((CUDA.zeros(Float32,2),CUDA.zeros(Float32,2),CUDA.zeros(Float32,2)),
        ntuple(_->CUDA.zeros(Float32,2),3),(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=4,dt=0.001,fused=true)
end
end
