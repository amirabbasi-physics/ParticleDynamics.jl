module DeviceArgonPolicyGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists

function decide(kernel,largest::T,threshold::T;valid=1,error=0,age=1000,interval=20) where {T}
    s=zeros(Int64,22); s[NL.DN_VALID]=valid; s[NL.DN_ERROR]=error
    s[NL.DN_STEP]=age; s[NL.DN_LAST_BUILD]=0; s[NL.DN_INTERVAL]=interval; s[NL.DN_DIAGNOSTICS]=1
    state=CuArray(s); scratch=CuArray(T[largest,zero(T),largest/2])
    @cuda threads=256 blocks=1 kernel(state,scratch,threshold)
    return Array(state)
end

@testset "Displacement-only decision: strict threshold, invalid, nonfinite, no age trigger" begin
    for T in (Float32,Float64)
        thr=T(0.0756)
        s=decide(DA._displacement_decide_kernel!,thr,thr)        # equality: keep
        @test s[NL.DN_REQUEST]==0 && s[NL.DN_VALID]==1 && s[NL.DN_INTERVAL]==20
        a=decide(NL._device_decide_kernel!,thr,thr)                # control: age rebuild
        @test a[NL.DN_REQUEST]==1 && a[NL.DN_REASON]==NL.DN_REASON_AGE
        s=decide(DA._displacement_decide_kernel!,nextfloat(thr),thr)
        @test s[NL.DN_REQUEST]==1 && s[NL.DN_VALID]==0 && s[NL.DN_REASON]==NL.DN_REASON_DISPLACEMENT
        @test s[NL.DN_INTERVAL]==20 # no adaptation
        s=decide(DA._displacement_decide_kernel!,zero(T),thr;valid=0)
        @test s[NL.DN_REQUEST]==1 && s[NL.DN_REASON]==NL.DN_REASON_INVALID
        s=decide(DA._displacement_decide_kernel!,T(NaN),thr)
        @test s[NL.DN_ERROR]==NL.DN_COORDINATE_ERROR && s[NL.DN_REQUEST]==0 && s[NL.DN_VALID]==0
        s=decide(DA._displacement_decide_kernel!,T(10),thr;error=NL.DN_CAPACITY_ERROR)
        @test s[NL.DN_REQUEST]==0 && s[NL.DN_ERROR]==NL.DN_CAPACITY_ERROR && s[NL.DN_CHECKS]==0
        @test DA.validate_policy(:displacement)===nothing
        @test_throws ArgumentError DA.validate_policy(:age)
    end
end

lattice(::Type{T},N,a) where {T}=ntuple(d->CuArray(T[a*(mod((i-1)÷7^(d-1),7)-3)+0.02*sin(i*(0.5+0.1d)) for i in 1:N]),3)

function full_force_error(run)
    N=length(run.r[1]); ids=CuArray(Int32.(1:N))
    ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,run.neighbors.box,ids,run.sigma,run.epsilon,run.neighbors.list.cutoff)
    return norm(permutedims(hcat(run.f...))-ref.F)/max(norm(ref.F),1), maximum(abs,ref.F)
end

@testset "Every-step all-atom forces under displacement-only maintenance" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true)
        N=257; box=ntuple(_->T(20),3); ids=CuArray(Int32.(1:N))
        v=ntuple(d->T(1.5).*sin.(T.(ids).*T(0.3+0.1d)),3)
        run=DA.Run(lattice(T,N,1.3),v,box;cutoff=2.5,skin=0.12,cap=128,dt=0.002,
            spatial_reorder=sorted,reorder_interval=7,rebuild_policy=:displacement)
        graph=DA.capture_step(run;conditional)
        worst=0.0
        for step in 1:300
            CUDA.launch(graph)
            err,_=full_force_error(run); worst=max(worst,err)
        end
        @test worst<=(T===Float32 ? 2e-5 : 3e-12)
        d=NL.device_neighbor_diagnostics(run.neighbors)
        @test NL.device_neighbor_status(run.neighbors).step==300
        @test d.builds>3 && d.age_builds==0
        @test d.builds==d.attempts==d.invalid_builds+d.displacement_builds
    end
end

@testset "Approaching pair starting just outside rc+skin is never missed" begin
    for T in (Float32,Float64), conditional in (false,true), lanes in (1,8)
        cutoff=2.5; skin=0.3; gap=T(cutoff+skin+0.02)
        r=(CuArray(T[-gap/2,gap/2]),CUDA.zeros(T,2),CUDA.zeros(T,2))
        v=(CuArray(T[2,-2]),CUDA.zeros(T,2),CUDA.zeros(T,2))
        run=DA.Run(r,v,ntuple(_->T(20),3);cutoff,skin,cap=8,dt=0.002,
            rebuild_policy=:displacement,build_lanes=lanes,force_lanes=lanes)
        @test all(run.neighbors.list.counts .== 0) # pair starts outside the list radius
        graph=DA.capture_step(run;conditional)
        interacting=0; worst=0.0
        for step in 1:150
            CUDA.launch(graph)
            err,peak=full_force_error(run); worst=max(worst,err)
            peak>0 && (interacting+=1)
        end
        @test interacting>10 # the pair entered the cutoff and was evaluated
        @test worst<=(T===Float32 ? 2e-5 : 3e-12)
        d=NL.device_neighbor_diagnostics(run.neighbors)
        @test d.displacement_builds>=1 && d.age_builds==0
    end
end

@testset "Displacement-only trajectory equals adaptive control within rounding" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true)
        N=257; box=ntuple(_->T(20),3); ids=CuArray(Int32.(1:N))
        v=ntuple(d->T(0.3).*sin.(T.(ids).*T(0.3+0.1d)),3)
        args=(;cutoff=2.5,skin=0.08,cap=96,dt=0.002,spatial_reorder=sorted,reorder_interval=3)
        control=DA.Run(lattice(T,N,1.4),v,box;args...)
        run=DA.Run(lattice(T,N,1.4),v,box;args...,rebuild_policy=:displacement)
        graphs=DA.capture_step(run;conditional),DA.capture_step(run;conditional,energy=true)
        controls=DA.capture_step(control;conditional),DA.capture_step(control;conditional,energy=true)
        for step in 1:100
            CUDA.launch(graphs[step%5==0 ? 2 : 1]);CUDA.launch(controls[step%5==0 ? 2 : 1])
        end
        tags=sorted ? run.ordering.tags : ids; ctags=sorted ? control.ordering.tags : ids
        perm=sortperm(tags); cperm=sortperm(ctags)
        tol=T===Float32 ? 1e-4 : 3e-11
        for (a,b) in zip((run.r...,run.v...,run.f...),(control.r...,control.v...,control.f...))
            @test norm(a[perm]-b[cperm])<=tol*max(norm(b),1)
        end
        @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-11
        @test NL.device_neighbor_diagnostics(run.neighbors).age_builds==0
        @test NL.device_neighbor_status(run.neighbors).builds<=NL.device_neighbor_status(control.neighbors).builds
    end
end

@testset "Displacement-only graphs stop consumers on a latched capacity failure" begin
    for conditional in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3));v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,
            rebuild_policy=:displacement,spatial_reorder=true,reorder_interval=1)
        graph=DA.capture_step(run;conditional,energy=true)
        DA.run_steps!(run,graph,1)
        forces=map(copy,run.f); stamp=copy(run.observables.stamp)
        copyto!(run.r[1],Float32[-1,0,1]) # large jump forces a displacement build that overflows
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        @test all(all(a .== b) for (a,b) in zip(run.f,forces))
        @test all(run.observables.stamp .== stamp)
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test all(run.observables.stamp .== stamp)
    end
end
end
