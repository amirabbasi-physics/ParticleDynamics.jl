module DeviceArgonObservablesGPUChecks
using Test, CUDA, ParticleDynamics
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists

@testset "Persistent global reductions: partial blocks, large final reduction, reuse" begin
    for T in (Float32,Float64), N in (1,33,256,257,65537)
        ids=CuArray(Int32.(1:N))
        energy=sin.(T.(ids)); virial=hcat([cos.(T.(ids).*T(q/10)) for q in 1:6]...)
        v=ntuple(q->sin.(T.(ids).*T(q/7)),3)
        f=ntuple(q->cos.(T.(ids).*T(q/9)),3)
        state=CuArray(Int64[0,1,0,17,0,0,0,0,0])
        o=DA.ObservableWorkspace(N)
        run=(;energy,virial,v,f,mass=T(2.7),dt=T(0.3),
            neighbors=(;state,box=(T(11),T(12),T(13))),observables=o)
        pointers=map(pointer,(o.partial,o.values,o.stamp))
        for rep in 1:3
            # Poison scratch/output; every used element must be overwritten.
            fill!(o.partial,NaN); fill!(o.values,NaN)
            DA.enqueue_observables!(run)
            @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-12
            @test Array(o.stamp)==[17,rep]
            @test map(pointer,(o.partial,o.values,o.stamp))==pointers
            # Half-step averaging must differ detectably from K(v(t)).
            @test all(o.values[15:15] .> o.values[14:14])
            for x in v; x .*= T(0.9); end
        end
    end
end

@testset "Observable graphs: centering, sample stamps, sorting, capture and stale boundaries" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true)
        N=257; ids=CuArray(Int32.(1:N)); box=ntuple(_->T(20),3)
        r=ntuple(d->T(1.4).*(T.(mod.(div.(ids.-1,7^(d-1)),7)).-3),3)
        v=ntuple(d->T(0.2).*sin.(T.(ids).*T(0.3+0.1d)),3)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,mass=2,
            spatial_reorder=sorted,reorder_interval=3)
        @test_throws ErrorException DA.observable_status(run)
        plain=DA.capture_step(run;conditional)
        sampled=DA.capture_step(run;conditional,energy=true)
        @test Array(run.observables.stamp)==[-1,0]
        addresses=map(pointer,(run.observables.partial,run.observables.values,run.observables.stamp))
        last_values=copy(run.observables.values)
        # Includes unsampled steps after a sample, repeated samples and rebuilds.
        for step in 1:21
            sampled_step=step in (1,5,6,10,20)
            CUDA.launch(sampled_step ? sampled : plain)
            if sampled_step
                status=DA.observable_status(run;expected_step=step)
                @test status.samples==count(s->s<=step,(1,5,6,10,20))
                ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,box,ids,1,1,2.5)
                @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-12
                @test GPUForceReference.observable_errors(run;pair_reference=ref)["max_scaled_error"]<(T===Float32 ? 5e-5 : 3e-11)
                copyto!(last_values,run.observables.values)
            else
                @test all(run.observables.values .== last_values)
                @test_throws ErrorException DA.observable_status(run;expected_step=step)
            end
            @test map(pointer,(run.observables.partial,run.observables.values,run.observables.stamp))==addresses
        end
        sorted && @test DA.ordering_status(run).count>1
    end
end

@testset "Failed sampled graphs preserve prior global values and sample counters" begin
    for conditional in (false,true), sorted in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3))
        v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,
            spatial_reorder=sorted,reorder_interval=1)
        graph=DA.capture_step(run;conditional,energy=true)
        DA.run_steps!(run,graph,1)
        @test DA.observable_status(run;expected_step=1).samples==1
        values=copy(run.observables.values); stamp=copy(run.observables.stamp)
        copyto!(run.r[1],Float32[-1,0,1])
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        @test_throws NL.NeighborCapacityError DA.observable_status(run)
        @test all(run.observables.values .== values)
        @test all(run.observables.stamp .== stamp)
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test all(run.observables.values .== values)
        @test all(run.observables.stamp .== stamp)
    end
end
end
