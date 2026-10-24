module DeviceArgonBuilderGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists

function positions(::Type{T},N,L,kind) where {T}
    # Irregular occupancy: a perturbed lattice, a dense cluster with empty cells,
    # and atoms placed exactly at periodic faces. Generated on the GPU.
    ids=CuArray(Float64.(1:N))
    if kind==:lattice
        m=ceil(Int,cbrt(N)); a=L/m
        return ntuple(d->T.(a.*(mod.(div.(ids.-1,m^(d-1)),m).+0.5).-L/2 .+0.07a.*sin.(ids.*(0.7+0.3d))),3)
    elseif kind==:cluster
        return ntuple(d->T.(0.3L.*sin.(ids.*(1.3+0.17d)).*abs.(cos.(ids.*0.11d))),3)
    else
        r=ntuple(d->T.(0.49L.*sin.(ids.*(2.1+0.3d))),3)
        k=min(N,8)
        for d in 1:3; r[d][1:k].=T(L/2).*T.(((-1).^(1:k)))[1:k]|>CuArray; end
        return r
    end
end

function row_equal(flat_a,flat_b,counts,N,cap)
    a=reshape(flat_a,N,cap); b=reshape(flat_b,N,cap)
    used=CuArray(reshape(collect(0:cap-1),1,cap)) .< min.(counts,Int32(cap))
    return all((a .== b) .| .!used)
end

@testset "Cooperative rows equal the serial builder (order, counts, candidates, truncation)" begin
    truncated=Set{Tuple{DataType,Int}}()
    for T in (Float32,Float64), (N,kind) in ((1,:lattice),(17,:faces),(33,:cluster),(257,:lattice),(4099,:cluster),(4099,:faces)),
        (cap,skin) in ((96,0.55),(7,0.35)), G in (2,4,8,16,32)
        L=20.0; box=ntuple(_->T(L),3)
        r=positions(T,N,L,kind); v=ntuple(_->CUDA.zeros(T,N),3)
        run=try
            DA.Run(r,v,box;cutoff=2.5,skin,cap,dt=0.002)
        catch err
            err isa NL.NeighborCapacityError || rethrow()
            nothing
        end
        # A capacity failure latches before rows can be compared through Run;
        # use an uncommitted workspace instead for the truncation fixtures.
        ws=run===nothing ? NL.DeviceNeighborWorkspace(T,N,box;cutoff=2.5,skin,cap) : run.neighbors
        coords=run===nothing ? r : run.r
        nb=ws.list; refs=(nb.rref_x,nb.rref_y,nb.rref_z::CuArray{T,1})
        ws.state[NL.DN_ERROR:NL.DN_ERROR].=0
        ws.state[NL.DN_REQUEST:NL.DN_REQUEST].=1
        NL._device_bin!(ws,coords)
        NL._device_finish_rebuild!(ws,coords,refs)
        counts=copy(nb.counts); flat=copy(nb.neighbors_flat); tested=copy(ws.candidates)
        ws.state[NL.DN_ERROR:NL.DN_ERROR].=0
        ws.state[NL.DN_REQUEST:NL.DN_REQUEST].=1
        fill!(nb.counts,Int32(-1)); fill!(nb.neighbors_flat,Int32(-7)); fill!(ws.candidates,-1)
        DA.finish_rebuild_cooperative!(ws,coords,refs,Val(G))
        @test all(nb.counts .== counts)
        @test all(ws.candidates .== tested)
        @test row_equal(nb.neighbors_flat,flat,counts,N,cap)
        maximum(counts)>cap && push!(truncated,(T,G))
    end
    # Capacity truncation must be exercised for every precision and lane count.
    @test truncated==Set((T,G) for T in (Float32,Float64) for G in (2,4,8,16,32))
end

@testset "Cooperative builder graph replay versus serial control" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true), G in (2,4,8,16,32), F in (1,8)
        N=257; box=ntuple(_->T(20),3)
        r=positions(T,N,20.0,:lattice)
        ids=CuArray(Int32.(1:N)); v=ntuple(d->T(0.3).*sin.(T.(ids).*T(0.3+0.1d)),3)
        control=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,spatial_reorder=sorted,
            reorder_interval=3,force_lanes=F)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,spatial_reorder=sorted,
            reorder_interval=3,force_lanes=F,build_lanes=G)
        graphs=DA.capture_step(run;conditional),DA.capture_step(run;conditional,energy=true)
        controls=DA.capture_step(control;conditional),DA.capture_step(control;conditional,energy=true)
        addresses=map(pointer,(run.r...,run.v...,run.f...,run.neighbors.list.neighbors_flat,run.neighbors.list.counts))
        for step in 1:100
            CUDA.launch(graphs[step%5==0 ? 2 : 1]);CUDA.launch(controls[step%5==0 ? 2 : 1])
        end
        status=NL.device_neighbor_status(run.neighbors)
        @test status.step==100 && status.builds>1
        d=NL.device_neighbor_diagnostics(run.neighbors)
        @test d.attempts==d.builds==d.invalid_builds+d.displacement_builds+d.age_builds
        @test addresses==map(pointer,(run.r...,run.v...,run.f...,run.neighbors.list.neighbors_flat,run.neighbors.list.counts))
        tags=sorted ? run.ordering.tags : ids; ctags=sorted ? control.ordering.tags : ids
        perm=sortperm(tags); cperm=sortperm(ctags)
        tol=T===Float32 ? 1e-4 : 3e-11
        for (a,b) in zip((run.r...,run.v...,run.f...),(control.r...,control.v...,control.f...))
            @test norm(a[perm]-b[cperm])<=tol*max(norm(b),1)
        end
        ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,box,ids,1,1,2.5)
        @test norm(permutedims(hcat(run.f...))-ref.F)<=(T===Float32 ? 2e-5 : 3e-12)*max(norm(ref.F),1)
        @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-11
    end
end

@testset "Cooperative builder stops consumers on a latched capacity failure" begin
    for G in (2,4,8,16,32), conditional in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3));v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,build_lanes=G,
            spatial_reorder=true,reorder_interval=1)
        graph=DA.capture_step(run;conditional,energy=true)
        DA.run_steps!(run,graph,1)
        forces=map(copy,run.f); stamp=copy(run.observables.stamp)
        copyto!(run.r[1],Float32[-1,0,1])
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        @test all(all(a .== b) for (a,b) in zip(run.f,forces))
        @test all(run.observables.stamp .== stamp)
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test all(run.observables.stamp .== stamp)
    end
end
end
