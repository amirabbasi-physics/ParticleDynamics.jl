module DeviceArgonTraversalGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists
const VARIANTS=((1,:ell),(1,:tiled),(2,:ell),(2,:tiled),(4,:ell),(4,:tiled),(8,:ell),(8,:tiled))

# Independent inverse of the physical tiled address. Audit every populated
# destination against synthetic slot-major contents, without production helpers.
function packed_reference_check!(ok,packed,counts,groups,::Val{G}) where {G}
    k=(blockIdx().x-1)*blockDim().x+threadIdx().x
    k>length(packed) && return
    index=k-1; warp=index÷32; lane=index%32
    atom=(warp÷groups)*(32÷G)+lane÷G+1
    slot=(warp%groups)*G+lane%G
    valid=atom<=length(counts) && slot<counts[atom]
    @inbounds ok[k]=!valid || packed[k]==slot*length(counts)+atom
    return nothing
end

@testset "Tiled packing: irregular counts, padding, reuse and stable addresses" begin
    for G in (1,2,4,8), N in (1,17,33,257)
        cap=13; ws=DA.TraversalWorkspace(N,cap,Val(G),Val(:tiled))
        ids=CuArray(Int32.(1:N)); counts=mod.(ids,Int32(cap+1))
        ell=CuArray(Int32.(1:N*cap)); state=CuArray(Int64[1,1,0,0,0,20,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        addresses=map(pointer,(ws.packed,ws.packs))
        for _ in 1:3
            fill!(ws.packed,Int32(-999))
            @cuda threads=128 blocks=cld(N,128) DA._pack_tiled_neighbors_kernel!(state,ws.packed,ell,counts,ws.slot_groups,Val(G))
            ok=CUDA.zeros(Bool,length(ws.packed))
            @cuda threads=128 blocks=cld(length(ok),128) packed_reference_check!(ok,ws.packed,counts,ws.slot_groups,Val(G))
            @test all(ok)
            @test map(pointer,(ws.packed,ws.packs))==addresses
            counts=mod.(counts.+Int32(3),Int32(cap+1))
        end
    end
end

function compare_reference(run)
    N=length(run.r[1]); T=eltype(run.r[1]); ids=CuArray(Int32.(1:N))
    DA.enqueue_force!(run;energy=true)
    DA.enqueue_observables!(run)
    ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,run.neighbors.box,ids,
        run.sigma,run.epsilon,run.neighbors.list.cutoff)
    err=GPUForceReference.gpu_force_errors(run.f...,ids,ref.F)
    scale=sqrt(sum(abs2,ref.pair_scale)/N)
    @test err["rms_absolute"] <= (T===Float32 ? 2e-7 : 3e-14)*scale
    @test abs(sum(Float64.(run.energy))-sum(ref.energy))<=
        (T===Float32 ? 2e-5 : 3e-12)*max(abs(sum(ref.energy)),1)
    @test norm(vec(sum(Float64.(run.virial);dims=1))-vec(sum(ref.virial;dims=2)))<=
        (T===Float32 ? 2e-5 : 3e-12)*max(norm(ref.virial),1)
    @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-11
    return nothing
end

@testset "Cooperative force U/W: partial warps, empty rows and boundary pairs" begin
    for T in (Float32,Float64), N in (1,2,17,33,257), (G,L) in VARIANTS
        box=ntuple(_->T(20),3)
        pos=N==2 ? (T[9.98,-8.62],T[0,0],T[0,0]) :
            ntuple(d->T[1.4*(mod((i-1)÷7^(d-1),7)-3)+0.013*sin(i*(0.5+0.1d)) for i in 1:N],3)
        if N==17
            for d in 1:3; pos[d][end]=T(8); end
        elseif N==33
            for i in 29:33
                pos[1][i]=T(-8+4*(i-29));pos[2][i]=T(8);pos[3][i]=T(8)
            end
        end
        r=map(CuArray,pos); v=ntuple(_->CUDA.zeros(T,N),3)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,force_lanes=G,force_layout=L)
        # N=1 is empty; the other fixtures have nonuniform row lengths and
        # sizes not divisible by the atom-group or warp particle count.
        N==1 && @test all(run.neighbors.list.counts .== 0)
        if N in (17,33)
            @test minimum(run.neighbors.list.counts)==0
            @test maximum(run.neighbors.list.counts)>0
        end
        compare_reference(run)
        status=DA.traversal_status(run)
        @test status.packs==(L===:tiled ? 1 : 0)
        DA.enqueue_force!(run;energy=false)
        ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,box,CuArray(Int32.(1:N)),1,1,2.5)
        @test norm(permutedims(hcat(run.f...))-ref.F)<=
            (T===Float32 ? 2e-5 : 3e-12)*max(norm(ref.F),1)
    end
end

@testset "Cooperative graph replay, sorted identities and packed rebuild lifetime" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true), G in (1,2,4,8)
        N=33; box=ntuple(_->T(20),3)
        r=ntuple(d->CuArray(T[1.4*(mod((i-1)÷4^(d-1),4)-1.5)+0.01sin(i*(0.5+0.1d)) for i in 1:N]),3)
        ids=CuArray(Int32.(1:N)); v=ntuple(d->T(0.3).*sin.(T.(ids).*T(0.3+0.1d)),3)
        control=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,
            spatial_reorder=sorted,reorder_interval=3)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt=0.002,
            spatial_reorder=sorted,reorder_interval=3,force_lanes=G,force_layout=:tiled)
        before=map(copy,(run.r...,run.v...)); pack_before=DA.traversal_status(run).packs
        graphs=DA.capture_step(run;conditional),DA.capture_step(run;conditional,energy=true)
        controls=DA.capture_step(control;conditional),DA.capture_step(control;conditional,energy=true)
        @test DA.traversal_status(run).packs==pack_before
        @test all(all(a .== b) for (a,b) in zip(before,(run.r...,run.v...)))
        addresses=map(pointer,(run.r...,run.v...,run.f...,run.traversal.packed,run.traversal.packs))
        for step in 1:100
            CUDA.launch(graphs[step%5==0 ? 2 : 1]);CUDA.launch(controls[step%5==0 ? 2 : 1])
        end
        status=NL.device_neighbor_status(run.neighbors)
        @test status.step==100 && status.builds>1
        @test DA.traversal_status(run).packs==status.builds
        @test addresses==map(pointer,(run.r...,run.v...,run.f...,run.traversal.packed,run.traversal.packs))
        tags=sorted ? run.ordering.tags : ids; ctags=sorted ? control.ordering.tags : ids
        # Compare original identity on CUDA; sorting may choose a different
        # storage permutation even though the physical systems correspond.
        perm=sortperm(tags); cperm=sortperm(ctags)
        tol=T===Float32 ? 1e-4 : 3e-11
        for (a,b) in zip((run.r...,run.v...,run.f...),(control.r...,control.v...,control.f...))
            @test norm(a[perm]-b[cperm])<=tol*max(norm(b),1)
        end
        @test GPUForceReference.observable_errors(run)["max_scaled_error"]<3e-11
        compare_reference(run)
        sorted && @test DA.ordering_status(run).count>1
    end
end

@testset "Cooperative consumers/packing stop on a latched capacity failure" begin
    for G in (1,2,4,8), L in (:ell,:tiled), conditional in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3));v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,
            force_lanes=G,force_layout=L,spatial_reorder=true,reorder_interval=1)
        graph=DA.capture_step(run;conditional,energy=true)
        DA.run_steps!(run,graph,1)
        forces=map(copy,run.f); packs=copy(run.traversal.packs)
        packed=L===:tiled ? copy(run.traversal.packed) : nothing
        copyto!(run.r[1],Float32[-1,0,1])
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        @test all(all(a .== b) for (a,b) in zip(run.f,forces))
        @test all(run.traversal.packs .== packs)
        L===:tiled && @test all(run.traversal.packed .== packed)
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test all(run.traversal.packs .== packs)
    end
end
end
