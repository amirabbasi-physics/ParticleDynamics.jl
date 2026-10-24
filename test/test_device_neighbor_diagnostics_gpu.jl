module DeviceNeighborDiagnosticsGPUChecks
using Test, CUDA, ParticleDynamics
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
const DA=DeviceArgonNVE
const NL=ParticleDynamics.NeighborLists

# Independent all-particle CUDA enumeration of cell candidates and accepted
# distances. No production row traversal or reduction tree is used.
function search_reference!(out,coords,cells,ncells,box,radius2::Float64)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(cells) && return
    D=length(coords); tests=Int64(0); accepted=Int64(0)
    @inbounds for j in 1:length(cells)
        i==j && continue
        ci=cells[i]; cj=cells[j]; near=true; r2=0.0
        for d in 1:D
            a=ci%ncells[d]; b=cj%ncells[d]
            gap=abs(a-b); near &= min(gap,ncells[d]-gap)<=1
            ci÷=ncells[d]; cj÷=ncells[d]
            dx=Float64(coords[d][j])-Float64(coords[d][i])
            L=Float64(box[d]); dx-=L*round(dx/L); r2+=dx*dx
        end
        tests+=Int64(near); accepted+=Int64(near && r2<=radius2)
    end
    @inbounds begin out[i,1]=tests; out[i,2]=accepted; end
    return
end

function check_search(ws,r)
    N=length(r[1]); nb=ws.list; D=length(r)
    out=CUDA.zeros(Int64,N,2)
    nc=D==2 ? (nb.nx,nb.ny) : (nb.nx,nb.ny,nb.nz)
    @cuda threads=128 blocks=cld(N,128) search_reference!(out,r,nb.cell_of_particle,nc,
        ws.box,Float64((nb.cutoff+nb.skin)^2))
    d=NL.device_neighbor_diagnostics(ws)
    @test all(ws.candidates .== out[:,1])
    @test all(nb.counts .== out[:,2])
    @test d.last_candidates==sum(out[:,1])
    @test d.last_accepted==sum(out[:,2])
    @test d.last_max_row==maximum(out[:,2])
    @test d.invalid_builds+d.displacement_builds+d.age_builds==d.builds==d.attempts
end

@testset "Exclusive rebuild causes, simultaneous triggers, and search totals" begin
    for T in (Float32,Float64), D in (2,3), row_threads in (64,128,256)
        r=(CuArray(T[-1.5,0,1.5]),ntuple(_->CUDA.zeros(T,3),D-1)...)
        ws=NL.DeviceNeighborWorkspace(T,3,ntuple(_->T(16),D);
            cutoff=2.5,skin=0.5,cap=8,interval=5,row_threads)
        NL.update_device_neighbors!(ws,r...)
        d=NL.device_neighbor_diagnostics(ws)
        @test (d.invalid_builds,d.displacement_builds,d.age_builds)==(1,0,0)
        check_search(ws,r)
        initial=d
        for _ in 1:4; NL.advance_device_neighbor_step!(ws); end
        # No build before the exact age threshold.
        NL.update_device_neighbors!(ws,r...)
        @test NL.device_neighbor_diagnostics(ws)==initial
        NL.advance_device_neighbor_step!(ws)
        NL.update_device_neighbors!(ws,r...)
        d=NL.device_neighbor_diagnostics(ws)
        @test (d.invalid_builds,d.displacement_builds,d.age_builds)==(1,0,1)
        check_search(ws,r)
        # Exact skin/2 equality retains coverage, independent of age.
        r[1] .+= T(0.25)
        NL.update_device_neighbors!(ws,r...)
        @test NL.device_neighbor_diagnostics(ws)==d
        r[1] .+= T(0.125)
        NL.update_device_neighbors!(ws,r...)
        d=NL.device_neighbor_diagnostics(ws)
        @test (d.invalid_builds,d.displacement_builds,d.age_builds)==(1,1,1)
        @test d.displacement_age_builds==0
        check_search(ws,r)
        for _ in 1:NL.device_neighbor_status(ws).interval; NL.advance_device_neighbor_step!(ws); end
        r[1] .+= T(0.375)
        NL.update_device_neighbors!(ws,r...)
        d=NL.device_neighbor_diagnostics(ws)
        @test (d.invalid_builds,d.displacement_builds,d.age_builds)==(1,2,1)
        @test d.displacement_age_builds==1
        check_search(ws,r)
        # Force invalidity with unchanged positions; tests only the first-list
        # branch and does not alter coverage policy in production.
        ws.state[NL.DN_VALID:NL.DN_VALID] .= Int64(0)
        NL.update_device_neighbors!(ws,r...)
        d=NL.device_neighbor_diagnostics(ws)
        @test (d.invalid_builds,d.displacement_builds,d.age_builds)==(2,2,1)
        @test d.total_candidates==5d.last_candidates
        @test d.total_accepted==5d.last_accepted
        check_search(ws,r)
    end
end

@testset "Search diagnostics span partial row blocks and capacity tiles" begin
    for T in (Float32,Float64), row_threads in (64,128,256), N in (1,33,257,513)
        box=(T(48),T(48),T(48))
        r=ntuple(d->CuArray(T[1.35*(mod((i-1)÷9^(d-1),9)-4)+0.01*sin(i*(0.3+0.1d)) for i in 1:N]),3)
        ws=NL.DeviceNeighborWorkspace(T,N,box;cutoff=2.5,skin=0.4,cap=96,row_threads)
        NL.update_device_neighbors!(ws,r...)
        check_search(ws,r)
    end
end

@testset "Diagnostics preserve errors and report untruncated attempted rows" begin
    for row_threads in (64,128,256)
        r=(CuArray(Float32[-1,0,1]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3))
        ws=NL.DeviceNeighborWorkspace(Float32,3,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,row_threads)
        NL.update_device_neighbors!(ws,r...)
        @test_throws NL.NeighborCapacityError NL.device_neighbor_diagnostics(ws)
        d=NL.device_neighbor_diagnostics(ws;check=false)
        @test d.attempts==1 && d.builds==0 && d.invalid_builds==0
        @test d.last_candidates==d.last_accepted==6 && d.max_row==2
        NL.update_device_neighbors!(ws,r...)
        @test NL.device_neighbor_diagnostics(ws;check=false)==d
    end
end

@testset "Counter-on/off graphs have identical trajectories and stable buffers" begin
    for T in (Float32,Float64), conditional in (false,true), sorted in (false,true)
        # One particle/cell makes atomic cell scatter unambiguous. Interacting
        # pairs occupy neighboring cells; this isolates counters from scatter
        # permutation nondeterminism present in denser independent runs.
        N=66; cell=T(2.58); box=ntuple(_->T(48),3)
        r=ntuple(3) do d
            CuArray(T[ d==1 ? -24+2cell*(2+mod((i-1)÷2,5))+(isodd(i) ? -0.7 : 0.7) :
                d==2 ? -24+2cell*(2+mod((i-1)÷10,5))+0.4 :
                       -24+2cell*(2+(i-1)÷50)+0.4 for i in 1:N])
        end
        v=(CUDA.fill(T(0.5),N),CUDA.zeros(T,N),CUDA.zeros(T,N))
        runs=[DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=16,dt=0.002,
            spatial_reorder=sorted,reorder_interval=3,diagnostics=enabled) for enabled in (false,true)]
        graphs=[(DA.capture_step(run;conditional),DA.capture_step(run;conditional,energy=true)) for run in runs]
        candidate=runs[2]; ws=candidate.neighbors
        ptrs=map(pointer,(ws.state,ws.candidates,ws.statistics_partial,ws.capacity))
        for step in 1:100
            for (run,g) in zip(runs,graphs); CUDA.launch(g[step%5==0 ? 2 : 1]); end
            for (a,b) in zip((runs[1].r...,runs[1].v...,runs[1].f...),
                             (runs[2].r...,runs[2].v...,runs[2].f...))
                # Bit comparisons also distinguish -0.0 from 0.0.
                U=T===Float32 ? UInt32 : UInt64
                @test all(reinterpret(U,a) .== reinterpret(U,b))
            end
            @test NL.device_neighbor_status(runs[1].neighbors)==NL.device_neighbor_status(ws)
            @test map(pointer,(ws.state,ws.candidates,ws.statistics_partial,ws.capacity))==ptrs
            if step%5==0
                @test all(reinterpret(UInt64,runs[1].observables.values) .== reinterpret(UInt64,candidate.observables.values))
            end
        end
        d=NL.device_neighbor_diagnostics(ws)
        @test d.attempts==d.builds==d.invalid_builds+d.displacement_builds+d.age_builds
        @test d.sorting_builds==DA.ordering_status(candidate).count
        @test d.builds>1
    end
end
end
