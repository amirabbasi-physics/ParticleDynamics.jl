module DeviceNeighborGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
const NL = ParticleDynamics.NeighborLists
const NB = ParticleDynamics.NonBondedForces
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
using .GPUForceReference

# 2D serial LJ over ELL rows (the staged kernel, kept here as the test's reference force).
function lj2_ell_noE_reference_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    neighbors_index::CuDeviceVector{Int32},
    neighbors_flat::CuDeviceVector{Int32},
    counts::CuDeviceVector{Int32}, cap::Int32,
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end

    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)

    nlist = counts[i]

    @inbounds for t in 0:Int(nlist-1)
        j = neighbors_flat[NL._ell_index(i, t, N)]
        dx = NB.periodic_delta(xi, rx[j], halfLx, Lx)
        dy = NB.periodic_delta(yi, ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end

    fx[i] = accx; fy[i] = accy
    return
end

function guarded_force!(ws, r, f)
    nb=ws.list; box=ws.box; T=eltype(r[1]); half=map(x->T(0.5)*x,box)
    if length(r)==2
        @cuda threads=128 blocks=cld(length(r[1]),128) NL._device_if_valid!(
            ws.state,lj2_ell_noE_reference_kernel!,r...,f...,nb.neighbors_index,
            nb.neighbors_flat,nb.counts,nb.cap,box...,half...,T(1),T(1),nb.cutoff^2)
    else
        @cuda threads=128 blocks=cld(length(r[1]),128) NL._device_if_valid!(
            ws.state,NB._lj3_ell_noE_kernel!,r...,f...,nb.neighbors_index,
            nb.neighbors_flat,nb.counts,nb.cap,box...,half...,T(1),T(1),nb.cutoff^2)
    end
    return nothing
end

function check_against_gpu_oracle(ws,r,f)
    T=eltype(r[1]); N=length(r[1]); D=length(r)
    ids=CuArray(Int32.(1:N))
    r3=D==2 ? (r...,CUDA.zeros(T,N)) : r
    box3=D==2 ? (ws.box...,T(100)) : ws.box
    ref=sampled_lj_oracle_gpu(r3...,box3,ids,1,1,ws.list.cutoff)
    F=permutedims(hcat(f...))
    tol=T===Float32 ? 2e-5 : 5e-12
    @test norm(Float64.(F)-ref.F[1:D,:]) <= tol*max(norm(ref.F),1)
    listref=sampled_lj_oracle_gpu(r3...,box3,ids,1,1,ws.list.cutoff+ws.list.skin)
    @test all(ws.list.counts .== listref.counts)
    # Independent library scan and permutation check both execute on CUDA.
    @test all(ws.list.cell_offsets[2:end] .== cumsum(ws.list.cell_counts).+Int32(1))
    @test minimum(ws.list.cell_offsets)==1
    @test maximum(ws.list.cell_offsets)==N+1
    @test all(sort(ws.list.particle_ids_sorted) .== ids)
end

@testset "Device ELL rows, persistent scan and force reference" begin
    for T in (Float32,Float64), D in (2,3), N in (1,33,257)
        box=ntuple(_->T(48),D)
        # Input generation only; all reference forces and list checks use CUDA.
        side=D==2 ? 17 : 7
        r=ntuple(d->CuArray(T[1.35*(mod((i-1)÷side^(d-1),side)-(side-1)/2)+
                             0.015*sin(i*(0.31+0.1d)) for i in 1:N]),D)
        f=ntuple(_->CUDA.zeros(T,N),D)
        ws=NL.DeviceNeighborWorkspace(T,N,box;cutoff=2.5,skin=0.4,cap=96)
        NL.update_device_neighbors!(ws,r...)
        guarded_force!(ws,r,f)
        status=NL.device_neighbor_status(ws)
        @test status.valid && status.builds==1 && status.checks==1
        @test !ws.list.valid # ordinary force entry points must reject this list
        @test_throws ArgumentError NL.require_valid_neighbors(ws.list)
        check_against_gpu_oracle(ws,r,f)
        pointers=map(pointer,(ws.state,ws.displacement,ws.capacity,
            ws.list.neighbors_flat,ws.list.cell_offsets,ws.scan_values...,ws.scan_totals...))

        # Warm every specialization before capture; capture itself executes none
        # of the recorded work. Rebuild/no-rebuild are selected only on device.
        NL.update_device_neighbors!(ws,r...)
        guarded_force!(ws,r,f)
        NL.advance_device_neighbor_step!(ws)
        CUDA.synchronize()
        g=NL.capture_graph() do
            NL.update_device_neighbors!(ws,r...)
            guarded_force!(ws,r,f)
            NL.advance_device_neighbor_step!(ws)
        end
        exec=CUDA.instantiate(g)
        before=NL.device_neighbor_status(ws)
        for _ in 1:3; CUDA.launch(exec); end
        after=NL.device_neighbor_status(ws)
        @test after.step==before.step+3
        @test after.checks==before.checks+3
        @test after.builds==before.builds
        @test pointers==map(pointer,(ws.state,ws.displacement,ws.capacity,
            ws.list.neighbors_flat,ws.list.cell_offsets,ws.scan_values...,ws.scan_totals...))

        # A displacement above skin/2 must take the rebuild path in the SAME
        # captured graph before its force kernel executes.
        r[1] .+= T(0.3)
        CUDA.launch(exec)
        after=NL.device_neighbor_status(ws)
        @test after.builds==before.builds+1
        @test after.interval==18
        check_against_gpu_oracle(ws,r,f)
    end
end

@testset "Three-level persistent GPU scan" begin
    T=Float32; N=33; box=(T(200),T(200),T(200))
    r=(CuArray(T[1.3i-23 for i in 1:N]),CUDA.zeros(T,N),CUDA.zeros(T,N))
    ws=NL.DeviceNeighborWorkspace(T,N,box;cutoff=2.5,skin=0.4,cap=16)
    @test length(ws.scan_values)==3
    NL.update_device_neighbors!(ws,r...)
    @test NL.device_neighbor_status(ws).valid
    f=ntuple(_->CUDA.zeros(T,N),3)
    guarded_force!(ws,r,f)
    check_against_gpu_oracle(ws,r,f)
end

@testset "Device threshold, periodic crossing and adaptive interval" begin
    for T in (Float32,Float64), D in (2,3)
        box=ntuple(_->T(16),D)
        r=(CuArray(T[7.875]),ntuple(_->CUDA.zeros(T,1),D-1)...)
        ws=NL.DeviceNeighborWorkspace(T,1,box;cutoff=1,skin=0.5,cap=1)
        NL.update_device_neighbors!(ws,r...)
        NL.advance_device_neighbor_step!(ws)
        fill!(r[1],T(-7.875)) # wrapped displacement exactly skin/2
        NL.update_device_neighbors!(ws,r...)
        s=NL.device_neighbor_status(ws)
        @test s.builds==1 && s.interval==20
        fill!(r[1],T(-7.75))
        NL.update_device_neighbors!(ws,r...)
        s=NL.device_neighbor_status(ws)
        @test s.builds==2 && s.interval==18 && s.last_build==1
        for _ in 1:18; NL.advance_device_neighbor_step!(ws); end
        NL.update_device_neighbors!(ws,r...)
        s=NL.device_neighbor_status(ws)
        @test s.builds==3 && s.interval==20 && s.last_build==19
        @test all(ws.list.rref_x .== r[1])
    end
end

@testset "Device overflow and coordinate errors stop consumers" begin
    for T in (Float32,Float64)
        r=(CuArray(T[-1,0,1]),CUDA.zeros(T,3),CUDA.zeros(T,3))
        f=ntuple(_->CUDA.fill(T(7),3),3)
        ws=NL.DeviceNeighborWorkspace(T,3,(T(16),T(16),T(16));cutoff=2.5,skin=0.4,cap=1)
        # The initial build fails: no references or force output may be committed.
        fill!(ws.list.rref_x,T(9))
        NL.update_device_neighbors!(ws,r...)
        guarded_force!(ws,r,f)
        NL.advance_device_neighbor_step!(ws)
        s=NL.device_neighbor_status(ws;check=false)
        @test !s.valid && s.error==NL.DN_CAPACITY_ERROR && s.required==2
        @test s.step==0 && s.builds==0
        @test all(ws.list.rref_x .== T(9))
        @test all(all(x .== T(7)) for x in f)
        @test_throws NL.NeighborCapacityError NL.device_neighbor_status(ws)
        # Even if a later configuration fits, the first error stays latched.
        r[1] .*= T(5)
        NL.update_device_neighbors!(ws,r...)
        guarded_force!(ws,r,f)
        @test NL.device_neighbor_status(ws;check=false)==s
        @test all(all(x .== T(7)) for x in f)

        for bad in (T(NaN),T(Inf),T(9))
            r=(CUDA.zeros(T,1),CUDA.zeros(T,1),CUDA.zeros(T,1))
            f=ntuple(_->CUDA.fill(T(7),1),3)
            ws=NL.DeviceNeighborWorkspace(T,1,(T(16),T(16),T(16));cutoff=1,skin=0.5,cap=1)
            NL.update_device_neighbors!(ws,r...)
            guarded_force!(ws,r,f)
            CUDA.synchronize()
            g=NL.capture_graph() do
                NL.update_device_neighbors!(ws,r...)
                guarded_force!(ws,r,f)
                NL.advance_device_neighbor_step!(ws)
            end
            exec=CUDA.instantiate(g)
            fill!(r[1],bad)
            foreach(x->fill!(x,T(7)),f)
            CUDA.launch(exec)
            s=NL.device_neighbor_status(ws;check=false)
            @test !s.valid && s.error==NL.DN_COORDINATE_ERROR && s.step==0
            @test all(all(x .== T(7)) for x in f)
            @test_throws ArgumentError NL.device_neighbor_status(ws)
        end
    end
end
end
