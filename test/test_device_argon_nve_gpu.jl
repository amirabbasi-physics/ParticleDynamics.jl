module DeviceArgonNVEGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
include(joinpath(@__DIR__,"../benchmark/gromacs/DeviceArgonNVE.jl"))
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
const DA=DeviceArgonNVE
const SC=ParticleDynamics.SimulationCore

@testset "Device ELL NVE graph versus staged GPU integration" begin
    for T in (Float32,Float64), N in (2,257), conditional in (false,true), spatial_reorder in (false,true)
        box=ntuple(_->T(20),3); dt=T(0.002); mass=T(2)
        st=build_simulation(;N,box,nonbonded=:lj,cutoff=2.5,skin=0.08,cap=Int32(96),mass,
            spatial_reorder=false,temperature=0,precision=T===Float32 ? :f32 : :f64)
        pos=N==2 ? (T[9.98,-8.62],T[0,0],T[0,0]) :
            ntuple(d->T[1.4*(mod((i-1)÷7^(d-1),7)-3)+0.013*sin(i*(0.5+0.1d)) for i in 1:N],3)
        for (r,x) in zip((st.rx,st.ry,st.rz),pos); copyto!(r,x); end
        ids=CuArray(Int32.(1:N))
        for (d,v) in enumerate((st.vx,st.vy,st.vz))
            v .= T(0.2).*sin.(T.(ids).*T(0.3+0.1d))
        end
        N==2 && fill!(st.vx,T(1))
        r=(st.rx,st.ry,st.rz); v=(st.vx,st.vy,st.vz)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.08,cap=96,dt,mass,spatial_reorder,reorder_interval=3)
        before=map(copy,(run.r...,run.v...))
        arrays=(run.r...,run.v...,run.f...,run.energy,run.virial,
            run.neighbors.list.cell_of_particle,run.neighbors.list.particle_ids_sorted)
        addresses=map(pointer,arrays)
        initial_order=DA.ordering_status(run)
        graph=DA.capture_step(run;conditional)
        energy_graph=DA.capture_step(run;conditional,energy=true)
        @test all(all(a .== b) for (a,b) in zip(before,(run.r...,run.v...)))
        @test ParticleDynamics.NeighborLists.device_neighbor_status(run.neighbors).step==0
        @test DA.ordering_status(run)==initial_order
        spec=SC.nve(st)
        total=0
        # Odd/even boundaries, an empty batch, forced early rebuilds, and PBC.
        for n in (1,2,0,17,31)
            for i in 1:n
                CUDA.launch((total+i)%5==0 ? energy_graph : graph)
            end
            status=ParticleDynamics.NeighborLists.device_neighbor_status(run.neighbors)
            for _ in 1:n; SC.step!(st,spec,dt;compute_energy=(st.step+1)%5==0); end
            total+=n
            @test status.step==st.step==total
            tags=spatial_reorder ? run.ordering.tags : ids
            # CUDA gather compares physical identities despite storage changes.
            @test all(sort(copy(tags)) .== ids)
            @test map(pointer,arrays)==addresses
            # Also check the fields currently held by run, not just saved arrays.
            @test map(pointer,(run.r...,run.v...,run.f...,run.energy,run.virial,
                run.neighbors.list.cell_of_particle,run.neighbors.list.particle_ids_sorted))==addresses
            tol=T===Float32 ? 4e-5 : 2e-11
            for (a,b) in zip((run.r...,run.v...,run.f...),
                              (st.rx,st.ry,st.rz,st.vx,st.vy,st.vz,st.fx,st.fy,st.fz))
                @test norm(a-b[tags]) <= tol*max(norm(b),1)
            end
            # Independent Float64 CUDA force/energy/virial reference at the
            # actual replayed coordinates, not the staged trajectory's state.
            if total%5==0 && n>0
                @test abs(sum(run.energy)-SC.total_potential_energy(st)) <= tol*max(abs(sum(run.energy)),1)
                @test norm(run.virial-st.virial_nonbonded[tags,:]) <= tol*max(norm(run.virial),1)
            end
            DA.enqueue_force!(run;energy=true)
            ref=GPUForceReference.sampled_lj_oracle_gpu(run.r...,box,ids,1,1,2.5)
            F=permutedims(hcat(run.f...))
            @test norm(F-ref.F) <= tol*max(norm(ref.F),1)
            @test abs(sum(run.energy)-sum(ref.energy)) <= tol*max(abs(sum(ref.energy)),1)
            @test norm(vec(sum(run.virial;dims=1))-vec(sum(ref.virial;dims=2))) <= tol*max(norm(ref.virial),1)
        end
        @test ParticleDynamics.NeighborLists.device_neighbor_status(run.neighbors).builds>1
        if spatial_reorder
            order=DA.ordering_status(run)
            @test 1<order.count<=1+total÷3
            @test 0<order.last_step<total
        end
        N==2 && @test maximum(run.r[1])<0
    end
end

@testset "Device sorting permutes all state and respects its interval" begin
    for T in (Float32,Float64)
        N=33; box=ntuple(_->T(40),3)
        ids=CuArray(Int32.(1:N))
        r=(T.(mod.(ids.*13, N)).-T(16),CUDA.zeros(T,N),CUDA.zeros(T,N))
        v=ntuple(d->T.(ids).*T(0.01d),3)
        run=DA.Run(r,v,box;cutoff=2.5,skin=0.4,cap=16,dt=0.001,spatial_reorder=true,reorder_interval=5)
        @test DA.ordering_status(run)==(count=1,last_step=0)
        # Unique per-identity values catch accidental velocity/force/observable
        # loss. Exercise only sorting, so force recomputation cannot hide loss.
        fields=(run.r...,run.v...,run.f...,run.energy)
        for d in 4:10; fields[d] .= T(d).*T.(run.ordering.tags); end
        for d in 1:6; run.virial[:,d] .= T(d+10).*T.(run.ordering.tags); end
        NL=ParticleDynamics.NeighborLists
        for _ in 1:5; NL.advance_device_neighbor_step!(run.neighbors); end
        # Change cell order and force a rebuild request on device. This is
        # explicit fault-free test setup, outside any captured trajectory.
        run.r[1] .*= -one(T)
        NL.decide_device_neighbors!(run.neighbors,run.r...)
        NL._device_bin!(run.neighbors,run.r)
        DA.enqueue_ordering!(run,run.ordering)
        @test DA.ordering_status(run)==(count=2,last_step=5)
        @test all(sort(copy(run.ordering.tags)) .== ids)
        @test all(run.r[1] .== -r[1][run.ordering.tags])
        for d in 4:10
            @test all(fields[d] .== T(d).*T.(run.ordering.tags))
        end
        for d in 1:6
            @test all(run.virial[:,d] .== T(d+10).*T.(run.ordering.tags))
        end
        @test all(run.neighbors.list.particle_ids_sorted .== ids)
        cells=run.neighbors.list.cell_of_particle
        @test all(cells[2:end] .>= cells[1:end-1])
        # A second request at the same step must not sort again.
        DA.enqueue_ordering!(run,run.ordering)
        @test DA.ordering_status(run)==(count=2,last_step=5)
    end
end

@testset "NVE graph stops on capacity failure after a valid build" begin
    NL=ParticleDynamics.NeighborLists
    for conditional in (false,true), spatial_reorder in (false,true)
        r=(CuArray(Float32[-5,0,5]),CUDA.zeros(Float32,3),CUDA.zeros(Float32,3))
        v=ntuple(_->CUDA.zeros(Float32,3),3)
        run=DA.Run(r,v,(16f0,16f0,16f0);cutoff=2.5,skin=0.4,cap=1,dt=0.001,
            spatial_reorder,reorder_interval=1)
        graph=DA.capture_step(run;conditional)
        refs=copy(run.neighbors.list.rref_x)
        forces=map(copy,run.f)
        # Let sorting be due on the failing rebuild, if enabled.
        spatial_reorder && NL.advance_device_neighbor_step!(run.neighbors)
        initial_step=spatial_reorder ? 1 : 0
        # Deliberate fault injection: the next checked configuration needs two
        # entries per row. No force, final kick, or step commit may follow it.
        copyto!(run.r[1],Float32[-1,0,1])
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,1)
        failed=NL.device_neighbor_status(run.neighbors;check=false)
        @test failed.step==initial_step && failed.builds==1 && failed.required==2
        @test all(run.neighbors.list.rref_x .== refs)
        @test all(all(a .== b) for (a,b) in zip(run.f,forces))
        @test all(all(x .== 0) for x in run.v)
        order_failed=DA.ordering_status(run)
        @test_throws NL.NeighborCapacityError DA.run_steps!(run,graph,3)
        @test NL.device_neighbor_status(run.neighbors;check=false)==failed
        @test DA.ordering_status(run)==order_failed
    end
end
end
