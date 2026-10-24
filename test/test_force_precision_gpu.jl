module ForcePrecisionGPUChecks
using Test, CUDA, ParticleDynamics, LinearAlgebra
const SC = ParticleDynamics.SimulationCore
include(joinpath(@__DIR__,"../benchmark/gromacs/GPUForceReference.jl"))
using .GPUForceReference

@testset "Independent CUDA reference analytic checks" begin
    for T in (Float32,Float64)
        z=CUDA.zeros(T,2); ids=CuArray(Int32[1,2])
        r=sampled_lj_oracle_gpu(CuArray(T[0,1]),z,z,(10.0,10.0,10.0),ids,1.0,1.0,2.5)
        @test Array(r.F) == [-24.0 24.0; 0.0 0.0; 0.0 0.0]
        @test Array(r.energy) == [0.0,0.0]
        @test Array(sum(r.virial;dims=2))[:] == [24.0,0.0,0.0,0.0,0.0,0.0]
        @test Array(r.counts) == [1.0,1.0]
        r=sampled_lj_oracle_gpu(CuArray(T[0,2]),z,z,(10.0,10.0,10.0),ids,1.0,1.0,2.5)
        @test Array(r.F) == [93/512 -93/512; 0.0 0.0; 0.0 0.0]
        @test Array(r.energy) == [-63/2048,-63/2048]
        @test Array(sum(r.virial;dims=2))[:] == [-93/256,0.0,0.0,0.0,0.0,0.0]
    end
end

@testset "LJ periodic force precision and cluster packing (GPU)" begin
    positions = [-73.96331 74.49271 -30.13671 -28.84318 2.14 2.18 40.2 40.25 -5.1309 -3.8938;
                 0.2 0.24 25.25 25.28 -74.13124 73.93639 30.14 30.2 -5.6216 -5.6417;
                 0.3 0.35 25.32 25.4 35.17 35.2 -74.04009 74.11345 -5.2937 -5.2491]
    for T in (Float32,Float64), backend in (:ell,:allpairs,:cluster), image in (:centered,:positive)
        L=T(149.38015821857212); N=size(positions,2)
        R=T.(positions)
        if image === :positive
            R .+= L/T(2)
        end
        st=build_simulation(N=N,box=(L,L,L),nonbonded=:lj,cutoff=2.5,sigma=1,epsilon=1,
            skin=0.4,cap=Int32(64),neigh_interval=20,spatial_reorder=false,
            use_neighborlist=backend !== :allpairs,precision=T === Float32 ? :f32 : :f64,
            mass=1,temperature=0)
        for (d,r) in enumerate((st.rx,st.ry,st.rz)); copyto!(r,R[d,:]); end
        backend === :cluster && set_nonbonded_backend!(st,:cluster;prune_buffer=0.05)
        ids=CuArray(Int32.(1:N))
        tol=T === Float32 ? 2e-5 : 5e-12
        for phase in (:initial,:small_move,:crossing)
            if phase !== :initial
                for (d,r) in enumerate((st.rx,st.ry,st.rz))
                    if phase === :small_move
                        r .+= T.(0.017 .* sin.(Float64.(ids).*(0.73+0.11d)))
                    else
                        r .+= T(0.33)
                        # The resulting stored coordinates are the reference;
                        # this test does not assume a wrap is bitwise reversible.
                        r .= r .- L.*floor.(r./L .+ T(0.5))
                    end
                end
            end
            SC.evaluate_forces_into_f!(st,true)
            reference=sampled_lj_oracle_gpu(st.rx,st.ry,st.rz,st.box3,ids,1.0,1.0,2.5)
            F=permutedims(hcat(st.fx,st.fy,st.fz))
            @test norm(F-reference.F)/norm(reference.F) < tol
            @test norm(sum(F;dims=2))/norm(reference.F) < tol
            @test abs(SC.total_potential_energy(st)-sum(reference.energy))/max(abs(sum(reference.energy)),1) < tol
            W=vec(sum(st.virial_nonbonded;dims=1))
            Wref=vec(sum(reference.virial;dims=2))
            @test norm(W-Wref)/max(norm(Wref),1) < tol
            SC.evaluate_forces_into_f!(st,false)
            Fno=permutedims(hcat(st.fx,st.fy,st.fz))
            @test norm(Fno-reference.F)/norm(reference.F) < tol
            if backend === :cluster && image === :centered && phase !== :crossing
                order=Array(st.cluster.list.perm)
                slots=findall(!iszero,order)
                for (d,packed) in enumerate((st.cluster.list.xs,st.cluster.list.ys,st.cluster.list.zs))
                    @test Array(packed)[slots] == Array((st.rx,st.ry,st.rz)[d])[order[slots]]
                end
            end
        end
        st=nothing; GC.gc(); CUDA.reclaim()
    end
end
end
