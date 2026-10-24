module ClusterPruningGPUChecks
# S5 regression: eager distance-only pruning must still evaluate pair forces.
# Reference forces, energies and virials use the separate GPU ELL implementation.
using Test, CUDA, ParticleDynamics, Random
const SC = ParticleDynamics.SimulationCore
const DE = ParticleDynamics.DeviceEngine
const CP = ParticleDynamics.ClusterPairs

function fixture(::Type{T}, backend, intervals) where {T}
    # The cell builder requires at least three cells at cutoff + skin = 2.9.
    n = 8; a = T(1.12); L = n * a; N = n^3
    rng = MersenneTwister(1605)
    R = hcat([T.(((i,j,k) .+ 0.5) .* a .- L/2) .+ T(0.02) .* randn(rng,T,3)
              for i in 0:n-1 for j in 0:n-1 for k in 0:n-1]...)
    st = SC.build_simulation(;N, box=(L,L,L), nonbonded=:lj, cutoff=T(2.5), skin=T(0.4),
        cap=Int32(160), epsilon=T(1), sigma=T(1), temperature=T(0), dt=T(0.001),
        precision=T===Float64 ? :f64 : :f32, spatial_reorder=false)
    for (d,x) in enumerate((st.rx,st.ry,st.rz)); copyto!(x,R[d,:]); end
    for v in (st.vx,st.vy,st.vz); copyto!(v,T(0.1).*randn(rng,T,N)); end
    backend===:cluster && set_nonbonded_backend!(st,:cluster;prune_buffer=0.1,list_intervals=intervals)
    return st
end

function compare!(A,B;energy=true)
    for (a,b) in zip((A.rx,A.ry,A.rz),(B.rx,B.ry,B.rz)); copyto!(b,a); end
    SC.evaluate_forces_into_f!(B,energy)
    tol = eltype(A.rx)===Float64 ? 1e-11 : 5e-5
    for name in (energy ? (:fx,:fy,:fz,:Epot,:virial_nonbonded) : (:fx,:fy,:fz))
        x = Array(getfield(A,name)); y = Array(getfield(B,name))
        @test all(isfinite,x)
        @test maximum(abs.(x.-y)) <= tol * max(maximum(abs.(y)),1.0)
    end
end

@testset verbose=true "Distance-only pruning: eager evaluation and graph steps" begin
    original = CP.PRUNE_ONLY[]
    try
        @testset "$T, distance-only=$prune_only, intervals=$intervals" for T in (Float32,Float64),
                prune_only in (false,true), intervals in (nothing,(8,2))
            CP.PRUNE_ONLY[] = prune_only
            A = fixture(T,:cluster,intervals); B = fixture(T,:ell,nothing)
            SC.evaluate_forces_into_f!(A,true)
            @test A.cluster.list.prune_only == prune_only
            compare!(A,B)

            # Eager prune without rebuilding: motion exceeds prune_buffer/2 but not skin/2.
            L = A.cluster.list; before = Array(L.counts)
            A.rx .+= T(0.06)
            SC.evaluate_forces_into_f!(A,true)
            after = Array(L.counts)
            @test after[CP.CL_NBUILD] == before[CP.CL_NBUILD]
            @test after[CP.CL_NPRUNE] == before[CP.CL_NPRUNE] + 1
            compare!(A,B)

            # Explicit rebuild, and then an unchanged configuration (no prune).
            CP.request_build!(L)
            SC.evaluate_forces_into_f!(A,true)
            @test Array(L.counts)[CP.CL_NBUILD] == after[CP.CL_NBUILD] + 1
            compare!(A,B)
            before = Array(L.counts)
            SC.evaluate_forces_into_f!(A,false)
            @test Array(L.counts)[CP.CL_NPRUNE] == before[CP.CL_NPRUNE]
            compare!(A,B;energy=false)

            # Every step is checked, including fixed-interval rebuild/prune phases.
            sp = SC.nve(A)
            for step in 1:24
                energy = step % 3 == 0
                SC.step!(A,sp,A.dt;compute_energy=energy)
                DE.synchronize_state!(A)
                compare!(A,B;energy)
                if step % 5 == 0
                    # Force an eager prune while the stepping engine owns the state.
                    CP.request_build!(A.cluster.list)
                    SC.evaluate_forces_into_f!(A,energy)
                    compare!(A,B;energy)
                end
            end
        end
    finally
        CP.PRUNE_ONLY[] = original
    end
end
end
