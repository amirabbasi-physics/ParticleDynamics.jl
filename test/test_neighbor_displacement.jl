@testset "Fused neighbor displacement reduction" begin
    NL = ParticleDynamics.NeighborLists
    rng = MersenneTwister(0xD152)
    # Partial blocks, single particles, and a multi-block Argon-sized reduction.
    for T in (Float32, Float64), D in (2, 3), N in (1, 33, 256, 257, 10000, 100003)
        box = ntuple(d -> T(12 + d), D)
        old = ntuple(d -> T.(rand(rng, N) .* box[d] .- box[d]/2), D)
        pos = ntuple(d -> T.(rand(rng, N) .* box[d] .- box[d]/2), D)
        expected = maximum(1:N) do i
            sum(1:D) do d
                dx = Float64(pos[d][i]) - Float64(old[d][i])
                abs2(dx - round(dx / box[d]) * box[d])
            end
        end
        device_pos, device_old = map(CuArray, pos), map(CuArray, old)
        got = NL._max_displacement2(device_pos..., device_old..., box)
        @test got ≈ expected rtol=8eps(T)
        scratch = CUDA.fill(T(NaN), N)
        preallocated = NL._max_displacement2!(scratch, device_pos, device_old, box)
        @test preallocated ≈ expected rtol=8eps(T)
        @test preallocated == got
        # Reused scratch must not retain a previous call's larger displacement.
        @test NL._max_displacement2!(scratch, device_old, device_old, box) == zero(T)
    end

    # Exact binary values exercise skin/2 equality and periodic wrapping.
    for T in (Float32, Float64), D in (2, 3), stencil in (false, true)
        box = ntuple(_ -> T(16), D)
        old = (T[7.875], ntuple(_ -> T[0], D-1)...)
        pos = map(CuArray, old)
        nb = stencil ? NL.build_neighbors_stencil!(pos...; box,
            rcut_particle=T[1], skin=T(0.5), cap=Int32(1)) :
            NL.build_neighbors_dense!(pos...; box, cutoff=T(1), skin=T(0.5), cap=Int32(1))
        kwargs = D == 2 ? (; Lx=box[1], Ly=box[2]) : (; Lx=box[1], Ly=box[2], Lz=box[3])
        copyto!(pos[1], T[-7.875]) # MIC displacement exactly skin/2.
        @test !NL.update_needed!(nb, pos...; skin=T(0.5), step=1, kwargs...)
        copyto!(pos[1], T[-7.75])
        @test NL.update_needed!(nb, pos...; skin=T(0.5), step=1, kwargs...)
        @test nb.target_interval == 18
        copyto!(pos[1], old[1])
        @test NL.update_needed!(nb, pos...; skin=T(0.5), step=18, kwargs...)
        @test nb.target_interval == 20
        nb.valid = false
        @test NL.update_needed!(nb, pos...; skin=T(0.5), step=1, kwargs...)
    end
end
