@testset "GSD IO" begin
    seed_all!(0xC0222)

    mktempdir() do tmp
        path = joinpath(tmp, "tiny_traj.gsd")

        st = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=5, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0.2f0
        )
        st.typeid .= CuArray(Int32[1, 2, 1, 2, 1, 2, 1, 2])

        h = NonEqSimGPU.gsd_open(path)
        try
            NonEqSimGPU.write_gsd_frame!(h, st; step=7, types_names=["A", "B"])
        finally
            NonEqSimGPU.gsd_close(h)
        end

        frame = NonEqSimGPU.read_gsd_frame!(path)
        @test frame.step == 7
        @test frame.N == 8
        @test frame.D == 2
        @test length(frame.rx) == 8
        @test length(frame.ry) == 8
        @test frame.rz === nothing
        @test isempty(frame.types) || frame.types == ["A", "B"]

        rx_host = Array(st.rx)
        ry_host = Array(st.ry)
        tid_host = Array(st.typeid)
        @test isapprox(sum(frame.rx), sum(rx_host); atol=1e-5, rtol=1e-6)
        @test isapprox(sum(frame.ry), sum(ry_host); atol=1e-5, rtol=1e-6)
        @test frame.typeid == tid_host
    end
end
