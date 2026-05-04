using GSDFiles

@testset "GSD IO" begin
    seed_all!(0xC0222)

    mktempdir() do tmp
        path = joinpath(tmp, "tiny_traj.gsd")

        st = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=5, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0.2f0
        )
        st.typeid .= CuArray(Int32[1, 2, 1, 2, 1, 2, 1, 2])

        h = ParticleDynamics.gsd_open(path)
        try
            ParticleDynamics.write_gsd_frame!(h, st; step=7, types_names=["A", "B"])
        finally
            ParticleDynamics.gsd_close(h)
        end

        frame = ParticleDynamics.read_gsd_frame!(path)
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

    mktempdir() do tmp
        path = joinpath(tmp, "tiny_traj_append.gsd")

        st = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=5, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0.2f0
        )

        h1 = ParticleDynamics.gsd_open(path)
        try
            ParticleDynamics.write_gsd_frame!(h1, st; step=7, types_names=["A"])
        finally
            ParticleDynamics.gsd_close(h1)
        end

        h2 = ParticleDynamics.gsd_open(path; append=true)
        try
            ParticleDynamics.write_gsd_frame!(h2, st; step=8, types_names=["A"])
        finally
            ParticleDynamics.gsd_close(h2)
        end

        frame_last = ParticleDynamics.read_gsd_frame!(path)
        frame_first = ParticleDynamics.read_gsd_frame!(path; step=1)
        @test frame_first.step == 7
        @test frame_last.step == 8

        r = GSDFiles.open_read(path)
        try
            @test GSDFiles.nframes(r) == 2
        finally
            GSDFiles.close(r)
        end
    end

    mktempdir() do tmp
        path = joinpath(tmp, "tiny_traj_virial.gsd")

        st = build_tiny2d(
            N=8, T=Float32, box=(20f0, 20f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=1, use_neighborlist=true, nonbonded=:wca, gamma=1f0, temperature=0f0
        )

        vv = Simulation.velocityverlet(st; gamma=1f0, temperature=0f0, dt=1f-4)
        step!(st, vv, 1f-4; compute_energy=true)
        expected_virial = Array(st.virial_tensor)

        h = ParticleDynamics.gsd_open(path)
        try
            ParticleDynamics.write_gsd_frame!(h, st; step=st.step, types_names=["A"], write_virial=true)
        finally
            ParticleDynamics.gsd_close(h)
        end

        frame = ParticleDynamics.read_gsd_frame!(path)
        @test haskey(frame.particle_properties, :virial)
        got_virial = frame.particle_properties[:virial]
        @test size(got_virial) == size(expected_virial)
        @test isapprox(got_virial, expected_virial; atol=1e-6, rtol=1e-6)

        r = GSDFiles.open_read(path)
        try
            ents = GSDFiles._entries_for_frame(r, UInt64(0))

            function read_chunk_matrix(name::AbstractString)
                entry = GSDFiles._maybe_one(r, ents, name)
                @test entry !== nothing
                entry === nothing && return nothing
                @test Int(entry.N) == size(expected_virial, 1)
                @test Int(entry.M) == size(expected_virial, 2)
                @test entry.type == GSDFiles._R_FLOAT32
                seek(r.io, entry.location)
                flat = read!(r.io, Array{Float32}(undef, Int(entry.N) * Int(entry.M)))
                return reshape(flat, (Int(entry.M), Int(entry.N)))'
            end

            raw_virial = read_chunk_matrix("particles/virial")
            prop_virial = read_chunk_matrix("particles/property/virial")
            @test raw_virial !== nothing
            @test prop_virial !== nothing
            @test isapprox(raw_virial, Float32.(expected_virial); atol=1e-6, rtol=1e-6)
            @test isapprox(prop_virial, Float32.(expected_virial); atol=1e-6, rtol=1e-6)
        finally
            GSDFiles.close(r)
        end
    end
end
