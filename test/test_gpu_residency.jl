@testset "GPU Residency and Backend Smoke" begin
    seed_all!(0x605005)

    @testset "CUDA-backed state and OU buffers stay on device" begin
        T = Float32
        dt = T(1e-3)

        st = build_tiny2d(
            N=10, T=T, box=(24f0, 24f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=2, use_neighborlist=true, nonbonded=:wca,
            epsilon=1f0, sigma=1f0, gamma=1f0, temperature=0.5f0, dt=dt
        )

        @test ParticleDynamics.Backends.storage_backend(st) isa ParticleDynamics.Backends.CUDABackend
        @test Simulation.backend(st) isa ParticleDynamics.Backends.CUDABackend
        @test ParticleDynamics.Backends.storage_backend(st.rx) isa ParticleDynamics.Backends.CUDABackend
        @test ParticleDynamics.Backends.storage_backend(st.virial_tensor) isa ParticleDynamics.Backends.CUDABackend
        @test st.rx isa CuArray{T,1}
        @test st.ry isa CuArray{T,1}
        @test st.vx isa CuArray{T,1}
        @test st.vy isa CuArray{T,1}
        @test st.fx isa CuArray{T,1}
        @test st.fy isa CuArray{T,1}
        @test st.Epot isa CuArray{T,1}
        @test st.Ekin isa CuArray{T,1}
        @test st.virial_tensor isa CuArray{T,2}
        @test st.typeid isa CuArray{Int32,1}

        spec = Simulation.velocityverlet(st; gamma=1f0, temperature=0.5f0, dt=dt)
        selected = Filters.Indices(collect(1:2:length(st.rx)))
        Filters.set_ou_spectrum!(spec, st, T[0.05, 0.20], T[0.40, 0.10]; filter=selected, dt=dt)

        @test spec.params.gamma isa CuArray{T,1}
        @test spec.params.noise_scale isa CuArray{T,1}
        @test spec.params.corr_time === nothing
        @test spec.params.ou !== nothing
        @test spec.params.ou.active_idx isa CuArray{Int32,1}
        @test spec.params.ou.tau isa CuArray{T,2}
        @test spec.params.ou.scale isa CuArray{T,2}
        @test spec.params.ou.coeff_a isa CuArray{T,2}
        @test spec.params.ou.coeff_c isa CuArray{T,2}

        Simulation.step!(st, spec, dt; compute_energy=false)

        @test spec.workspace.rf_x isa CuArray{T,1}
        @test spec.workspace.rf_y isa CuArray{T,1}
        @test spec.workspace.rf_z === nothing
        @test spec.workspace.ou_x isa CuArray{T,2}
        @test spec.workspace.ou_y isa CuArray{T,2}
        @test spec.workspace.ou_z === nothing
    end

    @testset "Thermostat workspaces stay on device" begin
        T = Float32
        dt = T(1e-3)

        st_nhc = build_tiny2d(
            N=12, T=T, box=(24f0, 24f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=2, use_neighborlist=true, nonbonded=:wca,
            epsilon=1f0, sigma=1f0, gamma=1f0, temperature=0.5f0, dt=dt
        )
        nhc = Simulation.nosehooverchain(st_nhc; temperature=0.5f0, tau=0.25f0, chain_length=4, substeps=3)

        @test nhc.workspace.xi isa CuArray{T,2}
        @test nhc.workspace.eta isa CuArray{T,2}
        @test nhc.workspace.target_temperature isa CuArray{T,1}
        @test nhc.workspace.particle_bath_id isa CuArray{Int32,1}
        @test nhc.workspace.last_velocity_scale_per_bath isa CuArray{T,1}

        Simulation.step!(st_nhc, nhc.params, dt; compute_energy=false)
        @test state_allfinite(st_nhc)

        st_csvr = build_tiny2d(
            N=12, T=T, box=(24f0, 24f0), cutoff=2.5f0, skin=0.3f0, cap=Int32(32),
            neigh_interval=2, use_neighborlist=true, nonbonded=:wca,
            epsilon=1f0, sigma=1f0, gamma=1f0, temperature=0.5f0, dt=dt
        )
        csvr = Simulation.csvr(st_csvr; temperature=0.5f0, tau=0.25f0)

        @test csvr.workspace.target_temperature isa CuArray{T,1}
        @test csvr.workspace.tau isa CuArray{T,1}
        @test csvr.workspace.particle_bath_id isa CuArray{Int32,1}
        @test csvr.workspace.last_velocity_scale_per_bath isa CuArray{T,1}

        Simulation.step!(st_csvr, csvr.params, dt; compute_energy=false)
        @test state_allfinite(st_csvr)
    end
end
