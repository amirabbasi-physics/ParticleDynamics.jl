@testset "External potential providers" begin
    seed_all!(0xE9002)

    # Mock provider: isotropic harmonic trap centered at `c` with stiffness `k`.
    # Pure broadcasts, so it exercises the external-force seam without any
    # Python/MLIP dependency.
    struct HarmonicTrap{T} <: ParticleDynamics.AbstractExternalPotential
        k::T
        cx::T; cy::T; cz::T
    end

    function ParticleDynamics.external_forces!(trap::HarmonicTrap{T},
                                               st::ParticleDynamics.SimulationState{T},
                                               compute_energy::Bool) where {T}
        st.fx .= .-trap.k .* (st.rx .- trap.cx)
        st.fy .= .-trap.k .* (st.ry .- trap.cy)
        st.fz .= .-trap.k .* (st.rz .- trap.cz)
        if compute_energy
            E = T(0.5) * trap.k * (sum(abs2, st.rx .- trap.cx) +
                                   sum(abs2, st.ry .- trap.cy) +
                                   sum(abs2, st.rz .- trap.cz))
            fill!(st.Epot, E / length(st.rx))
            fill!(st.virial_nonbonded, zero(T))
        end
        return nothing
    end

    dt = 1.0e-3
    k = 1.0
    L = 20.0
    # positions live in [-L/2, L/2): center the trap at the origin so the
    # trajectory (amplitude ~1.5) never crosses the wrap boundary
    c = 0.0
    kwargs = (
        N=4,
        T=Float64,
        box=(L, L, L),
        cutoff=2.5,
        skin=0.3,
        cap=Int32(8),
        neigh_interval=1,
        use_neighborlist=false,
        epsilon=1.0,
        sigma=1.0,
        gamma=0.0,
        temperature=0.0,
        nonbonded=:lj,
        dt=dt,
    )

    rx = c .+ Float64[-1.20, -0.10, 1.05, 0.55]
    ry = c .+ Float64[-0.80, 0.95, -0.45, 1.20]
    rz = c .+ Float64[0.35, -0.65, 0.85, -1.10]
    vx = Float64[0.18, -0.07, 0.05, -0.11]
    vy = Float64[-0.09, 0.06, -0.04, 0.13]
    vz = Float64[0.02, 0.11, -0.08, -0.05]

    @testset "attach guards" begin
        st = build_tiny3d(; kwargs...)
        trap = HarmonicTrap(k, c, c, c)

        # spatial-reorder guard: tag !== nothing must be rejected
        st_reorder = build_simulation(; N=64, box=(L, L, L), cutoff=2.5, skin=0.3,
                                      cap=Int32(16), neigh_interval=1,
                                      use_neighborlist=true, spatial_reorder=true,
                                      gamma=0.0, temperature=0.0,
                                      precision=:f64, dt=dt)
        if st_reorder.tag !== nothing
            @test_throws ErrorException ParticleDynamics.attach_external_potential!(st_reorder, trap)
        end

        # bonds guard
        st_bonded = build_simulation(; N=4, box=(L, L, L), cutoff=2.5, skin=0.3,
                                     cap=Int32(8), neigh_interval=1,
                                     use_neighborlist=false, spatial_reorder=false,
                                     gamma=0.0, temperature=0.0,
                                     bonds=[(Int32(1), Int32(2))],
                                     bonding=ParticleDynamics.harmonic_bond(k=300.0, r0=1.0),
                                     precision=:f64, dt=dt)
        @test_throws ErrorException ParticleDynamics.attach_external_potential!(st_bonded, trap)

        # clean attach + detach round trip
        @test ParticleDynamics.attach_external_potential!(st, trap) === st
        @test st.external_potential === trap
        @test ParticleDynamics.detach_external_potential!(st) === st
        @test st.external_potential === nothing
    end

    @testset "forces overwrite and energy fill" begin
        st = build_tiny3d(; kwargs...)
        set_positions_3d!(st, rx, ry, rz)
        set_velocities_3d!(st, vx, vy, vz)
        trap = HarmonicTrap(k, c, c, c)
        ParticleDynamics.attach_external_potential!(st, trap)

        # dirty force buffers, then evaluate twice: identical (overwrite, not accumulate)
        fill!(st.fx, 123.0); fill!(st.fy, -7.0); fill!(st.fz, 55.0)
        SimulationCore.evaluate_forces_into_f!(st, true)
        f1 = (Array(st.fx), Array(st.fy), Array(st.fz))
        SimulationCore.evaluate_forces_into_f!(st, true)
        f2 = (Array(st.fx), Array(st.fy), Array(st.fz))
        @test f1[1] == f2[1] && f1[2] == f2[2] && f1[3] == f2[3]

        # exact analytic values
        @test f1[1] ≈ -k .* (rx .- c) atol=1e-14
        @test f1[2] ≈ -k .* (ry .- c) atol=1e-14
        @test f1[3] ≈ -k .* (rz .- c) atol=1e-14
        E_expected = 0.5 * k * (sum(abs2, rx .- c) + sum(abs2, ry .- c) + sum(abs2, rz .- c))
        @test sum(Array(st.Epot)) ≈ E_expected rtol=1e-13

        # detach restores the internal LJ path (different forces)
        ParticleDynamics.detach_external_potential!(st)
        SimulationCore.evaluate_forces_into_f!(st, true)
        @test Array(st.fx) != f1[1]
    end

    @testset "NVE with external potential: conservation + analytic trajectory" begin
        st = build_tiny3d(; kwargs...)
        set_positions_3d!(st, rx, ry, rz)
        set_velocities_3d!(st, vx, vy, vz)
        trap = HarmonicTrap(k, c, c, c)
        ParticleDynamics.attach_external_potential!(st, trap)

        spec = SimulationCore.nve(st; dt=dt)
        SimulationCore.step!(st, spec, dt; compute_energy=true)
        SimulationCore._refresh_kinetic_buffer!(st)
        E0 = sum(Array(st.Epot)) + sum(Array(st.Ekin))

        nsteps = 2000
        for i in 2:nsteps
            SimulationCore.step!(st, spec, dt; compute_energy=(i % 100 == 0))
            if i % 100 == 0
                SimulationCore._refresh_kinetic_buffer!(st)
                E = sum(Array(st.Epot)) + sum(Array(st.Ekin))
                # velocity Verlet on a harmonic trap: bounded energy
                # oscillation of order (ω dt)^2 = 1e-6 relative, no drift
                @test abs(E - E0) / abs(E0) < 5e-6
            end
        end
        @test state_allfinite(st)

        # analytic solution: x(t) = c + A cos(ωt) + (v0/ω) sin(ωt), ω = √(k/m) = 1
        t = nsteps * dt
        x_exact = c .+ (rx .- c) .* cos(t) .+ vx .* sin(t)
        @test Array(st.rx) ≈ x_exact atol=1e-5
    end
end
