@testset "NVE integrator" begin
    seed_all!(0xE9001)

    dt = 1.0e-3
    kwargs = (
        N=4,
        T=Float64,
        box=(20.0, 20.0),
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

    st_nve = build_tiny2d(; kwargs...)
    st_vv = build_tiny2d(; kwargs...)

    rx = Float64[-1.20, -0.10, 1.05, 0.55]
    ry = Float64[-0.80, 0.95, -0.45, 1.20]
    vx = Float64[0.18, -0.07, 0.05, -0.11]
    vy = Float64[-0.09, 0.06, -0.04, 0.13]
    set_positions_2d!(st_nve, rx, ry)
    set_positions_2d!(st_vv, rx, ry)
    set_velocities_2d!(st_nve, vx, vy)
    set_velocities_2d!(st_vv, vx, vy)
    ParticleDynamics.NeighborLists.update_neighbors_inplace!(st_nve.nbh, st_nve.rx, st_nve.ry; box=st_nve.box2, step=st_nve.step)
    ParticleDynamics.NeighborLists.update_neighbors_inplace!(st_vv.nbh, st_vv.rx, st_vv.ry; box=st_vv.box2, step=st_vv.step)

    nve = SimulationCore.nve(st_nve; dt=dt)
    vv_zero = SimulationCore.velocityverlet(st_vv; gamma=0.0, temperature=0.0, dt=dt)

    @test ParticleDynamics.IntegratorInterfaces.integrator_name(nve) == :nve
    @test ParticleDynamics.IntegratorInterfaces.stage_sequence(nve) == (:kick1, :drift, :force, :kick2)

    for _ in 1:6
        SimulationCore.step!(st_nve, nve, dt; compute_energy=true)
        SimulationCore.step!(st_vv, vv_zero, dt; compute_energy=true)
    end

    @test Array(st_nve.rx) ≈ Array(st_vv.rx) atol=1e-12 rtol=1e-12
    @test Array(st_nve.ry) ≈ Array(st_vv.ry) atol=1e-12 rtol=1e-12
    @test Array(st_nve.vx) ≈ Array(st_vv.vx) atol=1e-12 rtol=1e-12
    @test Array(st_nve.vy) ≈ Array(st_vv.vy) atol=1e-12 rtol=1e-12
    # NVE maintains `Ekin` lazily; it is refreshed at sampling time.
    SimulationCore._refresh_kinetic_buffer!(st_nve)
    @test Array(st_nve.Ekin) ≈ Array(st_vv.Ekin) atol=1e-12 rtol=1e-12
    @test Array(st_nve.Epot) ≈ Array(st_vv.Epot) atol=1e-12 rtol=1e-12
    @test state_allfinite(st_nve)
    @test st_nve.last_integrator == UInt8(5)

    obs = SimulationCore.collect_step_observables(st_nve, nve)
    @test obs.integrator == :nve
    @test obs.thermostatted_dof == 0
    @test obs.thermostat_kind == :none
    @test isfinite(obs.Etot)
    @test CUDA.sum(abs.(st_nve.dq)) == 0.0
    @test CUDA.sum(abs.(st_nve.dU)) == 0.0
end
