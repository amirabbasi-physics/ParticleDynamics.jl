@testset "Force Kernels Smoke and Invariants" begin
    seed_all!(0xC1001)

    st_lj = build_tiny2d(
        N=2, T=Float32, box=(12f0, 12f0), use_neighborlist=false,
        nonbonded=:lj, gamma=0f0, temperature=0f0, epsilon=1f0, sigma=1f0
    )
    set_positions_2d!(st_lj, Float32[-0.75, 0.75], Float32[0.0, 0.0])
    set_velocities_2d!(st_lj, zeros(Float32, 2), zeros(Float32, 2))
    Simulation.step!(st_lj, Simulation.velocityverlet(st_lj; gamma=0f0, temperature=0f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_lj)
    @test abs(Float64(CUDA.sum(st_lj.fx))) <= 1e-5
    @test abs(Float64(CUDA.sum(st_lj.fy))) <= 1e-5

    st_wca = build_tiny2d(
        N=4, T=Float32, box=(16f0, 16f0), use_neighborlist=true,
        nonbonded=:wca, gamma=1f0, temperature=0.2f0, epsilon=2f0, sigma=1f0
    )
    Simulation.step!(st_wca, Simulation.velocityverlet(st_wca; gamma=1f0, temperature=0.2f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_wca)

    st_soft = build_tiny2d(
        N=4, T=Float32, box=(16f0, 16f0), use_neighborlist=true,
        nonbonded=:soft_repulsive, gamma=1f0, temperature=0.2f0, epsilon=10f0, sigma=1f0
    )
    Simulation.step!(st_soft, Simulation.velocityverlet(st_soft; gamma=1f0, temperature=0.2f0, dt=1f-3), 1f-3; compute_energy=true)
    @test state_allfinite(st_soft)
end
