@testset "Workflow barostats" begin
    function fluid(; N=216, ρ=0.5, seed=1)
        n = round(Int, cbrt(N)); L = cbrt(N / ρ); a = L / n
        pos = [((i - 0.5) * a - L / 2, (j - 0.5) * a - L / 2, (k - 0.5) * a - L / 2) for i in 1:n, j in 1:n, k in 1:n][:]
        return ParticleSystem(pos; box=PeriodicBox((L, L, L))), L
    end
    wca() = WCA(epsilon=1.0, sigma=1.0, pairs=:neighborlist, neighborlist=CellList(buffer=0.3, capacity=96, rebuild_interval=10))

    function sim_with(barostat; scheme=BAOAB(), methods=nothing, forces=nothing, seed=3, precision=Float64)
        sys, _ = fluid()
        allp = Group(:all, AllSelection())
        m = methods === nothing ? [Langevin(allp; gamma=1.0, kT=1.0)] : methods
        return Simulation(sys; groups=Groups(allp), precision=precision, seed=seed,
                          integrator=Integrator(dt=0.004, scheme=scheme, forces=(forces === nothing ? [wca()] : forces),
                                                methods=m, barostat=barostat))
    end

    mc() = MonteCarloBarostat(pressure=1.0, kT=1.0, interval=5, log_volume_step=0.02, seed=21)
    cr() = StochasticCellRescaling(pressure=1.0, kT=1.0, compressibility=0.5, tau=0.5, interval=5, seed=22)

    @testset "Compatibility gates" begin
        allp = Group(:all, AllSelection())
        @test_throws ArgumentError prepare!(sim_with(mc(); scheme=nothing))                  # not BAOAB
        @test_throws ArgumentError prepare!(sim_with(mc(); methods=[Langevin(allp; gamma=1.0, kT=2.0)]))
        @test_throws ArgumentError prepare!(sim_with(StochasticCellRescaling(pressure=1.0, kT=1.0, compressibility=0.5,
                                                                             tau=0.5, seed=1, mode=:semiisotropic)))
        @test_throws ArgumentError prepare!(sim_with(MonteCarloBarostat(pressure=1.0, kT=1.0, seed=1, scale=:molecules)))
        @test_throws ArgumentError prepare!(sim_with(StochasticCellRescaling(pressure=1.0, kT=1.0, compressibility=-1.0,
                                                                             tau=0.5, seed=1)))
        sys2d = ParticleSystem([(0.0, 0.0), (1.5, 0.0)]; box=PeriodicBox((10.0, 10.0)))
        sim2d = Simulation(sys2d; groups=Groups(allp), integrator=Integrator(dt=0.004, scheme=BAOAB(), forces=[wca()],
                           methods=[Langevin(allp; gamma=1.0, kT=1.0)], barostat=mc()))
        @test_throws ArgumentError prepare!(sim2d)
    end

    @testset "$(nameof(typeof(b))) runs, changes volume, stays coherent" for b in (mc(), cr())
        sim = sim_with(b)
        run!(sim, Stage(:npt, steps=400; progress=false))
        st = state(sim)
        d = barostat_diagnostics(sim)
        @test state_allfinite(st)
        @test d.volume != fluid()[2]^3
        @test st.force_valid
        if b isa MonteCarloBarostat
            @test d.attempts == 80
            @test 0 < d.acceptance < 1
        else
            @test d.events == 80
            @test isfinite(d.last_pressure)
        end
        # forces in the active slot belong to the current cell and coordinates
        f = Array(st.fx)
        SimulationCore.invalidate_forces!(st)
        SimulationCore.evaluate_configuration!(st)
        @test Array(st.fx) ≈ f
    end

    @testset "Restart continues the barostat stream" begin
        # continuous reference
        ref = sim_with(mc(); seed=9)
        run!(ref, Stage(:a, steps=200; progress=false))
        rec_mid = barostat_state_record(ref)
        run!(ref, Stage(:b, steps=200; progress=false))
        d_ref = barostat_diagnostics(ref)
        # restart: GSD frame + barostat record, new simulation
        path = joinpath(mktempdir(), "npt.gsd")
        sim = sim_with(mc(); seed=9)
        allp = Group(:all, AllSelection())
        sim = Simulation(sim.system; groups=Groups(allp), precision=Float64, seed=9, integrator=sim.integrator,
                         writers=[GSDWriter(path; every=200)])
        run!(sim, Stage(:a, steps=200; progress=false))
        rec = read_gsd_log(path, "log/barostat/state")
        @test rec == barostat_state_record(sim)
        @test rec[3] == rec_mid[3]                                  # same attempt counter
        sys2 = ParticleSystem.from_gsd(path)
        sim2 = Simulation(sys2; groups=Groups(allp), precision=Float64, seed=10, integrator=sim.integrator)
        restore_barostat!(sim2, rec)
        @test barostat_diagnostics(sim2).attempts == 40
        # GSD stores configuration/box as float32
        @test isapprox(barostat_diagnostics(sim2).volume, barostat_diagnostics(sim).volume; rtol=1e-6)
        run!(sim2, Stage(:b, steps=200; progress=false))
        @test barostat_diagnostics(sim2).attempts == d_ref.attempts
        @test_throws ArgumentError restore_barostat!(sim_with(MonteCarloBarostat(pressure=1.0, kT=1.0, interval=5, seed=99)), rec)
    end

    @testset "Reproducible barostat streams" begin
        # Langevin noise comes from the CUDA device RNG, which the workflow seed
        # does not reset, so compare the barostat stream on identical states.
        sim = sim_with(mc(); seed=5)
        prepare!(sim)
        st = state(sim)
        ws = SimulationCore.TrialWorkspace(st)
        SimulationCore.save_trial_state!(ws, st)
        decisions(c) = [SimulationCore.monte_carlo_volume_move!(st, c) for _ in 1:30]
        mk() = SimulationCore.MonteCarloVolumeMove(st; pressure=1.0, kT=1.0, log_volume_step=0.05, seed=77)
        d1 = decisions(mk()); V1 = cell_volume(st)
        SimulationCore.restore_trial_state!(ws, st)
        d2 = decisions(mk()); V2 = cell_volume(st)
        @test d1 == d2 && V1 == V2
        @test 0 < sum(d1) < 30          # (`count` clashes with Filters.count in the full suite)
    end
end
