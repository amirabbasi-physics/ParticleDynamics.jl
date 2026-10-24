using Statistics: mean, var

@testset "Barostats" begin
    SC = SimulationCore

    function ideal_state(T, N, L)
        st = SC.build_simulation(N=N, box=(T(L), T(L), T(L)), use_neighborlist=false, nonbonded=:none,
            gamma=T(1), temperature=T(1), precision=(T === Float64 ? :f64 : :f32), spatial_reorder=false)
        rng = Random.MersenneTwister(1)
        copyto!(st.rx, T.(L .* (rand(rng, N) .- 0.5))); copyto!(st.ry, T.(L .* (rand(rng, N) .- 0.5)))
        copyto!(st.rz, T.(L .* (rand(rng, N) .- 0.5)))
        SC.invalidate_forces!(st)
        return st
    end

    # Block standard error for correlated samples.
    function block_sem(x; nblocks=50)
        n = length(x) ÷ nblocks
        b = [mean(@view x[(k - 1) * n + 1:k * n]) for k in 1:nblocks]
        return std_(b) / sqrt(nblocks)
    end
    std_(v) = sqrt(var(v))

    N = 4; P = 1.0; kT = 1.0
    Vmean = (N + 1) * kT / P; Vvar = (N + 1) * (kT / P)^2

    @testset "MC acceptance rule and exact rejection" begin
        st = ideal_state(Float64, N, 2.0)
        mc = SC.MonteCarloVolumeMove(st; pressure=P, kT=kT, log_volume_step=0.3, seed=7)
        # reproduce the first decision with an independent RNG copy
        rng = Random.Xoshiro(0)
        Random.seed!(rng, SC._event_seed(UInt64(7), 0))     # stream of attempt 0
        η = 0.3 * (2 * rand(rng) - 1); logu = log(rand(rng))
        V = cell_volume(st)
        x0 = Array(st.rx)
        acc = SC.monte_carlo_volume_move!(st, mc)
        Vt = V * exp(η)
        @test acc == (logu < min(0.0, -P * (Vt - V) / kT + (N + 1) * η))
        if acc
            @test cell_volume(st) ≈ Vt rtol=1e-12
            @test Array(st.rx) ≈ x0 .* exp(η / 3)
        else
            @test cell_volume(st) == V && Array(st.rx) == x0
        end
        @test mc.attempts == 1
        # a hopeless move (huge pressure) is rejected and leaves the state bit-identical
        mc2 = SC.MonteCarloVolumeMove(st; pressure=1e9, kT=kT, log_volume_step=0.3, seed=8)
        snap = (Array(st.rx), Array(st.ry), Array(st.rz), st.box3)
        for _ in 1:20
            if !SC.monte_carlo_volume_move!(st, mc2)
                @test (Array(st.rx), Array(st.ry), Array(st.rz), st.box3) == snap
            else
                snap = (Array(st.rx), Array(st.ry), Array(st.rz), st.box3)
            end
        end
        @test mc2.attempts == 20
    end

    @testset "MC ideal gas volume distribution ($T)" for T in (Float64, Float32)
        st = ideal_state(T, N, cbrt(Vmean))
        mc = SC.MonteCarloVolumeMove(st; pressure=P, kT=kT, log_volume_step=0.8, seed=11)
        for _ in 1:2_000; SC.monte_carlo_volume_move!(st, mc); end
        nsamp = 60_000
        Vs = Vector{Float64}(undef, nsamp)
        for k in 1:nsamp
            SC.monte_carlo_volume_move!(st, mc)
            Vs[k] = cell_volume(st)
        end
        m = mean(Vs); v = var(Vs)
        sem = block_sem(Vs)
        @info "MC ideal gas $T" mean=m expected=Vmean sem var=v expected_var=Vvar acceptance=SC.acceptance_ratio(mc)
        @test abs(m - Vmean) < 4 * sem
        @test isapprox(v, Vvar; rtol=0.08)
    end

    @testset "C-rescale ideal gas volume distribution" begin
        T = Float64
        st = ideal_state(T, N, cbrt(Vmean))
        cr = SC.StochasticCellRescalingState(st; pressure=P, kT=kT, compressibility=1.0, tau=1.0, seed=13)
        h = 0.01
        for _ in 1:5_000; SC.stochastic_cell_rescale!(st, cr, h; compute_energy=false); end
        nsamp = 150_000
        Vs = Vector{Float64}(undef, nsamp)
        for k in 1:nsamp
            SC.stochastic_cell_rescale!(st, cr, h; compute_energy=false)
            Vs[k] = cell_volume(st)
        end
        m = mean(Vs); v = var(Vs); sem = block_sem(Vs)
        @info "C-rescale ideal gas" mean=m expected=Vmean sem var=v expected_var=Vvar
        @test abs(m - Vmean) < 4 * sem + 0.01 * Vmean      # allow O(h) bias
        @test isapprox(v, Vvar; rtol=0.1)
    end

    @testset "Pressure diagnostics" begin
        st = ideal_state(Float64, N, 3.0)
        copyto!(st.vx, [1.0, -1.0, 0.5, 0.0]); fill!(st.vy, 0.0); fill!(st.vz, 0.0)
        @test SC.total_kinetic_energy(st) ≈ 0.5 * (1 + 1 + 0.25)
        @test SC.mechanical_pressure(st) ≈ 2 * 1.125 / (3 * 27.0)
    end
end
