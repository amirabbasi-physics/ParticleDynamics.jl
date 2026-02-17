@testset "Phase 4B: Stochastic Physics Validation (GPU)" begin
    seed_all!(0x4B0001)

    params = ParamsFromExamples.recommended_test_params()

    @testset "4B-1 Brownian free diffusion MSD slope" begin
        p = params.brownian
        dt = clamp(Float64(p.dt), 2.0e-4, 5.0e-4)
        gamma = clamp(Float64(p.gamma), 5.0, 20.0)
        temperature = clamp(Float64(p.temperature), 1.0, 5.0)
        boxL = max(200.0, Float64(p.boxL))
        integrator = p.integrator in (:eulerheun, :eulermaruyama) ? p.integrator : :eulerheun

        N = 256
        steps = 2400
        sample_stride = 40

        st = Simulation.build_simulation(
            N=N, box=(boxL, boxL),
            cutoff=1.0, skin=0.2, cap=Int32(8), neigh_interval=20,
            use_neighborlist=false, epsilon=0.0, sigma=1.0,
            gamma=gamma, temperature=temperature, dt=dt,
            nonbonded=:lj, precision=:f64, unwrapped_positions=true
        )

        nside = ceil(Int, sqrt(N))
        rx = Vector{Float64}(undef, N)
        ry = similar(rx)
        for i in 1:N
            ix = (i - 1) % nside
            iy = (i - 1) ÷ nside
            rx[i] = (ix + 0.5) * boxL / nside - boxL / 2
            ry[i] = (iy + 0.5) * boxL / nside - boxL / 2
        end
        copyto!(st.rx, rx)
        copyto!(st.ry, ry)
        copyto!(st.vx, zeros(Float64, N))
        copyto!(st.vy, zeros(Float64, N))
        Simulation.sync_unwrapped!(st)

        spec = integrator == :eulermaruyama ? Simulation.eulermaruyama(st) : Simulation.eulerheun(st)
        rx0 = copy(st.rx_unwrap)
        ry0 = copy(st.ry_unwrap)

        ts = Float64[]
        msd = Float64[]
        for s in 1:steps
            Simulation.step!(st, spec, dt; compute_energy=false)
            if s % sample_stride == 0
                push!(ts, s * dt)
                v = CUDA.sum((st.rx_unwrap .- rx0).^2 .+ (st.ry_unwrap .- ry0).^2) / Float64(N)
                push!(msd, Float64(v))
            end
        end

        n = length(ts)
        @test n >= 20
        i1 = max(2, Int(floor(0.2 * n)))
        i2 = max(i1 + 2, Int(floor(0.8 * n)))
        x = ts[i1:i2]
        y = msd[i1:i2]

        mx = sum(x) / length(x)
        my = sum(y) / length(y)
        ss_xx = sum((xj - mx)^2 for xj in x)
        ss_xy = sum((xj - mx) * (yj - my) for (xj, yj) in zip(x, y))
        slope = ss_xy / ss_xx
        yhat = [my + slope * (xj - mx) for xj in x]
        ss_res = sum((yj - yh)^2 for (yj, yh) in zip(y, yhat))
        ss_tot = sum((yj - my)^2 for yj in y)
        r2 = 1 - ss_res / ss_tot

        d = 2
        theory = 2 * d * (temperature / gamma)
        rel_err = abs(slope - theory) / theory

        @test isfinite(slope)
        @test isfinite(r2)
        @test rel_err <= 0.15
        @test r2 >= 0.985
    end

    @testset "4B-2 Langevin equipartition (VV + BAOAB)" begin
        p = params.langevin
        dt = clamp(Float64(p.dt), 5.0e-6, 2.0e-5)
        gamma = clamp(Float64(p.gamma), 200.0, 800.0)
        temperature = clamp(Float64(p.temperature), 10.0, 100.0)
        boxL = max(150.0, Float64(p.boxL))

        N = 256
        burn_steps = 1500
        sample_steps = 2500
        sample_stride = 10

        function run_and_measure(spec_symbol::Symbol)
            seed_all!(0x4B0200 + hash(spec_symbol) % Int(100))
            st = Simulation.build_simulation(
                N=N, box=(boxL, boxL),
                cutoff=1.0, skin=0.2, cap=Int32(8), neigh_interval=40,
                use_neighborlist=false, epsilon=0.0, sigma=1.0,
                gamma=gamma, temperature=temperature, dt=dt,
                nonbonded=:lj, precision=:f64
            )

            nside = ceil(Int, sqrt(N))
            rx = Vector{Float64}(undef, N)
            ry = similar(rx)
            for i in 1:N
                ix = (i - 1) % nside
                iy = (i - 1) ÷ nside
                rx[i] = (ix + 0.5) * boxL / nside - boxL / 2
                ry[i] = (iy + 0.5) * boxL / nside - boxL / 2
            end
            copyto!(st.rx, rx)
            copyto!(st.ry, ry)
            copyto!(st.vx, zeros(Float64, N))
            copyto!(st.vy, zeros(Float64, N))

            spec = spec_symbol == :vv ? Simulation.velocityverlet(st) : Simulation.baoab(st)

            for _ in 1:burn_steps
                Simulation.step!(st, spec, dt; compute_energy=false)
            end

            acc = 0.0
            nsamp = 0
            for s in 1:sample_steps
                Simulation.step!(st, spec, dt; compute_energy=false)
                if s % sample_stride == 0
                    acc += Float64(CUDA.sum(st.vx.^2 .+ st.vy.^2)) / N
                    nsamp += 1
                end
            end
            mean_v2 = acc / nsamp
            return (mean_v2=mean_v2, mass=Float64(st.vv.mass))
        end

        vv = run_and_measure(:vv)
        bao = run_and_measure(:baoab)
        v2_theory = 2 * temperature / vv.mass

        err_vv = abs(vv.mean_v2 - v2_theory) / v2_theory
        err_bao = abs(bao.mean_v2 - v2_theory) / v2_theory

        @test isfinite(vv.mean_v2)
        @test isfinite(bao.mean_v2)
        @test err_vv <= 0.10
        @test err_bao <= 0.10
    end

    @testset "4B-3 OU autocorrelation decay (OU-enabled path)" begin
        p = params.ou

        dt = clamp(Float64(p.dt), 1.0e-3, 2.0e-3)
        gamma = clamp(Float64(p.gamma), 5.0, 20.0)
        temperature = clamp(Float64(p.temperature), 1.0, 5.0)
        tau = clamp(Float64(p.tau), 0.2, 0.5)
        boxL = max(160.0, Float64(p.boxL))

        N = 512
        burn_steps = 400
        lag_max = 300

        seed_all!(0x4B0303)
        st = Simulation.build_simulation(
            N=N, box=(boxL, boxL),
            cutoff=1.0, skin=0.2, cap=Int32(8), neigh_interval=20,
            use_neighborlist=false, epsilon=0.0, sigma=1.0,
            gamma=gamma, temperature=temperature, noise_corr_time=tau, dt=dt,
            nonbonded=:lj, precision=:f64
        )

        nside = ceil(Int, sqrt(N))
        rx = Vector{Float64}(undef, N)
        ry = similar(rx)
        for i in 1:N
            ix = (i - 1) % nside
            iy = (i - 1) ÷ nside
            rx[i] = (ix + 0.5) * boxL / nside - boxL / 2
            ry[i] = (iy + 0.5) * boxL / nside - boxL / 2
        end
        copyto!(st.rx, rx)
        copyto!(st.ry, ry)
        copyto!(st.vx, zeros(Float64, N))
        copyto!(st.vy, zeros(Float64, N))

        spec = Simulation.velocityverlet(st)
        for _ in 1:burn_steps
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        v0x = copy(st.vx)
        v0y = copy(st.vy)
        c0v = Float64(CUDA.sum(v0x.^2 .+ v0y.^2)) / N

        ou0 = copy(st.ou_y)
        c0ou = Float64(CUDA.sum(ou0.^2)) / N

        t_lags = Float64[]
        c_vel = Float64[]
        c_ou = Float64[]
        for lag in 1:lag_max
            Simulation.step!(st, spec, dt; compute_energy=false)
            cv = Float64(CUDA.sum(v0x .* st.vx .+ v0y .* st.vy)) / N
            co = Float64(CUDA.sum(ou0 .* st.ou_y)) / N
            push!(t_lags, lag * dt)
            push!(c_vel, cv / c0v)
            push!(c_ou, co / c0ou)
        end

        # Velocity ACF sanity: finite, bounded, and overall decays.
        @test all(isfinite, c_vel)
        @test all((c_vel .>= -1.2) .& (c_vel .<= 1.2))
        @test c_vel[end] < c_vel[1]

        # Fit OU state autocorrelation to exp(-t/tau) over a mid-range.
        i1 = 20
        i2 = 220
        x = t_lags[i1:i2]
        y = log.(max.(c_ou[i1:i2], 1e-12))
        mx = sum(x) / length(x)
        my = sum(y) / length(y)
        slope = sum((xj - mx) * (yj - my) for (xj, yj) in zip(x, y)) / sum((xj - mx)^2 for xj in x)
        tau_fit = -1 / slope
        tau_rel_err = abs(tau_fit - tau) / tau

        @test all(isfinite, c_ou)
        @test all((c_ou .>= -1.2) .& (c_ou .<= 1.05))
        @test c_ou[end] < c_ou[1]
        @test tau_rel_err <= 0.30
    end

    @testset "4B-4 Weak convergence trend (deterministic limit, harmonic spring)" begin
        p = params.brownian
        gamma = clamp(Float64(p.gamma), 5.0, 20.0)
        boxL = max(300.0, Float64(p.boxL))

        dt0 = 4.0e-3
        total_time = 0.5
        kspring = 20.0
        delta = 0.75
        N = 256

        function msd_to_anchor(dt::Float64)
            steps = Int(round(total_time / dt))
            st = Simulation.build_simulation(
                N=N, box=(boxL, boxL),
                cutoff=1.0, skin=0.2, cap=Int32(8), neigh_interval=20,
                use_neighborlist=false, epsilon=0.0, sigma=1.0,
                gamma=gamma, temperature=0.0, dt=dt,
                nonbonded=:lj, precision=:f64
            )

            nside = ceil(Int, sqrt(N))
            rx = Vector{Float64}(undef, N)
            ry = similar(rx)
            for i in 1:N
                ix = (i - 1) % nside
                iy = (i - 1) ÷ nside
                rx[i] = (ix + 0.5) * boxL / nside - boxL / 2
                ry[i] = (iy + 0.5) * boxL / nside - boxL / 2
            end
            copyto!(st.rx, rx)
            copyto!(st.ry, ry)
            copyto!(st.vx, zeros(Float64, N))
            copyto!(st.vy, zeros(Float64, N))

            Filters.freeze_particles!(st; filter=Filters.All(), mode=:spring, k=kspring, steps=typemax(Int), include_energy=true)
            st.rx .+= delta

            spec = Simulation.eulerheun(st)
            for _ in 1:steps
                Simulation.step!(st, spec, dt; compute_energy=false)
            end
            return Float64(CUDA.sum((st.rx .- st.freeze_rx).^2 .+ (st.ry .- st.freeze_ry).^2)) / N
        end

        m_dt = msd_to_anchor(dt0)
        m_dt2 = msd_to_anchor(dt0 / 2)
        m_dt4 = msd_to_anchor(dt0 / 4)
        exact = delta^2 * exp(-2 * kspring * total_time / gamma)

        e_dt = abs(m_dt - exact)
        e_dt2 = abs(m_dt2 - exact)
        e_dt4 = abs(m_dt4 - exact)

        @test isfinite(m_dt)
        @test isfinite(m_dt2)
        @test isfinite(m_dt4)
        @test e_dt2 < e_dt
        @test e_dt4 < e_dt2
    end
end
