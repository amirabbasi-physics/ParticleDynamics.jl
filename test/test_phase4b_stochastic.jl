@testset "Phase 4B: Stochastic Physics Validation (GPU)" begin
    seed_all!(0x4B0001)

    params = ParamsFromExamples.recommended_test_params()

    function report_D(noise_scale::Float64, tau::Float64, dt::Float64)
        return noise_scale^2 * tau / dt^2
    end

    function _resonant(mass::Float64, gamma::Float64, tau::Float64)
        scale = max(abs(mass), abs(gamma * tau), 1.0)
        return abs(mass - gamma * tau) <= 1.0e-12 * scale
    end

    function vacf_report_1d(t::Float64, mass::Float64, gamma::Float64,
                            tau::Float64, D::Float64)
        t1 = abs(t)
        if _resonant(mass, gamma, tau)
            return D / (2 * gamma * mass) *
                   (1 + (gamma / mass) * t1) * exp(-gamma * t1 / mass)
        end
        return D / (mass^2 - gamma^2 * tau^2) *
               ((mass / gamma) * exp(-gamma * t1 / mass) - tau * exp(-t1 / tau))
    end

    function msd_report_1d(t::Float64, mass::Float64, gamma::Float64,
                           tau::Float64, D::Float64)
        if _resonant(mass, gamma, tau)
            return D / gamma^2 * (2 * t - 3 * tau + (t + 3 * tau) * exp(-t / tau))
        end
        pref = 2 * D / (mass^2 - gamma^2 * tau^2)
        return pref * (
            (mass^3 / gamma^3) * (exp(-gamma * t / mass) - 1) +
            (mass^2 / gamma^2) * t -
            tau^3 * (exp(-t / tau) - 1) -
            tau^2 * t
        )
    end

    @testset "4B-1 Brownian free diffusion MSD slope" begin
        p = params.brownian
        dt = clamp(Float64(p.dt), 2.0e-4, 5.0e-4)
        gamma = clamp(Float64(p.gamma), 5.0, 20.0)
        temperature = clamp(Float64(p.temperature), 1.0, 5.0)
        boxL = max(200.0, Float64(p.boxL))
        # Use the midpoint Brownian path for the free-diffusion slope check.
        # Euler-Maruyama is still covered elsewhere, but its larger weak-error
        # bias makes this short regression too sensitive to example drift.
        integrator = :eulerheun

        N = 384
        steps = 3200
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

        spec = integrator == :eulermaruyama ?
               Simulation.eulermaruyama(st; gamma=gamma, temperature=temperature, dt=dt) :
               Simulation.eulerheun(st; gamma=gamma, temperature=temperature, dt=dt)
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

            spec = spec_symbol == :vv ?
                   Simulation.velocityverlet(st; gamma=gamma, temperature=temperature, dt=dt) :
                   Simulation.baoab(st; gamma=gamma, temperature=temperature, dt=dt)

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
            return (mean_v2=mean_v2, mass=Float64(st.mass))
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

        spec = Simulation.velocityverlet(st; gamma=gamma, temperature=temperature, noise_corr_time=tau, dt=dt)
        for _ in 1:burn_steps
            Simulation.step!(st, spec, dt; compute_energy=false)
        end

        v0x = copy(st.vx)
        v0y = copy(st.vy)
        c0v = Float64(CUDA.sum(v0x.^2 .+ v0y.^2)) / N

        ou0 = copy(vec(spec.workspace.ou_y))
        c0ou = Float64(CUDA.sum(ou0.^2)) / N

        t_lags = Float64[]
        c_vel = Float64[]
        c_ou = Float64[]
        for lag in 1:lag_max
            Simulation.step!(st, spec, dt; compute_energy=false)
            cv = Float64(CUDA.sum(v0x .* st.vx .+ v0y .* st.vy)) / N
            co = Float64(CUDA.sum(ou0 .* vec(spec.workspace.ou_y))) / N
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

            spec = Simulation.eulerheun(st; gamma=gamma, temperature=0.0, dt=dt)
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

    @testset "4B-5 Free VV OU MSD/VACF match report formulas" begin
        dt = 1.0e-3
        gamma = 100.0
        temperature = 0.0
        tau = 2.25
        noise_scale = 2.0
        mass = 1.0

        N = 512
        burn_steps = 7000
        steps = 60
        sample_stride = 20
        nside = ceil(Int, sqrt(N))
        boxL = 2048.0
        D = report_D(noise_scale, tau, dt)

        st = Simulation.build_simulation(
            N = N, box = (boxL, boxL),
            cutoff = 1.0, skin = 0.5, cap = Int32(256), neigh_interval = 50,
            use_neighborlist = true, epsilon = 0.0, sigma = 1.0,
            gamma = gamma, temperature = temperature, dt = dt,
            mass = mass, nonbonded = :soft_repulsive, precision = :f64,
            unwrapped_positions = true
        )

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
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box = st.box2, step = st.step)
        Simulation.sync_unwrapped!(st)
        spec = Simulation.velocityverlet(st; gamma=gamma, temperature=temperature, noise_corr_time=tau, dt=dt)
        Filters.set_noise_scale!(spec, noise_scale)

        for _ in 1:burn_steps
            Simulation.step!(st, spec, dt; compute_energy = false)
        end

        rx0 = copy(st.rx_unwrap)
        ry0 = copy(st.ry_unwrap)
        vx0 = copy(st.vx)
        vy0 = copy(st.vy)

        msd_num = Float64[]
        msd_ref = Float64[]
        vacf_num = Float64[]
        vacf_ref = Float64[]
        for step in 1:steps
            Simulation.step!(st, spec, dt; compute_energy = false)
            if step % sample_stride == 0
                t = step * dt
                push!(msd_num, Float64(CUDA.sum((st.rx_unwrap .- rx0).^2 .+ (st.ry_unwrap .- ry0).^2) / N))
                push!(msd_ref, 2 * msd_report_1d(t, mass, gamma, tau, D))
                push!(vacf_num, Float64(CUDA.sum(vx0 .* st.vx .+ vy0 .* st.vy) / N))
                push!(vacf_ref, 2 * vacf_report_1d(t, mass, gamma, tau, D))
            end
        end

        @test length(msd_num) >= 3
        rel_msd = [abs(xn - xr) / max(abs(xr), 1.0e-12) for (xn, xr) in zip(msd_num, msd_ref)]
        rel_vacf = [abs(xn - xr) / max(abs(xr), 1.0e-12) for (xn, xr) in zip(vacf_num, vacf_ref)]
        mean_rel_msd = sum(rel_msd) / length(rel_msd)
        mean_rel_vacf = sum(rel_vacf) / length(rel_vacf)

        @test all(isfinite, msd_num)
        @test all(isfinite, msd_ref)
        @test all(isfinite, vacf_num)
        @test all(isfinite, vacf_ref)
        @test mean_rel_msd <= 0.05
        @test maximum(rel_msd) <= 0.08
        @test mean_rel_vacf <= 0.05
        @test maximum(rel_vacf) <= 0.08
    end
end
