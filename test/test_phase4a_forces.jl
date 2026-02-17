@testset "Phase 4A: Analytic Two-Particle Forces" begin
    seed_all!(0x4A0101)

    lj_force_x(dx, r, eps, sig) = 24 * eps * (2 * (sig / r)^12 - (sig / r)^6) * (dx / (r * r))
    wca_rc(sig) = (2.0)^(1.0 / 6.0) * sig
    soft_force_x(dx, r, eps, sig) = (eps / sig) * (1 - r / sig) * (dx / r)

    function compute_pair_fx(;
        nonbonded::Symbol,
        r::Float64,
        eps::Float64=1.0,
        sig::Float64=1.0,
    )
        st = build_tiny2d(
            N=2, T=Float64, box=(20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
            use_neighborlist=false, nonbonded=nonbonded,
            epsilon=eps, sigma=sig, gamma=1.0, temperature=0.0, dt=1e-3
        )
        x1 = -r / 2
        x2 =  r / 2
        set_positions_2d!(st, [x1, x2], [0.0, 0.0])
        CUDA.fill!(st.fx, 0.0)
        CUDA.fill!(st.fy, 0.0)
        CUDA.fill!(st.Epot, 0.0)

        if nonbonded == :lj
            NonEqSimGPU.NonBondedForces.lj_forces_soa!(
                st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2, st.pair_lj
            )
        elseif nonbonded == :wca
            NonEqSimGPU.NonBondedForces.wca_forces_soa!(
                st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2, st.pair_lj
            )
        elseif nonbonded == :soft_repulsive
            NonEqSimGPU.NonBondedForces.harmonic_rep_forces_soa!(
                st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.box2, st.softrep
            )
        else
            error("unsupported nonbonded=$(nonbonded)")
        end

        fx_host = Array(st.fx)
        return (
            fx1=Float64(fx_host[1]),
            fx2=Float64(fx_host[2]),
            fy_sum=Float64(CUDA.sum(st.fy)),
            fx_sum=Float64(CUDA.sum(st.fx)),
            dx=(x1 - x2),
        )
    end

    @testset "LJ magnitude, sign, and momentum balance" begin
        eps = 2.5
        sig = 1.2
        for r in (0.95 * sig, 1.5 * sig)
            out = compute_pair_fx(nonbonded=:lj, r=r, eps=eps, sig=sig)
            fref = lj_force_x(out.dx, r, eps, sig)
            @test isapprox(out.fx1, fref; rtol=1e-11, atol=1e-11)
            @test isapprox(out.fx1, -out.fx2; rtol=1e-12, atol=1e-12)
            @test abs(out.fx_sum) <= 1e-12
            @test abs(out.fy_sum) <= 1e-12
        end

        out_rep = compute_pair_fx(nonbonded=:lj, r=0.95 * sig, eps=eps, sig=sig)
        out_att = compute_pair_fx(nonbonded=:lj, r=1.5 * sig, eps=eps, sig=sig)
        @test out_rep.fx1 < 0.0
        @test out_att.fx1 > 0.0
    end

    @testset "WCA repulsive branch and cutoff behavior" begin
        eps = 1.7
        sig = 1.0
        r_in = 0.9 * sig
        out_in = compute_pair_fx(nonbonded=:wca, r=r_in, eps=eps, sig=sig)
        fref_in = lj_force_x(out_in.dx, r_in, eps, sig)
        @test isapprox(out_in.fx1, fref_in; rtol=1e-11, atol=1e-11)
        @test out_in.fx1 < 0.0
        @test isapprox(out_in.fx1, -out_in.fx2; rtol=1e-12, atol=1e-12)
        @test abs(out_in.fx_sum) <= 1e-12
        @test abs(out_in.fy_sum) <= 1e-12

        r_out = 1.01 * wca_rc(sig)
        out_out = compute_pair_fx(nonbonded=:wca, r=r_out, eps=eps, sig=sig)
        @test abs(out_out.fx1) <= 1e-12
        @test abs(out_out.fx2) <= 1e-12
        @test abs(out_out.fx_sum) <= 1e-12
    end

    @testset "Soft-repulsive analytic force and sign" begin
        eps = 8.0
        sig = 1.5
        r_in = 0.8 * sig
        out_in = compute_pair_fx(nonbonded=:soft_repulsive, r=r_in, eps=eps, sig=sig)
        fref_in = soft_force_x(out_in.dx, r_in, eps, sig)
        @test isapprox(out_in.fx1, fref_in; rtol=1e-11, atol=1e-11)
        @test out_in.fx1 < 0.0
        @test isapprox(out_in.fx1, -out_in.fx2; rtol=1e-12, atol=1e-12)
        @test abs(out_in.fx_sum) <= 1e-12
        @test abs(out_in.fy_sum) <= 1e-12

        r_out = 1.05 * sig
        out_out = compute_pair_fx(nonbonded=:soft_repulsive, r=r_out, eps=eps, sig=sig)
        @test abs(out_out.fx1) <= 1e-12
        @test abs(out_out.fx2) <= 1e-12
        @test abs(out_out.fx_sum) <= 1e-12
    end
end
