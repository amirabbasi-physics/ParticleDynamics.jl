using NonEqSimGPU: Simulation, accumulate_virial!, virial_components, virial_tensor, harmonic_bond, fene_bond

@testset "Configurational Virial" begin
    seed_all!(0xC0FFEE)

    @inline function mic(dx::T, L::T) where {T<:AbstractFloat}
        half = L / T(2)
        dx > half && (dx -= L)
        dx < -half && (dx += L)
        return dx
    end

    @inline function pair_virial2(dx::T, dy::T, fx::T, fy::T) where {T<:AbstractFloat}
        return (xx=dx * fx, yy=dy * fy, xy=dx * fy)
    end

    @inline function pair_virial3(dx::T, dy::T, dz::T, fx::T, fy::T, fz::T) where {T<:AbstractFloat}
        return (xx=dx * fx, yy=dy * fy, zz=dz * fz, xy=dx * fy, xz=dx * fz, yz=dy * fz)
    end

    @inline function zero_virial2(::Type{T}) where {T<:AbstractFloat}
        return (xx=zero(T), yy=zero(T), xy=zero(T))
    end

    @inline function zero_virial3(::Type{T}) where {T<:AbstractFloat}
        return (xx=zero(T), yy=zero(T), zz=zero(T), xy=zero(T), xz=zero(T), yz=zero(T))
    end

    @inline function trace(v::NamedTuple)
        return haskey(v, :zz) ? (v.xx + v.yy + v.zz) : (v.xx + v.yy)
    end

    function assert_virial2_close(got::NamedTuple, expected::NamedTuple; atol::Real=1e-10, rtol::Real=1e-10)
        @test isapprox(got.xx, expected.xx; atol, rtol)
        @test isapprox(got.yy, expected.yy; atol, rtol)
        @test isapprox(got.xy, expected.xy; atol, rtol)
    end

    function assert_virial3_close(got::NamedTuple, expected::NamedTuple; atol::Real=1e-10, rtol::Real=1e-10)
        @test isapprox(got.xx, expected.xx; atol, rtol)
        @test isapprox(got.yy, expected.yy; atol, rtol)
        @test isapprox(got.zz, expected.zz; atol, rtol)
        @test isapprox(got.xy, expected.xy; atol, rtol)
        @test isapprox(got.xz, expected.xz; atol, rtol)
        @test isapprox(got.yz, expected.yz; atol, rtol)
    end

    function lj_force_2d(dx::T, dy::T, ϵ::T, σ::T; cutoff::T=T(2.5) * σ) where {T<:AbstractFloat}
        r2 = muladd(dx, dx, dy * dy)
        if !(zero(T) < r2 < cutoff * cutoff)
            return zero(T), zero(T)
        end
        invr2 = inv(r2)
        s2 = (σ * σ) * invr2
        s6 = s2 * s2 * s2
        s12 = s6 * s6
        f_over_r = T(24) * ϵ * (T(2) * s12 - s6) * invr2
        return f_over_r * dx, f_over_r * dy
    end

    function wca_force_3d(dx::T, dy::T, dz::T, ϵ::T, σ::T) where {T<:AbstractFloat}
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        rc2 = (T(1.122462048309373) * σ)^2
        if !(zero(T) < r2 < rc2)
            return zero(T), zero(T), zero(T)
        end
        invr2 = inv(r2)
        s2 = (σ * σ) * invr2
        s6 = s2 * s2 * s2
        s12 = s6 * s6
        f_over_r = T(24) * ϵ * (T(2) * s12 - s6) * invr2
        return f_over_r * dx, f_over_r * dy, f_over_r * dz
    end

    function softrep_force_2d(dx::T, dy::T, ϵ::T, σ::T) where {T<:AbstractFloat}
        r2 = muladd(dx, dx, dy * dy)
        if !(zero(T) < r2 < σ * σ)
            return zero(T), zero(T)
        end
        r = sqrt(r2)
        f_over_r = (ϵ / σ) * (one(T) - r / σ) / r
        return f_over_r * dx, f_over_r * dy
    end

    function harmonic_bond_force_2d(dx::T, dy::T, k::T, r0::T) where {T<:AbstractFloat}
        r2 = muladd(dx, dx, dy * dy)
        r2 > zero(T) || return zero(T), zero(T)
        r = sqrt(r2)
        f_over_r = -k * (r - r0) / r
        return f_over_r * dx, f_over_r * dy
    end

    function fene_bond_force_3d(dx::T, dy::T, dz::T, k::T, R0::T) where {T<:AbstractFloat}
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        r2 > zero(T) || return zero(T), zero(T), zero(T)
        denom = max(one(T) - r2 / (R0 * R0), T(1e-6))
        f_over_r = -k / denom
        return f_over_r * dx, f_over_r * dy, f_over_r * dz
    end

    function lj_virial_sum_2d(rx::Vector{T}, ry::Vector{T}, Lx::T, Ly::T, ϵ::T, σ::T, cutoff::T) where {T<:AbstractFloat}
        acc = zero_virial2(T)
        for i in 1:length(rx)-1, j in i+1:length(rx)
            dx = mic(rx[i] - rx[j], Lx)
            dy = mic(ry[i] - ry[j], Ly)
            fx, fy = lj_force_2d(dx, dy, ϵ, σ; cutoff)
            pair = pair_virial2(dx, dy, fx, fy)
            acc = (xx=acc.xx + pair.xx, yy=acc.yy + pair.yy, xy=acc.xy + pair.xy)
        end
        return acc
    end

    function update_neighbors_host!(st)
        if st.rz === nothing
            NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
        else
            NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
        end
        return st
    end

    function refresh_forces!(st, mode::Symbol; dt::Real=zero(eltype(st.rx)))
        dtT = eltype(st.rx)(dt)
        T = eltype(st.rx)
        gamma = one(T)
        temperature = zero(T)
        if mode == :vv
            Simulation.step!(st, Simulation.velocityverlet(st; gamma=gamma, temperature=temperature, dt=dtT), dtT; compute_energy=true)
        elseif mode == :baoa
            Simulation.step!(st, Simulation.baoa(st; gamma=gamma, temperature=temperature, dt=dtT), dtT; compute_energy=true)
        elseif mode == :baoab
            Simulation.step!(st, Simulation.baoab(st; gamma=gamma, temperature=temperature, dt=dtT), dtT; compute_energy=true)
        elseif mode == :eh
            Simulation.step!(st, Simulation.eulerheun(st; gamma=gamma, temperature=temperature, dt=dtT), dtT; compute_energy=true)
        elseif mode == :em
            Simulation.step!(st, Simulation.eulermaruyama(st; gamma=gamma, temperature=temperature, dt=dtT), dtT; compute_energy=true)
        else
            error("unknown refresh mode $(mode)")
        end
        CUDA.synchronize()
        return st
    end

    @testset "Component metadata" begin
        st2 = build_tiny2d(N=2, T=Float64, gamma=1.0, temperature=0.0)
        st3 = build_tiny3d(N=2, T=Float64, gamma=1.0, temperature=0.0)
        @test virial_components(st2) == (:xx, :yy, :xy)
        @test virial_components(st3) == (:xx, :yy, :zz, :xy, :xz, :yz)
    end

    @testset "2D LJ pair via velocity Verlet" begin
        T = Float64
        st = build_tiny2d(
            N=2, T=T, box=(T(12), T(12)), cutoff=T(2.5), skin=T(0.4), cap=Int32(4),
            neigh_interval=1, epsilon=T(1), sigma=T(1), gamma=T(1), temperature=T(0),
            nonbonded=:lj,
        )
        set_positions_2d!(st, T[0.0, 1.25], T[0.0, 0.0])
        set_velocities_2d!(st, T[0.0, 0.0], T[0.0, 0.0])
        update_neighbors_host!(st)
        refresh_forces!(st, :vv)

        dx = T(-1.25)
        dy = zero(T)
        fx, fy = lj_force_2d(dx, dy, T(1), T(1))
        expected = pair_virial2(dx, dy, fx, fy)

        got_total = virial_tensor(st)
        got_nb = virial_tensor(st; part=:nonbonded)
        got_bonded = virial_tensor(st; part=:bonded)

        assert_virial2_close(got_total, expected)
        assert_virial2_close(got_nb, expected)
        assert_virial2_close(got_bonded, zero_virial2(T))
        @test isapprox(sum(Array(st.virial)), trace(expected); atol=1e-10, rtol=1e-10)

        fill!(st.virial_accum, zero(T))
        fill!(st.virial_tensor_accum, zero(T))
        accumulate_virial!(st)
        assert_virial2_close(virial_tensor(st; accumulated=true), expected)
    end

    @testset "2D harmonic bond pair via BAOA" begin
        T = Float64
        k = T(7)
        r0 = T(1.1)
        st = Simulation.build_simulation(
            N=2,
            box=(T(12), T(12)),
            cutoff=T(2.5),
            skin=T(0.4),
            cap=Int32(4),
            neigh_interval=1,
            use_neighborlist=true,
            epsilon=T(0),
            sigma=T(1),
            gamma=T(1),
            temperature=T(0),
            bonds=[(Int32(1), Int32(2))],
            bonding=harmonic_bond(k=k, r0=r0),
            nonbonded=:lj,
            precision=:f64,
        )
        set_positions_2d!(st, T[0.0, 1.3], T[0.0, 0.4])
        set_velocities_2d!(st, T[0.0, 0.0], T[0.0, 0.0])
        update_neighbors_host!(st)
        refresh_forces!(st, :baoa; dt=T(1e-4))

        rx = Array(st.rx)
        ry = Array(st.ry)
        dx = mic(rx[1] - rx[2], T(12))
        dy = mic(ry[1] - ry[2], T(12))
        fx, fy = harmonic_bond_force_2d(dx, dy, k, r0)
        expected = pair_virial2(dx, dy, fx, fy)

        assert_virial2_close(virial_tensor(st), expected)
        assert_virial2_close(virial_tensor(st; part=:bonded), expected)
        assert_virial2_close(virial_tensor(st; part=:nonbonded), zero_virial2(T))
    end

    @testset "Minimum-image virial in Brownian EH and EM" begin
        T = Float64
        L = T(10)
        expected_dx = T(0.5)
        expected_dy = zero(T)
        fx, fy = softrep_force_2d(expected_dx, expected_dy, T(8), T(1))
        expected = pair_virial2(expected_dx, expected_dy, fx, fy)
        wrapped = pair_virial2(T(-9.5), expected_dy, fx, fy)

        for mode in (:eh, :em)
            st = build_tiny2d(
                N=2, T=T, box=(L, L), cutoff=T(1), skin=T(0.3), cap=Int32(4),
                neigh_interval=1, epsilon=T(8), sigma=T(1), gamma=T(1), temperature=T(0),
                nonbonded=:soft_repulsive,
            )
            set_positions_2d!(st, T[-4.75, 4.75], T[0.0, 0.0])
            update_neighbors_host!(st)
            refresh_forces!(st, mode)

            got = virial_tensor(st)
            assert_virial2_close(got, expected)
            @test !isapprox(got.xx, wrapped.xx; atol=1e-6, rtol=1e-6)
        end
    end

    @testset "3D WCA pair via BAOAB is symmetric" begin
        T = Float64
        st = build_tiny3d(
            N=2, T=T, box=(T(14), T(14), T(14)), cutoff=T(2.5), skin=T(0.4), cap=Int32(4),
            neigh_interval=1, epsilon=T(1.5), sigma=T(1), gamma=T(1), temperature=T(0),
            nonbonded=:wca,
        )
        set_positions_3d!(st, T[0.0, 0.6], T[0.0, 0.8], T[0.0, 0.4])
        set_velocities_3d!(st, T[0.0, 0.0], T[0.0, 0.0], T[0.0, 0.0])
        update_neighbors_host!(st)
        refresh_forces!(st, :baoab; dt=T(1e-4))

        rx = Array(st.rx)
        ry = Array(st.ry)
        rz = Array(st.rz)
        dx = mic(rx[1] - rx[2], T(14))
        dy = mic(ry[1] - ry[2], T(14))
        dz = mic(rz[1] - rz[2], T(14))
        fx, fy, fz = wca_force_3d(dx, dy, dz, T(1.5), T(1))
        expected = pair_virial3(dx, dy, dz, fx, fy, fz)
        got = virial_tensor(st)

        assert_virial3_close(got, expected)
        assert_virial3_close(virial_tensor(st; part=:nonbonded), expected)
        @test isapprox(got.xy, dy * fx; atol=1e-10, rtol=1e-10)
        @test isapprox(got.xz, dz * fx; atol=1e-10, rtol=1e-10)
        @test isapprox(got.yz, dz * fy; atol=1e-10, rtol=1e-10)
    end

    @testset "3D FENE bond pair" begin
        T = Float64
        k = T(9)
        R0 = T(1.8)
        st = Simulation.build_simulation(
            N=2,
            box=(T(14), T(14), T(14)),
            cutoff=T(2.5),
            skin=T(0.4),
            cap=Int32(4),
            neigh_interval=1,
            use_neighborlist=true,
            epsilon=T(0),
            sigma=T(1),
            gamma=T(1),
            temperature=T(0),
            bonds=[(Int32(1), Int32(2))],
            bonding=fene_bond(k=k, r0=R0),
            nonbonded=:lj,
            precision=:f64,
        )
        set_positions_3d!(st, T[0.0, 0.6], T[0.0, 0.8], T[0.0, 0.2])
        set_velocities_3d!(st, T[0.0, 0.0], T[0.0, 0.0], T[0.0, 0.0])
        update_neighbors_host!(st)
        refresh_forces!(st, :vv)

        dx = T(-0.6)
        dy = T(-0.8)
        dz = T(-0.2)
        fx, fy, fz = fene_bond_force_3d(dx, dy, dz, k, R0)
        expected = pair_virial3(dx, dy, dz, fx, fy, fz)

        assert_virial3_close(virial_tensor(st), expected)
        assert_virial3_close(virial_tensor(st; part=:bonded), expected)
        assert_virial3_close(virial_tensor(st; part=:nonbonded), zero_virial3(T))
    end

    @testset "No double counting over multiple LJ pairs" begin
        T = Float64
        L = T(20)
        st = build_tiny2d(
            N=3, T=T, box=(L, L), cutoff=T(2.5), skin=T(0.4), cap=Int32(8),
            neigh_interval=1, epsilon=T(1), sigma=T(1), gamma=T(1), temperature=T(0),
            nonbonded=:lj,
        )
        rx = T[0.0, 1.25, 2.1]
        ry = T[0.0, 0.0, 0.0]
        set_positions_2d!(st, rx, ry)
        set_velocities_2d!(st, zeros(T, 3), zeros(T, 3))
        update_neighbors_host!(st)
        refresh_forces!(st, :vv)

        expected = lj_virial_sum_2d(rx, ry, L, L, T(1), T(1), T(2.5))
        assert_virial2_close(virial_tensor(st; part=:nonbonded), expected)
        @test isapprox(sum(Array(st.virial)), trace(expected); atol=1e-10, rtol=1e-10)
    end
end
