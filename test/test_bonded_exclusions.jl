@testset "Bonded Exclusion Correctness and Guardrails" begin
    seed_all!(0xB0AD)

    function cpu_softrep_excl_forces_2d(rx::Vector{T}, ry::Vector{T},
                                        bonds::Vector{Tuple{Int32,Int32}},
                                        eps::T, sig::T) where {T<:AbstractFloat}
        N = length(rx)
        fx = zeros(T, N)
        fy = zeros(T, N)
        bonded = Set{Tuple{Int,Int}}()
        for (i32, j32) in bonds
            i = Int(i32); j = Int(j32)
            i == j && continue
            a, b = minmax(i, j)
            push!(bonded, (a, b))
        end
        @inbounds for i in 1:N
            xi = rx[i]; yi = ry[i]
            for j in 1:N
                i == j && continue
                a, b = minmax(i, j)
                (a, b) in bonded && continue
                dx = xi - rx[j]
                dy = yi - ry[j]
                r2 = muladd(dx, dx, dy*dy)
                if (r2 > zero(T)) & (r2 < sig*sig)
                    r = sqrt(r2)
                    f_over_r = (eps/sig) * (one(T) - r/sig) / r
                    fx[i] += f_over_r * dx
                    fy[i] += f_over_r * dy
                end
            end
        end
        return fx, fy
    end

    @testset "2D soft-repulsive exclusions match CPU reference" begin
        T = Float64
        bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
        st = Simulation.build_simulation(
            N=4,
            box=(T(20), T(20)),
            cutoff=T(1),
            skin=T(0.4),
            cap=Int32(16),
            neigh_interval=5,
            use_neighborlist=true,
            epsilon=T(10),
            sigma=T(1),
            gamma=T(0),
            temperature=T(0),
            bonds=bonds,
            bonding=harmonic_bond(k=T(300), r0=T(1)),
            nonbonded=:soft_repulsive,
            precision=:f64,
        )

        rx = T[0.0, 0.6, 1.2, 0.6]
        ry = T[0.0, 0.0, 0.0, 0.6]
        copyto!(st.rx, rx)
        copyto!(st.ry, ry)
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)

        fill!(st.fx, zero(T)); fill!(st.fy, zero(T)); fill!(st.Epot, zero(T))
        NonEqSimGPU.NonBondedForces.harmonic_rep_forces_soa_excl!(
            st.rx, st.ry, st.fx, st.fy, st.Epot, st.nbh, st.bonds, st.box2, st.softrep
        )
        CUDA.synchronize()

        fx_gpu = Array(st.fx)
        fy_gpu = Array(st.fy)
        fx_ref, fy_ref = cpu_softrep_excl_forces_2d(rx, ry, bonds, T(10), T(1))

        @test maximum(abs.(fx_gpu .- fx_ref)) < 1e-10
        @test maximum(abs.(fy_gpu .- fy_ref)) < 1e-10
        @test abs(sum(fx_gpu)) < 1e-10
        @test abs(sum(fy_gpu)) < 1e-10
    end

    @testset "Polymer slowdown guardrail (GPU smoke)" begin
        # Broad threshold to avoid flaky failures, but catches severe regressions.
        T = Float32
        N = 2000
        bonds = Tuple{Int32,Int32}[(Int32(i), Int32(i+1)) for i in 1:(N÷2 - 1)]

        st_poly = Simulation.build_simulation(
            N=N,
            box=(T(90), T(90)),
            cutoff=T(1),
            skin=T(0.55),
            cap=Int32(128),
            neigh_interval=10,
            use_neighborlist=true,
            epsilon=T(1e4),
            sigma=T(1),
            gamma=T(1),
            temperature=T(0),
            bonds=bonds,
            bonding=harmonic_bond(k=T(300), r0=T(1)),
            nonbonded=:soft_repulsive,
            precision=:f32,
        )
        st_gas = Simulation.build_simulation(
            N=N,
            box=(T(90), T(90)),
            cutoff=T(1),
            skin=T(0.55),
            cap=Int32(128),
            neigh_interval=10,
            use_neighborlist=true,
            epsilon=T(1e4),
            sigma=T(1),
            gamma=T(1),
            temperature=T(0),
            nonbonded=:soft_repulsive,
            precision=:f32,
        )

        rx, ry = let s = range(-20f0, 20f0; length=ceil(Int, sqrt(N)))
            xv = Vector{T}(undef, N)
            yv = Vector{T}(undef, N)
            n = length(s)
            for k in 1:N
                i = (k - 1) % n + 1
                j = (k - 1) ÷ n + 1
                xv[k] = s[i]
                yv[k] = s[j]
            end
            xv, yv
        end
        copyto!(st_poly.rx, rx); copyto!(st_poly.ry, ry)
        copyto!(st_gas.rx, rx);  copyto!(st_gas.ry, ry)
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st_poly.nbh, st_poly.rx, st_poly.ry; box=st_poly.box2, step=st_poly.step)
        NonEqSimGPU.NeighborLists.update_neighbors_inplace!(st_gas.nbh, st_gas.rx, st_gas.ry; box=st_gas.box2, step=st_gas.step)

        spec_poly = Simulation.velocityverlet(st_poly)
        spec_gas = Simulation.velocityverlet(st_gas)
        dt = T(2.5e-6)

        for _ in 1:8
            Simulation.step!(st_poly, spec_poly, dt; compute_energy=true)
            Simulation.step!(st_gas, spec_gas, dt; compute_energy=true)
        end
        CUDA.synchronize()

        function avg_step_ms(st, spec; reps::Int=12)
            vals = Float64[]
            for _ in 1:reps
                CUDA.synchronize()
                t0 = time_ns()
                Simulation.step!(st, spec, dt; compute_energy=true)
                CUDA.synchronize()
                push!(vals, (time_ns() - t0) / 1e6)
            end
            return sum(vals) / length(vals)
        end

        gas_ms = avg_step_ms(st_gas, spec_gas)
        poly_ms = avg_step_ms(st_poly, spec_poly)
        @test poly_ms <= 1.6 * gas_ms
    end
end
