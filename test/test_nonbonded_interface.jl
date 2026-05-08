@testset "Nonbonded Interaction Interface" begin
    seed_all!(0x4A0606)
    NBI = ParticleDynamics.NonBondedInteractions
    NBF = ParticleDynamics.NonBondedForces

    function zero_nonbonded!(st)
        T = eltype(st.fx)
        fill!(st.fx, zero(T))
        fill!(st.fy, zero(T))
        st.fz === nothing || fill!(st.fz, zero(T))
        fill!(st.Epot, zero(T))
        fill!(st.virial_nonbonded, zero(T))
        return st
    end

    function snapshot_nonbonded(st)
        return (
            fx=Array(st.fx),
            fy=Array(st.fy),
            fz=st.fz === nothing ? nothing : Array(st.fz),
            epot=Array(st.Epot),
            virial=Array(st.virial_nonbonded),
        )
    end

    function refresh_neighbors!(st)
        if st.rz === nothing
            ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry; box=st.box2, step=st.step)
        else
            ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, st.rx, st.ry, st.rz; box=st.box3, step=st.step)
        end
        return st
    end

    function run_interface!(st, compute_energy::Bool)
        zero_nonbonded!(st)
        interaction = SimulationCore._nonbonded_interaction(st)
        if st.rz === nothing
            if compute_energy
                NBI.compute_nonbonded!(st.rx, st.ry, st.fx, st.fy,
                                       st.Epot, st.virial_nonbonded,
                                       st.nbh, st.box2,
                                       interaction, NBI.ForceEnergyVirial())
            else
                NBI.compute_nonbonded!(st.rx, st.ry, st.fx, st.fy,
                                       st.nbh, st.box2,
                                       interaction, NBI.ForceOnly())
            end
        else
            if compute_energy
                NBI.compute_nonbonded!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                       st.Epot, st.virial_nonbonded,
                                       st.nbh, st.box3,
                                       interaction, NBI.ForceEnergyVirial())
            else
                NBI.compute_nonbonded!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                       st.nbh, st.box3,
                                       interaction, NBI.ForceOnly())
            end
        end
        CUDA.synchronize()
        return interaction, snapshot_nonbonded(st)
    end

    function run_direct!(st, compute_energy::Bool)
        zero_nonbonded!(st)
        if st.rz === nothing
            if st.nb_kind == SimulationCore.NB_KIND_LJ && st.sigma_pair !== nothing
                if compute_energy
                    NBF.lj_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                             st.nbh, st.box2, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NBF.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                 st.nbh, st.box2, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_LJ && st.sigma_particle !== nothing
                if compute_energy
                    NBF.lj_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                             st.nbh, st.box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NBF.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                 st.nbh, st.box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA && st.sigma_pair !== nothing
                if compute_energy
                    NBF.wca_forces_soa_pairs!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                              st.nbh, st.box2, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NBF.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.fx, st.fy,
                                                  st.nbh, st.box2, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA && st.sigma_particle !== nothing
                if compute_energy
                    NBF.wca_forces_soa_mixed!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                              st.nbh, st.box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NBF.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.fx, st.fy,
                                                  st.nbh, st.box2, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NBF.lj_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                           st.nbh, st.box2, st.pair_lj)
                    else
                        NBF.lj_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                st.nbh, st.bonds, st.box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NBF.lj_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2, st.pair_lj)
                    else
                        NBF.lj_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2, st.pair_lj)
                    end
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NBF.wca_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                            st.nbh, st.box2, st.pair_lj)
                    else
                        NBF.wca_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.bonds, st.box2, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NBF.wca_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2, st.pair_lj)
                    else
                        NBF.wca_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2, st.pair_lj)
                    end
                end
            else
                if compute_energy
                    if st.bonds === nothing
                        NBF.harmonic_rep_forces_soa!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                     st.nbh, st.box2, st.softrep)
                    else
                        NBF.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.fx, st.fy, st.Epot, st.virial_nonbonded,
                                                          st.nbh, st.bonds, st.box2, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NBF.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.box2, st.softrep)
                    else
                        NBF.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.fx, st.fy, st.nbh, st.bonds, st.box2, st.softrep)
                    end
                end
            end
        else
            if st.nb_kind == SimulationCore.NB_KIND_LJ && st.sigma_pair !== nothing
                if compute_energy
                    NBF.lj_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                             st.nbh, st.box3, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NBF.lj_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                 st.nbh, st.box3, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_LJ && st.sigma_particle !== nothing
                if compute_energy
                    NBF.lj_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                             st.nbh, st.box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NBF.lj_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                 st.nbh, st.box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA && st.sigma_pair !== nothing
                if compute_energy
                    NBF.wca_forces_soa_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                              st.nbh, st.box3, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                else
                    NBF.wca_forces_soa_noE_pairs!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                  st.nbh, st.box3, st.typeid, st.sigma_pair, st.epsilon_pair, st.rcut_pair)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA && st.sigma_particle !== nothing
                if compute_energy
                    NBF.wca_forces_soa_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                              st.nbh, st.box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                else
                    NBF.wca_forces_soa_noE_mixed!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz,
                                                  st.nbh, st.box3, st.pair_lj.ϵ, st.sigma_particle, st.rcut_factor)
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_LJ
                if compute_energy
                    if st.bonds === nothing
                        NBF.lj_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                           st.nbh, st.box3, st.pair_lj)
                    else
                        NBF.lj_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                st.nbh, st.bonds, st.box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NBF.lj_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3, st.pair_lj)
                    else
                        NBF.lj_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3, st.pair_lj)
                    end
                end
            elseif st.nb_kind == SimulationCore.NB_KIND_WCA
                if compute_energy
                    if st.bonds === nothing
                        NBF.wca_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                            st.nbh, st.box3, st.pair_lj)
                    else
                        NBF.wca_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                 st.nbh, st.bonds, st.box3, st.pair_lj)
                    end
                else
                    if st.bonds === nothing
                        NBF.wca_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3, st.pair_lj)
                    else
                        NBF.wca_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3, st.pair_lj)
                    end
                end
            else
                if compute_energy
                    if st.bonds === nothing
                        NBF.harmonic_rep_forces_soa!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                     st.nbh, st.box3, st.softrep)
                    else
                        NBF.harmonic_rep_forces_soa_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.Epot, st.virial_nonbonded,
                                                          st.nbh, st.bonds, st.box3, st.softrep)
                    end
                else
                    if st.bonds === nothing
                        NBF.harmonic_rep_forces_soa_noE!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.box3, st.softrep)
                    else
                        NBF.harmonic_rep_forces_soa_noE_excl!(st.rx, st.ry, st.rz, st.fx, st.fy, st.fz, st.nbh, st.bonds, st.box3, st.softrep)
                    end
                end
            end
        end
        CUDA.synchronize()
        return snapshot_nonbonded(st)
    end

    function assert_snapshot_equal(a, b)
        @test a.fx == b.fx
        @test a.fy == b.fy
        @test a.fz == b.fz
        @test a.epot == b.epot
        @test a.virial == b.virial
    end

    @testset "Descriptor classification" begin
        st_lj = build_tiny2d(N=6, T=Float64, nonbonded=:lj, temperature=0.0)
        inter_lj = SimulationCore._nonbonded_interaction(st_lj)
        @test inter_lj.potential isa NBI.LennardJonesPotential
        @test inter_lj.coefficients isa NBI.UniformLJCoefficients{Float64}
        @test inter_lj.exclusions isa NBI.NoExclusions

        bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
        st_soft = SimulationCore.build_simulation(
            N=4, box=(20.0, 20.0), cutoff=1.0, skin=0.3, cap=Int32(16), neigh_interval=2,
            use_neighborlist=true, epsilon=8.0, sigma=1.0, gamma=0.0, temperature=0.0,
            bonds=bonds, nonbonded=:soft_repulsive, precision=:f64,
        )
        set_positions_2d!(st_soft, [0.0, 0.6, 1.2, 0.6], [0.0, 0.0, 0.0, 0.6])
        ParticleDynamics.NeighborLists.update_neighbors_inplace!(st_soft.nbh, st_soft.rx, st_soft.ry; box=st_soft.box2, step=st_soft.step)
        inter_soft = SimulationCore._nonbonded_interaction(st_soft)
        @test inter_soft.potential isa NBI.SoftRepulsivePotential
        @test inter_soft.coefficients isa NBI.UniformSoftRepCoefficients{Float64}
        @test inter_soft.exclusions isa NBI.BondExclusions

        st_pair = build_tiny2d(N=2, T=Float64, nonbonded=:wca, temperature=0.0)
        st_pair.typeid .= CuArray(Int32[1, 2])
        st_pair.sigma_pair = CuArray(Float64[0.90 1.10; 1.10 0.95])
        st_pair.epsilon_pair = CuArray(Float64[1.00 2.50; 2.50 1.20])
        st_pair.rcut_pair = CuArray(Float64[1.01 1.60; 1.60 1.01])
        inter_pair = SimulationCore._nonbonded_interaction(st_pair)
        @test inter_pair.potential isa NBI.WCAPotential
        @test inter_pair.coefficients isa NBI.PairMatrixCoefficients{Float64}

        st_mixed = build_tiny2d(N=4, T=Float64, nonbonded=:lj, cutoff=3.0, temperature=0.0)
        st_mixed.sigma_particle = CuArray(Float64[0.8, 1.0, 1.2, 1.1])
        inter_mixed = SimulationCore._nonbonded_interaction(st_mixed)
        @test inter_mixed.potential isa NBI.LennardJonesPotential
        @test inter_mixed.coefficients isa NBI.MixedSigmaCoefficients{Float64}

        @test !isdefined(NBF, :lj_forces_oa_pairs_bugfix)
    end

    @testset "Interface parity with legacy wrappers" begin
        cases = Pair{String,Function}[]

        push!(cases, "2D LJ uniform" => function ()
            st = build_tiny2d(N=4, T=Float64, box=(20.0, 20.0), cutoff=2.5, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.7, sigma=1.1, gamma=0.0, temperature=0.0)
            set_positions_2d!(st, [-0.9, -0.1, 0.8, 1.6], [0.0, 0.0, 0.0, 0.0])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D LJ exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(20.0, 20.0), cutoff=2.5, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=1.7, sigma=1.1, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:lj, precision=:f64,
            )
            set_positions_2d!(st, [0.0, 0.8, 1.6, 0.6], [0.0, 0.0, 0.0, 0.9])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D WCA uniform" => function ()
            st = build_tiny2d(N=4, T=Float64, box=(18.0, 18.0), cutoff=2.2, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.4, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_2d!(st, [-0.8, 0.0, 0.9, 1.7], [0.0, 0.0, 0.0, 0.0])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D WCA exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(18.0, 18.0), cutoff=2.2, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=1.4, sigma=1.0, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:wca, precision=:f64,
            )
            set_positions_2d!(st, [0.0, 0.7, 1.4, 0.5], [0.0, 0.0, 0.0, 0.8])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D LJ pair matrix" => function ()
            st = build_tiny2d(N=2, T=Float64, box=(20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.0, sigma=1.0, gamma=0.0, temperature=0.0)
            st.typeid .= CuArray(Int32[1, 2])
            set_positions_2d!(st, [-0.5, 0.5], [0.0, 0.0])
            st.sigma_pair = CuArray(Float64[0.90 1.10; 1.10 0.95])
            st.epsilon_pair = CuArray(Float64[1.00 2.50; 2.50 1.20])
            st.rcut_pair = CuArray(Float64[1.01 1.60; 1.60 1.01])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D WCA pair matrix" => function ()
            st = build_tiny2d(N=2, T=Float64, box=(20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.0, sigma=1.0, gamma=0.0, temperature=0.0)
            st.typeid .= CuArray(Int32[1, 2])
            set_positions_2d!(st, [-0.5, 0.5], [0.0, 0.0])
            st.sigma_pair = CuArray(Float64[0.90 1.10; 1.10 0.95])
            st.epsilon_pair = CuArray(Float64[1.00 2.50; 2.50 1.20])
            st.rcut_pair = CuArray(Float64[1.01 1.60; 1.60 1.01])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D LJ mixed sigma" => function ()
            st = build_tiny2d(N=4, T=Float64, box=(24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.8, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_2d!(st, [-1.0, -0.2, 0.7, 1.6], [0.0, 0.0, 0.0, 0.0])
            st.sigma_particle = CuArray(Float64[0.8, 1.0, 1.2, 1.1])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D WCA mixed sigma" => function ()
            st = build_tiny2d(N=4, T=Float64, box=(24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.5, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_2d!(st, [-1.0, -0.2, 0.7, 1.6], [0.0, 0.0, 0.0, 0.0])
            st.sigma_particle = CuArray(Float64[0.85, 1.0, 1.15, 1.05])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D soft-repulsive uniform" => function ()
            st = build_tiny2d(N=4, T=Float64, box=(20.0, 20.0), cutoff=1.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:soft_repulsive, epsilon=8.0, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_2d!(st, [0.0, 0.6, 1.2, 0.6], [0.0, 0.0, 0.0, 0.6])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "2D soft-repulsive exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(20.0, 20.0), cutoff=1.0, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=8.0, sigma=1.0, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:soft_repulsive, precision=:f64,
            )
            set_positions_2d!(st, [0.0, 0.6, 1.2, 0.6], [0.0, 0.0, 0.0, 0.6])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D LJ uniform" => function ()
            st = build_tiny3d(N=4, T=Float64, box=(24.0, 24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.2, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_3d!(st, [-0.8, 0.2, 1.1, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D LJ exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(24.0, 24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=1.2, sigma=1.0, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:lj, precision=:f64,
            )
            set_positions_3d!(st, [0.0, 0.8, 1.6, 0.6], [0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.4])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D WCA uniform" => function ()
            st = build_tiny3d(N=4, T=Float64, box=(20.0, 20.0, 20.0), cutoff=2.2, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.3, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_3d!(st, [-0.8, 0.0, 0.9, 1.7], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D WCA exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(20.0, 20.0, 20.0), cutoff=2.2, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=1.3, sigma=1.0, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:wca, precision=:f64,
            )
            set_positions_3d!(st, [0.0, 0.7, 1.4, 0.5], [0.0, 0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 0.3])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D LJ pair matrix" => function ()
            st = build_tiny3d(N=2, T=Float64, box=(20.0, 20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.0, sigma=1.0, gamma=0.0, temperature=0.0)
            st.typeid .= CuArray(Int32[1, 2])
            set_positions_3d!(st, [-0.5, 0.5], [0.0, 0.0], [0.0, 0.0])
            st.sigma_pair = CuArray(Float64[0.90 1.10; 1.10 0.95])
            st.epsilon_pair = CuArray(Float64[1.00 2.50; 2.50 1.20])
            st.rcut_pair = CuArray(Float64[1.01 1.60; 1.60 1.01])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D WCA pair matrix" => function ()
            st = build_tiny3d(N=2, T=Float64, box=(20.0, 20.0, 20.0), cutoff=3.0, skin=0.3, cap=Int32(8),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.0, sigma=1.0, gamma=0.0, temperature=0.0)
            st.typeid .= CuArray(Int32[1, 2])
            set_positions_3d!(st, [-0.5, 0.5], [0.0, 0.0], [0.0, 0.0])
            st.sigma_pair = CuArray(Float64[0.90 1.10; 1.10 0.95])
            st.epsilon_pair = CuArray(Float64[1.00 2.50; 2.50 1.20])
            st.rcut_pair = CuArray(Float64[1.01 1.60; 1.60 1.01])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D LJ mixed sigma" => function ()
            st = build_tiny3d(N=4, T=Float64, box=(24.0, 24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:lj, epsilon=1.8, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_3d!(st, [-1.0, -0.2, 0.7, 1.6], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
            st.sigma_particle = CuArray(Float64[0.8, 1.0, 1.2, 1.1])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D WCA mixed sigma" => function ()
            st = build_tiny3d(N=4, T=Float64, box=(24.0, 24.0, 24.0), cutoff=3.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:wca, epsilon=1.5, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_3d!(st, [-1.0, -0.2, 0.7, 1.6], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
            st.sigma_particle = CuArray(Float64[0.85, 1.0, 1.15, 1.05])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D soft-repulsive uniform" => function ()
            st = build_tiny3d(N=4, T=Float64, box=(20.0, 20.0, 20.0), cutoff=1.0, skin=0.3, cap=Int32(16),
                              use_neighborlist=true, nonbonded=:soft_repulsive, epsilon=8.0, sigma=1.0, gamma=0.0, temperature=0.0)
            set_positions_3d!(st, [0.0, 0.6, 1.2, 0.6], [0.0, 0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 0.3])
            refresh_neighbors!(st)
            return st
        end)

        push!(cases, "3D soft-repulsive exclusions" => function ()
            bonds = Tuple{Int32,Int32}[(Int32(1), Int32(2)), (Int32(2), Int32(3))]
            st = SimulationCore.build_simulation(
                N=4, box=(20.0, 20.0, 20.0), cutoff=1.0, skin=0.3, cap=Int32(16), neigh_interval=2,
                use_neighborlist=true, epsilon=8.0, sigma=1.0, gamma=0.0, temperature=0.0,
                bonds=bonds, nonbonded=:soft_repulsive, precision=:f64,
            )
            set_positions_3d!(st, [0.0, 0.6, 1.2, 0.6], [0.0, 0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 0.3])
            refresh_neighbors!(st)
            return st
        end)

        for (label, build_case) in cases
            @testset "$label" begin
                st = build_case()
                for compute_energy in (false, true)
                    direct = run_direct!(st, compute_energy)
                    interaction, via_interface = run_interface!(st, compute_energy)
                    @test interaction isa NBI.NonBondedInteraction
                    assert_snapshot_equal(via_interface, direct)
                end
            end
        end
    end
end
