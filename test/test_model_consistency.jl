# Independent host arithmetic checks the physical result, not another GPU path.
function mixed_model_reference(coords, sigmas, rc, excluded, wca)
    N, D = length(sigmas), length(coords)
    force = zeros(Float64, N, D); energy = zeros(Float64, N)
    components = D == 2 ? ((1,1),(2,2),(1,2)) : ((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))
    virial = zeros(Float64, N, length(components))
    for i in 1:N, j in 1:N
        (i == j || minmax(i,j) in excluded) && continue
        d = [Float64(c[i])-Float64(c[j]) for c in coords]
        d .-= 20 .* round.(d ./ 20)
        r2 = sum(abs2,d); sigma = (Float64(sigmas[i])+Float64(sigmas[j]))/2
        0 < r2 < (rc*sigma)^2 || continue
        s6 = (sigma^2/r2)^3
        f = 24*(2*s6^2-s6)/r2 .* d
        force[i,:] .+= f
        energy[i] += (4*(s6^2-s6)+(wca ? 1 : 0))/2
        for (n,(a,b)) in enumerate(components)
            virial[i,n] += d[a]*f[b]/2
        end
    end
    return force, energy, virial
end

@testset "Mixed-size physical consistency" begin
    NL = ParticleDynamics.NeighborLists
    NF = ParticleDynamics.NonBondedForces
    @testset "$T D=$D $style $potential excluded=$exclude" for T in (Float32,Float64), D in (2,3),
        style in (:dense,:stencil,:allpairs), potential in (:lj,:wca), exclude in (false,true)
        box = ntuple(_ -> T(20), D)
        st = build_simulation(;N=4, box, cutoff=T(3), skin=T(.4), cap=Int32(8),
            temperature=0, precision=T === Float32 ? :f32 : :f64,
            nonbonded=potential, use_neighborlist=style != :allpairs,
            bonds=Tuple{Int32,Int32}[(1,2),(2,3)], bonding=harmonic_bond(k=0.,r0=1.), exclude_bonded_pairs=exclude)
        # Include a pair crossing the periodic boundary.
        coords = (T[-9.7,9.3,8.3,-9.7],T[0,0,.3,1.1],T[0,.1,.2,.3])[1:D]
        dest = (st.rx,st.ry,st.rz)[1:D]
        foreach(s -> copyto!(s[1],s[2]), zip(dest,coords))
        sigmas = T[.8,1.2,.9,1.1]
        st.sigma_particle = CuArray(sigmas)
        st.rcut_factor = potential == :lj ? T(2.5) : T(2)^(one(T)/T(6))
        if style == :stencil
            st.nbh = NL.build_neighbors_stencil!(dest...; box,
                rcut_particle=fill(T(3),4),skin=T(.4),cap=Int32(8))
        end
        f,e,v = mixed_model_reference(coords,sigmas,st.rcut_factor,
            exclude ? Set([(1,2),(2,3)]) : Set{Tuple{Int,Int}}(),potential == :wca)
        tol = T === Float32 ? 4e-5 : 2e-12
        for with_energy in (false,true)
            SimulationCore.evaluate_forces_into_f!(st,with_energy)
            actual = hcat(map(Array,(st.fx,st.fy,st.fz)[1:D])...)
            @test actual ≈ f rtol=tol atol=tol
            @test vec(sum(actual;dims=1)) ≈ zeros(D) atol=tol
            if with_energy
                @test Array(st.Epot) ≈ e rtol=tol atol=tol
                @test Array(st.virial_nonbonded) ≈ v rtol=tol atol=tol
            end
        end
        # The energy-only low-level entry point has a distinct kernel.
        energy_kernel = potential == :lj ? NF.lj_forces_soa_mixed! : NF.wca_forces_soa_mixed!
        energy_kernel(dest..., (st.fx,st.fy,st.fz)[1:D]..., st.Epot, st.nbh,
            box,one(T),st.sigma_particle,st.rcut_factor; bonds=exclude ? st.bonds : nothing)
        @test Array(st.Epot) ≈ e rtol=tol atol=tol
        @test hcat(map(Array,(st.fx,st.fy,st.fz)[1:D])...) ≈ f rtol=tol atol=tol
    end
end

@testset "Stochastic timestep consistency" begin
    SC = SimulationCore
    constructors = (SC.velocityverlet,SC.baoab,SC.baoa,SC.gsm,SC.eulerheun,SC.eulermaruyama)
    for T in (Float32,Float64), D in (2,3), constructor in constructors, mode in (:white,:legacy,:spectrum)
        st = build_simulation(N=2,box=ntuple(_ -> T(20),D),temperature=0,
            epsilon=0,precision=T === Float32 ? :f32 : :f64)
        dt = T(.01)
        options = mode == :white ? (;) : mode == :legacy ? (;noise_corr_time=T(.1)) :
            (;noise_corr_time=T[.1,.3],ou_scales=T[.2,.4])
        spec = constructor(st;gamma=1,temperature=1,dt,options...)
        before = (Array(st.rx),Array(st.vx),st.step,st.force_valid,spec.params,spec.workspace)
        for bad_dt in (T(.02),zero(T),T(-1),T(Inf),T(NaN))
            @test_throws ArgumentError step!(st,spec,bad_dt)
        end
        @test_throws ArgumentError SC.step_graph!(st,spec,T(.02))
        @test_throws ArgumentError Filters.set_temperature!(spec,T(.02),2)
        @test_throws ArgumentError Filters.set_temperature!(spec,st,T(.02),2;filter=Filters.Indices([1]))
        @test_throws ArgumentError Filters.set_ou_spectrum!(spec,st,T(.2),T(.3);dt=T(.02))
        @test (Array(st.rx),Array(st.vx),st.step,st.force_valid) == before[1:4]
        @test spec.params === before[5] && spec.workspace === before[6]
        @test Array(spec.params.noise_scale) ≈ fill(sqrt(T(2)*dt),2)
        step!(st,spec,dt;compute_energy=false)
        @test st.step == before[3]+1
        @test all(isfinite,Array(st.rx))
    end
end

@testset "FENE domain and energy gradient" begin
    BF = ParticleDynamics.BondedForces
    for T in (Float32,Float64), D in (2,3), mode in (:force,:energy,:virial)
        box = ntuple(_ -> T(20),D)
        coords = ntuple(_ -> CUDA.zeros(T,2),D)
        force = ntuple(_ -> CUDA.zeros(T,2),D)
        e = CUDA.zeros(T,2); v = CUDA.zeros(T,2,D == 2 ? 3 : 6)
        bonds = BF.build_bondlist(2,[(1,2)])
        params = FENEParams{T}(T(30),T(1.5))
        function evaluate(r; p=params)
            copyto!(coords[1],T[0,r])
            foreach(a -> fill!(a,zero(T)),(force...,e,v))
            if mode == :force
                BF.fene_forces_soa_noE!(coords...,force...,bonds,box,p)
            elseif mode == :energy
                BF.fene_forces_soa!(coords...,force...,e,bonds,box,p)
            else
                BF.fene_forces_soa!(coords...,force...,e,v,bonds,box,p)
            end
            return Array(force[1])[1],sum(Array(e))
        end
        # Include a valid extension below the old 1e-6 clamp and tiny energy.
        for r in (zero(T),T(1e-5),T(.9),prevfloat(params.R0))
            f,energy = evaluate(r)
            R02 = params.R0^2; r2 = r*r
            expected = params.k*r/((R02-r2)/R02)
            @test f ≈ expected rtol=20eps(T)
            if mode != :force
                expected_energy = Float64(-BigFloat(params.k)*BigFloat(R02)/2*log1p(-BigFloat(r2)/BigFloat(R02)))
                @test energy ≈ expected_energy rtol=T === Float32 ? 2e-6 : 2e-9
            end
            if mode == :virial
                @test sum(Array(v)[:,1]) ≈ -r*f rtol=20eps(T)
            end
        end
        if mode != :force
            r = T(.9); h = T === Float32 ? T(.001) : T(1e-5)
            f,_ = evaluate(r)
            _,ep = evaluate(r+h); _,em = evaluate(r-h)
            @test f ≈ (ep-em)/(2h) rtol=T === Float32 ? 2e-4 : 2e-9
        end
        for r in (T(1.5),T(1.6),T(Inf),T(NaN))
            @test_throws DomainError evaluate(r)
            @test all(a -> all(iszero,Array(a)),(force...,e,v))
        end
        for p in (FENEParams{T}(T(-1),T(1.5)),FENEParams{T}(T(30),zero(T)),FENEParams{T}(T(30),T(Inf)))
            @test_throws ArgumentError evaluate(T(.5);p)
        end
        @test first(evaluate(T(.5))) > 0 # CUDA context remains usable after errors.
    end
end

@testset "Step validation failure boundaries" begin
    SC = SimulationCore
    st = build_simulation(N=2,box=(20.,20.),temperature=0.,epsilon=0.,precision=:f64,
        bonds=Tuple{Int32,Int32}[(1,2)],bonding=fene_bond(k=30.,r0=1.5))
    copyto!(st.rx,[0.,.9]); fill!(st.ry,0.)
    fill!(st.vx,0.); fill!(st.vy,0.)
    nve_spec = SC.nve(st)
    for dt in (0.,-1.,Inf,NaN)
        @test_throws ArgumentError step!(st,nve_spec,dt)
        @test st.step == 0 && Array(st.rx) == [0.,.9]
    end
    SC.evaluate_forces_into_f!(st,false)
    @test st.force_valid
    copyto!(st.rx,[0.,1.6]); invalidate_forces!(st)
    @test_throws DomainError step!(st,nve_spec,.001)
    @test !st.force_valid && st.step == 0
    copyto!(st.rx,[0.,.9]); invalidate_forces!(st)
    step!(st,nve_spec,.001)
    @test st.force_valid && st.step == 1
    # Compatibility stepping with raw parameters uses the same dt guard.
    em = SC.eulermaruyama(st;gamma=1.,temperature=1.,dt=.001)
    x = Array(st.rx)
    @test_throws ArgumentError step!(st,em.params,.002)
    @test Array(st.rx) == x && st.step == 1
end

@testset "Low-level stochastic timestep guards" begin
    LI = ParticleDynamics.LangevinIntegrators
    BI = ParticleDynamics.BrownianIntegrators
    for T in (Float32,Float64), D in (2,3)
        st = build_simulation(N=2,box=ntuple(_ -> T(20),D),temperature=0,
            epsilon=0,precision=T === Float32 ? :f32 : :f64)
        coords = (st.rx,st.ry,st.rz)[1:D]
        velocity = (st.vx,st.vy,st.vz)[1:D]
        forces = (st.fx,st.fy,st.fz)[1:D]
        noise = ntuple(_ -> CUDA.zeros(T,2),D)
        mass = CUDA.ones(T,2)
        dt = T(.01); box = ntuple(_ -> T(20),D)
        vv = SimulationCore.velocityverlet(st;gamma=1,temperature=0,dt).params
        bao = SimulationCore.baoab(st;gamma=1,temperature=0,dt).params
        em = SimulationCore.eulermaruyama(st;gamma=1,temperature=0,dt).params
        ou_step = D == 2 ? LI.baoab_OU_2d! : LI.baoab_OU_3d!
        em_step = D == 2 ? BI.em_step_2d! : BI.em_step_3d!
        calls = (
            h -> LI.vv_positions_soa!(coords...,velocity...,forces...,noise...,vv,h,box),
            h -> LI.vv_positions_soa!(coords...,velocity...,forces...,noise...,mass,vv,h,box),
            h -> LI.vv_velocities_soa!(velocity...,forces...,forces...,noise...,st.dq,st.dU,st.Ekin,vv,h),
            h -> LI.vv_velocities_soa!(velocity...,forces...,forces...,noise...,st.dq,st.dU,st.Ekin,mass,mass,vv,h),
            h -> ou_step(velocity...,noise...,bao,h,st.dq),
            h -> ou_step(velocity...,noise...,mass,bao,h,st.dq),
            h -> em_step(coords...,forces...,em,h,st.dq,st.dU,box),
        )
        for call in calls
            before = map(Array,(coords...,velocity...,st.dq,st.dU,st.Ekin))
            @test_throws ArgumentError call(2dt)
            @test map(Array,(coords...,velocity...,st.dq,st.dU,st.Ekin)) == before
            call(dt)
            @test all(a -> all(isfinite,Array(a)),(coords...,velocity...))
        end
    end
end
