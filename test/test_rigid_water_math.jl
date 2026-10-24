using StaticArrays, LinearAlgebra

@testset "Rigid water constraint pressure estimator (CPU)" begin
    RW = ParticleDynamics.RigidWaterConstraints
    rng = MersenneTwister(8172)
    for T in (Float32, Float64)
        oh, hh = T(0.1), T(0.16330)
        x = (SVector{3,T}(0,0,0), SVector{3,T}(sqrt(oh^2-hh^2/4),hh/2,0),
             SVector{3,T}(sqrt(oh^2-hh^2/4),-hh/2,0))
        masses = T[15.9994,1.008,1.008]
        w = Tuple(inv.(masses))
        J = zeros(3,9)
        for (k,(a,b)) in enumerate(((1,2),(1,3),(2,3)))
            J[k,3a-2:3a] = x[a]-x[b]
            J[k,3b-2:3b] = x[b]-x[a]
        end
        M = Diagonal(repeat(Float64.(collect(w)); inner=3))
        tol = T === Float32 ? 5e-5 : 1e-11
        for _ in 1:25
            v, ok = RW.rattle_water(x, ntuple(_ -> SVector{3,T}(randn(rng,3)),3), w)
            @test ok
            f = ntuple(_ -> SVector{3,T}(randn(rng,3)),3)
            fc, ok = RW.constraint_force_water(x,v,f,w)
            @test ok
            curvature = [sum(abs2,v[a]-v[b]) for (a,b) in ((1,2),(1,3),(2,3))]
            reference = -J' * ((J*M*J') \ (J*M*Float64.(vcat(f...)) + curvature))
            @test vcat(fc...) ≈ reference rtol=tol atol=tol
            @test norm(sum(fc)) < 10tol
            @test norm(sum(cross(x[k],fc[k]) for k in 1:3)) < tol
            @test abs(sum(dot(v[k],fc[k]) for k in 1:3)) < 10tol
            W = RW.constraint_tensor_water(x,fc)
            com = sum(masses[k]*x[k] for k in 1:3)/sum(masses)
            vcom = sum(masses[k]*v[k] for k in 1:3)/sum(masses)
            kin_internal2 = sum(masses[k]*sum(abs2,v[k]-vcom) for k in 1:3)
            force_internal_virial = sum(dot(x[k]-com,f[k]) for k in 1:3)
            # Atomic kinetic + atomic virial collapses to the molecular COM
            # estimator; rotation must not add an ideal-gas pressure term.
            @test sum(W[1:3]) ≈ -kin_internal2-force_internal_virial rtol=tol atol=tol
            shift = SVector{3,T}(1,2,3)
            @test RW.constraint_tensor_water(ntuple(k -> x[k]+shift,3),fc) ≈ W rtol=tol atol=tol
        end
        @test !RW.constraint_force_water(ntuple(_ -> zero(x[1]),3),x,x,w)[2]
    end
end

@testset "Rigid water local constraint algebra (CPU)" begin
    RW = ParticleDynamics.RigidWaterConstraints
    rng = MersenneTwister(9147)
    for T in (Float32, Float64)
        oh, hh = T(0.1), T(0.16330)
        l2 = (oh^2, oh^2, hh^2)
        tol = T === Float32 ? T(2e-6) : T(1e-12)
        atol = T === Float32 ? 2e-6 : 1e-11
        masses = T[15.9994, 1.008, 1.008]
        w = Tuple(inv.(masses))
        x = (SVector{3,T}(0, 0, 0), SVector{3,T}(sqrt(oh^2 - hh^2/4), hh/2, 0),
             SVector{3,T}(sqrt(oh^2 - hh^2/4), -hh/2, 0))
        for _ in 1:30
            v = ntuple(_ -> SVector{3,T}(randn(rng, 3)), 3)
            vp, ok = RW.rattle_water(x, v, w)
            @test ok
            # Independent dense mass-metric projection in Float64.
            J = zeros(3, 9)
            for (k, (a, b)) in enumerate(((1,2), (1,3), (2,3)))
                J[k, 3a-2:3a] = x[a] - x[b]
                J[k, 3b-2:3b] = x[b] - x[a]
            end
            M = Diagonal(repeat(Float64.(collect(w)); inner=3))
            flat = Float64.(vcat(v...))
            reference = flat - M * J' * ((J * M * J') \ (J * flat))
            @test vcat(vp...) ≈ reference atol=atol rtol=atol
            @test sum(masses[k] * vp[k] for k in 1:3) ≈ sum(masses[k] * v[k] for k in 1:3) atol=10atol
            @test sum(masses[k] * cross(x[k], vp[k]) for k in 1:3) ≈
                  sum(masses[k] * cross(x[k], v[k]) for k in 1:3) atol=atol
            @test sum(masses[k] * dot(vp[k], vp[k]) for k in 1:3) <=
                  sum(masses[k] * dot(v[k], v[k]) for k in 1:3) + 10atol

            trial = ntuple(k -> x[k] + T(0.002) * vp[k], 3)
            corrected, success, _ = RW.shake_water(x, trial, w, l2, tol, 50)
            @test success
            @test RW.distance_error(corrected, l2) <= tol
            @test sum(masses[k] * (corrected[k] - trial[k]) for k in 1:3) ≈ zeros(3) atol=atol
            # SHAKE corrections must lie along the OLD bond gradients.
            delta = Float64.(vcat((masses[k] * (corrected[k] - trial[k]) for k in 1:3)...))
            @test norm(delta - J' * (J' \ delta)) < 10atol
        end
        bad = ntuple(_ -> zero(x[1]), 3)
        @test !RW.shake_water(bad, bad, w, l2, tol, 20)[2]
        @test !RW.rattle_water(bad, x, w)[2]
        trial = (x[1], x[2] * T(1.1), x[3])
        @test !RW.shake_water(x, trial, w, l2, tol, 0)[2]
        box = (T(2), T(3), T(4))
        translated = ntuple(k -> x[k] + SVector{3,T}(T(0.98), T(1.48), T(1.99)), 3)
        wrapped = ntuple(k -> RW._image(translated[k], box), 3)
        localx = RW.local_water(wrapped, wrapped[1], box)
        @test all(isapprox(localx[k], x[k]; atol=atol) for k in 1:3)
    end

    # Free asymmetric rotor: constrained velocity Verlet must be reversible
    # and have bounded energy error. A geometric fit can pass length tests
    # while dissipating rotation; this test distinguishes it from RATTLE.
    oh, hh = 0.1, 0.16330
    x0 = (SVector(0.,0.,0.), SVector(sqrt(oh^2-hh^2/4),hh/2,0.), SVector(sqrt(oh^2-hh^2/4),-hh/2,0.))
    m = (15.9994, 1.008, 1.008); w = inv.(m); l2 = (oh^2, oh^2, hh^2)
    com = sum(m[k] * x0[k] for k in 1:3) / sum(m)
    x0 = ntuple(k -> x0[k] - com, 3)
    v0 = ntuple(k -> cross(SVector(1.,2.,3.), x0[k]), 3)
    energy(v) = sum(m[k] * dot(v[k], v[k]) for k in 1:3) / 2
    function free_step(x, v, dt)
        trial = ntuple(k -> x[k] + dt * v[k], 3)
        y, ok, _ = RW.shake_water(x, trial, w, l2, 1e-13, 50)
        ok || error("Free rotor SHAKE failed")
        vh = ntuple(k -> v[k] + (y[k] - trial[k]) / dt, 3)
        vnew, ok = RW.rattle_water(y, vh, w)
        ok || error("Free rotor RATTLE failed")
        return y, vnew
    end
    x, v = x0, v0
    max_energy_error = 0.0
    for _ in 1:2000
        x, v = free_step(x, v, 0.001)
        max_energy_error = max(max_energy_error, abs(energy(v) / energy(v0) - 1))
    end
    @test max_energy_error < 1e-4
    v = map(-, v)
    for _ in 1:2000; x, v = free_step(x, v, 0.001); end
    @test all(isapprox(x[k], x0[k]; atol=1e-8) for k in 1:3)
    @test all(isapprox(v[k], -v0[k]; atol=1e-7) for k in 1:3)
end
