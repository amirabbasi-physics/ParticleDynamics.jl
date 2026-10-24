@testset "Rigid water CUDA projection" begin
    RW = ParticleDynamics.RigidWaterConstraints
    for T in (Float32, Float64)
        # Multiple disjoint waters, including one split over a periodic face.
        oh, hh, h = T(0.1), T(0.16330), T(0.002)
        tol = T === Float32 ? 2e-5 : 2e-10
        shape = (SVector{3,T}(0,0,0), SVector{3,T}(sqrt(oh^2-hh^2/4),hh/2,0),
                 SVector{3,T}(sqrt(oh^2-hh^2/4),-hh/2,0))
        box = (T(2), T(2), T(2))
        positions = [RW._image(shape[k] + SVector{3,T}(o,0,0), box) for o in (T(0.98), T(-0.3)) for k in 1:3]
        vel = [SVector{3,T}(0.2, -0.1, 0.05) + cross(SVector{3,T}(1,2,3), shape[k]) for _ in 1:2 for k in 1:3]
        posdev = ntuple(d -> CuArray([p[d] for p in positions]), 3)
        veldev = ntuple(d -> CuArray([v[d] for v in vel]), 3)
        unwrap = map(copy, posdev)
        ws = RW.WaterWorkspace(T, [(1,2,3),(4,5,6)], repeat([15.9994,1.008,1.008], 2); oh, hh)
        @test_throws ArgumentError RW.constrain_positions!(ws, posdev..., veldev..., box, h)
        RW.constrain_velocities!(ws, posdev..., veldev..., box)
        RW.save_reference!(ws, posdev...)
        for d in 1:3
            posdev[d] .+= h .* veldev[d]
            unwrap[d] .+= h .* veldev[d]
        end
        RW.constrain_positions!(ws, posdev..., veldev..., box, h; unwrapped=unwrap)
        @test_throws ArgumentError RW.constrain_positions!(ws, posdev..., veldev..., box, h)
        RW.constrain_velocities!(ws, posdev..., veldev..., box)
        p, v, u = map(x -> map(Array, x), (posdev, veldev, unwrap))
        for ids in ((1,2,3),(4,5,6))
            raw = ntuple(k -> SVector(ntuple(d -> p[d][ids[k]], 3)), 3)
            x = RW.local_water(raw, raw[1], box)
            vv = ntuple(k -> SVector(ntuple(d -> v[d][ids[k]], 3)), 3)
            @test RW.distance_error(x, (oh^2,oh^2,hh^2)) < tol
            @test maximum(abs(dot(a,b)) for (a,b) in zip(RW._edges(x), RW._edges(vv))) < tol
            # Same stage on the CPU, including the SHAKE velocity impulse.
            old = RW.local_water(ntuple(k -> positions[ids[k]], 3), positions[ids[1]], box)
            wi = (inv(T(15.9994)), inv(T(1.008)), inv(T(1.008)))
            pv, ok = RW.rattle_water(old, ntuple(k -> vel[ids[k]], 3), wi)
            @test ok
            trial = ntuple(k -> old[k] + h * pv[k], 3)
            corrected, ok, _ = RW.shake_water(old,trial,wi,ws.lengths2,ws.tolerance,50)
            @test ok
            expected, ok = RW.rattle_water(corrected, ntuple(k -> pv[k] + (corrected[k]-trial[k])/h,3), wi)
            @test ok
            @test all(isapprox(vv[k],expected[k]; atol=10tol,rtol=10tol) for k in 1:3)
        end
        for j in 1:6, d in 1:3
            @test abs((u[d][j]-p[d][j])/box[d] - round((u[d][j]-p[d][j])/box[d])) < tol
        end
        @test_throws ArgumentError RW.WaterWorkspace(T, [(1,2,3),(3,4,5)], ones(5); oh, hh)
    end
end

@testset "Constrained NVE integration and pressure" begin
    RW = ParticleDynamics.RigidWaterConstraints
    for T in (Float32, Float64)
        oh, hh, h = T(0.1), T(0.16330), T(0.0002)
        box = (T(4),T(4),T(4))
        masses = T[15.9994,1.008,1.008]
        shape = (SVector{3,T}(0,0,0), SVector{3,T}(sqrt(oh^2-hh^2/4),hh/2,0),
                 SVector{3,T}(sqrt(oh^2-hh^2/4),-hh/2,0))
        initial = ntuple(k -> RW._image(shape[k]+SVector{3,T}(1.98,0,0),box),3)
        com = sum(masses[k]*shape[k] for k in 1:3)/sum(masses)
        vcom = SVector{3,T}(0.2,-0.1,0.05)
        vel = ntuple(k -> vcom + cross(SVector{3,T}(1,2,3),shape[k]-com),3)
        # Nonzero conservative intramolecular forces exercise both kicks;
        # constraints should cancel their stretching acceleration and virial.
        st = build_simulation(N=3, box=box, cutoff=T(0.4), cap=Int32(8),
            epsilon=T(0), sigma=T(0.1), gamma=T(1), temperature=T(0), mass=masses,
            nonbonded=:none, precision=T === Float64 ? :f64 : :f32, dt=h,
            bonds=[(Int32(1),Int32(2)),(Int32(1),Int32(3)),(Int32(2),Int32(3))],
            bonding=ParticleDynamics.harmonic_bond(k=20.0,r0=0.12), spatial_reorder=false)
        for (d,a) in enumerate((st.rx,st.ry,st.rz)); copyto!(a,T[x[d] for x in initial]); end
        for (d,a) in enumerate((st.vx,st.vy,st.vz)); copyto!(a,T[x[d] for x in vel]); end
        attach_rigid_water!(st, [(1,2,3)]; oh, hh)
        @test kinetic_dof(st) == 6
        @test kinetic_dof(st; remove_com=true) == 3
        @test_throws ArgumentError attach_rigid_water!(st,[(1,2,3)];oh,hh)
        @test_throws ArgumentError set_cell_and_positions!(st,(2.1,2.1,2.1))
        @test_throws ArgumentError SimulationCore._validate_rigid_integrator!(st,nothing)
        @test_throws ArgumentError SimulationCore._check_barostat_state(st,"MC")     # atom-wise (C-rescale)
        @test SimulationCore._check_barostat_state(st,"MC";molecular=true)===nothing  # molecular MC moves
        @test_throws ArgumentError minimize!(st; method=FIRE(dt_initial=0.001,dt_max=0.01),force_tolerance=1e-6,max_displacement=0.01)
        # Writer must reject before touching its handle or losing topology.
        @test_throws ArgumentError write_gsd_frame!(nothing,st)
        E0 = total_potential_energy(st)+total_kinetic_energy(st)
        spec = nve(st; dt=h)
        tol = T === Float32 ? 5e-3 : 2e-6
        for i in 1:200
            step!(st,spec,h;compute_energy=(i%10==0))
            if i%25==0
                r,v = map(a -> map(Array,a),((st.rx,st.ry,st.rz),(st.vx,st.vy,st.vz)))
                raw = ntuple(k -> SVector(ntuple(d -> r[d][k],3)),3)
                x = RW.local_water(raw,raw[1],box)
                vv = ntuple(k -> SVector(ntuple(d -> v[d][k],3)),3)
                @test RW.distance_error(x,(oh^2,oh^2,hh^2)) < (T === Float32 ? 3e-5 : 2e-10)
                @test maximum(abs(dot(a,b)) for (a,b) in zip(RW._edges(x),RW._edges(vv))) < tol
                @test sum(masses[k]*vv[k] for k in 1:3) ≈ sum(masses)*vcom atol=tol
                @test abs(total_potential_energy(st)+total_kinetic_energy(st)-E0) < tol
                @test mechanical_pressure(st) ≈ sum(masses)*sum(abs2,vcom)/(3prod(box)) atol=tol
                W = total_configurational_virial(st)
                @test W.xx+W.yy+W.zz ≈ total_virial_trace(st) atol=tol
            end
        end
        @test st.step == 200
        @test st.force_valid
        wc = constraint_virial_trace(st)
        accumulate_virial!(st)
        @test constraint_virial_trace(st;accumulated=true) ≈ wc atol=tol
        # Velocity changes alone must refresh the constraint pressure term,
        # without inserting constraint forces into the conservative buffers.
        f_before = map(Array,(st.fx,st.fy,st.fz))
        for a in (st.vx,st.vy,st.vz); a .*= T(2); end
        @test mechanical_pressure(st) ≈ 4sum(masses)*sum(abs2,vcom)/(3prod(box)) atol=tol
        @test map(Array,(st.fx,st.fy,st.fz)) == f_before
        # On the device engine a failed constraint latches during the step and the
        # staged DomainError is thrown at the next host boundary (as FENE domain errors).
        st.rigid_water.workspace.maxiter = Int32(0)
        step!(st,spec,T(0.02);compute_energy=true)
        @test st.engine isa ParticleDynamics.DeviceEngine.StateEngine
        @test_throws DomainError ParticleDynamics.DeviceEngine.synchronize_state!(st)
    end
end
