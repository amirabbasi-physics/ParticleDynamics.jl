# Deterministic reference cases isolate force lifecycle from thermostat sampling.
function lifecycle_pair(T, D; style=:dense, distance=2.91, counter=0, stride=1000, reorder=false)
    box = ntuple(_ -> T(20), D)
    st = build_simulation(N=2, box=box, cutoff=T(2.5), skin=T(0.4), cap=Int32(8),
        temperature=zero(T), precision=T === Float32 ? :f32 : :f64,
        use_neighborlist=style != :allpairs, spatial_reorder=reorder,
        reorder_interval=1, neigh_interval=stride)
    copyto!(st.rx, T[-distance/2, distance/2]); fill!(st.ry, zero(T))
    st.rz === nothing || fill!(st.rz, zero(T))
    fill!(st.vx, zero(T)); fill!(st.vy, zero(T))
    st.vz === nothing || fill!(st.vz, zero(T))
    st.step = counter
    coords = D == 2 ? (st.rx, st.ry) : (st.rx, st.ry, st.rz)
    if style == :stencil
        st.nbh = ParticleDynamics.NeighborLists.build_neighbors_stencil!(coords...;
            box, rcut_particle=fill(T(2.5), 2), skin=T(0.4), cap=Int32(8))
    else
        ParticleDynamics.NeighborLists.update_neighbors_inplace!(st.nbh, coords...; box, step=counter)
    end
    return st
end

function lifecycle_spec(st, kind, dt)
    kind == :nve && return SimulationCore.nve(st; dt)
    kind == :nhc && return SimulationCore.nosehooverchain(st; temperature=1., tau=1.)
    # Negligible CSVR coupling isolates force reuse from stochastic scaling.
    kind == :csvr && return SimulationCore.csvr(st; temperature=1., tau=1e30)
    constructors = (vv=SimulationCore.velocityverlet, baoab=SimulationCore.baoab,
                    baoa=SimulationCore.baoa, gsm=SimulationCore.gsm,
                    midpoint=SimulationCore.eulerheun, em=SimulationCore.eulermaruyama)
    return getproperty(constructors, kind)(st; gamma=1., temperature=0., dt)
end

struct LifecycleZeroProvider <: SimulationCore.AbstractExternalPotential end
function SimulationCore.external_forces!(::LifecycleZeroProvider, st, energy::Bool)
    fill!(st.fx, 0); fill!(st.fy, 0)
    st.fz === nothing || fill!(st.fz, 0)
    energy && fill!(st.Epot, 0)
    return nothing
end

@testset "Force-time neighbor coverage and force-cache lifecycle" begin
    SC = SimulationCore
    NL = ParticleDynamics.NeighborLists
    @testset "Approaching pair: no force can use the pre-drift list" begin
        @testset "$T D=$D $style $kind" for T in (Float32, Float64), D in (2,3), style in (:dense, :stencil),
            kind in (:nve, :vv, :baoab, :baoa, :gsm, :nhc, :csvr)
            st = lifecycle_pair(T, D; style)
            ref = lifecycle_pair(T, D; style=:allpairs)
            for s in (st, ref)
                copyto!(s.vx, T[0.4, -0.4])
                # Large dt is intentional: forces start at zero and the pair
                # enters the cutoff in this one step, even with stride=1000.
                step!(s, lifecycle_spec(s, kind, one(T)), one(T); compute_energy=true)
            end
            tol = T === Float32 ? 2e-5 : 1e-12
            @test Array(st.rx) ≈ Array(ref.rx) rtol=tol atol=tol
            @test Array(st.fx) ≈ Array(ref.fx) rtol=tol atol=tol
            @test Array(st.vx) ≈ Array(ref.vx) rtol=tol atol=tol
            @test st.force_valid
            @test maximum(abs, Array(ref.fx)) > 0
        end
    end
    @testset "Midpoint coverage, storage restoration, and physical collision history" begin
        for T in (Float32,Float64), D in (2,3), style in (:dense,:stencil)
            st = lifecycle_pair(T,D; style)
            ref = lifecycle_pair(T,D; style=:allpairs, distance=2.31)
            ParticleDynamics.enable_collision_counting!(st; ntypes=1)
            x, v = st.rx, st.vx
            copyto!(st.vx, T[-1.155,1.155])
            SC.evaluate_midpoint_forces_into_f0!(st)
            SC.evaluate_forces_into_f!(ref, false)
            @test st.rx === x && st.vx === v
            @test Array(st.f0x) ≈ Array(ref.fx)
            @test !st.force_valid
            @test ParticleDynamics.collisions_read_counts!(st) == [0]
            copyto!(st.rx, T[-1.155,1.155])
            SC.apply_post_position_hooks!(st, :after_final_position)
            @test ParticleDynamics.collisions_read_counts!(st) == [1]
            # Rebuild in a new storage order at the same physical positions.
            st.nbh.valid = false
            SC.ensure_force_neighbors!(st)
            SC.apply_post_position_hooks!(st, :after_final_position)
            @test ParticleDynamics.collisions_read_counts!(st) == [1]
            ParticleDynamics.disable_collision_counting!(st)
            @test st.coll_ref_x === nothing && st.coll_ref_y === nothing && st.coll_ref_z === nothing
        end
    end
    @testset "Collision entry on the force-time rebuild step" begin
        for D in (2,3), style in (:dense,:stencil)
            st = lifecycle_pair(Float64,D; style)
            ParticleDynamics.enable_collision_counting!(st; ntypes=1)
            copyto!(st.vx, [.3,-.3])
            step!(st, SC.nve(st), 1.; compute_energy=false)
            @test ParticleDynamics.collisions_read_counts!(st) == [1]
        end
    end
    @testset "First step independent of restart counter; cached force survives counter reset" begin
        for D in (2,3), kind in (:nve,:vv,:baoab,:baoa,:gsm,:nhc,:csvr,:midpoint,:em)
            a = lifecycle_pair(Float64,D; distance=1.2, counter=0)
            b = lifecycle_pair(Float64,D; distance=1.2, counter=100)
            sa, sb = lifecycle_spec(a,kind,.001), lifecycle_spec(b,kind,.001)
            for _ in 1:3
                step!(a,sa,.001; compute_energy=true)
                step!(b,sb,.001; compute_energy=true)
                @test Array(a.rx) ≈ Array(b.rx) atol=1e-12 rtol=1e-12
                @test Array(a.vx) ≈ Array(b.vx) atol=1e-12 rtol=1e-12
                @test a.force_valid && b.force_valid
            end
            b.step = 0
            step!(a,sa,.001); step!(b,sb,.001)
            @test Array(a.rx) ≈ Array(b.rx) atol=1e-12 rtol=1e-12
            @test Array(a.vx) ≈ Array(b.vx) atol=1e-12 rtol=1e-12
        end
    end
    @testset "Manual edits and provider transitions invalidate first kick" begin
        a = lifecycle_pair(Float64,2; distance=1.2, counter=100)
        SC.evaluate_forces_into_f!(a,false)
        ParticleDynamics.attach_external_potential!(a,LifecycleZeroProvider())
        @test !a.force_valid
        step!(a,SC.nve(a),.01)
        @test Array(a.vx) == [0.,0.]
        @test Array(a.rx) == [-.6,.6]
        ParticleDynamics.detach_external_potential!(a)
        @test !a.force_valid
        b = lifecycle_pair(Float64,2; distance=1.2)
        step!(a,SC.nve(a),.01); step!(b,SC.nve(b),.01)
        @test Array(a.rx) ≈ Array(b.rx)
        @test Array(a.vx) ≈ Array(b.vx)
        copyto!(a.rx, [-.7,.7]); fill!(a.vx,0)
        @test ParticleDynamics.invalidate_forces!(a) === a
        b = lifecycle_pair(Float64,2; distance=1.4)
        step!(a,SC.nve(a),.01); step!(b,SC.nve(b),.01)
        @test Array(a.rx) ≈ Array(b.rx)
        @test Array(a.vx) ≈ Array(b.vx)
    end
    @testset "Spring expiration and removal invalidate the cached initial kick" begin
        for expire in (false,true)
            a = lifecycle_pair(Float64,2; distance=1.2)
            ParticleDynamics.Filters.freeze_particles!(a; mode=:spring, k=10., steps=expire ? 1 : nothing)
            spec = SC.nve(a)
            step!(a,spec,.01)
            @test a.force_valid && a.force_freeze_spring
            b = lifecycle_pair(Float64,2; distance=1.2, counter=a.step)
            copyto!(b.rx,a.rx); copyto!(b.vx,a.vx)
            if !expire
                ParticleDynamics.Filters.unfreeze_particles!(a)
                @test !a.force_valid
            end
            step!(a,spec,.01); step!(b,SC.nve(b),.01)
            @test Array(a.rx) ≈ Array(b.rx) atol=1e-12 rtol=1e-12
            @test Array(a.vx) ≈ Array(b.vx) atol=1e-12 rtol=1e-12
            @test !a.force_freeze_spring
        end
    end
    @testset "Overflow during motion fails before force consumption" begin
        st = build_simulation(N=4, box=(20.,20.), cutoff=2.5, skin=.4, cap=Int32(2),
            temperature=0., precision=:f64, spatial_reorder=false, neigh_interval=1000)
        copyto!(st.rx, [-6.,-2.,2.,6.]); fill!(st.ry,0); fill!(st.vy,0)
        copyto!(st.vx, [6.,2.8,-.4,-3.6])
        @test_throws NL.NeighborCapacityError step!(st,SC.nve(st),1.)
        @test !st.force_valid && !st.nbh.valid && st.step == 0
        # Failed midpoint evaluation must restore physical slots and must not
        # mark the midpoint result as a reusable physical force.
        copyto!(st.rx, [-6.,-2.,2.,6.])
        copyto!(st.vx, [0.,.8,1.6,2.4])
        x, v, f, f0 = st.rx, st.vx, st.fx, st.f0x
        @test_throws NL.NeighborCapacityError SC.evaluate_midpoint_forces_into_f0!(st)
        @test st.rx === x && st.vx === v && st.fx === f && st.f0x === f0
        @test !st.force_valid
    end
    @testset "Boundary reordering preserves identity and current force" begin
        a = lifecycle_pair(Float64,2; distance=1.2, reorder=true)
        b = lifecycle_pair(Float64,2; distance=1.2, style=:allpairs)
        copyto!(a.rx, [-.7,-1.9]); copyto!(b.rx, [-.7,-1.9])
        sa, sb = SC.nve(a), SC.nve(b)
        for _ in 1:4
            step!(a,sa,.01); step!(b,sb,.01)
            order = sortperm(Array(a.tag))
            @test Array(a.rx)[order] ≈ Array(b.rx)
            @test Array(a.vx)[order] ≈ Array(b.vx)
        end
        @test Array(a.tag) == [2,1]
        @test a.last_reorder_step == 3
    end
end
