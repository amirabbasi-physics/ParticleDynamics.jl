module DeviceEngineCSVRReportingGPUChecks
using Test, CUDA, ParticleDynamics, Random
const SC=ParticleDynamics.SimulationCore
const DE=ParticleDynamics.DeviceEngine

CUDA.functional() || error("CSVR reporting checks require a CUDA device")

function state(::Type{T}) where {T}
    N=343; rng=MersenneTwister(71)
    st=SC.build_simulation(;N,box=(11,11,11),nonbonded=:lj,cutoff=T(2.5),skin=T(0.3),
        cap=Int32(256),epsilon=T(1),sigma=T(1),mass=T[isodd(i) ? 1 : 2 for i in 1:N],
        dt=T(0.002),precision=T===Float32 ? :f32 : :f64,spatial_reorder=false)
    sites=collect(Iterators.product(0:6,0:6,0:6))[:]
    for (d,r,v) in ((1,st.rx,st.vx),(2,st.ry,st.vy),(3,st.rz,st.vz))
        copyto!(r,T[1.4*sites[i][d]+0.5 for i in 1:N])
        copyto!(v,T.(0.1.*randn(rng,N)))
    end
    set_nonbonded_backend!(st,:cluster;prune_buffer=0.05,list_intervals=(8,2))
    st
end

@testset "Cluster CSVR observation-gated reporting" begin
    saved=(DE.CSVR_LAZY_REPORT[],DE.FOLD_TAIL[])
    try
        for T in (Float32,Float64), interval in (1,10), fold in (false,true)
            DE.FOLD_TAIL[]=fold
            states=[state(T),state(T),state(T)] # eager/lazy plus independent eager noise control
            specs=[SC.csvr(st;temperatures=T[0.2,0.4],taus=T[0.1,0.2],interval) for st in states]
            for (lazy,st,sp) in zip((false,true,false),states,specs)
                SC.ensure_integrator_workspace!(sp,st)
                copyto!(sp.workspace.particle_bath_id,Int32[isodd(i) ? 1 : 2 for i in eachindex(st.rx)])
                DE.CSVR_LAZY_REPORT[]=lazy
                Random.seed!(0x71c5) # StateEngine draws its CSVR seed at first capture
                SC.step!(st,sp,st.dt;compute_energy=false) # capture independent variants
            end
            DE.CSVR_LAZY_REPORT[]=true
            A,B,C=states; sa,sb,sc=specs
            @test A.engine.run.thermostat.seed==B.engine.run.thermostat.seed
            @test !DE.csvr_lazy_report(A.engine.run)
            @test DE.csvr_lazy_report(B.engine.run)
            @test !DE.csvr_lazy_report(C.engine.run)
            if interval==10
                # Read CuArrays directly, without a SimulationState boundary: this
                # proves eager and lazy captured graphs really execute different work.
                for sp in specs; fill!(sp.workspace.kinetic_total_per_bath,T(NaN)); end
                for (st,sp) in zip(states,specs); SC.step!(st,sp,st.dt;compute_energy=false); end
                @test all(isfinite,Array(sa.workspace.kinetic_total_per_bath))
                @test all(isnan,Array(sb.workspace.kinetic_total_per_bath))
                for st in states; DE.synchronize_state!(st); end
            end
            firststep=interval==10 ? 3 : 2
            for i in firststep:270 # coupling, sampling, build/prune and periodic boundary crossings
                for (st,sp) in zip(states,specs)
                    SC.step!(st,sp,st.dt;compute_energy=i%7==0)
                end
            end
            for st in states; DE.synchronize_state!(st); end
            tol=T===Float32 ? 2e-5 : 1e-11
            for field in (:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:Ekin)
                a,b,c=map(st->Array(getfield(st,field)),states)
                ab=maximum(abs.(a.-b)); aa=maximum(abs.(a.-c))
                # Atomic force/bath sums are not bitwise reproducible, even eager/eager.
                # One eager/eager sample is not a worst-case bound for 270 steps of
                # rounding feedback. Allow a small double-precision accumulation
                # floor, while also enforcing the trajectory accuracy tolerance.
                scale=max(one(T),maximum(abs.(a)),maximum(abs.(c)))
                floor=(T===Float32 ? 64 : 1024)*eps(T)*scale
                println("trajectory error T=$T interval=$interval fold=$fold field=$field AB=$ab AA=$aa floor=$floor")
                @test ab<=min(max(8aa,floor),tol*scale)
            end
            # Isolate the changed kernel from nondeterministic force/thermostat feedback:
            # omitting reporting must leave its velocity and Ekin arithmetic bitwise equal.
            run=B.engine.run; th=run.thermostat
            velocities=[map(copy,run.v) for _ in 1:2]
            energies=[similar(B.Ekin) for _ in 1:2]
            for j in 1:2
                k=CUDA.zeros(Float64,2); n=CUDA.zeros(Int32,2)
                @cuda threads=256 blocks=cld(length(B.rx),256) DE._csvr_kick_measure_kernel!(velocities[j],run.f,energies[j],
                    th.bath_id,th.scale,k,n,(run.dt/T(2))/run.mass,run.mass,Val(2),run.fx,Val(true),Val(j==1))
            end
            @test all(Array(a)==Array(b) for (a,b) in zip(velocities...))
            @test Array(energies[1])==Array(energies[2])
            @test A.step==B.step==270
            for field in (:cumulative_energy_exchange_per_bath,:last_velocity_scale_per_bath,:kinetic_total_per_bath)
                a,b,c=map(sp->Array(getfield(sp.workspace,field)),specs)
                ab=maximum(abs.(a.-b)); aa=maximum(abs.(a.-c))
                scale=max(one(T),maximum(abs.(a)),maximum(abs.(c)))
                floor=(T===Float32 ? 64 : 1024)*eps(T)*scale
                println("bath error T=$T interval=$interval fold=$fold field=$field AB=$ab AA=$aa floor=$floor")
                @test ab<=min(max(8aa,floor),tol*scale)
            end
            ids=Array(sb.workspace.particle_bath_id); ek=Array(B.Ekin)
            @test isapprox(Array(sb.workspace.kinetic_total_per_bath),[sum(ek[ids.==b]) for b in 1:2];rtol=tol)
            # An arbitrary public observation after an unsampled step must refresh bath K.
            SC.step!(B,sb,B.dt;compute_energy=false)
            obs=SC.collect_step_observables(B,sb)
            @test isapprox(obs.physical_kinetic,sum(Array(B.Ekin));rtol=tol)
            @test isapprox(sum(Array(sb.workspace.kinetic_total_per_bath)),obs.physical_kinetic;rtol=tol)
            SC.step!(B,sb,B.dt;compute_energy=false)
            fill!(sb.workspace.kinetic_total_per_bath,T(NaN))
            # A switch change affects new runs only. Existing lazy graphs still
            # require a fresh report at the public observation boundary.
            DE.CSVR_LAZY_REPORT[]=false
            @test DE.csvr_lazy_report(B.engine.run)
            direct=SC.collect_integrator_observables(sb,B)
            @test isapprox(direct.physical_kinetic,sum(Array(B.Ekin));rtol=tol)
            @test all(isfinite,Array(sb.workspace.kinetic_total_per_bath))
            @test isapprox(sum(Array(sb.workspace.kinetic_total_per_bath)),direct.physical_kinetic;rtol=tol)
            @test all(iszero,Array(B.engine.run.thermostat.kfull))
            @test all(iszero,Array(B.engine.run.thermostat.nfull))
        end
        @testset "Pressure control retains eager reporting" begin
            DE.CSVR_LAZY_REPORT[]=true
            st=state(Float32)
            sp=SC.csvr(st;temperature=0.2f0,tau=0.1f0,interval=10)
            pc=SC.StochasticCellRescalingState(st;pressure=0.01,kT=0.2,
                compressibility=0.01,tau=5.0,seed=29)
            SC.attach_pressure_control!(st,pc;interval=10)
            SC.step!(st,sp,st.dt;compute_energy=false)
            @test st.engine.run.cluster.barostat!==nothing
            @test !DE.csvr_lazy_report(st.engine.run)
            fill!(sp.workspace.kinetic_total_per_bath,Float32(NaN))
            SC.step!(st,sp,st.dt;compute_energy=false)
            @test all(isfinite,Array(sp.workspace.kinetic_total_per_bath))
        end
    finally
        DE.CSVR_LAZY_REPORT[]=saved[1]; DE.FOLD_TAIL[]=saved[2]
    end
end
end
