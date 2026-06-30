using Printf
using Statistics

function _canonical_mode_vectors(taus, scales)
    tau_vals = taus isa Real ? Float64[Float64(taus)] : Float64.(collect(taus))
    scale_vals = scales isa Real ? Float64[Float64(scales)] : Float64.(collect(scales))
    M = max(length(tau_vals), length(scale_vals))
    (length(tau_vals) == 1 || length(tau_vals) == M) ||
        throw(ArgumentError("OU taus must be scalar or have length $(M)."))
    (length(scale_vals) == 1 || length(scale_vals) == M) ||
        throw(ArgumentError("OU scales must be scalar or have length $(M)."))
    length(tau_vals) == M || (tau_vals = fill(tau_vals[1], M))
    length(scale_vals) == M || (scale_vals = fill(scale_vals[1], M))
    return tau_vals, scale_vals
end

function _ou_msd_weight(steps::Integer, a::Float64)
    steps <= 0 && return 0.0
    total = Float64(steps)
    a_pow = a
    for lag in 1:(steps - 1)
        total += 2.0 * (steps - lag) * a_pow
        a_pow *= a
    end
    return total
end

function free_aoup_msd_1d(steps::Integer;
                          gamma::Real,
                          dt::Real,
                          thermal_temperature::Real=0.0,
                          taus,
                          scales)
    n = max(steps, 0)
    g = Float64(gamma)
    dt1 = Float64(dt)
    temp = Float64(thermal_temperature)
    tau_vals, scale_vals = _canonical_mode_vectors(taus, scales)

    thermal = 2.0 * temp * Float64(n) * dt1 / g
    active = 0.0
    for (tau, scale) in zip(tau_vals, scale_vals)
        a = tau <= 0 ? 0.0 : exp(-dt1 / tau)
        active += (scale / g)^2 * _ou_msd_weight(n, a)
    end
    return thermal + active
end

free_aoup_msd_2d(steps::Integer; kwargs...) = 2.0 * free_aoup_msd_1d(steps; kwargs...)

function run_free_aoup_msd_check(; label::AbstractString,
                                 csv_name::AbstractString,
                                 N::Integer,
                                 dt::Real,
                                 n_steps::Integer,
                                 sample_stride::Integer,
                                 gamma::Real,
                                 thermal_temperature::Real,
                                 taus,
                                 scales)
    tau_vals, scale_vals = _canonical_mode_vectors(taus, scales)
    n_side = ceil(Int, sqrt(Float64(max(N, 1))))
    boxL = max(2048.0, 16.0 * n_side)
    box = (boxL, boxL)

    all_particles = Group(:all, AllSelection())
    msd = MSDObservable(all_particles; name=:msd)
    method = if length(tau_vals) == 1 && length(scale_vals) == 1
        ActiveOrnsteinUhlenbeck(all_particles; gamma=gamma, tau=tau_vals[1], noise_scale=scale_vals[1])
    else
        ActiveOrnsteinUhlenbeck(all_particles; gamma=gamma, spectrum=OUSpectrum(tau_vals, scale_vals))
    end

    system = ParticleSystem(
        square_lattice_positions(N, box);
        box=PeriodicBox(box),
        types=[:C],
        typeids=fill(Int32(1), N),
        masses=Dict(:C => 0.0),
        velocities=[(0.0, 0.0) for _ in 1:N],
    )

    sim = Simulation(
        system;
        groups=Groups(all_particles),
        observables=[msd],
        integrator=Integrator(
            dt=dt,
            scheme=EulerHeun(),
            methods=[
                Brownian(all_particles; gamma=gamma, kT=thermal_temperature),
                method,
            ],
        ),
        precision=Float64,
        seed=0xA0B0,
    )

    prepare!(sim)
    reset_step!(sim, 0)
    reset_observables!(sim)

    csv_path = joinpath(@__DIR__, csv_name)
    open(csv_path, "w") do io
        println(io, "step,time,msd_numeric,msd_exact,msd_rel_err")
    end

    msd0 = ParticleDynamics.Workflow.sample_observable(sim, msd; fields=[:msd])
    open(csv_path, "a") do io
        @printf(io, "%d,%.8e,%.8e,%.8e,%.8e\n", 0, 0.0, msd0.msd, 0.0, 0.0)
    end

    errs = Float64[]
    final_num = 0.0
    final_ref = 0.0
    sampled_steps = 0
    while sampled_steps + sample_stride <= n_steps
        run!(sim, Stage(:sample, steps=sample_stride; progress=false))
        sampled_steps += sample_stride
        sample = ParticleDynamics.Workflow.sample_observable(sim, msd; fields=[:elapsed_steps, :elapsed_time, :msd])
        msd_ref = free_aoup_msd_2d(sample.elapsed_steps;
                                   gamma=gamma,
                                   dt=dt,
                                   thermal_temperature=thermal_temperature,
                                   taus=tau_vals,
                                   scales=scale_vals)
        err = abs(sample.msd - msd_ref) / max(abs(msd_ref), 1.0e-12)
        push!(errs, err)
        final_num = sample.msd
        final_ref = msd_ref
        open(csv_path, "a") do io
            @printf(io, "%d,%.8e,%.8e,%.8e,%.8e\n",
                    sample.elapsed_steps, sample.elapsed_time, sample.msd, msd_ref, err)
        end
    end

    mean_err = isempty(errs) ? 0.0 : mean(errs)
    max_err = isempty(errs) ? 0.0 : maximum(errs)
    final_err = abs(final_num - final_ref) / max(abs(final_ref), 1.0e-12)

    println(label)
    println(" - N = $(N), dt = $(dt), steps = $(n_steps), sample_stride = $(sample_stride)")
    println(" - gamma = $(gamma), thermal kT = $(thermal_temperature)")
    println(" - OU taus   = $(tau_vals)")
    println(" - OU scales = $(scale_vals)")
    @printf(" - mean relative MSD error  = %.4e\n", mean_err)
    @printf(" - max relative MSD error   = %.4e\n", max_err)
    @printf(" - final relative MSD error = %.4e\n", final_err)
    println("Wrote MSD comparison to $(csv_path)")

    return nothing
end
