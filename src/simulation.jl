export hr_min_sec
export run_simulation!
export Force
export kinetic



@inline function hr_min_sec(time::Float64)
    hours = trunc(Int64, time / 3600.0)
    minutes = trunc(Int64, mod(time, 3600.0) / 60.0)
    seconds = trunc(Int64, mod(time, 60.0))

    return string(hours < 10 ? "0" : "", hours,
                  minutes < 10 ? ":0" : ":", minutes,
                  seconds < 10 ? ":0" : ":", seconds)
end


function run_simulation!(simulation::Simulation; message_interval::Float64 = 30.0)
    println("Number of threads available: ", Threads.nthreads())
    println("Number of particles: ", length(simulation.particles))
    println("Description: ", simulation.descriptor)
    println("")

	periodicity = simulation.periodicity
	dim = size(periodicity,1)
    prev_step = 0
    time_elapsed = 0.0
    interval_start = time()
	avg_param_counter = 0
	Force_tot = 0.0
	kinetic_tot = 0.0
	kinetic_theo = 0.0
	kinetic_tol = 0.0
    for step = 1 : simulation.num_steps
		if mod(step, 100) == 0
			Force_tot = 0.0
			kinetic_tot = 0.0
			kinetic_theo = 0.0
			kinetic_tol = 0.0
			avg_param_counter = 0
		end
		for interaction in simulation.interactions
			compute_interactions!(interaction, periodicity)
		end
		for particle in simulation.particles
			Force_tot += sum([abs.(particle.f[i]) for i=1:dim])
		end
        for integrator in simulation.integrators
			avg_param_counter += 1
			Update_Particles!(integrator, periodicity)
			num_parts = size(simulation.particles,1)
			[kinetic_tot += sum(kinetic(simulation.particles[i].v,simulation.particles[i].τm,simulation.particles[i].τD)) for i = 1:num_parts]
			[kinetic_theo += size(periodicity,1)*0.5*sum(simulation.particles[i].α + 1.0) for i = 1:num_parts]
        end


		#################### Revise R!!!!!! #########################
		if mod(step, simulation.save_interval) == 0
			write_xyz(simulation.output_file, size(simulation.particles,1), step, dim, simulation.particles)
		end
		kinetic_tol = 100.0*(kinetic_tot-kinetic_theo)/kinetic_theo

        interval_time = time() - interval_start
        if interval_time > message_interval || step == simulation.num_steps
            time_elapsed += interval_time
			step_diff = step - prev_step
            rate = step_diff / interval_time
            println(hr_min_sec(time_elapsed), " | ",
                    step, "/", simulation.num_steps, " (", round(step / simulation.num_steps * 100, digits = 1), "%) | ",
                    round(rate, digits = 1), " steps/s | ",
                    hr_min_sec((simulation.num_steps - step) / rate)," | ","Total Absolute Force : ", round(Force_tot/avg_param_counter,digits = 2))
            prev_step = step
            interval_start = time()
        end
    end
    println("Average steps/s: ", round(simulation.num_steps / time_elapsed, digits = 1))
end


@inline function kinetic(v::SVector,τm::Float64,τD::Float64)
	return (τm/(2.0*τD))*dot(v,v)
end
