function Update_Particles!(brownian::Brownian, periodicity::SVector)
	dim = size(periodicity,1)
    @inbounds Threads.@threads for particle in brownian.particles
		if !isnothing(particle.τΓ)
			particle.v      = -(particle.τD/particle.τΓ) .*(particle.r - particle.r_pseu) .+ particle.f  .+ sqrt(2.0/brownian.dt) .* SRandns(dim)
			particle.v 		=  particle.free .* particle.v
			particle.v_pseu =  (particle.τD/particle.τΓ) .*(particle.r - particle.r_pseu)  .+ sqrt(2.0 * (particle.α+1)/brownian.dt) .* SRandns(dim)

			particle.r = particle.r .+ brownian.dt .* particle.v
	        particle.r_pseu = particle.r_pseu .+ brownian.dt .* particle.v_pseu

			PBC_ActiveTP!(particle, periodicity)

	        particle.f = SZeros(dim)

		else
			particle.v = particle.f .+ sqrt(2.0 * (particle.α + 1.0)/brownian.dt) .* SRandns(dim)
			particle.v = particle.free .* particle.v
			particle.r = particle.r .+ brownian.dt .* particle.v

	        PBC_PassiveBP!(particle, periodicity)

	        particle.f = SZeros(dim)
		end
    end
end


function Update_Particles!(langevin::Langevin, periodicity::SVector)
	dim = size(periodicity,1)
    @inbounds Threads.@threads for particle in langevin.particles
		if !isnothing(particle.τΓ)
			coeff1 = (langevin.dt * particle.τD / particle.τm)
			coeff2 = (particle.τD / particle.τΓ)
			particle.v      =  (1.0-coeff1) .* particle.v + coeff1 .* (-coeff2 .*(particle.r - particle.r_pseu) .+ particle.f  .+ sqrt(2.0/langevin.dt) .* SRandns(dim))
			particle.v 		=  particle.free .* particle.v
			particle.v_pseu =  coeff2 .*(particle.r - particle.r_pseu)  .+ sqrt(2.0 * (particle.α+1)/langevin.dt) .* SRandns(dim)

			particle.r = particle.r .+ langevin.dt .* particle.v
	        particle.r_pseu = particle.r_pseu .+ langevin.dt .* particle.v_pseu


			PBC_ActiveTP!(particle, periodicity)

	        particle.f = SZeros(dim)

		else
			coeff = (langevin.dt*particle.τD/particle.τm)
			particle.v = (1.0 - coeff) .* particle.v .+ coeff .* (particle.f .+ sqrt(2.0 * (particle.α + 1.0)/langevin.dt) .* SRandns(dim))
			particle.v = particle.free .* particle.v
			particle.r = particle.r .+ langevin.dt .* particle.v

	        PBC_PassiveBP!(particle, periodicity)

	        particle.f = SZeros(dim)
		end
    end
end
