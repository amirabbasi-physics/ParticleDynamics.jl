

export PBC_PassiveBP!
export PBC_ActiveTP!


"""
    PBC_PassiveBP!(position; period)

Returns a new position after applying periodic boundary conditions.  The
periodicity is given by `period`.  If `period < 0`, then no periodic boundary
condition is applied.
"""


@inline function PBC_PassiveBP!(Particle::Particle, periodicity::SVector)
	dim = size(Particle.r,1)
	Particle.r = SVector{dim,Float64}(mod.(Particle.r, periodicity))
end

"""
@inline function PBC_PassiveBP!(Particle::Particle, periodicity::SVector)
	dim = size(Particle.r,1)
	a = zeros(dim)
	if dim == 2
		for i in 1:2
			if periodicity[i] > 0.0
	        	a[i] = mod(Particle.r[i], periodicity[i])
			else
				a[i] = Particle.r[i]
	    	end
		end
		Particle.r = SVector{2,Float64}(a)
	elseif dim == 3
		for i in 1:3
			if periodicity[i] > 0.0
	        	a[i] = mod(Particle.r[i], periodicity[i])
			else
				a[i] = Particle[i]
	    	end
		end
		Particle.r = SVector{3,Float64}(a)
	end
end
"""

@inline function PBC_ActiveTP!(ActiveTP::Particle, periodicity::SVector)
	dim = size(ActiveTP.r,1)
	a = zeros(dim)
	b = zeros(dim)
	if dim == 2
		@inbounds for i in 1:2
			if periodicity[i] > 0.0
	        	a[i] = mod(ActiveTP.r[i], periodicity[i])
				b[i] = a[i] - ActiveTP.r[i] + ActiveTP.r_pseu[i]
			else
				a[i] = ActiveTP.r[i]
				b[i] = ActiveTP.r_pseu[i]
	    	end
		end
		ActiveTP.r = SVector{2,Float64}(a)
		ActiveTP.r_pseu = SVector{2,Float64}(b)
	elseif dim == 3
		@inbounds for i in 1:3
			if periodicity[i] > 0.0
	        	a[i] = mod(ActiveTP.r[i], periodicity[i])
				b[i] = a[i] - ActiveTP.r[i] + ActiveTP.r_pseu[i]
			else
				a[i] = ActiveTP.r[i]
				b[i] = ActiveTP.r_pseu[i]
	    	end
		end
		ActiveTP.r = SVector{3,Float64}(a)
		ActiveTP.r_pseu = SVector{3,Float64}(b)
	end
end
