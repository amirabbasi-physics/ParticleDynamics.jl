

export compute_interactions!


function compute_interactions!(wca::WCA, periodicity::SVector)
	dim = size(periodicity,1)
	if dim == 2
	    @inbounds @use_threads wca.multithreaded for particle in wca.particles
	        f_x, f_y = 0.0, 0.0
			for neighbor in wca.particles
				Δx = wrap_displacement(particle.r[1] - neighbor.r[1]; period = periodicity[1])
	            Δy = wrap_displacement(particle.r[2] - neighbor.r[2]; period = periodicity[2])
	            Δr² = Δx^2 + Δy^2
				σ² = 0.25*(particle.σ + neighbor.σ)^2
				rc² = wca.rc^2
	            if 0.0 < Δr² < rc²
					inv² = σ² / Δr²
	                inv⁶ = inv² * inv² * inv²
	                coef = wca.ϵ * (48.0 * inv⁶ - 24.0) * inv⁶ / Δr²
	                f_x += (_f_x = coef * Δx)
	                f_y += (_f_y = coef * Δy)
	                if wca.use_newton_3rd
	                    neighbor.f = neighbor.f - SVector{2,Float64}(f_x,f_y)
	                end
	            end
	        end
			particle.f = particle.f + SVector{2,Float64}(f_x,f_y)
		end
"""	elseif dim == 3
		@inbounds @use_threads wca.multithreaded for particle in wca.particles
	        i = floor(Int64, particle.r[1] / wca.cell_list.cell_dr[1])
	        j = floor(Int64, particle.r[2] / wca.cell_list.cell_dr[2])
			k = floor(Int64, particle.r[3] / wca.cell_list.cell_dr[3])
	        f_x, f_y,f_z = 0.0, 0.0, 0.0
	        for dk = -1 : 1, dj = -1 : 1, di = -1 : 1
	            idi = mod(i + di, wca.cell_list.num_cells[1]) + 1
	            jdj = mod(j + dj, wca.cell_list.num_cells[2]) + 1
				kdk = mod(k + dk, wca.cell_list.num_cells[3]) + 1
	            pid = wca.cell_list.start_pid[idi, jdj, kdk]
	            while pid > 0
	                neighbor = wca.cell_list.particles[pid]
	                Δx = wrap_displacement(particle.r[1] - neighbor.r[1]; period = periodicity[1])
	                Δy = wrap_displacement(particle.r[2] - neighbor.r[2]; period = periodicity[2])
					Δz = wrap_displacement(particle.r[3] - neighbor.r[3]; period = periodicity[3])
	                Δr² = Δx^2 + Δy^2 + Δz^2

	                σ² =  0.25*(particle.σ + neighbor.σ)^2
					rc² = wca.rc^2
	                if 0.0 < Δr² < rc²
	                    inv² = σ² / Δr²
	                    inv⁶ = inv² * inv² * inv²
	                    coef = wca.ϵ * (48.0 * inv⁶ - 24.0) * inv⁶ / Δr²

	                    f_x += (_f_x = coef * Δx)
	                    f_y += (_f_y = coef * Δy)
						f_z += (_f_z = coef * Δz)
	                    if wca.use_newton_3rd
	                        neighbor.f = neighbor.f - SVector{3,Float64}(f_x,f_y,f_z)
	                    end
	                end
	                pid = wca.cell_list.next_pid[pid]
	            end
	        end
	        particle.f = particle.f + SVector{3,Float64}(f_x,f_y,f_z)
	    end"""
	end
end



function compute_interactions!(harm_rep::Harmonic_Repulsive, periodicity::SVector)
	dim = size(periodicity,1)
	if dim == 2
		if dim == 2
		    @inbounds @use_threads harm_rep.multithreaded for particle in harm_rep.particles
		        f_x, f_y = 0.0, 0.0
				for neighbor in wca.particles
					Δx = wrap_displacement(particle.r[1] - neighbor.r[1]; period = periodicity[1])
		            Δy = wrap_displacement(particle.r[2] - neighbor.r[2]; period = periodicity[2])
					Δr² = Δx^2 + Δy^2
					Δr  = sqrt(Δr²)
					rc² = harm_rep.rc^2
					if 0.0 < Δr² < rc²
						coef = harm_rep.k*(harm_rep.rc - Δr)/Δr
						f_x += (_f_x = coef * Δx )
						f_y += (_f_y = coef * Δy )
						if harm_rep.use_newton_3rd
							neighbor.f = neighbor.f - SVector{2,Float64}(f_x,f_y)
						end
					end
				end
			end
			particle.f = particle.f + SVector{2,Float64}(f_x,f_y)
		end
"""	elseif dim == 3
		@inbounds @use_threads harm_rep.multithreaded for particle in harm_rep.particles
	        i = trunc(Int64, particle.r[1] / harm_rep.cell_list.cell_dr[1])
	        j = trunc(Int64, particle.r[2] / harm_rep.cell_list.cell_dr[2])
			k = trunc(Int64, particle.r[3] / harm_rep.cell_list.cell_dr[3])
	        f_x, f_y,f_z = 0.0, 0.0, 0.0
	        for dk = -1 : 1, dj = -1 : 1, di = -1 : 1
	            idi = mod(i + di, harm_rep.cell_list.num_cells[1]) + 1
	            jdj = mod(j + dj, harm_rep.cell_list.num_cells[2]) + 1
				kdk = mod(k + dk, harm_rep.cell_list.num_cells[3]) + 1
	            pid = harm_rep.cell_list.start_pid[idi, jdj, kdk]
	            while pid > 0
	                neighbor = harm_rep.cell_list.particles[pid]
	                Δx = wrap_displacement(particle.r[1] - neighbor.r[1]; period = periodicity[1])
	                Δy = wrap_displacement(particle.r[2] - neighbor.r[2]; period = periodicity[2])
					Δz = wrap_displacement(particle.r[3] - neighbor.r[3]; period = periodicity[3])
					Δr² = Δx^2 + Δy^2 + Δz^2
					Δr  = sqrt(Δr²)

					rc² = harm_rep.rc^2
					if 0.0 < Δr² < rc²
						coef = harm_rep.k*(harm_rep.rc - Δr)/Δr

	                    f_x += (_f_x = coef * Δx)
	                    f_y += (_f_y = coef * Δy)
						f_z += (_f_z = coef * Δz)
	                    if harm_rep.use_newton_3rd
	                        neighbor.f = neighbor.f - SVector{3,Float64}(f_x,f_y,f_z)
	                    end
	                end
	                pid = harm_rep.cell_list.next_pid[pid]
	            end
	        end
	        particle.f = particle.f + SVector{3,Float64}(f_x,f_y,f_z)
	    end"""
	end
end
