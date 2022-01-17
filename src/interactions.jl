

export compute_interactions!

"""
function compute_interactions!(harm_rep.particles, periodicity::SVector)
	dim = size(periodicity,1)
	if dim == 2
		N = nPart
    	tid = threadIdx().x
    	gtid = (blockIdx().x - 1) * blockDim().x + tid  # global thread id
    	shared = @cuStaticSharedMem(T, TH)
		full_blocks = N ÷ blockDim().x
    	rest = N % blockDim().x

		@inbounds begin
        if gtid <= N
            pos = particle[gtid]
        else
            pos = zero(T)
        end
        acc = zero(T)

		@inbounds @use_threads harm_rep.multithreaded for particle in harm_rep.particles
			i = trunc(Int64, particle.r[1] / harm_rep.cell_list.cell_dr[1])
			j = trunc(Int64, particle.r[2] / harm_rep.cell_list.cell_dr[2])
			f_x, f_y = 0.0, 0.0
			for dj = -1 : 1, di = -1 : 1
				idi = mod(i + di, harm_rep.cell_list.num_cells[1]) + 1
				jdj = mod(j + dj, harm_rep.cell_list.num_cells[2]) + 1
				pid = harm_rep.cell_list.start_pid[idi, jdj]
				while pid > 0
					neighbor = harm_rep.cell_list.particles[pid]
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
					pid = harm_rep.cell_list.next_pid[pid]
				end
			end
			particle.f = particle.f + SVector{2,Float64}(f_x,f_y)
		end
	elseif dim == 3
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
	    end
	end
end
"""
