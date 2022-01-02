export update_cell_list!


function update_cell_list!(cell_list::CellList,dim::Int64)
    fill!(cell_list.start_pid, -1)
    fill!(cell_list.next_pid, -1)
	if dim == 2
		@inbounds for (n, particle) in enumerate(cell_list.particles)
	        i = floor(Int64, particle.r[1] / cell_list.cell_dr[1]) + 1
	        j = floor(Int64, particle.r[2] / cell_list.cell_dr[2]) + 1
	        if cell_list.start_pid[i, j] > 0
	            cell_list.next_pid[n] = cell_list.start_pid[i, j]
	        end
	        cell_list.start_pid[i, j] = n
	    end
	elseif dim == 3
		@inbounds for (n, particle) in enumerate(cell_list.particles)
			i = trunc(Int64, particle.r[1] / cell_list.cell_dr[1]) + 1
			j = trunc(Int64, particle.r[2] / cell_list.cell_dr[2]) + 1
			k = trunc(Int64, particle.r[3] / cell_list.cell_dr[3]) + 1

			if cell_list.start_pid[i, j, k] > 0
				cell_list.next_pid[n] = cell_list.start_pid[i, j, k]
			end
			cell_list.start_pid[i, j, k] = n
		end
	end
end
