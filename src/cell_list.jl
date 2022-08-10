

"""
struct CellList
    start_pid::Array{Int64, 2}
    next_pid::Array{Int64, 1}

    particles::Array{Particle, 1}

    num_cells_x::Int64
    num_cells_y::Int64
    cell_spacing_x::Float64
    cell_spacing_y::Float64

    function CellList(; particles::Array{Particle, 1}, L_x::Float64, L_y::Float64, cutoff::Float64)
        num_cells_x = trunc(Int64, L_x / cutoff)
        num_cells_y = trunc(Int64, L_y / cutoff)
        cell_spacing_x = L_x / num_cells_x
        cell_spacing_y = L_y / num_cells_y

        start_pid = -ones(Int64, num_cells_x, num_cells_y)
        next_pid = -ones(Int64, length(particles))

        for (n, particle) in enumerate(particles)
            i = trunc(Int64, particle.x / cell_spacing_x) + 1
            j = trunc(Int64, particle.y / cell_spacing_y) + 1

            if start_pid[i, j] > 0
                next_pid[n] = start_pid[i, j]
            end
            start_pid[i, j] = n
        end

        new(start_pid, next_pid, particles, num_cells_x, num_cells_y, cell_spacing_x, cell_spacing_y)
    end
end
"""
