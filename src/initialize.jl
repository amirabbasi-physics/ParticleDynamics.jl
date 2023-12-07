
export initialization, Box, random_pos, slab_create
 
function Box(; dim::I, Npart::I, ϕ::T, sigma::T) where {I,T}    
    if dim == 2
        L = sqrt(π*sigma^2*Npart/(4*ϕ))
        return SVector{2,T}([L,L])
    elseif dim == 3
        L = (π*sigma^3*Npart/(6*ϕ))^(1/3)
        return SVector{3,T}([L,L,L])
    end
end

function initialization(;
    homogeneous::Union{Bool,String} = true,
    dim::Int = 2,
    Npart::Int = 10000,
    ϕ::T = 0.5f0,
    fraction::T = 0.5f0,
    sigma::T = 1.0f0,
    random_positions::Bool = true,
    cold_frac::T = 0.5,
    ) where T

    box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma)

    if homogeneous == true
		if random_positions && (Npart <= 100000)			
    		num_cold = floor(Int, Npart*fraction)
			r_init = [random_pos(box)]
			for _ in 1:Npart-1
				pos = random_pos(box)    
				# Check for overlap with other particles
				while check_overlap(pos, r_init, box, sigma * T(1.05))
					pos = random_pos(box)
				end        
				# Append the position to the list of positions
				push!(r_init,pos)
			end
		else
    		num_cold = ceil(Int, Npart*fraction)
            if dim == 2
                r_init = rectangular_lattice(Npart,box)
                r_init = sort_pos_by_dist(r_init, zero(T), zero(T))
            elseif dim == 3
                r_init = simplecubic_lattice(Npart,box)
                r_init = sort_pos_by_dist(r_init, zero(T), zero(T), zero(T))
            end
		end
		shuffle!(r_init)
    
    elseif homogeneous == "slab"
        num_cold = floor(Int, Npart * fraction)
        num_cold_inhomo = floor(Int, Npart * fraction * cold_frac)
        slab_length = 50
        if dim == 2
            box, r_inhomogeneous, num_cold_inhomo, x_off_sets = slab_create(num_cold_inhomo, slab_length, sigma, box)
        end
        num_random = Npart - num_cold_inhomo
        #r_random = generate_random_positions(num_random, box, sigma, r_inhomogeneous)
        r_random = generate_random_positions(num_random, box, sigma, x_off_sets)
        r_init = merge_positions(r_inhomogeneous, r_random)
        r_init = r_init
    else homogeneous == false
        num_cold = floor(Int, Npart * fraction)
        num_cold_inhomo = floor(Int, Npart * fraction * cold_frac)
        num_random = Npart - num_cold_inhomo 
        if dim == 2
            r_inhomogeneous = circular_cut_triangular_lattice(num_cold_inhomo, sigma)
        elseif dim == 3
            r_inhomogeneous = spherical_cut_hcp_lattice(num_cold_inhomo, sigma)
        else
            error("Unsupported dimension: $dim")
        end
        
        r_random = generate_random_positions(num_random, box, sigma, r_inhomogeneous)
        r_init = merge_positions(r_inhomogeneous, r_random)
    end

    return box, r_init, num_cold
end


function slab_create(num_cold_inhomo::I, slab_length::I, sigma::T, box::SVector{N,T}) where {I, N, T}
    L_y = slab_length * sigma
    L_x = box[1] * box[2] / L_y
    box = SVector{2,T}(L_x, L_y)

    num_layers = floor(Int, num_cold_inhomo/slab_length)
    num_cold_inhomo = num_layers * slab_length

    if num_layers % 2 == 0
        num_layers += 1
    end
    num_layers_half = floor(Int, num_layers/2)
    a = sigma * sqrt(3)/2
    lattice = SVector{2,T}[]

    
    for column in -num_layers_half:num_layers_half
        y_offset = (column % 2) * sigma / 2
        for atom in 1:slab_length
            push!(lattice, SVector{2, T}(column * a, (atom - 1) * sigma + y_offset))
        end
    end
    x_offset_min = -num_layers_half * a
    x_offset_max =  num_layers_half * a
    return box, lattice[1:num_cold_inhomo], num_cold_inhomo, SVector{2,T}(x_offset_min,x_offset_max)
end


function circular_cut_triangular_lattice(num_particles::I, sigma::T) where {I, T}
    lattice = SVector{2, T}[]
    seen = Set{SVector{2, T}}()
    a = sigma * sqrt(2 / sqrt(3))  # Lattice constant for close packing

    layer = 0
    while length(lattice) < num_particles
        for i in -layer:layer
            for j in -layer:layer
                x = i * a + j * a / 2
                y = j * sqrt(3) * a / 2
                point = SVector{2, T}(x, y)
                if norm(point) <= layer * a * sqrt(3) / 2
                    if point ∉ seen
                        push!(lattice, point)
                        push!(seen, point)
                    end
                end
                if length(lattice) == num_particles
                    return lattice
                end
            end
        end
        layer += 1
    end

    return lattice
end

function spherical_cut_hcp_lattice(num_particles::I, sigma::T) where {I, T}
    lattice = SVector{3, T}[]
    a = sigma  * 1.05# Lattice constant

    # Estimate a radius and a range to generate a sufficiently large lattice
    radius = cbrt(3 * num_particles / (4 * π)) * a
    range = ceil(Int, radius / a) + 5  # Add buffer to ensure enough points are generated

    # Generate a larger lattice
    for i in -range:range
        for j in -range:range
            for k in -range:range
                x = i * a + (j % 2) * a / 2 + (k % 2) * a / 2
                y = j * sqrt(3) * a / 2 + (k % 2) * sqrt(3) * a / 6
                z = k * sqrt(6) * a / 3
                point = SVector{3, T}(x, y, z)

                # Add point if it is within the spherical radius
                if norm(point) <= radius
                    push!(lattice, point)
                end
            end
        end
    end

    # Trim the lattice to contain only the desired number of particles
    sort!(lattice, by = p -> norm(p))
    return lattice[1:min(num_particles, length(lattice))]
end


"""
# Function to generate random positions
function generate_random_positions(num_particles::I, box::SVector{N,T}, sigma::T, exclude_region::Array{SVector{N,T}}) where {I, N, T}
    dim = length(box)
    positions = SVector{dim, T}[]

    while length(positions) < num_particles
        pos = random_pos(box) # Modify for 3D if needed
        if all([norm(pos - p) > sigma for p in exclude_region])
            push!(positions, pos)
        end
    end

    return positions
end

"""

function generate_random_positions(num_particles::I, box::SVector{N,T}, sigma::T, x_off_set::SVector{N,T}) where {I, N, T}
    dim = length(box)
    positions = SVector{dim, T}[]

    while length(positions) < num_particles
        pos = random_pos(box) # Modify for 3D if needed
        if (pos[1] < x_off_set[1] - sigma || pos[1] > x_off_set[2] + sigma)
            push!(positions, pos)
        end
    end

    return positions
end


function random_pos(box::SVector{N,T}) where {N,T}
    dim = length(box)
	return SVector{dim,T}((rand(dim)) .-T(0.5) ) .* box
end
# Function to merge positions
function merge_positions(positions1, positions2)
    return vcat(positions1, positions2)
end


export sort_pos_by_dist
function sort_pos_by_dist(positions::Array{SVector{2,T},1}, x0::T, y0::T) where T
    distances = [(pos[1]-x0)^2 + (pos[2]-y0)^2 for pos in positions]
    sorted_indices = sortperm(distances)
    return positions[sorted_indices]
end

function sort_pos_by_dist(positions::Array{SVector{3,T},1}, x0::T, y0::T, z0::T) where T
    distances = [(pos[1]-x0)^2 + (pos[2]-y0)^2 + (pos[2]-z0)^2 for pos in positions]
    sorted_indices = sortperm(distances)
    return positions[sorted_indices]
end

export shuffle_pos!
function shuffle_pos!(simulation)
    r_tmp = [simulation.particles[i].r for i = 1:length(simulation.particles)]
    shuffle!(r_tmp)
    [simulation.particles[i].r = r_tmp[i] for i = 1:length(simulation.particles)]
    return nothing
end

export rectangular_lattice

function rectangular_lattice(Npart::Int, box::SVector{2, T}) where T
    n_side = ceil(Int,sqrt(Npart))
    delta = box ./ n_side
    x = range(delta[1]/2, stop=box[1]-delta[1]/2, length=n_side)
    y = range(delta[2]/2, stop=box[2]-delta[2]/2, length=n_side)
    positions = [SVector{2, T}(xi, yj) .- box/2 for xi in x, yj in y]
    return positions[1:Npart]
end

export triangular_lattice


function triangular_lattice(Npart::Int, σ::T) where T
    lattice_const = σ
    positions = Array{SVector{2,T}, 1}()
    M_x = ceil(Int,sqrt(Npart)) 
    M_y = ceil(Int,sqrt(Npart))
    for i = 1:M_x
        for j = 1:M_y
            pos = SVector{2,T}([(i-1)*lattice_const + (j%2)*lattice_const/2, (j-1)*lattice_const*sqrt(3)/2]) 
            push!(positions, pos)
        end
    end
    max_x = maximum([pos[1] for pos in positions])
    min_x = minimum([pos[1] for pos in positions])

    max_y = maximum([pos[2] for pos in positions])
    min_y = minimum([pos[2] for pos in positions])

    L_x = max_x - min_x
    L_y = max_y - min_y

    size = @SVector [L_x, L_y]
    positions = [pos = pos .- size/2 for pos in positions]
    return positions[1:Npart]
end

function triangular_lattice(Npart::Int, box::SVector{N,T}, σ::T) where {N,T}
    positions = Array{SVector{2,T}, 1}()
    M_x = ceil(Int,box[1]/σ)
    M_y = ceil(Int,2box[2]/(sqrt(3)*σ))
    for i = 1:M_x
        for j = 1:M_y
            pos = SVector{2,T}([(i-1)*σ + (j%2)*σ/2, (j-1)*σ*sqrt(3)/2]) 
            push!(positions, pos)
        end
    end
    positions = [pos = pos .- box/2 for pos in positions]
    return positions[1:Npart]
end

export simplecubic_lattice

function simplecubic_lattice(Npart::Int, box::SVector{3, T}) where T
    n_side = ceil(Int, cbrt(Npart))  # Cube root for 3D lattice
    delta = box ./ n_side
    x = range(delta[1]/2, stop=box[1]-delta[1]/2, length=n_side)
    y = range(delta[2]/2, stop=box[2]-delta[2]/2, length=n_side)
    z = range(delta[3]/2, stop=box[3]-delta[3]/2, length=n_side)
    positions = [SVector{3, T}(xi, yj, zk) .- box/2 for xi in x, yj in y, zk in z]
    return positions[1:Npart]
end


export fcc_lattice
function fcc_lattice(box::SVector{N,T},σ::T,M_x::Int64, M_y::Int64, M_z::Int64) where {N,T}
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,T}, 1}()
    for i = 0:M_x-1
        for j = 0:M_y-1
            for k = 0:M_z-1
                pos = [pos_fcc(σ)[n] .+ @SVector [i * σ, j * σ, k * σ] for n = 1:4]
                pos = [pos[i] .- box ./2 for i=1:4]
                for nn = 1:4
                    push!(positions, pos[nn])
                end
            end
        end
    end
    return positions
end
export pos_fcc
function pos_fcc(a::T) where T
    """returns the positions (x,y,z) of the 4 atoms in a fcc unit cell with the lattice constant a."""
    p₁ = @SVector [0.f0, 0.f0, 0.f0]
    p₂ = @SVector [0.f0, 0.5f0*a, 0.5f0*a]
    p₃ = @SVector [0.5f0*a, 0.f0, 0.5f0*a]
    p₄ = @SVector [0.5f0*a, 0.5f0*a, 0.f0]
    return p₁, p₂, p₃, p₄
end


export isinsphere
function isinsphere(pos::SVector{3,T}, rad::T, r_margin::T) where T
    if norm(pos) < rad + r_margin
        return true
    else
        return false
    end
end

export isincircle
function isincircle(pos::SVector{2,T}, rad::T, r_margin::T) where T
    #mid_point = @SVector T[0.50*L, 0.50*L]                
    if norm(pos) < rad + r_margin
        return true
    else
        return false
    end
end

export check_overlap
function check_overlap(pos::SVector{N,T}, positions::Vector{SVector{N,T}}, box::SVector{N,T}, sigma::T) where {N,T}
    dim = length(box)

    if dim == 2
        for p in positions

            dx  = pos[1] - p[1]
            dy  = pos[2] - p[2]
    
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
    
            dr² = dx*dx + dy*dy
            if dr² <= sigma^2
                return true
            end
        end
        return false
    elseif dim == 3
        for p in positions

            dx  = pos[1] - p[1]
            dy  = pos[2] - p[2]
            dz  = pos[3] - p[3]
    
            dx = (2abs(dx) > box[1] ) ? dx - sign(dx) * box[1] : dx
            dy = (2abs(dy) > box[2] ) ? dy - sign(dy) * box[2] : dy
            dz = (2abs(dz) > box[2] ) ? dz - sign(dz) * box[2] : dz
    
            dr² = dx*dx + dy*dy + dz*dz
            if dr² <= sigma^2
                return true
            end
        end
        return false
    end
end




export random_positions_init
function random_positions_init(Npart, box, sigma, float_precision)
    r_init = Vector{SVector{box.dim, float_precision}}()
    for _ in 1:Npart
        pos = random_pos(box)
        while check_overlap(pos, r_init, box, sigma * float_precision(1.05))
            pos = random_pos(box)
        end
        push!(r_init, pos)
    end
    return r_init
end

export regular_positions_init
function regular_positions_init(Npart, box, sigma, float_precision)
	r_init = rectangular_lattice(Npart,box)
	r_init = sort_pos_by_dist(r_init, zero(float_precision), zero(float_precision))
end