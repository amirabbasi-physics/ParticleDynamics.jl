
export initialization

function initialization(;
    homogeneous::Bool = true,
    dim::Int = 2,
    Npart::Int = 10000,
    ϕ::T = 0.5f0,
    fraction::T = 0.5f0,
    sigma::T = 1.0f0,
    random_positions::Bool = true,
    cold_frac::T = 0.5
    ) where T
    if homogeneous
		if random_positions && (Npart <= 100000)
			box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
    		num_cold = ceil(Int, Npart*fraction)
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
			box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
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
    else
        println("Warning! It is not modified for 3D systems!")				
		box = Box(dim = dim, Npart = Npart, ϕ = ϕ, sigma = sigma )
		num_cold = floor(Int, Npart*fraction)
		r_init = cut_circle_sphere!(box, sigma, Npart, fraction, cold_frac)
		if length(r_init) <= num_cold
			rr_remain = rectangular_lattice(2Npart,box)
			rad = T(sqrt(ceil(Npart .* fraction))/(2π/(1.0675*2sqrt(3))))
			r_remain = circle_cut(rr_remain, rad, false)
			shuffle!(r_remain)
		else
			error("r_droplet size is more than cold particles size!")
		end
		n_remain = Npart - length(r_init)
		r_init = append!(r_init, r_remain[1:n_remain])
		r_init = r_init[1:Npart]
		if !random_positions
			r_init = sort_pos_by_dist(r_init, zero(T), zero(T))
		end
    end
    return box, r_init, num_cold
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


export cut_circle_sphere

function cut_circle_sphere!(box::SVector{N,T}, σ::T, Npart::Int, fraction::T,cold_frac::T) where {N,T}
    dim = length(box)
    R = T(σ/2)
    if dim == 2
        N_circle = Npart * fraction * cold_frac * T(0.9)
        if σ == T(1.0)
            #rad = T(sqrt(N_circle)/(1.055*π/(sqrt(3))))
            rad = T(sqrt(N_circle * 2sqrt(3)/π))
        else
            rad = T(sqrt(N_circle)/(2π/(1.0675*2sqrt(3))))
        end
        #rad = sqrt(N_circle)/(π/(2sqrt(3)))
        N_lattice = ceil(Int,2N_circle)
        r = triangular_lattice(N_lattice, σ)
        r_init  = circle_cut(r, rad, true)
    elseif dim == 3

        N_sphere = Int(ceil(Npart .* fraction))
        rad = T(cbrt(N_sphere))
        nn = Int(ceil(rad))
        lattice_const = T(sqrt(2)*σ)
        r = fcc_lattice(box,lattice_const, nn, nn, nn)
        
        r1  = sphere_cut(r, rad)

        num_pl = length(r1)
        n_remain = Int(ceil(length(r1) *(1/fraction -1)))
        Npart_new = n_remain + num_pl 
        r2  = sphere_cut(r, rad*cold_frac)
        n_remain = Npart_new-length(r2)

        r_init = Array{SVector{3,T}}(undef, Npart_new)
        [r_init[i]=r2[i] for i = 1:length(r2)]
        for ii in 1:n_remain
            # Generate a random position
            pos = random_pos(box)    
            # Check for overlap with other particles
            while check_overlap(pos, r_init, R)
                pos = random_pos(box)
            end        
            # Append the position to the list of positions
            r_init[length(r2)+ ii] = pos
        end
    end
    return r_init
end

export circle_cut
function circle_cut(r0::Array{SVector{N,T}}, rad::T, in::Bool) where {N,T}
    new_pos = Array{SVector{2,T}, 1}()
    if in
        for i = 1:length(r0)
            if norm(r0[i])/rad < T(1.0)
                push!(new_pos , r0[i])
            end
        end
    else
        for i = 1:length(r0)
            if norm(r0[i])/rad > 1.0001
                push!(new_pos , r0[i])
            end
        end
    end

    return new_pos
end

export sphere_cut
function sphere_cut(r0::Array{SVector{N,T}}, rad::T) where {N,T}
    r0_new = Array{SVector{3,T}, 1}()
    #r_center = @SVector T[0.5*L_box,0.5*L_box,0.5*L_box]
    for i = 1:length(r0)
        if norm(r0[i])/rad <= 1
            push!(r0_new , r0[i])
        end
    end
    return r0_new
end

export random_pos
function random_pos(box::SVector{N,T}) where {N,T}
    dim = length(box)
	return SVector{dim,T}((rand(dim)) .-T(0.5) ) .* box
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