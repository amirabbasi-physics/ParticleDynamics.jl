
export noise2D
export noise3D
@inline function noise2D(Npart::Int)
    return SVector{2,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart))
end

function noise3D(Npart::Int)
    return SVector{3,Float32}.(CUDA.randn(Float32,Npart) ,CUDA.randn(Float32,Npart),CUDA.randn(Float32,Npart))
end

export rectangular_lattice

function rectangular_lattice(s_x::Float32, s_y::Float32, M_x::Int64, M_y::Int64)
    positions = Array{SVector{2,Float32}, 1}()
    for i = 0 : M_x - 1, j = 0 : M_y - 1
        push!(positions, SVector{2,Float32}([(i + 0.5f0) * s_x, (j + 0.5f0) * s_y]))
    end
    return positions
end


export triangular_lattice

function triangular_lattice(L_box::T,lattice_const::T,M_x::Int64, M_y::Int64) where T
    """Calculates the positions of an hexagonal lattice with the lattice constant a
    in a square box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{2,Float32}, 1}()
    r_x = lattice_const*M_x/2
    r_y = lattice_const*M_y/2
    for i = 0:M_x-1
        for j = 0:M_y-1
            pos = [pos_triangulr(lattice_const)[n] .+ @SVector [i * lattice_const - r_x, j * lattice_const - r_y] for n = 1:4]
            for nn = 1:4
                #if pos_num < Npart
                    push!(positions, pos[nn])
                    #pos_num += 1
                #end
            end
        end
    end
    return positions
end


export pos_triangular
function pos_triangular(a::T) where T
    """returns the positions (x,y) of the 4 atoms in a hexagonal unit cell with the lattice constant a."""
    p₁ = @SVector [0.f0, 0.f0]
    p₂ = @SVector [0.f0, a]
    p₃ = @SVector [-0.5f0*a, 0.5f0 *sqrt(3)*a]
    p₄ = @SVector [0.5f0*a, 0.5f0 *sqrt(3)*a]
    return p₁, p₂, p₃, p₄
end


export triangular_circle
function triangular_circle(L_box::T, r0::Array{SVector{N,T}}, rad::T) where {N,T}
    new_pos = Array{SVector{2,T}, 1}()
    r_center = SVector{2,T}([0.5f0*L_box,0.5f0*L_box])
    for i = 1:length(r0)
        dist = r0[i] .- r_center
        if norm(dist)/rad <= 1
            push!(new_pos , r0[i])
        end
    end
    return new_pos
end


export simplecubic_lattice

function simplecubic_lattice(s_x::T, s_y::T, s_z::T, M_x::Int64, M_y::Int64, M_z::Int64) where T
    positions = Array{SVector{3,T}, 1}()
    for i = 0 : M_x - 1, j = 0 : M_y - 1, k = 0 : M_z - 1
        push!(positions, SVector{3,Float32}([(i + 0.5f0) * s_x, (j + 0.5f0) * s_y, (k + 0.5f0) * s_z]))
    end
    return positions
end


export fcc_sphere
function fcc_sphere(L_box::T, r0::Array{SVector{N,T}}, rad::T) where {N,T}
    new_pos = Array{SVector{3,T}, 1}()
    r_center = SVector{3,Float32}([0.5f0*L_box,0.5f0*L_box,0.5f0*L_box])
    for i = 1:length(r0)
        dist = r0[i] .- r_center
        if norm(dist)/rad <= 1
            push!(new_pos , r0[i])
        end
    end
    return new_pos
end

export fcc_lattice
function fcc_lattice(L_box::T,lattice_const::T,M_x::Int64, M_y::Int64, M_z::Int64) where T
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""
    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,Float32}, 1}()
    r_x = lattice_const*M_x/2
    r_y = lattice_const*M_y/2
    r_z = lattice_const*M_z/2
    for i = 0:M_x-1
        for j = 0:M_y-1
            for k = 0:M_z-1
                pos = [pos_fcc(lattice_const)[n] .+ @SVector [i * lattice_const - r_x, j * lattice_const - r_y, k * lattice_const - r_z] for n = 1:4]
                for nn = 1:4
                    #if pos_num < Npart
                        push!(positions, pos[nn])
                        #pos_num += 1
                    #end
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
function isinsphere(L::T, N::Int32, σ::T, pos::SVector{3,T}) where T
    mid_point = @SVector [0.5f0*L, 0.5f0*L, 0.5f0*L]
    sphere_rad = Float32((N/8)^(1/3)*σ)
    if norm(pos-mid_point) < sphere_rad
        return true
    else
        return false
    end
end

export volume
@inline function volume(R::T)::T where T
    return (4.0f0/3.0f0)*π*R*R*R
end

export friction

@inline function friction(η::T, R::T)::T where T
    return 6.0f0*π*η*R
end

export random_pos

function random_pos(dim::Int64, L::T) where T
	position = SVector{dim,T}(rand(dim)) .* L
	return position
end
