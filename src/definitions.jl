
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

export simplecubic_lattice

function simplecubic_lattice(s_x::Float32, s_y::Float32, s_z::Float32, M_x::Int64, M_y::Int64, M_z::Int64)
    positions = Array{SVector{3,Float32}, 1}()
    for i = 0 : M_x - 1, j = 0 : M_y - 1, k = 0 : M_z - 1
        push!(positions, SVector{3,Float32}([(i + 0.5f0) * s_x, (j + 0.5f0) * s_y, (k + 0.5f0) * s_z]))
    end
    return positions
end
export fcc_lattice

function fcc_lattice(Npart::Int64, L::T,σ::T,M_x::Int64, M_y::Int64, M_z::Int64) where T
    """Calculates the positions of an fcc lattice with the lattice constant a
    in a cubic box with the given dimensions"""

    # initialize coordinates: time 4 since there are 4 atoms in each unit cell
    positions = Array{SVector{3,Float32}, 1}()
    pos_num = 0
    for i = 1: M_x
        for j = 1: M_y
            for k = 1: M_z
                pos = [pos_fcc(σ)[n] .+ @SVector [i .*σ, j .*σ , k .*σ] for n = 1:4]
                for nn = 1:4
                    if pos_num < Npart
                        push!(positions, pos[nn])
                        pos_num += 1
                    end
                end
            end
        end
    end
    return positions
end
export pos_fcc
function pos_fcc(σ::Float32)
    """returns the positions (x,y,z) of the 4 atoms in a fcc unit cell with the lattice constant a."""
    p₁ = @SVector [0.f0, 0.f0, 0.f0]
    p₂ = @SVector [0.f0, 0.5f0*σ, 0.5f0*σ]
    p₃ = @SVector [0.5f0*σ, 0.f0, 0.5f0*σ]
    p₄ = @SVector [0.5f0*σ, 0.5f0*σ, 0.f0]
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
