
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


export volume
@inline function volume(R::T)::T where T
    return (4.0f0/3.0f0)*π*R*R*R
end

export friction

@inline function friction(η::T, R::T)::T where T
    return 6.0f0*π*η*R
end
