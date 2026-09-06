module NeighborReference

export reference_neighbor_rows

"""Independent host oracle: minimum-image distances, one row per particle."""
function reference_neighbor_rows(coords::Tuple, box::Tuple, radii::AbstractVector)
    N = length(first(coords))
    length(coords) == length(box) || throw(DimensionMismatch("Coordinate/box dimension mismatch"))
    all(x -> length(x) == N, coords) && length(radii) == N ||
        throw(DimensionMismatch("Particle count mismatch"))
    return [[j for j in 1:N if j != i &&
             sum((Float64(coords[d][j]) - Float64(coords[d][i]) -
                  Float64(box[d]) * round((Float64(coords[d][j]) - Float64(coords[d][i])) / Float64(box[d])))^2
                 for d in eachindex(coords)) <= Float64(radii[i])^2]
            for i in 1:N]
end

end
