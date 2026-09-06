# ============================================================================
# All-pairs "neighbor list" (sentinel for O(N^2) interactions)
# ============================================================================

"""
AllPairsNeighborMatrix

Lightweight sentinel type indicating that all-to-all interactions should be
computed (no neighbor list), using periodic MIC and excluding i==j.

Fields are kept minimal but include `skin` so existing calls that read
`st.nbh.skin` continue to work without changes. `examples/2D_allpairs_quicktest.jl`
uses this type when validating the WCA path without the cell list overhead.
"""
struct AllPairsNeighborMatrix{T<:AbstractFloat} <: AbstractNeighborMatrix
    skin::T
    N::Int32
    D::Int32
end

@inline require_valid_neighbors(::AllPairsNeighborMatrix) = nothing

"""
    build_neighbors_allpairs!(rx, ry[ , rz]; box, cutoff, cap, skin)

Construct an `AllPairsNeighborMatrix` sentinel. Arguments mirror the dense
builders to keep call-sites uniform.
"""
function build_neighbors_allpairs!(rx::CuArray{T,1}, ry::CuArray{T,1};
                                   box::NTuple{2,T}, cutoff::T, cap::Int32, skin::T) where {T<:AbstractFloat}
    N = Int32(length(rx))
    return AllPairsNeighborMatrix{T}(skin, N, Int32(2))
end

function build_neighbors_allpairs!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1};
                                   box::NTuple{3,T}, cutoff::T, cap::Int32, skin::T) where {T<:AbstractFloat}
    N = Int32(length(rx))
    return AllPairsNeighborMatrix{T}(skin, N, Int32(3))
end

# No-op update hooks so existing NL maintenance code paths remain valid
function update_neighbors_inplace!(nbh::AllPairsNeighborMatrix{T}, args...; kwargs...) where {T<:AbstractFloat}
    return nbh
end

function update_needed!(nbh::AllPairsNeighborMatrix{T}, args...; kwargs...) where {T<:AbstractFloat}
    return false
end
