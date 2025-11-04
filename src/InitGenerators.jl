module InitGenerators

using Random
using StaticArrays
using ..Definitions: Box2

export box_from_phi_2d,
       hex_random_2d,
       hex_circle_2d,
       hex_circle_plus_random_2d,
       hex_sites_in_box_2d,
       hex_circle_in_box_2d

"""
box_from_phi_2d(N, ϕ, σ; T=Float32)

Compute a square 2D box (L,L) that achieves the desired area fraction ϕ for N
disks of diameter σ. Returns an NTuple{2,T} compatible with the simulation API.
"""
function box_from_phi_2d(N::Integer, ϕ::Real, σ::Real; T=Float32)
    @assert ϕ > 0 "Area fraction must be positive"
    @assert σ > 0 "σ must be positive"
    A = N * (π * (σ^2) / 4) / ϕ
    L = T(sqrt(A))
    return (L, L)::Box2{T}
end

"""
hex_sites_in_box_2d(box, σ; T=eltype(box))

Generate all 2D hex (triangular) lattice sites inside a periodic square box
centered at the origin, using nearest-neighbor spacing a = σ. Coordinates are
returned as SVector{2,T} shifted to [-L/2, L/2) in each dimension.
"""
function hex_sites_in_box_2d(box::Box2{T}, σ::T) where {T<:AbstractFloat}
    Lx, Ly = box
    a = σ
    ax = a
    ay = a * sqrt(T(3)) / T(2)  # vertical spacing between rows

    # number of rows that fit
    ny = max(0, floor(Int, Ly / ay))

    sites = Vector{SVector{2,T}}()
    sites_cap = ceil(Int, (Lx * Ly) * (T(2) / (sqrt(T(3)) * a^2)))  # density of triangular lattice = 2/(√3 a^2)
    sizehint!(sites, sites_cap)

    for j in 0:ny
        y = T(j) * ay
        if y >= Ly - eps(T); break; end
        xoff = isodd(j) ? (ax / T(2)) : zero(T)
        # number of columns that fit given offset
        nx = max(0, floor(Int, (Lx - xoff) / ax))
        for i in 0:nx-1
            x = xoff + T(i) * ax
            # center to origin and push
            push!(sites, SVector{2,T}(x - Lx/T(2), y - Ly/T(2)))
        end
    end
    return sites
end

# Partial Fisher-Yates sampling without replacement (O(N) swaps, no big perm)
function _choose_indices_without_replacement(M::Integer, K::Integer; rng::AbstractRNG=Random.default_rng())
    @assert 0 <= K <= M "Cannot choose $K from $M"
    arr = collect(1:M)
    sel = Vector{Int}(undef, K)
    for t in 1:K
        k = rand(rng, t:M)
        arr[t], arr[k] = arr[k], arr[t]
        sel[t] = arr[t]
    end
    return sel
end

"""
hex_random_2d(N, σ, ϕ; T=Float32, rng=Random.default_rng())

Randomly place N particles on hex-lattice sites inside a square box sized from
area fraction ϕ and diameter σ. Uses a = σ for nearest-neighbor spacing (no
overlaps under periodic BCs). Returns a named tuple:
  (box, positions, indices)
where `box::NTuple{2,T}`, `positions::Vector{SVector{2,T}}`, and `indices` are
the selected site indices in the full site list (handy for further sampling).
"""
function hex_random_2d(N::Integer, σ::Real, ϕ::Real; T=Float32, rng::AbstractRNG=Random.default_rng())
    box = box_from_phi_2d(N, ϕ, σ; T=T)
    sites = hex_sites_in_box_2d(box, T(σ))
    @assert length(sites) >= N "Not enough lattice sites generated for given parameters"
    idx = _choose_indices_without_replacement(length(sites), N; rng=rng)
    pos = [sites[i] for i in idx]
    return (box=box, positions=pos, indices=idx, sites=sites)
end

"""
hex_circle_2d(N, σ, ϕ; T=Float32)

Place N particles as the innermost hex-lattice sites forming a circular cluster
around the box center. The box is sized from (N, ϕ, σ). Returns (box, positions).
"""
function hex_circle_2d(N::Integer, σ::Real, ϕ::Real; T=Float32)
    box = box_from_phi_2d(N, ϕ, σ; T=T)
    sites = hex_sites_in_box_2d(box, T(σ))
    @assert length(sites) >= N "Not enough lattice sites generated for given parameters"
    # sort by radial distance from center (box already centered)
    dist2 = map(p -> (p[1]^2 + p[2]^2), sites)
    order = sortperm(dist2)
    take = order[1:N]
    pos = [sites[i] for i in take]
    return (box=box, positions=pos, indices=take, sites=sites)
end

"""
hex_circle_plus_random_2d(N, σ, ϕ, frac_circle; T=Float32, rng=Random.default_rng())

Place floor(frac_circle*N) particles as the innermost circular hex cluster, and
place the remaining particles uniformly at random on hex-lattice sites outside
that circle. Returns (box, positions), with positions concatenated as
[circle_positions; random_outside].
"""
function hex_circle_plus_random_2d(N::Integer, σ::Real, ϕ::Real, frac_circle::Real;
                                   T=Float32, rng::AbstractRNG=Random.default_rng())
    @assert 0.0 <= frac_circle <= 1.0 "frac_circle must be in [0,1]"
    Nc = round(Int, clamp(frac_circle, 0, 1) * N)
    box = box_from_phi_2d(N, ϕ, σ; T=T)
    sites = hex_sites_in_box_2d(box, T(σ))
    @assert length(sites) >= N "Not enough lattice sites generated for given parameters"
    # circle: Nc innermost sites
    dist2 = map(p -> (p[1]^2 + p[2]^2), sites)
    order = sortperm(dist2)
    circle_idx = order[1:Nc]
    # threshold radius^2 so we can exclude the circular set
    r2_thr = Nc > 0 ? dist2[circle_idx[end]] : -one(T)
    outside_idx = [i for i in eachindex(sites) if dist2[i] > r2_thr]
    @assert length(outside_idx) >= (N - Nc) "Not enough sites outside the circle to place remaining particles"
    rand_idx = if N - Nc > 0
        _choose_indices_without_replacement(length(outside_idx), N - Nc; rng=rng)
    else
        Int[]
    end
    rand_sites = [outside_idx[i] for i in rand_idx]
    pos = Vector{SVector{2,T}}(undef, N)
    for k in 1:Nc
        pos[k] = sites[circle_idx[k]]
    end
    for k in 1:(N - Nc)
        pos[Nc + k] = sites[rand_sites[k]]
    end
    return (box=box, positions=pos, circle_indices=circle_idx, random_indices=rand_sites, sites=sites)
end

"""
hex_circle_in_box_2d(N, box; T=Float32, margin=zero(T))

Compute a hexagonal-lattice circle of N sites centered in a given fixed box.
This adapts the lattice spacing so that at least N lattice sites fall inside a
circle of radius R = min(box...)/2 - margin. Returns (positions, a, R, sites)
where positions are the selected N innermost SVector{2,T}.
"""
function hex_circle_in_box_2d(N::Integer, box::Box2{T}; margin::T=zero(T)) where {T<:AbstractFloat}
    Lx, Ly = box
    R = min(Lx, Ly)/2 - max(margin, zero(T))
    R <= zero(T) && error("Circle radius non-positive; increase box or reduce margin")

    # Solve for triangular lattice spacing a from N ≈ ρ π R^2, ρ=2/(√3 a^2)
    a = sqrt((2π * R^2) / (sqrt(T(3)) * N)) * T(0.98)
    sites = hex_sites_in_box_2d((T(Lx), T(Ly)), a)
    r2 = R^2
    inside = [p for p in sites if (p[1]*p[1] + p[2]*p[2]) <= r2]

    if length(inside) < N
        # densify iteratively until enough
        for _ in 1:30
            a *= T(0.95)
            sites = hex_sites_in_box_2d((T(Lx), T(Ly)), a)
            inside = [p for p in sites if (p[1]*p[1] + p[2]*p[2]) <= r2]
            length(inside) >= N && break
        end
    end
    length(inside) < N && error("Unable to fit $N sites inside circle; try larger box or smaller margin")

    # Pick innermost N
    dist2 = [(p[1]*p[1] + p[2]*p[2]) for p in inside]
    order = sortperm(dist2)
    pos = inside[order[1:N]]
    return (positions=pos, a=a, R=R, sites=sites)
end

end # module InitGenerators
