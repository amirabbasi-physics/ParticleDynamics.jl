"""
High-level initial configuration generators for 2D simulations.

The helper functions in this module reproduce the hexagonal arrangements used
throughout `examples/` (random hex placements, concentric circles, etc.) so that
scripts can lift proven particle packings without rewriting geometry code.
"""
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

Compute a square 2D box `(L, L)` whose area fraction matches the target `ϕ`
when populated with `N` disks of diameter `σ`. `examples/TwoT_2D_LD_VV.jl`
calls this with `(N=2000, ϕ=0.5, σ=1.0)` to fix the box before the random
hex-based placement.
"""
function box_from_phi_2d(N::Integer, ϕ::Real, σ::Real; T=Float32)
    @assert ϕ > 0 "Area fraction must be positive"
    @assert σ > 0 "σ must be positive"
    A = N * (π * (σ^2) / 4) / ϕ
    L = T(sqrt(A))
    return (L, L)::Box2{T}
end

function _hex_sites_count_2d(Lx::T, a::T, ny::Int) where {T<:AbstractFloat}
    ax = a
    count = 0
    for j in 0:(ny - 1)
        xoff = isodd(j) ? (ax / T(2)) : zero(T)
        nx = max(0, floor(Int, (Lx - xoff) / ax))
        count += nx
    end
    return count
end

function _hex_sites_in_box_2d(box::Box2{T}, a::T, ny::Int) where {T<:AbstractFloat}
    Lx, Ly = box
    ax = a
    ay = a * sqrt(T(3)) / T(2)  # vertical spacing between rows

    sites = Vector{SVector{2,T}}()
    sites_cap = ceil(Int, (Lx * Ly) * (T(2) / (sqrt(T(3)) * a^2)))  # density of triangular lattice = 2/(√3 a^2)
    sizehint!(sites, sites_cap)

    if ny == 0
        return sites
    end

    y0 = (Ly - (T(ny - 1) * ay)) / T(2)
    for j in 0:(ny - 1)
        y = y0 + T(j) * ay
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

"""
    hex_sites_in_box_2d(box, σ)

Enumerate hexagonal lattice sites inside a periodic square box of side lengths
`box`. `a = σ` fixes the nearest-neighbor spacing, which matches the setups
used in `examples/SingleT_2D_LD_VV.jl` and friends. Rows are centered in the
box, and if an odd row count would place periodic images closer than `a`,
the last row is dropped to avoid overlaps while minimizing empty space.
"""
function hex_sites_in_box_2d(box::Box2{T}, σ::T) where {T<:AbstractFloat}
    Lx, Ly = box
    a = σ
    ay = a * sqrt(T(3)) / T(2)  # vertical spacing between rows

    # number of full rows that fit; if odd rows would overlap across y-wrap, drop one
    ny = max(0, floor(Int, Ly / ay))
    if ny == 0
        return Vector{SVector{2,T}}()
    end

    gap = Ly - (T(ny - 1) * ay)
    if isodd(ny) && gap < a
        ny -= 1
        ny == 0 && return Vector{SVector{2,T}}()
    end

    return _hex_sites_in_box_2d(box, a, ny)
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

function _commensurate_hex_spacing_2d(box::Box2{T}, σ::T, N::Integer) where {T<:AbstractFloat}
    Lx, Ly = box
    sqrt3 = sqrt(T(3))

    # Largest even row count that keeps a >= σ (i.e., no overlaps).
    ny_max = floor(Int, (T(2) * Ly) / (sqrt3 * σ))
    ny_even = ny_max - (ny_max % 2)
    ny_even < 2 && error("Box too small to fit a hex lattice with spacing >= σ")

    a = (T(2) * Ly) / (sqrt3 * T(ny_even))
    count = _hex_sites_count_2d(Lx, a, ny_even)
    if count < N
        error("Cannot fit $N particles with spacing >= σ in this box; reduce N/ϕ or set fit_box=false")
    end

    return (a=a, ny=ny_even)
end

"""
    hex_random_2d(N, σ, ϕ; T=Float32, rng=Random.default_rng(), fit_box=true)

Sample `N` distinct hexagonal lattice sites at random inside the box computed
by [`box_from_phi_2d`](@ref). This is the entry point used by
`examples/TwoT_2D_LD_VV.jl` (with `σ = 1.0`, `ϕ = 0.5`) before assigning hot
and cold type IDs. When `fit_box=true`, the lattice spacing is adjusted so an
even row count fits exactly in `Ly`, avoiding overlaps across the periodic
y-wrap without leaving an empty strip. The spacing is only increased (never
reduced below `σ`); an error is raised if there is insufficient room.

# Returns
- `box`: square box tuple `(L, L)` in the requested precision.
- `positions`: vector of `SVector{2,T}` coordinates centered in `[-L/2, L/2)`.
- `indices`: indices into the full lattice site list (handy for re-sampling).
"""
function hex_random_2d(N::Integer, σ::Real, ϕ::Real; T=Float32, rng::AbstractRNG=Random.default_rng(), fit_box::Bool=true)
    box = box_from_phi_2d(N, ϕ, σ; T=T)
    if fit_box
        spacing = _commensurate_hex_spacing_2d(box, T(σ), N)
        sites = _hex_sites_in_box_2d(box, spacing.a, spacing.ny)
    else
        sites = hex_sites_in_box_2d(box, T(σ))
    end
    @assert length(sites) >= N "Not enough lattice sites generated for given parameters"
    idx = _choose_indices_without_replacement(length(sites), N; rng=rng)
    pos = [sites[i] for i in idx]
    return (box=box, positions=pos, indices=idx, sites=sites)
end

"""
    hex_circle_2d(N, σ, ϕ; T=Float32)

Return the innermost `N` hex-lattice sites (ordered by radius) so that the
configuration forms a densely packed circular cluster. The single-temperature
circles in `examples/SingleT_2D_LD_VV_Circle.jl` are assembled with this helper.
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

Hybrid generator used by `examples/TwoT_2D_LD_VV_frac.jl`: `frac_circle` of the
particles occupy the innermost circle while the rest are randomly drawn from
sites outside that radius. The return value bundles the circle and random
indices so scripts can assign different types or temperatures to each set.
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
    hex_circle_in_box_2d(N, box; margin=0)

Variant used when the box is fixed a priori (e.g. restart from a recorded GSD).
The helper shrinks the lattice spacing until at least `N` sites fall inside the
target circle of radius `min(box)/2 - margin`.
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
