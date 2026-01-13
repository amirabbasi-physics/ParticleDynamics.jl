"""
High-level initial configuration generators for 2D/3D simulations.

The helper functions in this module reproduce the hexagonal arrangements used
throughout `examples/` (random hex placements, concentric circles, etc.) and
provide 3D close-packed counterparts so scripts can lift proven particle
packings without rewriting geometry code.
"""
module InitGenerators

using Random
using StaticArrays
using ..Definitions: Box2, Box3

export box_from_phi_2d,
       box_from_phi_3d,
       hex_random_2d,
       hex_circle_2d,
       hex_circle_plus_random_2d,
       hex_sites_in_box_2d,
       hex_circle_in_box_2d,
       hex_slab_coexistence_2d,
       fcc_sites_in_box_3d,
       fcc_random_3d,
       fcc_slab_coexistence_3d

"""
    box_from_phi_2d(N, ϕ, σ; T=Float32, aspect=1)

Compute a rectangular 2D box `(Lx, Ly)` whose area fraction matches the target
`ϕ` when populated with `N` disks of diameter `σ`. `aspect = Lx/Ly` and defaults
to 1 (square). `examples/TwoT_2D_LD_VV.jl` calls this with `(N=2000, ϕ=0.5,
σ=1.0)` to fix the box before the random hex-based placement.
"""
function box_from_phi_2d(N::Integer, ϕ::Real, σ::Real; T=Float32, aspect::Real=1)
    @assert ϕ > 0 "Area fraction must be positive"
    @assert σ > 0 "σ must be positive"
    @assert aspect > 0 "aspect must be positive"
    A = N * (π * (σ^2) / 4) / ϕ
    aspectT = T(aspect)
    Ly = T(sqrt(A / aspectT))
    Lx = aspectT * Ly
    return (Lx, Ly)::Box2{T}
end

"""
    box_from_phi_3d(N, ϕ, σ; T=Float32, aspect=1)

Compute a 3D box `(Lx, Ly, Lz)` whose volume fraction matches the target `ϕ`
when populated with `N` spheres of diameter `σ`. `aspect = Lx/Ly`, while
`Ly = Lz` to keep the transverse directions equal.
"""
function box_from_phi_3d(N::Integer, ϕ::Real, σ::Real; T=Float32, aspect::Real=1)
    @assert ϕ > 0 "Volume fraction must be positive"
    @assert σ > 0 "σ must be positive"
    @assert aspect > 0 "aspect must be positive"
    V = N * (π * (σ^3) / 6) / ϕ
    aspectT = T(aspect)
    Ly = T(cbrt(V / aspectT))
    Lx = aspectT * Ly
    return (Lx, Ly, Ly)::Box3{T}
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

Enumerate hexagonal lattice sites inside a periodic rectangular box of side lengths
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

function _coexistence_counts(N::Integer, frac_cold::Real)
    @assert 0.0 <= frac_cold <= 1.0 "frac_cold must be in [0,1]"
    N_cold = round(Int, frac_cold * N)
    N_hot = N - N_cold
    N_cold_slab = fld(N_cold, 2)
    N_cold_left = N_cold - N_cold_slab
    N_hot_right = fld(N_hot, 2)
    N_hot_left = N_hot - N_hot_right
    return (N_cold=N_cold, N_hot=N_hot,
            N_cold_left=N_cold_left, N_cold_slab=N_cold_slab,
            N_hot_left=N_hot_left, N_hot_right=N_hot_right)
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
    hex_random_2d(N, σ, ϕ; T=Float32, rng=Random.default_rng(), fit_box=true, aspect=1)

Sample `N` distinct hexagonal lattice sites at random inside the box computed
by [`box_from_phi_2d`](@ref). This is the entry point used by
`examples/TwoT_2D_LD_VV.jl` (with `σ = 1.0`, `ϕ = 0.5`) before assigning hot
and cold type IDs. When `fit_box=true`, the lattice spacing is adjusted so an
even row count fits exactly in `Ly`, avoiding overlaps across the periodic
y-wrap without leaving an empty strip. The spacing is only increased (never
reduced below `σ`); an error is raised if there is insufficient room. Use
`aspect` to set `Lx/Ly` (defaults to 1).

# Returns
- `box`: rectangular box tuple `(Lx, Ly)` in the requested precision.
- `positions`: vector of `SVector{2,T}` coordinates centered in `[-Lx/2, Lx/2)` and `[-Ly/2, Ly/2)`.
- `indices`: indices into the full lattice site list (handy for re-sampling).
"""
function hex_random_2d(N::Integer, σ::Real, ϕ::Real; T=Float32, rng::AbstractRNG=Random.default_rng(), fit_box::Bool=true, aspect::Real=1)
    box = box_from_phi_2d(N, ϕ, σ; T=T, aspect=aspect)
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
    hex_circle_2d(N, σ, ϕ; T=Float32, aspect=1)

Return the innermost `N` hex-lattice sites (ordered by radius) so that the
configuration forms a densely packed circular cluster. The single-temperature
circles in `examples/SingleT_2D_LD_VV_Circle.jl` are assembled with this helper.
Use `aspect` to set `Lx/Ly` (defaults to 1).
"""
function hex_circle_2d(N::Integer, σ::Real, ϕ::Real; T=Float32, aspect::Real=1)
    box = box_from_phi_2d(N, ϕ, σ; T=T, aspect=aspect)
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
    hex_circle_plus_random_2d(N, σ, ϕ, frac_circle; T=Float32, rng=Random.default_rng(), aspect=1)

Hybrid generator used by `examples/TwoT_2D_LD_VV_frac.jl`: `frac_circle` of the
particles occupy the innermost circle while the rest are randomly drawn from
sites outside that radius. The return value bundles the circle and random
indices so scripts can assign different types or temperatures to each set. Use
`aspect` to set `Lx/Ly` (defaults to 1).
"""
function hex_circle_plus_random_2d(N::Integer, σ::Real, ϕ::Real, frac_circle::Real;
                                   T=Float32, rng::AbstractRNG=Random.default_rng(), aspect::Real=1)
    @assert 0.0 <= frac_circle <= 1.0 "frac_circle must be in [0,1]"
    Nc = round(Int, clamp(frac_circle, 0, 1) * N)
    box = box_from_phi_2d(N, ϕ, σ; T=T, aspect=aspect)
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

"""
    hex_slab_coexistence_2d(N, σ, ϕ, slab_height; frac_cold=0.5, T=Float32, rng=Random.default_rng(), fit_box=true)

Build a 2D direct-coexistence layout: the left half of the box is a homogeneous
mixture (cold/hot split by `frac_cold`), while the right half is split into a
cold slab (touching the midplane `x=0`) and a hot bath to its right. The slab
spans the full `y` direction (`slab_height`). The box width `Lx` is computed
from `ϕ` and `slab_height`. The slab width is set by taking the leftmost
`N_cold/2` lattice sites in `x >= 0`, so the slab is fully packed.
"""
function hex_slab_coexistence_2d(N::Integer, σ::Real, ϕ::Real, slab_height::Real;
                                 frac_cold::Real=0.5, T=Float32,
                                 rng::AbstractRNG=Random.default_rng(),
                                 fit_box::Bool=true)
    @assert ϕ > 0 "Area fraction must be positive"
    @assert σ > 0 "σ must be positive"
    Ly = T(slab_height)
    Ly > zero(T) || error("slab_height must be positive")

    sigmaT = T(σ)
    phiT = T(ϕ)
    area = T(N) * (π * sigmaT^2 / T(4)) / phiT
    Lx = area / Ly
    box = (Lx, Ly)::Box2{T}

    sites = if fit_box
        spacing = _commensurate_hex_spacing_2d(box, sigmaT, N)
        _hex_sites_in_box_2d(box, spacing.a, spacing.ny)
    else
        hex_sites_in_box_2d(box, sigmaT)
    end

    counts = _coexistence_counts(N, frac_cold)
    N_left = counts.N_cold_left + counts.N_hot_left
    N_right = counts.N_cold_slab + counts.N_hot_right
    @assert N_left + N_right == N

    left_pool = Int[]
    right_candidates = Int[]
    for i in eachindex(sites)
        x = sites[i][1]
        if x < zero(T)
            push!(left_pool, i)
        else
            push!(right_candidates, i)
        end
    end

    length(left_pool) >= N_left || error("Not enough lattice sites in left half for N_left=$(N_left)")
    length(right_candidates) >= N_right || error("Not enough lattice sites in right half for N_right=$(N_right)")

    left_sel = N_left > 0 ?
        _choose_indices_without_replacement(length(left_pool), N_left; rng=rng) : Int[]
    left_sites = [left_pool[i] for i in left_sel]

    slab_sites = Int[]
    slab_w = zero(T)
    if counts.N_cold_slab > 0
        slab_sorted = sort(right_candidates, by = i -> (sites[i][1], sites[i][2]))
        boundary_x = sites[slab_sorted[counts.N_cold_slab]][1]
        boundary_sites = Int[]
        for i in right_candidates
            x = sites[i][1]
            if x < boundary_x
                push!(slab_sites, i)
            elseif x == boundary_x
                push!(boundary_sites, i)
            end
        end
        need = counts.N_cold_slab - length(slab_sites)
        @assert need <= length(boundary_sites) "Slab boundary selection failed; check lattice generation"
        if need > 0
            boundary_sorted = sort(boundary_sites, by = i -> sites[i][2])
            append!(slab_sites, boundary_sorted[1:need])
        end
        slab_w = boundary_x
    end

    slab_mask = falses(length(sites))
    for i in slab_sites
        slab_mask[i] = true
    end

    right_pool = Int[]
    for i in right_candidates
        slab_mask[i] && continue
        push!(right_pool, i)
    end
    length(right_pool) >= counts.N_hot_right || error("Not enough lattice sites in right bath for N_hot_right=$(counts.N_hot_right)")

    right_sel = counts.N_hot_right > 0 ?
        _choose_indices_without_replacement(length(right_pool), counts.N_hot_right; rng=rng) : Int[]
    right_sites = [right_pool[i] for i in right_sel]

    positions = Vector{SVector{2,T}}(undef, N)
    for (k, idx) in enumerate(left_sites)
        positions[k] = sites[idx]
    end
    for (k, idx) in enumerate(slab_sites)
        positions[N_left + k] = sites[idx]
    end
    for (k, idx) in enumerate(right_sites)
        positions[N_left + counts.N_cold_slab + k] = sites[idx]
    end

    left_indices = collect(1:N_left)
    slab_indices = collect((N_left + 1):(N_left + counts.N_cold_slab))
    right_indices = collect((N_left + counts.N_cold_slab + 1):N)

    left_cold_local = counts.N_cold_left > 0 ?
        _choose_indices_without_replacement(N_left, counts.N_cold_left; rng=rng) : Int[]
    left_cold_mask = falses(N_left)
    for i in left_cold_local
        left_cold_mask[i] = true
    end
    left_cold = [left_indices[i] for i in left_cold_local]
    left_hot = [left_indices[i] for i in 1:N_left if !left_cold_mask[i]]

    cold_indices = vcat(left_cold, slab_indices)
    hot_indices = vcat(left_hot, right_indices)

    return (box=box, positions=positions, slab_width=slab_w, slab_height=Ly,
            cold_indices=cold_indices, hot_indices=hot_indices,
            left_indices=left_indices, slab_indices=slab_indices, right_indices=right_indices,
            sites=sites, counts=counts)
end

function _fcc_sites_in_box_3d(box::Box3{T}, a::T) where {T<:AbstractFloat}
    Lx, Ly, Lz = box
    nx = max(0, floor(Int, Lx / a))
    ny = max(0, floor(Int, Ly / a))
    nz = max(0, floor(Int, Lz / a))

    sites = Vector{SVector{3,T}}()
    if nx == 0 || ny == 0 || nz == 0
        return sites
    end

    sizehint!(sites, 4 * nx * ny * nz)
    ox = (Lx - T(nx) * a) / T(2)
    oy = (Ly - T(ny) * a) / T(2)
    oz = (Lz - T(nz) * a) / T(2)
    center = SVector{3,T}(Lx/T(2), Ly/T(2), Lz/T(2))
    half = a / T(2)

    b1 = SVector{3,T}(zero(T), zero(T), zero(T))
    b2 = SVector{3,T}(zero(T), half, half)
    b3 = SVector{3,T}(half, zero(T), half)
    b4 = SVector{3,T}(half, half, zero(T))

    for i in 0:(nx - 1), j in 0:(ny - 1), k in 0:(nz - 1)
        origin = SVector{3,T}(ox + T(i) * a, oy + T(j) * a, oz + T(k) * a)
        push!(sites, origin + b1 - center)
        push!(sites, origin + b2 - center)
        push!(sites, origin + b3 - center)
        push!(sites, origin + b4 - center)
    end
    return sites
end

"""
    fcc_sites_in_box_3d(box, σ)

Enumerate FCC lattice sites inside a periodic box of side lengths `box`.
The nearest-neighbor spacing is `σ` (lattice constant `a = σ√2`).
"""
function fcc_sites_in_box_3d(box::Box3{T}, σ::T) where {T<:AbstractFloat}
    a = σ * sqrt(T(2))
    return _fcc_sites_in_box_3d(box, a)
end

"""
    fcc_random_3d(N, σ, ϕ; T=Float32, rng=Random.default_rng(), aspect=1)

Sample `N` distinct FCC lattice sites at random inside the box computed by
[`box_from_phi_3d`](@ref). The nearest-neighbor spacing is `σ`. Use `aspect` to
set `Lx/Ly` while keeping `Ly = Lz` (defaults to 1).
"""
function fcc_random_3d(N::Integer, σ::Real, ϕ::Real; T=Float32, rng::AbstractRNG=Random.default_rng(), aspect::Real=1)
    box = box_from_phi_3d(N, ϕ, σ; T=T, aspect=aspect)
    sites = fcc_sites_in_box_3d(box, T(σ))
    @assert length(sites) >= N "Not enough FCC sites generated for given parameters"
    idx = _choose_indices_without_replacement(length(sites), N; rng=rng)
    pos = [sites[i] for i in idx]
    return (box=box, positions=pos, indices=idx, sites=sites)
end

"""
    fcc_slab_coexistence_3d(N, σ, ϕ, slab_height; frac_cold=0.5, T=Float32, rng=Random.default_rng())

3D counterpart of [`hex_slab_coexistence_2d`](@ref) using an FCC lattice. The
cold slab spans the full `y,z` extents, with `slab_height` setting both `Ly`
and `Lz`. The slab width is set by taking the leftmost `N_cold/2` lattice sites
in `x >= 0`, so the slab is fully packed.
"""
function fcc_slab_coexistence_3d(N::Integer, σ::Real, ϕ::Real, slab_height::Real;
                                 frac_cold::Real=0.5, T=Float32,
                                 rng::AbstractRNG=Random.default_rng())
    @assert ϕ > 0 "Volume fraction must be positive"
    @assert σ > 0 "σ must be positive"
    Ly = T(slab_height)
    Ly > zero(T) || error("slab_height must be positive")
    Lz = Ly

    sigmaT = T(σ)
    phiT = T(ϕ)
    volume = T(N) * (π * sigmaT^3 / T(6)) / phiT
    Lx = volume / (Ly * Lz)
    box = (Lx, Ly, Lz)::Box3{T}

    sites = fcc_sites_in_box_3d(box, sigmaT)

    counts = _coexistence_counts(N, frac_cold)
    N_left = counts.N_cold_left + counts.N_hot_left
    N_right = counts.N_cold_slab + counts.N_hot_right
    @assert N_left + N_right == N

    left_pool = Int[]
    right_candidates = Int[]
    for i in eachindex(sites)
        x = sites[i][1]
        if x < zero(T)
            push!(left_pool, i)
        else
            push!(right_candidates, i)
        end
    end

    length(left_pool) >= N_left || error("Not enough lattice sites in left half for N_left=$(N_left)")
    length(right_candidates) >= N_right || error("Not enough lattice sites in right half for N_right=$(N_right)")

    left_sel = N_left > 0 ?
        _choose_indices_without_replacement(length(left_pool), N_left; rng=rng) : Int[]
    left_sites = [left_pool[i] for i in left_sel]

    slab_sites = Int[]
    slab_w = zero(T)
    if counts.N_cold_slab > 0
        slab_sorted = sort(right_candidates, by = i -> (sites[i][1], sites[i][2], sites[i][3]))
        boundary_x = sites[slab_sorted[counts.N_cold_slab]][1]
        boundary_sites = Int[]
        for i in right_candidates
            x = sites[i][1]
            if x < boundary_x
                push!(slab_sites, i)
            elseif x == boundary_x
                push!(boundary_sites, i)
            end
        end
        need = counts.N_cold_slab - length(slab_sites)
        @assert need <= length(boundary_sites) "Slab boundary selection failed; check lattice generation"
        if need > 0
            boundary_sorted = sort(boundary_sites, by = i -> (sites[i][2], sites[i][3]))
            append!(slab_sites, boundary_sorted[1:need])
        end
        slab_w = boundary_x
    end

    slab_mask = falses(length(sites))
    for i in slab_sites
        slab_mask[i] = true
    end

    right_pool = Int[]
    for i in right_candidates
        slab_mask[i] && continue
        push!(right_pool, i)
    end
    length(right_pool) >= counts.N_hot_right || error("Not enough lattice sites in right bath for N_hot_right=$(counts.N_hot_right)")

    right_sel = counts.N_hot_right > 0 ?
        _choose_indices_without_replacement(length(right_pool), counts.N_hot_right; rng=rng) : Int[]
    right_sites = [right_pool[i] for i in right_sel]

    positions = Vector{SVector{3,T}}(undef, N)
    for (k, idx) in enumerate(left_sites)
        positions[k] = sites[idx]
    end
    for (k, idx) in enumerate(slab_sites)
        positions[N_left + k] = sites[idx]
    end
    for (k, idx) in enumerate(right_sites)
        positions[N_left + counts.N_cold_slab + k] = sites[idx]
    end

    left_indices = collect(1:N_left)
    slab_indices = collect((N_left + 1):(N_left + counts.N_cold_slab))
    right_indices = collect((N_left + counts.N_cold_slab + 1):N)

    left_cold_local = counts.N_cold_left > 0 ?
        _choose_indices_without_replacement(N_left, counts.N_cold_left; rng=rng) : Int[]
    left_cold_mask = falses(N_left)
    for i in left_cold_local
        left_cold_mask[i] = true
    end
    left_cold = [left_indices[i] for i in left_cold_local]
    left_hot = [left_indices[i] for i in 1:N_left if !left_cold_mask[i]]

    cold_indices = vcat(left_cold, slab_indices)
    hot_indices = vcat(left_hot, right_indices)

    return (box=box, positions=positions, slab_width=slab_w, slab_height=Ly,
            cold_indices=cold_indices, hot_indices=hot_indices,
            left_indices=left_indices, slab_indices=slab_indices, right_indices=right_indices,
            sites=sites, counts=counts)
end

end # module InitGenerators
