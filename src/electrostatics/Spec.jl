# =========================
#   Low-level electrostatics specification
# =========================

const INFLUENCE_KINDS = (:spme, :p3m_ad, :p3m_ik, :esp, :esp_deconvolution)

"""
    ElectrostaticsSpec

Validated, host-side description of a long-range Coulomb solver. Built by the
workflow descriptors `PME`, `PPPM` and `ESP`; only
the combinations those constructors produce are supported:

| `influence` | split | window | differentiation |
|---|---|---|---|
| `:spme` | Gaussian | B-spline | analytic |
| `:p3m_ad` | Gaussian | B-spline | analytic, optimal AD influence |
| `:p3m_ik` | Gaussian | B-spline | ik (three field transforms) |
| `:esp` | prolate | prolate | analytic, optimal AD influence (alias sums) |
| `:esp_deconvolution` | prolate | prolate | analytic, plain window deconvolution |

Fields: `coulomb_prefactor`, `relative_permittivity` (`C = prefactor/εᵣ`),
`real_cutoff`, `split`, `window`, `mesh` (`(K₁,K₂,K₃)`) or `mesh_spacing`,
`influence`, `alias_order` (P3M alias sums), `net_charge` (`:error` or
`:background`), `boundary` (`:tin_foil`), and `real_space_modifier`:
`:none` (unshifted), or `:potential_shift`, which subtracts the
short-range pair energy at the cutoff from every non-excluded pair inside it
(energy only, forces unchanged; the GROMACS `Potential-shift` convention).
Prolate splits vanish at the cutoff, so the shift is zero for ESP.
"""
struct ElectrostaticsSpec
    coulomb_prefactor::Float64
    relative_permittivity::Float64
    real_cutoff::Float64
    split::AbstractSplit
    window::AbstractWindow
    mesh::Union{Nothing,NTuple{3,Int}}
    mesh_spacing::Union{Nothing,Float64}
    influence::Symbol
    alias_order::Int
    net_charge::Symbol
    boundary::Symbol
    real_space_modifier::Symbol
    net_charge_tolerance::Float64
end

function ElectrostaticsSpec(; coulomb_prefactor::Real,
                              relative_permittivity::Real=1.0,
                              real_cutoff::Real,
                              split::AbstractSplit,
                              window::AbstractWindow,
                              mesh=nothing,
                              mesh_spacing=nothing,
                              influence::Symbol,
                              alias_order::Integer=2,
                              net_charge::Symbol=:error,
                              boundary::Symbol=:tin_foil,
                              real_space_modifier::Symbol=:none,
                              net_charge_tolerance::Real=1e-8)
    isfinite(coulomb_prefactor) && coulomb_prefactor > 0 ||
        throw(ArgumentError("coulomb_prefactor must be finite and positive (it carries the caller's E·L/Q² unit)."))
    isfinite(relative_permittivity) && relative_permittivity > 0 ||
        throw(ArgumentError("relative_permittivity must be finite and positive."))
    isfinite(real_cutoff) && real_cutoff > 0 || throw(ArgumentError("real_cutoff must be finite and positive."))
    influence in INFLUENCE_KINDS || throw(ArgumentError("Unknown influence kind $(influence)."))
    if influence === :esp || influence === :esp_deconvolution
        split isa ProlateSplit && window isa ProlateWindow ||
            throw(ArgumentError("ESP requires a prolate split and a prolate window."))
        isapprox(split.cutoff, real_cutoff; rtol=1e-12) ||
            throw(ArgumentError("The prolate split is defined for cutoff $(split.cutoff), not $(real_cutoff)."))
        # The window transform is band-limited to |θ| ≤ 2c₁/P (grid units) and the
        # mesh band reaches θ = π; a window without margin is deconvolved by
        # near-zero values.
        window.order <= max_prolate_order(window.shape) ||
            throw(ArgumentError("A prolate window of shape $(round(window.shape; digits=3)) supports at most $(max_prolate_order(window.shape)) grid points (P ≤ 2c/π − ½); got $(window.order). Use a larger window shape (tighter tolerance) or a narrower window."))
    else
        split isa GaussianSplit && window isa BSplineWindow ||
            throw(ArgumentError("PME/PPPM require a Gaussian split and a B-spline window."))
    end
    0 <= alias_order <= 6 || throw(ArgumentError("alias_order must be between 0 and 6."))
    net_charge in (:error, :background) ||
        throw(ArgumentError("net_charge must be :error or :background; got $(net_charge)."))
    boundary === :tin_foil ||
        throw(ArgumentError("Only conducting (tin-foil) boundary conditions are implemented; got boundary=$(boundary)."))
    real_space_modifier in (:none, :potential_shift) ||
        throw(ArgumentError("real_space_modifier must be :none or :potential_shift; got $(real_space_modifier)."))
    meshT = if mesh === nothing
        mesh_spacing === nothing && throw(ArgumentError("Provide either `mesh=(K1,K2,K3)` or `mesh_spacing`."))
        nothing
    else
        length(mesh) == 3 || throw(ArgumentError("mesh must have three dimensions; 2D and slab electrostatics are not implemented."))
        all(k -> k >= window_order(window), mesh) ||
            throw(ArgumentError("Every mesh dimension must be at least the assignment order $(window_order(window))."))
        ntuple(i -> Int(mesh[i]), 3)
    end
    spacing = mesh_spacing === nothing ? nothing : Float64(mesh_spacing)
    spacing === nothing || (isfinite(spacing) && spacing > 0) ||
        throw(ArgumentError("mesh_spacing must be finite and positive."))
    return ElectrostaticsSpec(Float64(coulomb_prefactor), Float64(relative_permittivity), Float64(real_cutoff),
                              split, window, meshT, spacing, influence, Int(alias_order),
                              net_charge, boundary, real_space_modifier, Float64(net_charge_tolerance))
end

coulomb_constant(spec::ElectrostaticsSpec) = spec.coulomb_prefactor / spec.relative_permittivity

"""
    max_prolate_order(shape) -> Int

Widest prolate assignment window (grid points) for window shape `c`. The
window transform is band-limited to `|θ| ≤ 2c/P` and must cover the mesh band
(`θ ≤ π`) with some margin: `P ≤ 2c/π − ½`. Windows at the bare limit were
measured to lose accuracy by orders of magnitude (the deconvolution divides by
values near the band edge).
"""
max_prolate_order(shape::Real) = floor(Int, 2 * shape / pi - 0.5 + 1e-9)

"""
    fft_friendly(n) -> Int

Smallest integer `≥ n` whose prime factors are all in `{2, 3, 5, 7}`.
"""
function fft_friendly(n::Integer)
    m = max(Int(n), 1)
    while true
        r = m
        for p in (2, 3, 5, 7)
            while r % p == 0
                r ÷= p
            end
        end
        r == 1 && return m
        m += 1
    end
end

"""
    resolve_mesh(spec, box) -> NTuple{3,Int}

Explicit `mesh`, or FFT-friendly dimensions from `mesh_spacing`.
"""
function resolve_mesh(spec::ElectrostaticsSpec, box::NTuple{3,<:Real})
    spec.mesh === nothing || return spec.mesh
    h = spec.mesh_spacing::Float64
    P = window_order(spec.window)
    return ntuple(d -> max(fft_friendly(ceil(Int, box[d] / h - 1e-9)), P), 3)
end

"""
    alpha_from_tolerance(cutoff, tol) -> α

Gaussian splitting parameter with `erfc(α r_c) = tol`, a real-space
potential-tail criterion (not an RMS force tolerance).
"""
function alpha_from_tolerance(cutoff::Real, tol::Real)
    0 < tol < 1 || throw(ArgumentError("real_space_tolerance must lie in (0, 1)."))
    f(x) = _host_erfc(x) - tol
    lo, hi = 0.0, 10.0
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi) / cutoff
end

# Host erfc via a converged continued fraction / series (setup only).
function _host_erfc(x::Float64)
    x < 0 && return 2.0 - _host_erfc(-x)
    if x < 2.0
        # erf(x) = 2/√π e^{−x²} Σ 2ⁿ x^{2n+1}/(1·3·…·(2n+1))
        term = x
        acc = x
        n = 0
        while term > 1e-18 * acc
            n += 1
            term *= 2.0 * x * x / (2n + 1)
            acc += term
        end
        return 1.0 - 2.0 / sqrt(pi) * exp(-x * x) * acc
    end
    # erfc(x) = e^{−x²}/√π / (x + (1/2)/(x + 1/(x + (3/2)/(x + …)))), evaluated backwards
    t = x
    for n in 400:-1:1
        t = x + (n / 2) / t
    end
    return exp(-x * x) / sqrt(pi) / t
end
_host_erfc(x::Real) = _host_erfc(Float64(x))

# -------------------------
# Exceptions (scaled / excluded pairs)
# -------------------------

"""
    normalize_exceptions(exceptions, N) -> Vector{Tuple{Int32,Int32,Float64}}

Validate and deduplicate `(i, j, scale)` triples (stable 1-based particle ids,
`i ≠ j`, `0 ≤ scale`). A pair listed twice must carry the same scale. Pairs
with `scale == 1` are dropped (they are ordinary interactions).
"""
function normalize_exceptions(exceptions, N::Integer)
    seen = Dict{Tuple{Int32,Int32},Float64}()
    for item in exceptions
        length(item) in (2, 3) || throw(ArgumentError("Each electrostatic exception must be (i, j) or (i, j, scale)."))
        i = Int(item[1])
        j = Int(item[2])
        s = length(item) == 3 ? Float64(item[3]) : 0.0
        1 <= i <= N && 1 <= j <= N || throw(ArgumentError("Exception pair ($(i), $(j)) is out of range for N=$(N)."))
        i != j || throw(ArgumentError("Exception pair ($(i), $(j)) refers to a single particle."))
        isfinite(s) && s >= 0 || throw(ArgumentError("Exception scale must be finite and nonnegative; got $(s)."))
        key = i < j ? (Int32(i), Int32(j)) : (Int32(j), Int32(i))
        if haskey(seen, key)
            seen[key] == s || throw(ArgumentError("Exception pair $(key) is listed with conflicting scales $(seen[key]) and $(s)."))
        else
            seen[key] = s
        end
    end
    out = Tuple{Int32,Int32,Float64}[]
    for (key, s) in seen
        s == 1.0 && continue
        push!(out, (key[1], key[2], s))
    end
    sort!(out)
    return out
end
