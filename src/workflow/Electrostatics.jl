using ..Electrostatics

"""
    ElectrostaticsForce

Abstract workflow descriptor of a long-range Coulomb solver ([`PME`](@ref),
[`PPPM`](@ref), [`ESP`](@ref)). At most one may be active, alongside at most
one short-range vdW force and one bonded force.
"""
abstract type ElectrostaticsForce <: Force end

"""
    PME(; coulomb_prefactor, real_cutoff, alpha=nothing, real_space_tolerance=nothing,
          mesh=nothing, mesh_spacing=nothing, order=4, relative_permittivity=1.0,
          net_charge=:error, boundary=:tin_foil, real_space_modifier=:none,
          exclusions=:topology, exclude_bonded=false, scaled_pairs=(),
          pairs=:neighborlist, neighborlist=nothing)

Smooth particle-mesh Ewald (Essmann et al. 1995) with analytic
differentiation, computed entirely on the GPU.

Units are the caller's: `U = C qᵢqⱼ/r` with
`C = coulomb_prefactor / relative_permittivity`, so `coulomb_prefactor` carries
`E·L/Q²` (≈138.935458 for nm/kJ mol⁻¹/e, ≈14.399645 for Å/eV/e, or 1 in
reduced units). `alpha` is in inverse length units; alternatively
`real_space_tolerance` sets it from `erfc(α r_c) = tol` (a real-space potential
tail criterion, not a force error). `mesh=(K₁,K₂,K₃)` or `mesh_spacing` (FFT
friendly sizes are chosen) and the B-spline `order` (stencil width) set the
reciprocal accuracy.

Exceptions: `exclusions=:topology` excludes `Topology.exclusions` pairs (scale
0); `exclude_bonded=true` also excludes bonded pairs; `scaled_pairs` lists
`(i, j, scale)` triples (e.g. 1–4 scaling) and overrides the other two. Each
exception removes or scales the pair's minimum-image interaction while its
periodic images remain.

`net_charge=:error` rejects charged systems; `:background` adds a uniform
neutralising background (energy and virial, no force).

`real_space_modifier=:potential_shift` subtracts the short-range pair energy at
the cutoff from every non-excluded pair inside it (energy only; forces are
unchanged), matching the GROMACS `coulomb-modifier = Potential-shift`
Hamiltonian; the default `:none` is the unshifted Ewald sum.
"""
@kwdef struct PME <: ElectrostaticsForce
    coulomb_prefactor = nothing
    real_cutoff = nothing
    alpha = nothing
    real_space_tolerance = nothing
    mesh = nothing
    mesh_spacing = nothing
    order::Int = 4
    relative_permittivity = 1.0
    net_charge::Symbol = :error
    boundary::Symbol = :tin_foil
    real_space_modifier::Symbol = :none
    exclusions = :topology
    exclude_bonded::Bool = false
    scaled_pairs = ()
    pairs::Symbol = :neighborlist
    neighborlist = nothing
end

"""
    PPPM(; coulomb_prefactor, real_cutoff, alpha=nothing, real_space_tolerance=nothing,
           mesh=nothing, mesh_spacing=nothing, order=5, differentiation=:ad,
           alias_order=2, kwargs...)

Particle–particle particle–mesh (P3M) with the Hockney–Eastwood optimal
influence function for B-spline assignment. `differentiation=:ad` uses
analytic gradients of the assignment weights (one forward and one inverse
transform, influence function optimised for AD); `:ik` forms the electric
field in reciprocal space (one forward and three inverse transforms). Alias
sums run over `|n| ≤ alias_order` images per dimension. All other keywords
match [`PME`](@ref).

The ik forces are not the exact gradient of the reported mesh energy; the
energy and virial are those of the discrete influence function.
"""
@kwdef struct PPPM <: ElectrostaticsForce
    coulomb_prefactor = nothing
    real_cutoff = nothing
    alpha = nothing
    real_space_tolerance = nothing
    mesh = nothing
    mesh_spacing = nothing
    order::Int = 5
    differentiation::Symbol = :ad
    alias_order::Int = 2
    relative_permittivity = 1.0
    net_charge::Symbol = :error
    boundary::Symbol = :tin_foil
    real_space_modifier::Symbol = :none
    exclusions = :topology
    exclude_bonded::Bool = false
    scaled_pairs = ()
    pairs::Symbol = :neighborlist
    neighborlist = nothing
end

"""
    ESP(; coulomb_prefactor, real_cutoff, tolerance=1e-4, split_shape=nothing,
          window_shape=nothing, order=nothing, mesh=nothing, mesh_spacing=nothing,
          kwargs...)

Ewald summation with prolates (Liang, Lu, Barnett, Greengard & Jiang 2026):
the Gaussian split is replaced by one built from the prolate spheroidal wave
function ψ₀ᶜ, and B-spline assignment by a prolate window of width `order`
grid points. The real-space part vanishes exactly at `real_cutoff`.

Defaults follow the paper's selection rules for a target relative force error
`tolerance` ε: `split_shape` c with `ψ₀ᶜ(1) = ε` (L²-normalised),
`window_shape` from `ε/2`, `order` from Table 2 of the paper (5, 5, 6, 7, 8 for
ε = 1e-3, 5e-4, 1e-4, 5e-5, 1e-5), and `mesh_spacing = π r_c / c`. All
dimensionless shapes are distinct from the Gaussian `alpha` (inverse length).
Treat the defaults as starting points and measure the error for production
settings.

`influence=:optimal` (default) uses the analytic-differentiation optimal
influence function with alias sums over `|n| ≤ alias_order` images of the
prolate window transform, which compensates the aliasing of the truncated
window. `influence=:deconvolution` uses the plain window deconvolution
`ĝ/φ̂²` of the paper's equation (9). The remaining keywords match [`PME`](@ref) (`alpha`/
`real_space_tolerance` do not apply).
"""
@kwdef struct ESP <: ElectrostaticsForce
    coulomb_prefactor = nothing
    real_cutoff = nothing
    tolerance = 1e-4
    split_shape = nothing
    window_shape = nothing
    order = nothing
    influence::Symbol = :optimal
    alias_order::Int = 2
    mesh = nothing
    mesh_spacing = nothing
    relative_permittivity = 1.0
    net_charge::Symbol = :error
    boundary::Symbol = :tin_foil
    real_space_modifier::Symbol = :none
    exclusions = :topology
    exclude_bonded::Bool = false
    scaled_pairs = ()
    pairs::Symbol = :neighborlist
    neighborlist = nothing
end

function _require_common(f::ElectrostaticsForce)
    f.coulomb_prefactor === nothing &&
        throw(ArgumentError("$(nameof(typeof(f))) requires an explicit `coulomb_prefactor` (no unit system is assumed)."))
    f.real_cutoff === nothing && throw(ArgumentError("$(nameof(typeof(f))) requires `real_cutoff`."))
    return Float64(f.coulomb_prefactor), Float64(f.real_cutoff)
end

function _gaussian_alpha(f::Union{PME,PPPM}, rc::Float64)
    if f.alpha !== nothing
        f.real_space_tolerance === nothing ||
            throw(ArgumentError("Give either `alpha` or `real_space_tolerance`, not both."))
        return Float64(f.alpha)
    end
    f.real_space_tolerance === nothing &&
        throw(ArgumentError("$(nameof(typeof(f))) requires `alpha` (inverse length) or `real_space_tolerance`."))
    return Electrostatics.alpha_from_tolerance(rc, Float64(f.real_space_tolerance))
end

"""
    esp_order_for_tolerance(ε) -> Int

Prolate window width: Table 2 of Liang et al. (2026) for ε ≥ 1e-5 (5, 5, 6,
7, 8 for ε = 1e-3, 5e-4, 1e-4, 5e-5, 1e-5); for tighter tolerances the width
follows the same margin below the band limit `2c₁/π` of the window shape
`c₁ = c(ε/2)` that the table keeps (about one grid point).
"""
function esp_order_for_tolerance(eps::Real)
    d = -log10(Float64(eps))
    d <= 3.35 && return 5
    d <= 4.05 && return 6
    d <= 4.35 && return 7
    d <= 5.05 && return 8
    c1 = Electrostatics.prolate_shape_for_tolerance(Float64(eps) / 2)
    return clamp(floor(Int, 2c1 / pi - 1), 8, 16)
end

"""
    electrostatics_spec(force) -> Electrostatics.ElectrostaticsSpec

Validated low-level specification of a workflow electrostatics descriptor.
"""
function electrostatics_spec(f::PME)
    pref, rc = _require_common(f)
    return Electrostatics.ElectrostaticsSpec(; coulomb_prefactor=pref,
        relative_permittivity=f.relative_permittivity, real_cutoff=rc,
        split=Electrostatics.GaussianSplit(_gaussian_alpha(f, rc)),
        window=Electrostatics.BSplineWindow(f.order),
        mesh=f.mesh, mesh_spacing=f.mesh_spacing, influence=:spme,
        net_charge=f.net_charge, boundary=f.boundary, real_space_modifier=f.real_space_modifier)
end

function electrostatics_spec(f::PPPM)
    pref, rc = _require_common(f)
    f.differentiation in (:ad, :ik) ||
        throw(ArgumentError("PPPM differentiation must be :ad or :ik; got $(f.differentiation)."))
    return Electrostatics.ElectrostaticsSpec(; coulomb_prefactor=pref,
        relative_permittivity=f.relative_permittivity, real_cutoff=rc,
        split=Electrostatics.GaussianSplit(_gaussian_alpha(f, rc)),
        window=Electrostatics.BSplineWindow(f.order),
        mesh=f.mesh, mesh_spacing=f.mesh_spacing,
        influence=f.differentiation === :ad ? :p3m_ad : :p3m_ik, alias_order=f.alias_order,
        net_charge=f.net_charge, boundary=f.boundary, real_space_modifier=f.real_space_modifier)
end

function electrostatics_spec(f::ESP)
    pref, rc = _require_common(f)
    if f.split_shape === nothing || f.window_shape === nothing || f.order === nothing
        f.tolerance === nothing &&
            throw(ArgumentError("ESP needs `tolerance`, or explicit `split_shape`, `window_shape` and `order`."))
    end
    eps = f.tolerance === nothing ? nothing : Float64(f.tolerance)
    c = f.split_shape === nothing ? Electrostatics.prolate_shape_for_tolerance(eps) : Float64(f.split_shape)
    c1 = f.window_shape === nothing ? Electrostatics.prolate_shape_for_tolerance(eps / 2) : Float64(f.window_shape)
    P = f.order === nothing ? esp_order_for_tolerance(eps) : Int(f.order)
    spacing = (f.mesh === nothing && f.mesh_spacing === nothing) ? pi * rc / c : f.mesh_spacing
    f.influence in (:optimal, :deconvolution) ||
        throw(ArgumentError("ESP influence must be :optimal or :deconvolution; got $(f.influence)."))
    return Electrostatics.ElectrostaticsSpec(; coulomb_prefactor=pref,
        relative_permittivity=f.relative_permittivity, real_cutoff=rc,
        split=Electrostatics.ProlateSplit(rc, c),
        window=Electrostatics.ProlateWindow(P, c1),
        mesh=f.mesh, mesh_spacing=spacing,
        influence=f.influence === :optimal ? :esp : :esp_deconvolution, alias_order=f.alias_order,
        net_charge=f.net_charge, boundary=f.boundary, real_space_modifier=f.real_space_modifier)
end

"""
    electrostatic_exceptions(system, force) -> Vector{Tuple{Int32,Int32,Float64}}

Exception list implied by a descriptor's `exclusions`, `exclude_bonded` and
`scaled_pairs` for `system` (build-time particle ids).
"""
function electrostatic_exceptions(system::ParticleSystem, f::ElectrostaticsForce)
    scales = Dict{Tuple{Int32,Int32},Float64}()
    key(i, j) = i < j ? (Int32(i), Int32(j)) : (Int32(j), Int32(i))
    excl = f.exclusions
    if excl === :topology
        for (i, j) in system.topology.exclusions
            scales[key(i, j)] = 0.0
        end
    elseif excl === :none || excl === nothing
        nothing
    else
        for item in excl
            scales[key(item[1], item[2])] = 0.0
        end
    end
    if f.exclude_bonded
        for (i, j) in system.topology.bonds
            scales[key(i, j)] = 0.0
        end
    end
    for item in f.scaled_pairs
        length(item) == 3 || throw(ArgumentError("scaled_pairs entries must be (i, j, scale)."))
        scales[key(item[1], item[2])] = Float64(item[3])
    end
    return Tuple{Int32,Int32,Float64}[(k[1], k[2], s) for (k, s) in sort!(collect(scales); by=first)]
end

"""
    attach_electrostatics!(st, force::ElectrostaticsForce; charges=nothing, exceptions=())

Low-level convenience: attach the solver described by a workflow descriptor
to an existing state.
"""
SimulationCore.attach_electrostatics!(st::SimulationState, f::ElectrostaticsForce; kwargs...) =
    SimulationCore.attach_electrostatics!(st, electrostatics_spec(f); kwargs...)

"""
    electrostatics_from_settings(record, exceptions=()) -> ElectrostaticsForce

Rebuild the exact workflow descriptor from a settings record (see
`Electrostatics.settings_record`). The stored mesh is used verbatim.
"""
function electrostatics_from_settings(rec::AbstractVector{<:Real}, exceptions=())
    length(rec) >= 14 && rec[1] == 1.0 || throw(ArgumentError("Unsupported electrostatics settings record."))
    method = Int(rec[2])
    mesh = (Int(rec[9]), Int(rec[10]), Int(rec[11]))
    net = rec[13] == 1.0 ? :background : :error
    modifier = (length(rec) >= 15 && rec[15] == 1.0) ? :potential_shift : :none
    pairs = Tuple{Int32,Int32,Float64}[(Int32(e[1]), Int32(e[2]), Float64(e[3])) for e in exceptions]
    common = (coulomb_prefactor=Float64(rec[3]), relative_permittivity=Float64(rec[4]),
              real_cutoff=Float64(rec[5]), mesh=mesh, net_charge=net, real_space_modifier=modifier,
              exclusions=:none, scaled_pairs=pairs)
    if method == 1
        return PME(; common..., alpha=Float64(rec[6]), order=Int(rec[7]))
    elseif method == 2 || method == 3
        return PPPM(; common..., alpha=Float64(rec[6]), order=Int(rec[7]),
                    differentiation=method == 2 ? :ad : :ik, alias_order=Int(rec[12]))
    elseif method == 4 || method == 5
        return ESP(; common..., tolerance=nothing, split_shape=Float64(rec[6]), order=Int(rec[7]),
                   window_shape=Float64(rec[8]), alias_order=Int(rec[12]),
                   influence=method == 4 ? :optimal : :deconvolution)
    end
    throw(ArgumentError("Unknown electrostatics method code $(method)."))
end

"""
    electrostatics_from_gsd(path; frame=-1) -> ElectrostaticsForce or nothing

Read the solver settings and exceptions stored in a GSD frame written while
electrostatics was attached, and return the matching descriptor (`nothing`
if the frame carries no electrostatics record).
"""
function electrostatics_from_gsd(path::AbstractString; frame::Integer=-1)
    rec = read_gsd_log(path, "log/electrostatics/settings"; frame=frame)
    rec === nothing && return nothing
    exc = read_gsd_log(path, "log/electrostatics/exceptions"; frame=frame)
    triples = exc === nothing ? () :
              [(exc[3k - 2], exc[3k - 1], exc[3k]) for k in 1:(length(exc) ÷ 3)]
    return electrostatics_from_settings(rec, triples)
end

# -------------------------
# Measured accuracy targeting
# -------------------------

# Real space (cluster list on cluster-backend states) plus the mesh of `solver`.
function _scratch_eval!(buf, solver, st)
    fx, fy, fz, E, V, Vg = buf
    SimulationCore._scratch_real_space!(st, solver, fx, fy, fz, E, V, false)
    Electrostatics.compute_mesh!(solver, st.rx, st.ry, st.rz, st.charge, fx, fy, fz, E, Vg, false)
    return nothing
end

function _scratch_forces!(buf, solver, st)
    fx, fy, fz, _, _, _ = buf
    _scratch_eval!(buf, solver, st)
    return vcat(Float64.(Array(fx))', Float64.(Array(fy))', Float64.(Array(fz))')
end

function _solver_for(st::SimulationState{T}, spec) where {T}
    q = Array(st.charge)
    return Electrostatics.ElectrostaticsSolver(spec, T, st.box3, q, Tuple{Int32,Int32,Float64}[], collect(1:length(q)))
end

_with_mesh(f::PME, mesh) = PME(; (k => getfield(f, k) for k in fieldnames(PME))..., mesh=mesh, mesh_spacing=nothing)
_with_mesh(f::PPPM, mesh) = PPPM(; (k => getfield(f, k) for k in fieldnames(PPPM))..., mesh=mesh, mesh_spacing=nothing)

"""
    tune_electrostatics(st, force; target, reference_tolerance=target/1000,
                        max_mesh=512, spacings=nothing) -> NamedTuple

Choose the cheapest accuracy setting of `force` that meets a *measured*
relative RMS force error `target` on the charges and coordinates of `st`
(a state whose neighbor list covers `force.real_cutoff`).

With `objective=:time` the search is over every knob that keeps the real-space
cutoff, and the *fastest complete force evaluation* of `st` (its own vdW
term and nonbonded backend included) that meets `target` wins:
* PME and PPPM: assignment order (`orders`, default 4–6 for PME and 5–7 for
  PPPM), the Ewald α through the real-space tail (`real_space_tolerances`,
  default `target × (0.01, 0.03, 0.1, 0.3)`) and the coarsest passing mesh
  spacing for each pair;
* ESP: the tolerance ε and the window width (default width and one point
  narrower or wider, within the shape's bound).

The error is measured on the Coulomb forces through the state's force path
(forces without electrostatics are subtracted), against an ESP reference at
`reference_tolerance` (Float64 states) or an order-6 PME on a grid of
spacing r_c/14 (Float32 states, ~5e-6 accurate). The state's solver is restored afterwards. Returns
`(force, error, time_ms, mesh, candidates)`, where `candidates` lists every
passing setting with its error and time.

The reference is an ESP solve at the same cutoff with prolate tolerance
`reference_tolerance`; ESP converges at fixed cutoff, so it reuses the
state's neighbor list. For [`PME`](@ref) and [`PPPM`](@ref) the mesh spacing
is refined (cutoff, `alpha`/`real_space_tolerance` and `order` are kept). For
[`ESP`](@ref) the tolerance ε is tightened with the paper's rules for shape,
width and spacing; refining an ESP mesh alone saturates near ε.

Returns `(force, error, time_ms, mesh, reference_error_bound)`. `force` is an
explicit descriptor (fixed `mesh`, no hidden retuning), `error` the measured
error against the reference, `time_ms` the median force-call time and
`reference_error_bound` the reference's own measured change when ε is
halved. Electrostatic exceptions are not included in the comparison (they are
identical for every candidate).
"""
function tune_electrostatics(st::SimulationState{T}, force::ElectrostaticsForce; target::Real,
                             reference_tolerance::Real=target / 1000, max_mesh::Integer=512,
                             spacings=nothing, objective::Symbol=:mesh, orders=nothing,
                             real_space_tolerances=nothing) where {T}
    st.charge === nothing && throw(ArgumentError("tune_electrostatics needs a charged state."))
    st.rz === nothing && throw(ArgumentError("tune_electrostatics needs a 3D state."))
    objective in (:mesh, :time) || throw(ArgumentError("objective must be :mesh or :time."))
    objective === :time && return _tune_for_time(st, force; target=Float64(target),
                                                 reference_tolerance=Float64(reference_tolerance),
                                                 max_mesh=Int(max_mesh), orders=orders,
                                                 tails=real_space_tolerances)
    N = length(st.rx)
    buf = (CUDA.zeros(T, N), CUDA.zeros(T, N), CUDA.zeros(T, N), CUDA.zeros(T, N),
           CUDA.zeros(T, N, 6), CUDA.zeros(T, 6))
    rc = Float64(force.real_cutoff)
    pref = Float64(force.coulomb_prefactor)
    box = Float64.(st.box3)
    refspec(eps) = electrostatics_spec(ESP(coulomb_prefactor=pref, relative_permittivity=force.relative_permittivity,
                                           real_cutoff=rc, tolerance=eps, net_charge=:background))
    Fref = _scratch_forces!(buf, _solver_for(st, refspec(reference_tolerance)), st)
    Fref2 = _scratch_forces!(buf, _solver_for(st, refspec(reference_tolerance / 2)), st)
    GC.gc()
    rel(F) = sqrt(Base.sum(abs2, F .- Fref) / Base.sum(abs2, Fref))
    ref_bound = rel(Fref2)
    function trial(f)
        spec = electrostatics_spec(f)
        maximum(Electrostatics.resolve_mesh(spec, box)) > max_mesh && return nothing
        solver = _solver_for(st, spec)
        err = rel(_scratch_forces!(buf, solver, st))
        return (solver=solver, err=err, mesh=Electrostatics.solver_mesh(solver))
    end
    chosen = nothing
    if force isa ESP
        for eps in Float64(target) .* (0.75 .^ (0:16))
            f = ESP(; (k => getfield(force, k) for k in fieldnames(ESP))..., tolerance=eps,
                    split_shape=nothing, window_shape=nothing, order=nothing, mesh=nothing, mesh_spacing=nothing)
            r = trial(f)
            r === nothing && break
            if r.err <= target
                spec = electrostatics_spec(f)
                chosen = (force=ESP(; (k => getfield(f, k) for k in fieldnames(ESP))..., tolerance=nothing,
                                    split_shape=spec.split.shape, window_shape=spec.window.shape,
                                    order=spec.window.order, mesh=r.mesh, mesh_spacing=nothing),
                          r...)
                break
            end
            GC.gc()
        end
    else
        hs = spacings === nothing ? (minimum(box) / 8) .* (0.9 .^ (0:60)) : spacings
        for h in hs
            f = _with_mesh(force, nothing)
            f = typeof(force)(; (k => getfield(f, k) for k in fieldnames(typeof(force)))..., mesh=nothing, mesh_spacing=h)
            r = trial(f)
            r === nothing && break
            if r.err <= target
                chosen = (force=_with_mesh(force, r.mesh), r...)
                break
            end
            GC.gc()
        end
    end
    chosen === nothing &&
        throw(ArgumentError("No setting within max_mesh=$(max_mesh) reaches the target $(target); increase the order, the cutoff or max_mesh."))
    solver = chosen.solver
    fx, fy, fz, E, V, Vg = buf
    call() = _scratch_eval!(buf, solver, st)
    call(); call()
    ts = sort!([1e3 * CUDA.@elapsed(call()) for _ in 1:15])
    return (force=chosen.force, error=chosen.err, time_ms=ts[8], mesh=chosen.mesh, reference_error_bound=ref_bound)
end

# -------------------------
# Time-optimal tuning through the state's own force path
# -------------------------

# Evaluations run on the device engine (an evaluation-only engine per configuration).
function _state_forces(st::SimulationState)
    SimulationCore.invalidate_forces!(st)
    SimulationCore.evaluate_current_configuration!(st, false)
    return vcat(Float64.(Array(st.fx))', Float64.(Array(st.fy))', Float64.(Array(st.fz))')
end

function _state_time_ms(st::SimulationState; reps::Integer=11)
    call() = (SimulationCore.invalidate_forces!(st); SimulationCore.evaluate_current_configuration!(st, false))
    call(); call(); CUDA.synchronize()
    ts = sort!([1e3 * CUDA.@elapsed(call()) for _ in 1:reps])
    return ts[cld(reps, 2)]
end

function _rebuild_force(f::ElectrostaticsForce; kw...)
    F = typeof(f)
    return F(; (k => getfield(f, k) for k in fieldnames(F))..., kw...)
end

function _tune_for_time(st::SimulationState{T}, force::ElectrostaticsForce; target::Float64,
                        reference_tolerance::Float64, max_mesh::Int, orders, tails) where {T}
    s0 = st.electrostatics
    exc = s0 === nothing ? Tuple{Int32,Int32,Float64}[] : copy(s0.exceptions)
    spec0 = s0 === nothing ? nothing : s0.spec
    rc = Float64(force.real_cutoff)
    box = Float64.(st.box3)
    attach(spec) = SimulationCore.attach_electrostatics!(st, spec; exceptions=exc)
    try
        s0 === nothing || SimulationCore.detach_electrostatics!(st)
        Fnb = _state_forces(st)
        # Reference: tight ESP in Float64. In Float32 wide prolate windows lose
        # accuracy (ESP at ε = 1e-7 is only good to ~1e-4 there), so the
        # reference is an order-6 PME on a fine grid (~5e-6 in Float32).
        refforce = T === Float64 ?
            ESP(coulomb_prefactor=force.coulomb_prefactor, relative_permittivity=force.relative_permittivity,
                real_cutoff=rc, tolerance=reference_tolerance, net_charge=:background) :
            PME(coulomb_prefactor=force.coulomb_prefactor, relative_permittivity=force.relative_permittivity,
                real_cutoff=rc, real_space_tolerance=1e-7, order=6, mesh_spacing=rc / 14, net_charge=:background)
        attach(electrostatics_spec(refforce))
        Fref = _state_forces(st) .- Fnb
        den = sqrt(Base.sum(abs2, Fref))
        err_of(F) = sqrt(Base.sum(abs2, F .- Fnb .- Fref)) / den
        candidates = NamedTuple[]
        function trial(f)
            spec = electrostatics_spec(f)
            maximum(Electrostatics.resolve_mesh(spec, box)) > max_mesh && return nothing
            attach(spec)
            err = err_of(_state_forces(st))
            return (err=err, mesh=Electrostatics.solver_mesh(st.electrostatics))
        end
        record!(f, r) = push!(candidates, (force=f, error=r.err, mesh=r.mesh, time_ms=_state_time_ms(st)))
        if force isa ESP
            eps_list = target .* (0.75 .^ (0:16))
            P0 = esp_order_for_tolerance(target)
            Ps = orders === nothing ? (P0 - 1, P0, P0 + 1) : Tuple(orders)
            for P in Ps
                P >= 3 || continue
                for eps in eps_list
                    shape = Electrostatics.prolate_shape_for_tolerance(eps / 2)
                    P <= Electrostatics.max_prolate_order(shape) || continue
                    f = _rebuild_force(force; tolerance=eps, split_shape=nothing, window_shape=nothing,
                                       order=P, mesh=nothing, mesh_spacing=nothing)
                    r = trial(f)
                    r === nothing && break
                    if r.err <= target
                        record!(f, r)
                        break
                    end
                end
            end
        else
            Ps = orders === nothing ? (force isa PME ? (4, 5, 6) : (5, 6, 7)) : Tuple(orders)
            tl = tails === nothing ? target .* (0.01, 0.03, 0.1, 0.3) : Tuple(tails)
            for P in Ps, tail in tl
                alpha = Electrostatics.alpha_from_tolerance(rc, tail)
                for h in (rc / 2) .* (0.9 .^ (0:40))
                    f = _rebuild_force(force; alpha=alpha, real_space_tolerance=nothing, order=P,
                                       mesh=nothing, mesh_spacing=h)
                    r = trial(f)
                    r === nothing && break
                    if r.err <= target
                        record!(_rebuild_force(f; mesh=r.mesh, mesh_spacing=nothing), r)
                        break
                    end
                end
            end
        end
        isempty(candidates) &&
            throw(ArgumentError("No setting within max_mesh=$(max_mesh) reaches the target $(target) on this state."))
        best = candidates[argmin([c.time_ms for c in candidates])]
        return (force=best.force, error=best.error, time_ms=best.time_ms, mesh=best.mesh,
                candidates=sort!(candidates; by=c -> c.time_ms))
    finally
        spec0 === nothing ? SimulationCore.detach_electrostatics!(st) : attach(spec0)
    end
end
