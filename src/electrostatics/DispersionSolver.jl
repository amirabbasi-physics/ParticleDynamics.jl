# ==============================================================================
#   DispersionSpec and DispersionSolver (GPU 1/r⁶ Lattice Dispersion Engine)
# ==============================================================================

const DISPERSION_INFLUENCE_KINDS = (:spme, :p3m_ad, :c2_esp, :c2_esp_deconvolution)

"""
    DispersionSpec

Validated host-side description of a long-range `1/r⁶` lattice dispersion solver
(`DispersionPME` or `DispersionESP`). Uses analytic differentiation (`1` forward
and `1` inverse FFT per evaluation) and includes the mandatory `k = 0` mode in
both energy and diagonal global virial.
"""
struct DispersionSpec
    real_cutoff::Float64
    split::AbstractDispersionSplit
    window::AbstractWindow
    mesh::Union{Nothing,NTuple{3,Int}}
    mesh_spacing::Union{Nothing,Float64}
    influence::Symbol
    alias_order::Int
    real_space_modifier::Symbol
    lb_switch_fraction::Float64
end

function DispersionSpec(; real_cutoff::Real,
                          split::AbstractDispersionSplit,
                          window::AbstractWindow,
                          mesh=nothing,
                          mesh_spacing=nothing,
                          influence::Symbol,
                          alias_order::Integer=2,
                          real_space_modifier::Symbol=:none,
                          lb_switch_fraction::Real=0.85)
    isfinite(real_cutoff) && real_cutoff > 0 || throw(ArgumentError("Dispersion real_cutoff must be finite and positive."))
    influence in DISPERSION_INFLUENCE_KINDS || throw(ArgumentError("Unknown dispersion influence kind $(influence)."))
    if influence === :c2_esp || influence === :c2_esp_deconvolution
        split isa C2ProlateDispersionSplit && window isa C2ProlateWindow ||
            throw(ArgumentError("DispersionESP requires C2ProlateDispersionSplit and C2ProlateWindow."))
        isapprox(split.cutoff, real_cutoff; rtol=1e-12) ||
            throw(ArgumentError("Dispersion split cutoff $(split.cutoff) does not match real_cutoff $(real_cutoff)."))
        window.order <= max_c2_prolate_order(window.shape) ||
            throw(ArgumentError("C2 prolate window width $(window.order) exceeds bandlimit bound $(max_c2_prolate_order(window.shape))."))
    else
        split isa GaussianDispersionSplit && window isa BSplineWindow ||
            throw(ArgumentError("DispersionPME requires GaussianDispersionSplit and BSplineWindow."))
    end
    0 <= alias_order <= 6 || throw(ArgumentError("alias_order must lie in 0..6."))
    real_space_modifier in (:none, :potential_shift) ||
        throw(ArgumentError("Dispersion real_space_modifier must be :none or :potential_shift."))
    (isfinite(lb_switch_fraction) && 0.0 < lb_switch_fraction < 1.0) ||
        throw(ArgumentError("lb_switch_fraction must lie in (0, 1); got $(lb_switch_fraction)."))
    meshT = if mesh === nothing
        mesh_spacing === nothing && throw(ArgumentError("Provide either `mesh=(K1,K2,K3)` or `mesh_spacing`."))
        nothing
    else
        length(mesh) == 3 || throw(ArgumentError("Dispersion mesh must be 3D."))
        all(k -> k >= window_order(window), mesh) ||
            throw(ArgumentError("Every dispersion mesh dimension must be ≥ window order $(window_order(window))."))
        ntuple(i -> Int(mesh[i]), 3)
    end
    spacing = mesh_spacing === nothing ? nothing : Float64(mesh_spacing)
    spacing === nothing || (isfinite(spacing) && spacing > 0) ||
        throw(ArgumentError("Dispersion mesh_spacing must be finite and positive."))
    return DispersionSpec(Float64(real_cutoff), split, window, meshT, spacing,
                          influence, Int(alias_order), real_space_modifier, Float64(lb_switch_fraction))
end

function resolve_mesh(spec::DispersionSpec, box::NTuple{3,<:Real})
    spec.mesh === nothing || return spec.mesh
    h = spec.mesh_spacing::Float64
    P = window_order(spec.window)
    return ntuple(d -> max(fft_friendly(ceil(Int, box[d] / h - 1e-9)), P), 3)
end

mutable struct DispersionSolver{T,ST,W<:MeshWorkspace}
    spec::DispersionSpec
    rc2::T
    selfc::T                  # Φ₆(0); self energy per particle is +½ selfc · s6ᵢ²
    eshift::T                 # V_{6,s}(r_c) when real_space_modifier === :potential_shift
    zcoef::Float64            # ½ Φ̂₆(0) for the mandatory k = 0 mode
    s6total::Float64          # Σ s6ᵢ
    s6sq::Float64             # Σ s6ᵢ²
    bg::T                     # −zcoef · s6total / V (per-particle k = 0 potential factor)
    ubg::Float64              # −zcoef · s6total² / V (total k = 0 energy & diagonal virial)
    box::NTuple{3,Float64}
    stencil::ST
    ws::W
    exceptions::Vector{Tuple{Int32,Int32,Float64}}
    exo::CuArray{Int32,1}
    exj::CuArray{Int32,1}
    exs::CuArray{T,1}
    identity_order::CuArray{Int32,1}
end

solver_mesh(s::DispersionSolver) = s.ws.mesh
gather_sets(::DispersionSolver) = true

function compute_dispersion_influence!(G::CuArray{Float64}, Vc::CuArray{Float64,2},
                                       spec::DispersionSpec, mesh::NTuple{3,Int},
                                       box::NTuple{3,Float64})
    K1, K2, K3 = mesh
    nmodes = (K1 ÷ 2 + 1) * K2 * K3
    threads = 128
    blocks = cld(nmodes, threads)
    keep = Any[]
    spectral = if spec.split isa GaussianDispersionSplit
        GaussianDispersionSpectral(spec.split.beta)
    else
        sp = spec.split::C2ProlateDispersionSplit
        rq = CuArray(sp.rq); wd = CuArray(sp.w_dphi)
        push!(keep, rq, wd)
        ProlateDispersionSpectral(sp.beta_ref, cudaconvert(rq), cudaconvert(wd))
    end
    # C = -1.0 because dispersion is attractive (-s6_i s6_j / r⁶)
    if spec.influence in (:spme, :c2_esp_deconvolution)
        D1 = CuArray(window_deconvolution(spec.window, K1))
        D2 = CuArray(window_deconvolution(spec.window, K2))
        D3 = CuArray(window_deconvolution(spec.window, K3))
        push!(keep, D1, D2, D3)
        @cuda threads=threads blocks=blocks _influence_ewald_kernel!(G, Vc, D1, D2, D3, K1, K2, K3,
                                                                    box, -1.0, spectral)
    else
        nmax = spec.alias_order
        U1 = CuArray(window_alias_table(spec.window, K1, nmax))
        U2 = CuArray(window_alias_table(spec.window, K2, nmax))
        U3 = CuArray(window_alias_table(spec.window, K3, nmax))
        push!(keep, U1, U2, U3)
        @cuda threads=threads blocks=blocks _influence_alias_kernel!(G, Vc, U1, U2, U3, K1, K2, K3,
                                                                    box, -1.0, spectral, nmax, false)
    end
    CUDA.synchronize()
    return nothing
end

const _DISP_INFLUENCE_CACHE = WeakKeyDict{Any,Any}()

function _dispersion_influence_tables(s::DispersionSolver)
    get!(_DISP_INFLUENCE_CACHE, s) do
        spec = s.spec; K1, K2, K3 = s.ws.mesh
        keep = Any[]
        spectral = if spec.split isa GaussianDispersionSplit
            GaussianDispersionSpectral(spec.split.beta)
        else
            sp = spec.split::C2ProlateDispersionSplit
            rq = CuArray(sp.rq); wd = CuArray(sp.w_dphi); push!(keep, rq, wd)
            exact = ProlateDispersionSpectral(sp.beta_ref, cudaconvert(rq), cudaconvert(wd))
            nal = spec.influence in (:spme, :c2_esp_deconvolution) ? 0 : spec.alias_order
            kmax = 2pi * sqrt(sum(((s.ws.mesh[d] ÷ 2 + nal * s.ws.mesh[d]) / (0.9 * s.box[d]))^2 for d in 1:3))
            du = 0.05; n = ceil(Int, sp.cutoff * kmax / du) + 3
            gt = CUDA.zeros(Float64, n); dgt = CUDA.zeros(Float64, n); push!(keep, gt, dgt)
            @cuda threads=128 blocks=cld(n, 128) _dispersion_table_kernel!(gt, dgt, exact, sp.cutoff, du)
            CUDA.synchronize()
            TabulatedDispersionSpectral(sp.cutoff, du, cudaconvert(gt), cudaconvert(dgt), exact)
        end
        tables = if spec.influence in (:spme, :c2_esp_deconvolution)
            (CuArray(window_deconvolution(spec.window, K1)),
             CuArray(window_deconvolution(spec.window, K2)),
             CuArray(window_deconvolution(spec.window, K3)))
        else
            nmax = spec.alias_order
            (CuArray(window_alias_table(spec.window, K1, nmax)),
             CuArray(window_alias_table(spec.window, K2, nmax)),
             CuArray(window_alias_table(spec.window, K3, nmax)))
        end
        (spectral=spectral, tables=tables, keep=keep)
    end
end

function device_dispersion_influence!(s::DispersionSolver, geom; launch::Bool=true)
    c = _dispersion_influence_tables(s); spec = s.spec; K1, K2, K3 = s.ws.mesh
    nmodes = (K1 ÷ 2 + 1) * K2 * K3; threads = 128; blocks = cld(nmodes, threads)
    D1, D2, D3 = c.tables
    if spec.influence in (:spme, :c2_esp_deconvolution)
        k = @cuda launch=false _influence_ewald_kernel!(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, -1.0, c.spectral)
        launch && k(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, -1.0, c.spectral; threads, blocks)
    else
        k = @cuda launch=false _influence_alias_kernel!(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, -1.0,
                                                         c.spectral, spec.alias_order, false, eltype(s.ws.G))
        launch && k(s.ws.G, s.ws.Vc, D1, D2, D3, K1, K2, K3, geom, -1.0, c.spectral, spec.alias_order,
                    false, eltype(s.ws.G); threads, blocks)
    end
    return nothing
end

function DispersionSolver(spec::DispersionSpec, ::Type{T}, box::NTuple{3,<:Real},
                          s6_weights::AbstractVector{<:Real},
                          exceptions::Vector{Tuple{Int32,Int32,Float64}},
                          slot_of::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    boxF = ntuple(d -> Float64(box[d]), 3)
    2 * spec.real_cutoff < minimum(boxF) ||
        throw(ArgumentError("Dispersion real_cutoff=$(spec.real_cutoff) must be < half shortest box length $(minimum(boxF))."))
    window_resolved(spec.window, T) ||
        throw(ArgumentError("Dispersion prolate window is attenuated below $(T) Nyquist resolution; increase window shape or use Float64."))
    mesh = resolve_mesh(spec, boxF)
    ws = MeshWorkspace(T, mesh, false)
    N = length(s6_weights)
    slots = Tuple{Int32,Int32,Float64}[(Int32(slot_of[i]), Int32(slot_of[j]), sc) for (i, j, sc) in exceptions]
    exo, exj, exs = _exception_rows(T, N, slots)
    stencil = device_stencil(spec.window, T)
    esh = spec.real_space_modifier === :potential_shift ? T(dispersion_cutoff_energy(spec.split, spec.real_cutoff)) : zero(T)
    solver = DispersionSolver{T,typeof(stencil),typeof(ws)}(
        spec, T(spec.real_cutoff^2), T(dispersion_self_coefficient(spec.split)), esh,
        dispersion_zero_mode_coefficient(spec.split), 0.0, 0.0, zero(T), 0.0, boxF,
        stencil, ws, exceptions, exo, exj, exs, CuArray(Int32.(1:N)))
    set_solver_dispersion_weights!(solver, s6_weights)
    refresh_geometry!(solver, boxF)
    return solver
end

function set_solver_dispersion_weights!(s::DispersionSolver{T}, s6_weights::AbstractVector{<:Real}) where {T}
    all(x -> isfinite(x) && x >= 0, s6_weights) || throw(ArgumentError("Dispersion √c6 weights must be finite and ≥ 0."))
    s.s6total = sum(Float64, s6_weights; init=0.0)
    s.s6sq = sum(x -> Float64(x)^2, s6_weights; init=0.0)
    _refresh_dispersion_zero_mode!(s)
    return s
end

function _refresh_dispersion_zero_mode!(s::DispersionSolver{T}) where {T}
    V = prod(s.box)
    s.bg = T(-s.zcoef * s.s6total / V)
    s.ubg = -s.zcoef * (s.s6total^2) / V
    return s
end

function refresh_geometry!(s::DispersionSolver{T}, box::NTuple{3,<:Real}) where {T}
    boxF = ntuple(d -> Float64(box[d]), 3)
    2 * s.spec.real_cutoff < minimum(boxF) ||
        throw(ArgumentError("Dispersion real_cutoff must stay below half the shortest box length."))
    s.box = boxF
    K1, K2, K3 = s.ws.mesh
    nmodes = (K1 ÷ 2 + 1) * K2 * K3
    G64 = CUDA.zeros(Float64, nmodes)
    V64 = CUDA.zeros(Float64, 6, nmodes)
    compute_dispersion_influence!(G64, V64, s.spec, s.ws.mesh, boxF)
    copyto!(s.ws.G, T.(G64))
    copyto!(s.ws.Vc, T.(V64))
    CUDA.unsafe_free!(G64)
    CUDA.unsafe_free!(V64)
    _refresh_dispersion_zero_mode!(s)
    return s
end

function _add_or_store_global_virial_kernel!(Vg, acc, geom, zcoef_s6sq::Float64, add::Bool)
    i = threadIdx().x
    if i <= 6
        L1, L2, L3 = map(Float64, Definitions.box_lengths(geom, Val(3)))
        ubg = -zcoef_s6sq / (L1 * L2 * L3)
        val = eltype(Vg)(acc[i + 1] + (i <= 3 ? ubg : 0.0))
        @inbounds Vg[i] = add ? Vg[i] + val : val
    end
    return
end

function _update_dispersion_global_virial!(Vglobal, s::DispersionSolver, geom; add::Bool=false)
    Lb = geom === nothing ? s.box : geom
    @cuda threads=6 blocks=1 _add_or_store_global_virial_kernel!(Vglobal, s.ws.acc, Lb, s.zcoef * (s.s6total^2), add)
    return nothing
end

# Dynamic k = 0 per-particle factor `-zcoef * s6total / V(geom)` when device barostat scales `geom`
function mesh_transform!(s::DispersionSolver{T}, rx, ry, rz, s6, compute_energy::Bool, order;
                         geom=nothing, part::Symbol=:all) where {T}
    ws = s.ws
    K1, K2, K3 = Int32.(ws.mesh)
    Lb = geom === nothing ? T.(s.box) : geom
    N = length(rx)
    part === :fft && @goto transforms
    part === :clear && (fill!(ws.grid, zero(T)); return nothing)
    part === :spread_only || fill!(ws.grid, zero(T))
    @cuda threads=128 blocks=cld(N * MESH_LANES, 128) _spread_lines_kernel!(ws.grid_flat, rx, ry, rz, s6, order, Lb,
                                                                            K1, K2, K3, s.stencil, nothing)
    (part === :spread || part === :spread_only) && return nothing
    @label transforms
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    nmodes = Int32(length(ws.G))
    compute_energy && fill!(ws.acc, 0.0)
    mthreads, mblocks = _launch1d(nmodes, 256)
    @cuda threads=mthreads blocks=mblocks _solve_kernel!(ws.spec_flat, ws.G, ws.Vc, ws.acc,
                                                         K1, K1 ÷ Int32(2) + Int32(1), nmodes,
                                                         Val(compute_energy))
    _c2r!(ws.grid, ws.plan_bwd, ws.spec)
    return nothing
end

function _gather_disp_ad_lines_kernel!(fx, fy, fz, Epot, rx, ry, rz, s6, order, grid::CuDeviceVector{T},
                                       box, K1::Int32, K2::Int32, K3::Int32, stencil::S,
                                       zcoef_s6tot::Float64, ::Val{ENERGY}, ::Val{SET}=Val(false)) where {T,S,ENERGY,SET}
    t, lane, i, s6i, phi, gx, gy, gz, Lx, Ly, Lz = _gather_ad_lines_sums(rx, ry, rz, s6, order, grid, box, K1, K2, K3, stencil)
    gx = _sum16(gx); gy = _sum16(gy); gz = _sum16(gz)
    ENERGY && (phi = _sum16(phi))
    bg = ENERGY ? T(-zcoef_s6tot / (Float64(Lx) * Float64(Ly) * Float64(Lz))) : zero(T)
    @inbounds if SET && lane == Int32(0) && t <= length(order)
        fx[i] = -(s6i * gx * (T(K1) / Lx))
        fy[i] = -(s6i * gy * (T(K2) / Ly))
        fz[i] = -(s6i * gz * (T(K3) / Lz))
        ENERGY && (Epot[i] = T(0.5) * s6i * phi + bg * s6i)
    elseif !SET && lane == Int32(0) && t <= length(order) && s6i != zero(T)
        fx[i] -= s6i * gx * (T(K1) / Lx)
        fy[i] -= s6i * gy * (T(K2) / Ly)
        fz[i] -= s6i * gz * (T(K3) / Lz)
        ENERGY && (Epot[i] += T(0.5) * s6i * phi + bg * s6i)
    end
    return
end

function mesh_gather!(s::DispersionSolver{T}, rx, ry, rz, s6, fx, fy, fz, Epot, Vglobal,
                      compute_energy::Bool, order; geom=nothing, set::Bool=false, add_virial::Bool=false) where {T}
    ws = s.ws
    K1, K2, K3 = Int32.(ws.mesh)
    Lb = geom === nothing ? T.(s.box) : geom
    @cuda threads=128 blocks=cld(length(rx) * MESH_LANES, 128) _gather_disp_ad_lines_kernel!(
        fx, fy, fz, Epot, rx, ry, rz, s6, order, ws.grid_flat, Lb, K1, K2, K3, s.stencil,
        s.zcoef * s.s6total, Val(compute_energy), Val(set))
    compute_energy && _update_dispersion_global_virial!(Vglobal, s, geom; add=add_virial)
    return nothing
end

dispersion_mesh_energy(s::DispersionSolver) = Array(view(s.ws.acc, 1:1))[1]
dispersion_self_energy(s::DispersionSolver) = 0.5 * Float64(s.selfc) * s.s6sq
dispersion_zero_mode_energy(s::DispersionSolver) = -s.zcoef * (s.s6total^2) / prod(s.box)

const _DISP_METHOD_CODE = Dict(:spme => 1.0, :p3m_ad => 2.0, :c2_esp => 3.0, :c2_esp_deconvolution => 4.0)

function dispersion_settings_record(s::DispersionSolver{T}) where {T}
    spec = s.spec
    shape = spec.split isa GaussianDispersionSplit ? spec.split.beta : spec.split.shape
    wshape = spec.window isa AnyProlateWindow ? spec.window.shape : 0.0
    mesh = s.ws.mesh
    return Float64[1.0, _DISP_METHOD_CODE[spec.influence], spec.real_cutoff, shape,
                   spec.window.order, wshape, mesh[1], mesh[2], mesh[3], spec.alias_order,
                   8 * sizeof(T), spec.real_space_modifier === :potential_shift ? 1.0 : 0.0,
                   spec.lb_switch_fraction]
end

function dispersion_exceptions_record(s::DispersionSolver)
    out = Float64[]
    for (i, j, sc) in s.exceptions
        push!(out, i, j, sc)
    end
    return out
end
