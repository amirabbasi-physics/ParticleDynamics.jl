# =========================
#   Solver: workspace ownership and the per-evaluation pipeline
# =========================

"""
    MeshWorkspace{T}

Reusable device storage of one mesh solver: the real charge/potential grid,
its R2C spectrum, influence function `G`, virial coefficients `Vc`, cuFFT
plans, the Float64 reduction buffer, and (ik only) three field grids plus a
spectral scratch array.

Aliasing rules: the charge grid is overwritten by the backward transform with
the potential; the spectrum is consumed (cuFFT C2R may overwrite its input),
so ik field spectra are formed in `scratch` before the potential transform.
"""
mutable struct MeshWorkspace{T,PF,PB}
    mesh::NTuple{3,Int}
    grid::CuArray{T,3}
    grid_flat::CuArray{T,1}
    spec::CuArray{Complex{T},3}
    spec_flat::CuArray{Complex{T},1}
    G::CuArray{T,1}
    Vc::CuArray{T,2}
    plan_fwd::PF
    plan_bwd::PB
    acc::CuArray{Float64,1}
    field::Union{Nothing,NTuple{3,CuArray{T,3}}}
    field_flat::Union{Nothing,NTuple{3,CuArray{T,1}}}
    scratch::Union{Nothing,CuArray{Complex{T},3}}
    scratch_flat::Union{Nothing,CuArray{Complex{T},1}}
end

function MeshWorkspace(::Type{T}, mesh::NTuple{3,Int}, ik::Bool) where {T<:AbstractFloat}
    K1, K2, K3 = mesh
    K1h = K1 ÷ 2 + 1
    nnodes = Int128(K1) * K2 * K3
    nmodes = Int128(K1h) * K2 * K3
    nnodes < typemax(Int32) && nmodes < typemax(Int32) ||
        throw(ArgumentError("Mesh $(mesh) is too large for 32-bit grid indexing."))
    bytes = nnodes * sizeof(T) * (ik ? 4 : 1) + nmodes * sizeof(Complex{T}) * (ik ? 2 : 1) +
            nmodes * 7 * sizeof(T)
    free = CUDA.free_memory() + CUDA.cached_memory()
    bytes < free ||
        throw(ArgumentError("Mesh $(mesh) needs ≈$(round(bytes / 2^20; digits=1)) MiB of device memory; only $(round(free / 2^20; digits=1)) MiB are free."))
    grid = CUDA.zeros(T, K1, K2, K3)
    spec = CUDA.zeros(Complex{T}, K1h, K2, K3)
    plan_fwd = CUDA.CUFFT.plan_rfft(grid)
    plan_bwd = CUDA.CUFFT.plan_brfft(spec, K1)
    G = CUDA.zeros(T, K1h * K2 * K3)
    Vc = CUDA.zeros(T, 6, K1h * K2 * K3)
    acc = CUDA.zeros(Float64, 7)
    field = nothing
    field_flat = nothing
    scratch = nothing
    scratch_flat = nothing
    if ik
        field = (CUDA.zeros(T, K1, K2, K3), CUDA.zeros(T, K1, K2, K3), CUDA.zeros(T, K1, K2, K3))
        field_flat = map(a -> reshape(a, :), field)
        scratch = CUDA.zeros(Complex{T}, K1h, K2, K3)
        scratch_flat = reshape(scratch, :)
    end
    return MeshWorkspace(mesh, grid, reshape(grid, :), spec, reshape(spec, :), G, Vc,
                         plan_fwd, plan_bwd, acc, field, field_flat, scratch, scratch_flat)
end

"""
    ElectrostaticsSolver{T}

Concretely typed solver attached to a `SimulationState` (`st.electrostatics`).
It holds no references to particle arrays: positions, charges, forces and
energy buffers are passed on every call, so force-slot swaps, midpoint
evaluations and spatial reordering are safe.
"""
mutable struct ElectrostaticsSolver{T,KR,ST,W<:MeshWorkspace}
    spec::ElectrostaticsSpec
    C::T
    rc2::T
    selfc::T
    eshift::T
    bgcoef::Float64
    qtotal::Float64
    qsq::Float64
    bg::T
    ubg::Float64
    box::NTuple{3,Float64}
    kernel::KR
    stencil::ST
    ws::W
    exceptions::Vector{Tuple{Int32,Int32,Float64}}   # stable particle ids
    exo::CuArray{Int32,1}
    exj::CuArray{Int32,1}
    exs::CuArray{T,1}
    identity_order::CuArray{Int32,1}
end

has_exceptions(s::ElectrostaticsSolver) = !isempty(s.exceptions)
solver_mesh(s::ElectrostaticsSolver) = s.ws.mesh
is_ik(s::ElectrostaticsSolver) = s.spec.influence === :p3m_ik

function _exception_rows(::Type{T}, N::Int, pairs_slots) where {T}
    counts = zeros(Int32, N)
    for (i, j, _) in pairs_slots
        counts[i] += 1
        counts[j] += 1
    end
    offsets = Vector{Int32}(undef, N + 1)
    offsets[1] = 1
    for i in 1:N
        offsets[i + 1] = offsets[i] + counts[i]
    end
    cursor = copy(offsets[1:N])
    js = Vector{Int32}(undef, offsets[end] - 1)
    ss = Vector{T}(undef, offsets[end] - 1)
    for (i, j, s) in pairs_slots
        js[cursor[i]] = j; ss[cursor[i]] = T(s); cursor[i] += 1
        js[cursor[j]] = i; ss[cursor[j]] = T(s); cursor[j] += 1
    end
    return CuArray(offsets), CuArray(js), CuArray(ss)
end

"""
    ElectrostaticsSolver(spec, T, box, charges, exceptions, slot_of)

Allocate the workspace, plan the FFTs, compute the influence function for
`box`, and compile exception rows. `charges` are host values in storage
order; `exceptions` are normalised `(i, j, scale)` in stable ids and
`slot_of[id]` gives the current storage slot.
"""
function ElectrostaticsSolver(spec::ElectrostaticsSpec, ::Type{T}, box::NTuple{3,<:Real},
                              charges::AbstractVector{<:Real},
                              exceptions::Vector{Tuple{Int32,Int32,Float64}},
                              slot_of::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    boxF = ntuple(d -> Float64(box[d]), 3)
    2 * spec.real_cutoff < minimum(boxF) ||
        throw(ArgumentError("real_cutoff=$(spec.real_cutoff) must be smaller than half the shortest box length $(minimum(boxF)) (minimum-image real space)."))
    mesh = resolve_mesh(spec, boxF)
    all(k -> k >= window_order(spec.window), mesh) ||
        throw(ArgumentError("Every mesh dimension must be at least the assignment order."))
    ws = MeshWorkspace(T, mesh, spec.influence === :p3m_ik)
    C = coulomb_constant(spec)
    N = length(charges)
    slots = Tuple{Int32,Int32,Float64}[(Int32(slot_of[i]), Int32(slot_of[j]), s) for (i, j, s) in exceptions]
    exo, exj, exs = _exception_rows(T, N, slots)
    solver = ElectrostaticsSolver{T,typeof(device_pair_kernel(spec.split, T)),
                                  typeof(device_stencil(spec.window, T)),typeof(ws)}(
        spec, T(C), T(spec.real_cutoff^2), T(self_coefficient(spec.split)),
        spec.real_space_modifier === :potential_shift ? T(cutoff_energy(spec.split, spec.real_cutoff)) : zero(T),
        background_coefficient(spec.split), 0.0, 0.0, zero(T), 0.0, boxF,
        device_pair_kernel(spec.split, T), device_stencil(spec.window, T), ws,
        exceptions, exo, exj, exs, CuArray(Int32.(1:N)))
    set_solver_charges!(solver, charges)
    refresh_geometry!(solver, boxF)
    return solver
end

"""
    set_solver_charges!(solver, charges)

Update the charge-dependent scalars (total charge, Σq², background factor)
and apply the net-charge policy.
"""
function set_solver_charges!(s::ElectrostaticsSolver{T}, charges::AbstractVector{<:Real}) where {T}
    all(isfinite, charges) || throw(ArgumentError("Charges must be finite."))
    Q = sum(Float64, charges)
    qabs = sum(abs, charges; init=0.0)
    if s.spec.net_charge === :error && abs(Q) > s.spec.net_charge_tolerance * max(qabs, 1.0)
        throw(ArgumentError("The system carries net charge $(Q). Neutralise it, or select net_charge=:background to add an explicit uniform neutralising background."))
    end
    s.qtotal = Q
    s.qsq = sum(x -> Float64(x)^2, charges; init=0.0)
    _refresh_background!(s)
    return s
end

function _refresh_background!(s::ElectrostaticsSolver{T}) where {T}
    V = prod(s.box)
    s.bg = s.spec.net_charge === :background ?
           T(-Float64(s.C) * s.bgcoef * s.qtotal / V) : zero(T)
    s.ubg = background_energy(s)
    return s
end

"""
    refresh_geometry!(solver, box)

Recompute the box-dependent influence function and virial coefficients
(reusing the FFT plans) after a box change with unchanged mesh dimensions.
"""
function refresh_geometry!(s::ElectrostaticsSolver{T}, box::NTuple{3,<:Real}) where {T}
    boxF = ntuple(d -> Float64(box[d]), 3)
    2 * s.spec.real_cutoff < minimum(boxF) ||
        throw(ArgumentError("real_cutoff must stay below half the shortest box length."))
    s.box = boxF
    K1, K2, K3 = s.ws.mesh
    nmodes = (K1 ÷ 2 + 1) * K2 * K3
    G64 = CUDA.zeros(Float64, nmodes)
    V64 = CUDA.zeros(Float64, 6, nmodes)
    compute_influence!(G64, V64, s.spec, s.ws.mesh, boxF, Float64(s.C))
    copyto!(s.ws.G, T.(G64))
    copyto!(s.ws.Vc, T.(V64))
    CUDA.unsafe_free!(G64)
    CUDA.unsafe_free!(V64)
    _refresh_background!(s)
    return s
end

# Thread → particle order for spreading and gathering: the cell-sorted order of
# the neighbor list (refreshed at every rebuild; particles move little between
# rebuilds), or the storage order without a cell list.
@inline _mesh_order(s::ElectrostaticsSolver, order::CuArray{Int32,1}) = order
@inline function _mesh_order(s::ElectrostaticsSolver, nbh)
    if nbh isa NeighborLists.CellListNeighborMatrix && nbh.valid &&
       length(nbh.particle_ids_sorted) == length(s.identity_order)
        return nbh.particle_ids_sorted
    end
    return s.identity_order
end

@inline _launch1d(n::Integer, threads::Integer=256) = (min(threads, max(n, 1)), cld(max(n, 1), min(threads, max(n, 1))))

"""
    compute_real_space!(solver, rx, ry, rz, q, fx, fy, fz, Epot, V, nbh, compute_energy)

Add real-space Coulomb forces, exception corrections and (with energies) the
self energy into the given buffers.
"""
function compute_real_space!(s::ElectrostaticsSolver{T}, rx, ry, rz, q, fx, fy, fz, Epot, V,
                             nbh, compute_energy::Bool) where {T}
    N = length(rx)
    L1, L2, L3 = T.(s.box)
    allpairs = nbh isa NeighborLists.AllPairsNeighborMatrix
    nbr = allpairs ? s.exj : nbh.neighbors_flat
    counts = allpairs ? s.exo : nbh.counts
    allpairs || NeighborLists.require_valid_neighbors(nbh)
    tpp = _threads_per_particle(N)
    args = (fx, fy, fz, Epot, V, rx, ry, rz, q, nbr, counts, s.exo, s.exj, s.exs,
            L1, L2, L3, s.C, s.rc2, s.selfc, s.eshift, s.kernel, Val(allpairs), Val(compute_energy))
    threads = 128
    blocks = cld(N * tpp, threads)
    if tpp == 1
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(1))
    elseif tpp == 2
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(2))
    elseif tpp == 4
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(4))
    elseif tpp == 8
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(8))
    elseif tpp == 16
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(16))
    else
        @cuda threads=threads blocks=blocks _coulomb_real_kernel!(args..., Val(32))
    end
    return nothing
end

const _RESIDENT_THREADS = Ref(0)

# Lanes per particle for the real-space kernel: enough threads to fill the
# device (SM count × 2048 resident threads) without splitting rows of large
# systems, which already saturate it with one thread per particle.
function _threads_per_particle(N::Integer)
    if _RESIDENT_THREADS[] == 0
        _RESIDENT_THREADS[] = 2048 * CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    end
    want = cld(_RESIDENT_THREADS[], max(N, 1))
    tpp = 1
    while tpp < want && tpp < 32
        tpp *= 2
    end
    return tpp
end

"""
    compute_mesh!(solver, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal, compute_energy, nbh=nothing)

Spread charges, transform, apply the influence function (reducing mesh energy
and virial when requested), transform back, and add reciprocal forces and
per-particle energies. With energies, `Vglobal[1:6]` receives the reciprocal
virial tensor `(xx, yy, zz, xy, xz, yz)`.
"""
function compute_mesh!(s::ElectrostaticsSolver{T}, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal,
                       compute_energy::Bool, nbh=nothing) where {T}
    order = _mesh_order(s, nbh)
    mesh_transform!(s, rx, ry, rz, q, compute_energy, order)
    mesh_gather!(s, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal, compute_energy, order)
    return nothing
end

"""
    mesh_transform!(solver, rx, ry, rz, q, compute_energy, order)

First half of [`compute_mesh!`](@ref): spread (tiled over blocks of `order`,
which should be spatially sorted), forward FFT, influence function (mesh
energy and virial with energies) and the inverse transforms. Touches only the
solver workspace, so it may run on its own stream concurrently with
short-range kernels.
"""
# `part`: :all, :spread (clear and spread), :clear, :spread_only (spread into a cleared grid) or
# :fft (transform, solve, inverse transform); the device engine overlaps the transforms with the
# pair force and the clear with the list check.
function mesh_transform!(s::ElectrostaticsSolver{T}, rx, ry, rz, q, compute_energy::Bool, order;
                         geom=nothing, part::Symbol=:all, xq=nothing) where {T}
    ws = s.ws
    K1, K2, K3 = Int32.(ws.mesh)
    # the box: a device geometry vector (engine runs, graphs stay valid when it changes)
    # or the solver's own lengths
    Lb = geom === nothing ? T.(s.box) : geom
    N = length(rx)
    part === :fft && @goto transforms
    part === :clear && (fill!(ws.grid, zero(T)); return nothing)
    part === :spread_only || fill!(ws.grid, zero(T))
    # large systems: 512-particle blocks with a 44 KB tile (Float32); small
    # systems keep more, smaller blocks in flight
    # small systems (too few blocks for tiles to pay off) deposit directly
    big = N >= 262_144 && sizeof(T) == 4
    if MESH_LINES[]
        @cuda threads=128 blocks=cld(N * MESH_LANES, 128) _spread_lines_kernel!(ws.grid_flat, rx, ry, rz, q, order, Lb,
                                                                                K1, K2, K3, s.stencil, xq)
    elseif N < 32_768
        xq === nothing || throw(ArgumentError("packed coordinates need the 16-lane spread (MESH_LINES)"))
        @cuda threads=128 blocks=cld(N, 128) _spread_kernel!(ws.grid_flat, rx, ry, rz, q, order, Lb,
                                                             K1, K2, K3, s.stencil)
    else
        th = big ? 512 : 128
        @cuda threads=th blocks=cld(N, th) _spread_tiled_kernel!(ws.grid_flat, rx, ry, rz, q, order, Lb,
                                                                 K1, K2, K3, s.stencil, big ? Val(11264) : Val(4096))
    end
    (part === :spread || part === :spread_only) && return nothing
    @label transforms
    mul!(ws.spec, ws.plan_fwd, ws.grid)
    nmodes = Int32(length(ws.G))
    compute_energy && fill!(ws.acc, 0.0)
    mthreads, mblocks = _launch1d(nmodes, 256)
    @cuda threads=mthreads blocks=mblocks _solve_kernel!(ws.spec_flat, ws.G, ws.Vc, ws.acc,
                                                         K1, K1 ÷ Int32(2) + Int32(1), nmodes,
                                                         Val(compute_energy))
    if is_ik(s)
        field = ws.field::NTuple{3,CuArray{T,3}}
        for d in 1:3
            @cuda threads=mthreads blocks=mblocks _ik_field_kernel!(ws.scratch_flat, ws.spec_flat,
                                                                    K1, K2, K3, Lb, Int32(d))
            _c2r!(field[d], ws.plan_bwd, ws.scratch::CuArray{Complex{T},3})
        end
        compute_energy && _c2r!(ws.grid, ws.plan_bwd, ws.spec)
    else
        _c2r!(ws.grid, ws.plan_bwd, ws.spec)
    end
    return nothing
end

# Complex-to-real transform without the input copy that `mul!` makes: the
# spectrum (or the ik scratch) is not read again after its inverse transform,
# so cuFFT may overwrite it.
@inline function _c2r!(y, p, x)
    CUDA.CUFFT.unsafe_execute!(p, x, y)
    return y
end

"""
    mesh_gather!(solver, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal, compute_energy, order)

Second half of [`compute_mesh!`](@ref): add reciprocal forces (and energies)
from the transformed grid, and store the global virial.
"""
# `set=true` (16-lane gather only, `gather_sets`): the gather writes the mesh force and energy
# of every particle of `order` instead of adding them (the caller adds the pair terms after).
gather_sets(s::ElectrostaticsSolver) = !is_ik(s) && MESH_LINES[]

function mesh_gather!(s::ElectrostaticsSolver{T}, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal,
                      compute_energy::Bool, order; geom=nothing, set::Bool=false) where {T}
    set && !gather_sets(s) && throw(ArgumentError("mesh_gather!(set=true) needs the 16-lane gather"))
    ws = s.ws
    K1, K2, K3 = Int32.(ws.mesh)
    Lb = geom === nothing ? T.(s.box) : geom
    pthreads, pblocks = _launch1d(length(rx), 128)
    if is_ik(s)
        ff = ws.field_flat::NTuple{3,CuArray{T,1}}
        @cuda threads=pthreads blocks=pblocks _gather_ik_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, order,
            ff[1], ff[2], ff[3], ws.grid_flat, Lb, K1, K2, K3, s.stencil, s.bg,
            Val(compute_energy))
    elseif MESH_LINES[]
        @cuda threads=128 blocks=cld(length(rx) * MESH_LANES, 128) _gather_ad_lines_kernel!(fx, fy, fz, Epot, rx, ry, rz,
            q, order, ws.grid_flat, Lb, K1, K2, K3, s.stencil, s.bg, Val(compute_energy), Val(set))
    else
        @cuda threads=pthreads blocks=pblocks _gather_ad_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, order,
            ws.grid_flat, Lb, K1, K2, K3, s.stencil, s.bg, Val(compute_energy))
    end
    compute_energy && _store_global_virial!(Vglobal, ws.acc, s.ubg)
    return nothing
end

# Global virial = mesh tensor plus the uniform-background term, whose
# configurational virial at fixed α is U_bg on the diagonal.
function _store_global_virial_kernel!(Vg, acc, ubg::Float64)
    i = threadIdx().x
    if i <= 6
        @inbounds Vg[i] = eltype(Vg)(acc[i + 1] + (i <= 3 ? ubg : 0.0))
    end
    return
end

function _store_global_virial!(Vglobal, acc, ubg::Float64)
    @cuda threads=6 blocks=1 _store_global_virial_kernel!(Vglobal, acc, ubg)
    return nothing
end

"""
    compute!(solver, rx, ry, rz, q, fx, fy, fz, Epot, Vpp, Vglobal, nbh, compute_energy)

Full electrostatic contribution, added on top of whatever the buffers already
hold (the short-range vdW writer runs first and initialises them).
"""
function compute!(s::ElectrostaticsSolver, rx, ry, rz, q, fx, fy, fz, Epot, Vpp, Vglobal,
                  nbh, compute_energy::Bool)
    compute_real_space!(s, rx, ry, rz, q, fx, fy, fz, Epot, Vpp, nbh, compute_energy)
    compute_mesh!(s, rx, ry, rz, q, fx, fy, fz, Epot, Vglobal, compute_energy, nbh)
    return nothing
end

"""
    mesh_energy(solver) -> Float64

Mesh energy `½ Σ G|b̂|²` from the most recent energy evaluation (host read;
diagnostic only).
"""
mesh_energy(s::ElectrostaticsSolver) = Array(view(s.ws.acc, 1:1))[1]

"""
    self_energy(solver) -> Float64

`−(C/2) χ(0)/r_c Σ qᵢ²` for the current charges.
"""
self_energy(s::ElectrostaticsSolver) = -0.5 * Float64(s.C) * Float64(s.selfc) * s.qsq

"""
    background_energy(solver) -> Float64

`−C π r_c² m₂ Q²/V` when `net_charge=:background`, otherwise zero.
"""
background_energy(s::ElectrostaticsSolver) =
    s.spec.net_charge === :background ? -Float64(s.C) * s.bgcoef * s.qtotal^2 / prod(s.box) : 0.0

const _METHOD_CODE = Dict(:spme => 1.0, :p3m_ad => 2.0, :p3m_ik => 3.0, :esp => 4.0, :esp_deconvolution => 5.0)

"""
    settings_record(solver) -> Vector{Float64}

Numeric record of every setting needed to rebuild the solver exactly (stored
in GSD frames as `log/electrostatics/settings`):
`[version, method (1 SPME, 2 P3M-AD, 3 P3M-ik, 4 ESP, 5 ESP with plain
  deconvolution), coulomb_prefactor,
  relative_permittivity, real_cutoff, alpha_or_split_shape, order,
  window_shape, K₁, K₂, K₃, alias_order, net_charge (0 error, 1 background),
  precision bits, real_space_modifier (0 none, 1 potential shift)]`.
"""
function settings_record(s::ElectrostaticsSolver{T}) where {T}
    spec = s.spec
    shape = spec.split isa GaussianSplit ? spec.split.alpha : spec.split.shape
    wshape = spec.window isa ProlateWindow ? spec.window.shape : 0.0
    mesh = s.ws.mesh
    return Float64[1.0, _METHOD_CODE[spec.influence], spec.coulomb_prefactor, spec.relative_permittivity,
                   spec.real_cutoff, shape, spec.window.order, wshape, mesh[1], mesh[2], mesh[3],
                   spec.alias_order, spec.net_charge === :background ? 1.0 : 0.0, 8 * sizeof(T),
                   spec.real_space_modifier === :potential_shift ? 1.0 : 0.0]
end

"""
    exceptions_record(solver) -> Vector{Float64}

Flattened `(i, j, scale)` exception triples in stable particle ids.
"""
function exceptions_record(s::ElectrostaticsSolver)
    out = Float64[]
    for (i, j, sc) in s.exceptions
        push!(out, i, j, sc)
    end
    return out
end
