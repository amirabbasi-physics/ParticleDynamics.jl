# Experimental fixed-cell ELL pipeline. Owns its list and scratch; the ordinary
# SimulationState path is unchanged until the graph/trajectory validation gate.
# Only device_neighbor_status performs a device-to-host transfer.
const DN_REQUEST = 1
const DN_VALID = 2
const DN_ERROR = 3
const DN_STEP = 4
const DN_LAST_BUILD = 5
const DN_INTERVAL = 6
const DN_CHECKS = 7
const DN_BUILDS = 8
const DN_REQUIRED = 9
const DN_REASON = 10 # bit mask for the currently requested build
const DN_ATTEMPTS = 11
const DN_DISPLACEMENT_BUILDS = 12
const DN_AGE_BUILDS = 13
const DN_INVALID_BUILDS = 14
const DN_SORT_BUILDS = 15 # overlapping subset, not a new policy trigger
const DN_OVERLAP_BUILDS = 16 # displacement and age were both satisfied
const DN_LAST_CANDIDATES = 17
const DN_TOTAL_CANDIDATES = 18
const DN_LAST_ACCEPTED = 19
const DN_TOTAL_ACCEPTED = 20
const DN_MAX_ROW = 21
const DN_DIAGNOSTICS = 22
const DN_SECTION = 23 # step section whose rebuild ran last (steps with several list checks; capacity recovery)
const DN_REASON_INVALID = 1
const DN_REASON_DISPLACEMENT = 2
const DN_REASON_AGE = 4
const DN_REASON_SORT = 8
const DN_CAPACITY_ERROR = 1
const DN_COORDINATE_ERROR = 2
const DN_FEATURE_ERROR = 3 # a device-engine feature check (e.g. FENE domain); the feature's boundary hook throws the details

mutable struct DeviceNeighborWorkspace{T,D}   # mutable: the box and cell-scan scratch follow box changes
    list::NeighborMatrix{T}
    box::NTuple{D,T}            # host copy of the box (decisions at host boundaries)
    geom::CuArray{T,1}          # device box lengths, read by captured kernels
    state::CuArray{Int64,1}
    displacement::CuArray{T,1}
    capacity::CuArray{Int32,1}
    scan_values::Vector{CuArray{Int32,1}}
    scan_totals::Vector{CuArray{Int32,1}}
    candidates::CuArray{Int64,1}
    statistics_partial::CuArray{Int64,2}
    row_threads::Int
    diagnostics::Bool
    allpairs::Bool     # small boxes: rows from all pairs, no cells (no sorting)
end

"""
    DeviceNeighborWorkspace(T, N, box; cutoff, skin, cap, step=0, interval=20)

Internal experimental GPU-owned dense ELL list, for a fixed 2D or 3D box and
centered coordinates. All storage is allocated here. Updates reuse it without
scalar readbacks or allocations. Rebuild decisions retain the existing strict
`max_displacement² > skin²/4` and adaptive-interval policy.

The list's host `valid` flag deliberately remains false: ordinary force entry
points must not consume it. Device consumers must use `_device_if_valid!`.
Errors latch until the workspace is discarded; no later rebuild clears them.
Read `device_neighbor_status` only at an explicit execution boundary.
"""
function DeviceNeighborWorkspace(::Type{T}, N::Integer, box::NTuple{D,T};
                                  cutoff, skin, cap::Integer, step::Integer=0,
                                  interval::Integer=20, row_threads::Integer=256,
                                  diagnostics::Bool=true, allpairs::Bool=false, margin::Real=1.0,
                                  sub::Integer=1) where {T<:AbstractFloat,D}
    D in (2,3) || throw(ArgumentError("Device ELL supports 2D/3D boxes"))
    row_threads in (64,128,256) || throw(ArgumentError("Row threads must be 64, 128 or 256"))
    0 <= step < typemax(Int64) || throw(ArgumentError("Invalid initial step"))
    1 <= interval <= 100 || throw(ArgumentError("Rebuild interval must be between 1 and 100"))
    0 < cap <= typemax(Int32) || throw(ArgumentError("Invalid ELL capacity"))
    isfinite(cutoff) && cutoff > 0 || throw(ArgumentError("Cutoff must be positive and finite"))
    nb = _alloc_neighbor_matrix(T, Int(N), D, box, cutoff, skin, Int32(cap); allpairs, margin, sub=D == 3 ? sub : 1)
    state = CuArray(vcat(Int64[0,0,0,step,step,interval,0,0,0],zeros(Int64,12),Int64(diagnostics),Int64(0)))
    values, totals = _scan_workspace(length(nb.cell_counts))
    return DeviceNeighborWorkspace{T,D}(nb, box, CuArray{T}(collect(box)), state,
        CuArray{T}(undef, cld(N,256)), CuArray{Int32}(undef, cld(N,256)), values, totals,
        CUDA.zeros(Int64,diagnostics ? N : 0),CUDA.zeros(Int64,diagnostics ? cld(N,256) : 0,2),
        Int(row_threads),diagnostics,allpairs)
end

# Multi-level scan scratch over n cells (one level per 256-fold reduction).
function _scan_workspace(n::Integer)
    values = CuArray{Int32,1}[]
    totals = CuArray{Int32,1}[]
    while true
        push!(values, CuArray{Int32}(undef, n))
        blocks = cld(n,256)
        push!(totals, CuArray{Int32}(undef, blocks))
        blocks == 1 && break
        n = blocks
    end
    return values, totals
end

"""
    set_device_box!(ws, box, coords) -> Bool

Change the periodic box (barostat events). The cell grid follows the box (new
cell storage only when the cell counts change) and the next coverage check rebuilds
the list for the new geometry ([`request_device_rebuild!`](@ref)); until then the list
stays valid, so validity-guarded kernels such as the drift still run. The caller
scales the coordinates. Returns whether storage was reallocated: graphs that
captured it must be re-instantiated.
"""
function set_device_box!(ws::DeviceNeighborWorkspace{T,D}, box::NTuple{D,T}, coords) where {T,D}
    nb = ws.list
    if ws.allpairs
        _validate_allpairs_radius(box, nb.cell_size)
        ws.box = box; copyto!(ws.geom, collect(box))
        request_device_rebuild!(ws, coords)
        return false
    end
    nx, ny, nz = _choose_grid(box, nb.cell_size, D)
    _validate_neighbor_grid(box, nb.cell_size, nx, ny, nz, D, "Device cell-list neighbor builder")
    # subdivided cells (width rlist/S) need 2S+1 cells per dimension
    S = round(Int, Float64(nb.cutoff + nb.skin) / Float64(nb.cell_size))
    S > 1 && any(n -> n < 2S + 1, (nx, ny, nz)[1:D]) && throw(ArgumentError(
        "The box became too small for the subdivided device cell grid ($(S) cells per list radius)"))
    realloc = (nx, ny, nz) != (nb.nx, nb.ny, nb.nz)
    if realloc
        ncell = Int(nx)*Int(ny)*Int(nz)
        nb.cell_offsets = CUDA.fill(Int32(1), ncell + 1)
        nb.cell_counts = CUDA.zeros(Int32, ncell)
        nb.nx, nb.ny, nb.nz = nx, ny, nz
        ws.scan_values, ws.scan_totals = _scan_workspace(ncell)
    end
    ws.box = box; copyto!(ws.geom, collect(box))
    request_device_rebuild!(ws, coords)
    return realloc
end

"""Make the next coverage check request a rebuild: the x references are set a
quarter box away from the current coordinates, so the measured displacement is
L/4 > skin/2 (also after later in-place coordinate changes smaller than that,
such as restoring a trial's coordinates); the rebuild stores fresh references.
Host, outside graph replay."""
request_device_rebuild!(ws::DeviceNeighborWorkspace, coords) = (ws.list.rref_x .= coords[1] .+ ws.box[1]/4; nothing)

@inline _device_rebuild_requested(s) = (@inbounds s[DN_REQUEST] != 0 && s[DN_ERROR] == 0)
@inline _device_list_valid(s) = (@inbounds s[DN_VALID] != 0 && s[DN_ERROR] == 0)

# The predicate is uniform across the grid and cannot change during a launch.
# Wrapped kernels may safely contain block barriers. Kernel boundaries provide
# the global ordering; never write the predicate from one of these consumers.
function _device_if_requested!(state, f::F, args...) where {F}
    if _device_rebuild_requested(state)
        # Call-site inlining keeps CuDeviceArray descriptors in kernel
        # parameters instead of copying them to a per-thread local stack.
        @inline f(args...)
    end
    return
end

function _device_if_valid!(state, f::F, args...) where {F}
    if _device_list_valid(state)
        @inline f(args...)
    end
    return
end

# Box lengths from a host tuple or the device geometry vector (kernels of graphs that a
# device barostat event may follow read the current box).
@inline _boxlen(b::Tuple, ::Val{D}) where {D} = b
@inline _boxlen(g::AbstractVector, ::Val{D}) where {D} = ntuple(i -> @inbounds(g[i]), Val(D))

function _device_displacement_kernel!(scratch::CuDeviceVector{T}, state, coords::NTuple{D},
                                      refs, box) where {T,D}
    box = _boxlen(box, Val(D))
    @inbounds state[DN_ERROR] != 0 && return
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    value = zero(T)
    if i <= length(coords[1])
        @inbounds begin
            valid = state[DN_VALID] != 0
            # Match the existing nested muladd order, including threshold ties.
            for d in D:-1:1
                x = coords[d][i]
                h = T(0.5)*box[d]
                if !(isfinite(x) && -h <= x <= h)
                    value = T(Inf)
                elseif valid
                    dx = mic_fast(x - refs[d][i], h, box[d])
                    value = muladd(dx,dx,value)
                end
            end
        end
    end
    largest = _displacement_block_max(value)
    if threadIdx().x == 1
        @inbounds scratch[blockIdx().x] = largest
    end
    return
end

function _device_decide_kernel!(state, scratch::CuDeviceVector{T}, threshold::T) where {T}
    @inbounds state[DN_ERROR] != 0 && return
    value = zero(T)
    i = threadIdx().x
    while i <= length(scratch)
        @inbounds value = max(value,scratch[i])
        i += blockDim().x
    end
    largest = _displacement_block_max(value)
    if threadIdx().x == 1
        @inbounds begin
            scratch[1] = largest
            state[DN_CHECKS] += 1
            state[DN_REQUEST] = 0
            diagnostic = state[DN_DIAGNOSTICS] != 0
            diagnostic && (state[DN_REASON] = 0)
            if !isfinite(largest)
                state[DN_ERROR] = DN_COORDINATE_ERROR
                state[DN_VALID] = 0
            elseif state[DN_VALID] == 0
                state[DN_REQUEST] = 1
                diagnostic && (state[DN_REASON] = DN_REASON_INVALID)
            elseif largest > threshold || state[DN_STEP]-state[DN_LAST_BUILD] >= state[DN_INTERVAL]
                state[DN_REQUEST] = 1
                state[DN_VALID] = 0
                if diagnostic
                    state[DN_REASON] = (largest > threshold ? DN_REASON_DISPLACEMENT : 0) |
                        (state[DN_STEP]-state[DN_LAST_BUILD] >= state[DN_INTERVAL] ? DN_REASON_AGE : 0)
                end
                if largest > threshold
                    state[DN_INTERVAL] = max(Int64(5), round(Int64,T(0.9)*state[DN_INTERVAL]))
                elseif largest < T(0.1)*threshold
                    state[DN_INTERVAL] = min(Int64(100), round(Int64,T(1.1)*state[DN_INTERVAL]))
                end
            end
        end
    end
    return
end

function _device_zero_kernel!(values)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(values)
        @inbounds values[i] = zero(eltype(values))
    end
    return
end

# Persistent hierarchical inclusive scan. Each level has its own values and
# block totals, allocated once. No CUDA.accumulate! temporary storage, views,
# dynamic launch sizes, or host reads are needed during graph capture/replay.
function _device_scan_tiles_kernel!(values, totals, input)
    t = threadIdx().x
    i = (blockIdx().x-1)*blockDim().x+t
    shared = CUDA.CuStaticSharedArray(Int32,256)
    @inbounds shared[t] = i <= length(input) ? input[i] : Int32(0)
    sync_threads()
    offset = 1
    while offset < 256
        @inbounds addend = t > offset ? shared[t-offset] : Int32(0)
        sync_threads()
        @inbounds shared[t] += addend
        sync_threads()
        offset *= 2
    end
    if i <= length(input)
        @inbounds values[i] = shared[t]
    end
    if t == 256
        @inbounds totals[blockIdx().x] = shared[t]
    end
    return
end

function _device_scan_add_kernel!(values, scanned_totals)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(values) && blockIdx().x > 1
        @inbounds values[i] += scanned_totals[blockIdx().x-1]
    end
    return
end

function _device_offsets_kernel!(offsets, counts, scanned)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(offsets)
        @inbounds offsets[i] = i == 1 ? Int32(1) : scanned[i-1]+Int32(1)
    end
    if i <= length(counts)
        @inbounds counts[i] = Int32(0) # scatter cursors reuse histogram storage
    end
    return
end

function _device_capacity_partial_kernel!(scratch, counts, candidates=nothing, totals=nothing)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    @inbounds value = i <= length(counts) ? counts[i] : Int32(0)
    largest = _displacement_block_max(value)
    if totals !== nothing
        sums=CUDA.CuStaticSharedArray(Int64,(256,2))
        t=threadIdx().x
        @inbounds begin
            sums[t,1]=i<=length(counts) ? candidates[i] : Int64(0)
            sums[t,2]=Int64(value)
        end
        sync_threads()
        stride=Int32(128)
        while stride>0
            if t<=stride
                @inbounds for q in 1:2; sums[t,q]+=sums[t+stride,q]; end
            end
            sync_threads()
            stride>>=1
        end
        if t==1
            @inbounds for q in 1:2; totals[blockIdx().x,q]=sums[1,q]; end
        end
    end
    if threadIdx().x == 1
        @inbounds scratch[blockIdx().x] = largest
    end
    return
end

function _device_commit_kernel!(state, scratch, cap::Int32, statistics=nothing)
    _device_rebuild_requested(state) || return
    value = Int32(0)
    i = threadIdx().x
    while i <= length(scratch)
        @inbounds value = max(value,scratch[i])
        i += blockDim().x
    end
    largest = _displacement_block_max(value)
    candidate_total=Int64(0); accepted_total=Int64(0)
    if statistics !== nothing
        j=threadIdx().x
        @inbounds while j<=size(statistics,1)
            candidate_total+=statistics[j,1]; accepted_total+=statistics[j,2]
            j+=blockDim().x
        end
        sums=CUDA.CuStaticSharedArray(Int64,(256,2))
        t=threadIdx().x
        @inbounds begin
            sums[t,1]=candidate_total; sums[t,2]=accepted_total
        end
        sync_threads()
        stride=Int32(128)
        while stride>0
            if t<=stride
                @inbounds for q in 1:2; sums[t,q]+=sums[t+stride,q]; end
            end
            sync_threads(); stride>>=1
        end
        @inbounds begin
            candidate_total=sums[1,1]; accepted_total=sums[1,2]
        end
    end
    if threadIdx().x == 1
        @inbounds begin
            state[DN_REQUIRED] = largest
            if statistics !== nothing
                state[DN_ATTEMPTS]+=1
                state[DN_LAST_CANDIDATES]=candidate_total
                state[DN_TOTAL_CANDIDATES]+=candidate_total
                state[DN_LAST_ACCEPTED]=accepted_total
                state[DN_TOTAL_ACCEPTED]+=accepted_total
                state[DN_MAX_ROW]=max(state[DN_MAX_ROW],Int64(largest))
            end
            if largest > cap
                state[DN_ERROR] = DN_CAPACITY_ERROR
                state[DN_VALID] = 0
            else
                state[DN_VALID] = 1
                state[DN_LAST_BUILD] = state[DN_STEP]
                state[DN_BUILDS] += 1
                if statistics !== nothing
                    reason=state[DN_REASON]
                    if reason & DN_REASON_INVALID != 0
                        state[DN_INVALID_BUILDS]+=1
                    elseif reason & DN_REASON_DISPLACEMENT != 0
                        state[DN_DISPLACEMENT_BUILDS]+=1
                    else
                        state[DN_AGE_BUILDS]+=1
                    end
                    state[DN_SORT_BUILDS]+=Int64(reason & DN_REASON_SORT != 0)
                    state[DN_OVERLAP_BUILDS]+=Int64(reason & (DN_REASON_DISPLACEMENT|DN_REASON_AGE) == (DN_REASON_DISPLACEMENT|DN_REASON_AGE))
                end
            end
        end
    end
    return
end

# `_device_if_valid!` that also advances the device step counter (block 1, thread 1, after its own
# work), for the last kernel of a step: no separate one-thread launch. The wrapped kernel must not
# read the step counter (other blocks may run after the increment).
function _device_if_valid_advance!(state, f::F, args...) where {F}
    if _device_list_valid(state)
        @inline f(args...)
        (blockIdx().x == 1 && threadIdx().x == 1) && (@inbounds state[DN_STEP] += 1)
    end
    return
end

function _device_neighbor_advance_kernel!(state)
    if _device_list_valid(state)
        @inbounds state[DN_STEP] += 1
    end
    return
end

function advance_device_neighbor_step!(ws::DeviceNeighborWorkspace)
    CUDA.@cuda threads=1 blocks=1 _device_neighbor_advance_kernel!(ws.state)
    return nothing
end

"""Enqueue coverage check and guarded rebuild; never reads a GPU flag on host.

All kernels are submitted, with the rebuild kernels returning immediately when
the device says keep. This is the predicated reference path. The experimental
Argon graph instead puts `_device_rebuild!` inside a CUDA conditional node.
Coordinates and workspace must stay alive, unmodified and at stable addresses
for the lifetime of any graph that captures this call.
"""
function update_device_neighbors!(ws::DeviceNeighborWorkspace{T,D}, coords::Vararg{CuArray{T,1},D}) where {T,D}
    decide_device_neighbors!(ws,coords...)
    nb = ws.list
    refs = D == 2 ? (nb.rref_x,nb.rref_y) : (nb.rref_x,nb.rref_y,nb.rref_z::CuArray{T,1})
    _device_rebuild!(ws,coords,refs)
    return nothing
end

function decide_device_neighbors!(ws::DeviceNeighborWorkspace{T,D}, coords::Vararg{CuArray{T,1},D}) where {T,D}
    nb = ws.list
    _validate_neighbor_arrays(nb, coords...)
    refs = D == 2 ? (nb.rref_x,nb.rref_y) : (nb.rref_x,nb.rref_y,nb.rref_z::CuArray{T,1})
    blocks = cld(Int(nb.N),256)
    CUDA.@cuda threads=256 blocks=blocks _device_displacement_kernel!(ws.displacement,ws.state,coords,refs,ws.geom)
    CUDA.@cuda threads=256 blocks=1 _device_decide_kernel!(ws.state,ws.displacement,T(0.25)*nb.skin*nb.skin)
    return nothing
end

function _device_bin!(ws::DeviceNeighborWorkspace{T,D}, coords) where {T,D}
    ws.allpairs && return nothing   # all-pairs rows need no cells
    nb = ws.list; s = ws.state; box = ws.box
    blocks = cld(Int(nb.N),256)
    CUDA.@cuda threads=256 blocks=cld(length(nb.cell_counts),256) _device_if_requested!(s,_device_zero_kernel!,nb.cell_counts)
    if D == 2
        CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_cell_ids2!,coords...,box...,inv(nb.cell_size),nb.nx,nb.ny,nb.cell_of_particle)
    else
        CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_cell_ids3g!,coords...,ws.geom,nb.nx,nb.ny,nb.nz,nb.cell_of_particle)
    end
    CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_cell_histogram!,nb.cell_of_particle,nb.cell_counts)
    for level in eachindex(ws.scan_values)
        input = level == 1 ? nb.cell_counts : ws.scan_totals[level-1]
        CUDA.@cuda threads=256 blocks=cld(length(input),256) _device_if_requested!(s,_device_scan_tiles_kernel!,ws.scan_values[level],ws.scan_totals[level],input)
    end
    for level in (length(ws.scan_values)-1):-1:1
        CUDA.@cuda threads=256 blocks=cld(length(ws.scan_values[level]),256) _device_if_requested!(s,_device_scan_add_kernel!,ws.scan_values[level],ws.scan_values[level+1])
    end
    CUDA.@cuda threads=256 blocks=cld(length(nb.cell_offsets),256) _device_if_requested!(s,_device_offsets_kernel!,nb.cell_offsets,nb.cell_counts,ws.scan_values[1])
    CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_scatter_by_cell!,nb.cell_of_particle,nb.cell_offsets,nb.cell_counts,nb.particle_ids_sorted)
    return nothing
end

# Device-box 3D binning: fractional cells ⌊x·n/L⌋ with the current box lengths, so the
# grid follows a barostat's box (cells stay ≥ the list radius while L ≥ margin·L_setup).
function _kernel_cell_ids3g!(rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T}, geom,
                             nx::Int32, ny::Int32, nz::Int32, cell_of_particle) where {T}
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i > length(rx) && return
    @inbounds begin
        Lx, Ly, Lz = _boxlen(geom, Val(3))
        x = rx[i] + T(0.5)*Lx; x -= floor(x/Lx)*Lx
        y = ry[i] + T(0.5)*Ly; y -= floor(y/Ly)*Ly
        z = rz[i] + T(0.5)*Lz; z -= floor(z/Lz)*Lz
        cx = unsafe_trunc(Int32, floor(x * (T(nx)/Lx))); cx = clamp(cx, Int32(0), nx-Int32(1))
        cy = unsafe_trunc(Int32, floor(y * (T(ny)/Ly))); cy = clamp(cy, Int32(0), ny-Int32(1))
        cz = unsafe_trunc(Int32, floor(z * (T(nz)/Lz))); cz = clamp(cz, Int32(0), nz-Int32(1))
        cell_of_particle[i] = (cz*ny + cy)*nx + cx
    end
    return
end
# Cell-row stencil of the device row kernels (15 Oct 2026). Cells have width h ≥ rlist/S
# per dimension (`sub` = S at allocation; S from the current box, ⌈rlist/h⌉, and at most
# (n−1)/2 so no cell is visited twice). For each (y, z) cell offset whose nearest distance
# lies within rlist, the x cells within reach are one contiguous range of the cell-sorted
# particles (two at the periodic seam), visited in increasing x offset. With S = 1 this is the
# 27-cell stencil in its visit order; S = 2 checks about 1.7× fewer candidates.
@inline _stencil_reach(rl::T, L::T, n::Int32) where {T} = n < Int32(3) ? Int32(1) :
    min(max(Int32(1), unsafe_trunc(Int32, ceil(rl * T(n) / L - T(1e-5)))), (n - Int32(1)) ÷ Int32(2))
@inline _stencil_gap(o::Int32, h::T) where {T} = T(max(abs(o) - Int32(1), Int32(0))) * h
@inline _wrap_cell(c::Int32, n::Int32) = (c -= (c >= n) * n; c += (c < Int32(0)) * n; c)
# x reach of row (oy, oz) and its one or two cell ranges [a, b] (an empty second range: a > b)
@inline function _row_ranges(cx::Int32, gyz2::T, rl2::T, hx::T, Sx::Int32, nx::Int32) where {T}
    ax = min(Sx, Int32(1) + unsafe_trunc(Int32, sqrt(max(rl2 - gyz2, zero(T))) / hx))
    lo = cx - ax; hi = cx + ax
    lo < Int32(0) && return (lo + nx, nx - Int32(1)), (Int32(0), hi)
    hi >= nx && return (lo, nx - Int32(1)), (Int32(0), hi - nx)
    return (lo, hi), (Int32(1), Int32(0))
end

# Cells per list radius at allocation (1: the 27-cell stencil kernels).
_cell_sub(nb) = round(Int, Float64(nb.cutoff + nb.skin) / Float64(nb.cell_size))

# Device-box rows: one thread per particle, the cell-row stencil with the current box lengths
# (S1: one cell per list radius, the 27-cell kernel).
function _kernel_neighbors3g!(rx::CuDeviceVector{T}, ry, rz, neighbors_index, neighbors_flat, counts, cell_offsets,
                              particle_ids_sorted, cell_of_particle, geom, nx::Int32, ny::Int32, nz::Int32, rl2::T,
                              cap::Int32, candidates, ::Val{S1}=Val(true)) where {T,S1}
    Lx, Ly, Lz = _boxlen(geom, Val(3))
    if S1
        @inline _kernel_neighbors3!(rx, ry, rz, neighbors_index, neighbors_flat, counts, cell_offsets, particle_ids_sorted,
                                    cell_of_particle, Lx, Ly, Lz, T(0.5)*Lx, T(0.5)*Ly, T(0.5)*Lz, nx, ny, nz, rl2, cap, candidates)
        return
    end
    halfLx = T(0.5) * Lx; halfLy = T(0.5) * Ly; halfLz = T(0.5) * Lz
    i1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); i1 > N && return
    @inbounds begin
        rl = sqrt(rl2); hx = Lx / T(nx); hy = Ly / T(ny); hz = Lz / T(nz)
        Sx = _stencil_reach(rl, Lx, nx); Sy = _stencil_reach(rl, Ly, ny); Sz = _stencil_reach(rl, Lz, nz)
        c0 = cell_of_particle[i1]
        cx = c0 % nx; tmp = c0 ÷ nx; cy = tmp % ny; cz = tmp ÷ ny
        xi = rx[i1]; yi = ry[i1]; zi = rz[i1]
        found = Int32(0); tested = Int64(0)
        for oz in -Sz:Sz
            cz2 = _wrap_cell(cz + oz, nz); gz = _stencil_gap(oz, hz)
            for oy in -Sy:Sy
                cy2 = _wrap_cell(cy + oy, ny); gy = _stencil_gap(oy, hy)
                g2 = muladd(gy, gy, gz * gz)
                g2 > rl2 && continue
                row = (cz2 * ny + cy2) * nx
                rg1, rg2 = _row_ranges(cx, g2, rl2, hx, Sx, nx)
                for seg in 1:2
                    a, b = seg == 1 ? rg1 : rg2
                    a > b && continue
                    for k in cell_offsets[row + a + 1]:(cell_offsets[row + b + 2] - 1)
                        j = particle_ids_sorted[k]
                        if j != i1
                            candidates === nothing || (tested += 1)
                            dx = mic_fast(rx[j] - xi, halfLx, Lx)
                            dy = mic_fast(ry[j] - yi, halfLy, Ly)
                            dz = mic_fast(rz[j] - zi, halfLz, Lz)
                            r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
                            if r2 <= rl2
                                found < cap && (neighbors_flat[_ell_index(i1, found, N)] = j)
                                found += 1
                            end
                        end
                    end
                end
            end
        end
        counts[i1] = found
        candidates === nothing || (candidates[i1] = tested)
    end
    return
end

# Separate binning from row construction so an owning graph can permute
# particle storage in-place at this boundary, before any new rows exist.
function _device_finish_rebuild!(ws::DeviceNeighborWorkspace{T,D}, coords, refs) where {T,D}
    nb = ws.list; s = ws.state; box = ws.box
    blocks = cld(Int(nb.N),256)
    halfbox = map(x -> T(0.5)*x,box)
    rl2 = (nb.cutoff+nb.skin)^2
    row_threads=ws.row_threads; row_blocks=cld(Int(nb.N),row_threads)
    candidates=ws.diagnostics ? ws.candidates : nothing
    statistics=ws.diagnostics ? ws.statistics_partial : nothing
    if ws.allpairs
        CUDA.@cuda threads=row_threads blocks=row_blocks _device_if_requested!(s,_kernel_neighbors_allpairs!,coords,nb.neighbors_flat,nb.counts,box,halfbox,rl2,nb.cap,candidates)
    elseif D == 2
        CUDA.@cuda threads=row_threads blocks=row_blocks _device_if_requested!(s,_kernel_neighbors2!,coords...,nb.neighbors_index,nb.neighbors_flat,nb.counts,nb.cell_offsets,nb.particle_ids_sorted,nb.cell_of_particle,box...,halfbox...,nb.nx,nb.ny,rl2,nb.cap,candidates)
    else
        CUDA.@cuda threads=row_threads blocks=row_blocks _device_if_requested!(s,_kernel_neighbors3g!,coords...,nb.neighbors_index,nb.neighbors_flat,nb.counts,nb.cell_offsets,nb.particle_ids_sorted,nb.cell_of_particle,ws.geom,nb.nx,nb.ny,nb.nz,rl2,nb.cap,candidates,Val(_cell_sub(nb)==1))
    end
    CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_device_capacity_partial_kernel!,ws.capacity,nb.counts,candidates,statistics)
    CUDA.@cuda threads=256 blocks=1 _device_commit_kernel!(s,ws.capacity,nb.cap,statistics)
    if D == 2
        CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_copy_refs_2d!,coords...,refs...)
    else
        CUDA.@cuda threads=256 blocks=blocks _device_if_requested!(s,_kernel_copy_refs_3d!,coords...,refs...)
    end
    return nothing
end

"""Boundary-only diagnostic readback. Causes are successful builds: invalid
first, then displacement, then age (exclusive precedence); sorting and joint
displacement/age are overlapping subsets. Attempt totals include a failed
capacity build and count directed distance tests/pairs before truncation.
Sorting never independently requests a build. No policy changes are made.
"""
function device_neighbor_diagnostics(ws::DeviceNeighborWorkspace;check::Bool=true)
    status=device_neighbor_status(ws;check)
    s=Array(ws.state)
    return (enabled=ws.diagnostics,builds=status.builds,attempts=s[DN_ATTEMPTS],
        displacement_builds=s[DN_DISPLACEMENT_BUILDS],age_builds=s[DN_AGE_BUILDS],
        invalid_builds=s[DN_INVALID_BUILDS],sorting_builds=s[DN_SORT_BUILDS],
        displacement_age_builds=s[DN_OVERLAP_BUILDS],last_candidates=s[DN_LAST_CANDIDATES],
        total_candidates=s[DN_TOTAL_CANDIDATES],last_accepted=s[DN_LAST_ACCEPTED],
        total_accepted=s[DN_TOTAL_ACCEPTED],last_max_row=status.required,
        max_row=s[DN_MAX_ROW],row_threads=ws.row_threads)
end

function _device_rebuild!(ws::DeviceNeighborWorkspace, coords, refs)
    _device_bin!(ws,coords)
    _device_finish_rebuild!(ws,coords,refs)
    return nothing
end

# Capacity recovery (device engine): sections mark themselves before rebuilding;
# after a capacity overflow the error is cleared and the failed rebuild requested again.
function _mark_section_kernel!(state,k::Int64)
    threadIdx().x==1 && blockIdx().x==1 && (@inbounds state[DN_SECTION]=k)
    return nothing
end
mark_section!(ws::DeviceNeighborWorkspace,k::Integer)=(CUDA.@cuda threads=1 blocks=1 _mark_section_kernel!(ws.state,Int64(k)); nothing)
function _rearm_rebuild_kernel!(state)
    if threadIdx().x==1 && blockIdx().x==1
        @inbounds begin state[DN_ERROR]=0; state[DN_REQUEST]=1 end
    end
    return nothing
end
"""Clear a latched capacity error and request the failed rebuild again (host, at a boundary)."""
function rearm_capacity_rebuild!(ws::DeviceNeighborWorkspace)
    CUDA.@cuda threads=1 blocks=1 _rearm_rebuild_kernel!(ws.state)
    return nothing
end
"""Replace the list storage by one with capacity `cap` (rows are rebuilt by the next
rebuild; kernels and graphs that captured the old storage must be recaptured)."""
function resize_capacity!(ws::DeviceNeighborWorkspace,cap::Integer)
    nb=ws.list; N=Int(nb.N)
    0<cap<=typemax(Int32) || throw(ArgumentError("Invalid ELL capacity"))
    nb.neighbors_flat=CUDA.fill(Int32(-1),N*Int(cap))
    nb.neighbors_index=CuArray(Int32[(i-1)*Int(cap) for i in 1:N])
    nb.cap=Int32(cap)
    return nothing
end
section_of(ws::DeviceNeighborWorkspace)=max(1,Int(only(Array(view(ws.state,DN_SECTION:DN_SECTION)))))

"""Explicit batch-boundary synchronization. Throws on latched device errors."""
function device_neighbor_status(ws::DeviceNeighborWorkspace; check::Bool=true)
    s = Array(ws.state)
    if check
        s[DN_ERROR] == DN_CAPACITY_ERROR && throw(NeighborCapacityError(Int(ws.list.cap),Int(s[DN_REQUIRED])))
        s[DN_ERROR] == DN_COORDINATE_ERROR && throw(ArgumentError("Device ELL requires finite, centered coordinates; execution stopped before rebuilding/forces"))
        s[DN_ERROR] == DN_FEATURE_ERROR && error("Device engine stopped by a feature check (e.g. FENE bond domain or bonded geometry); the engine boundary reports the details")
        s[DN_ERROR] == 0 || error("Device engine stopped with unknown error code $(s[DN_ERROR])")
        s[DN_VALID] != 0 || throw(ArgumentError("Device ELL has no valid list"))
    end
    return (valid=s[DN_VALID]!=0, error=s[DN_ERROR], requested=s[DN_REQUEST]!=0,
        step=s[DN_STEP],last_build=s[DN_LAST_BUILD],interval=s[DN_INTERVAL],
        checks=s[DN_CHECKS],builds=s[DN_BUILDS],required=s[DN_REQUIRED])
end
