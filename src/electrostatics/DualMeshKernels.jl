# ==============================================================================
#   Fused Dual-Mesh (q, √c6) Pipeline for Co-Grid Electrostatics + Dispersion
# ==============================================================================
# When Coulomb (1/r) and Dispersion (1/r⁶) share the same 3D mesh `(K₁, K₂, K₃)`
# and window stencil `S`, one particle traversal evaluates the 3P window weights
# once per particle, spreads `(qᵢ, s6ᵢ)` into a batched `(K₁, K₂, K₃, 2)` grid,
# runs a single batched 2-field cuFFT R2C/C2R pair, solves both spectra in one
# pass, and gathers `−qᵢ ∇φ_q − s6ᵢ ∇φ_d` in one pass.

mutable struct DualMeshWorkspace{T,PF,PB}
    mesh::NTuple{3,Int}
    grid2::CuArray{T,4}               # (K1, K2, K3, 2)
    grid_q_flat::SubArray{T,1,CuArray{T,1},Tuple{UnitRange{Int}},true}
    grid_d_flat::SubArray{T,1,CuArray{T,1},Tuple{UnitRange{Int}},true}
    spec2::CuArray{Complex{T},4}      # (K1÷2+1, K2, K3, 2)
    spec_q_flat::SubArray{Complex{T},1,CuArray{Complex{T},1},Tuple{UnitRange{Int}},true}
    spec_d_flat::SubArray{Complex{T},1,CuArray{Complex{T},1},Tuple{UnitRange{Int}},true}
    plan_fwd::PF
    plan_bwd::PB
    acc::CuArray{Float64,1}           # 7-element combined energy + 6 virial accumulator
end

function DualMeshWorkspace(::Type{T}, mesh::NTuple{3,Int}) where {T<:AbstractFloat}
    K1, K2, K3 = mesh
    K1h = K1 ÷ 2 + 1
    nnodes = K1 * K2 * K3
    nmodes = K1h * K2 * K3
    grid2 = CUDA.zeros(T, K1, K2, K3, 2)
    spec2 = CUDA.zeros(Complex{T}, K1h, K2, K3, 2)
    gflat = reshape(grid2, :)
    sflat = reshape(spec2, :)
    gq = view(gflat, 1:nnodes)
    gd = view(gflat, (nnodes + 1):(2 * nnodes))
    sq = view(sflat, 1:nmodes)
    sd = view(sflat, (nmodes + 1):(2 * nmodes))
    plan_fwd = CUDA.CUFFT.plan_rfft(grid2, (1, 2, 3))
    plan_bwd = CUDA.CUFFT.plan_brfft(spec2, K1, (1, 2, 3))
    acc = CUDA.zeros(Float64, 7)
    return DualMeshWorkspace(mesh, grid2, gq, gd, spec2, sq, sd, plan_fwd, plan_bwd, acc)
end

const _DUAL_WS_CACHE = WeakKeyDict{Any,Any}()

function dual_mesh_compatible(sq::ElectrostaticsSolver{T}, sd::DispersionSolver{T}) where {T}
    !is_ik(sq) && sq.ws.mesh == sd.ws.mesh && typeof(sq.stencil) === typeof(sd.stencil) &&
        sq.stencil == sd.stencil
end

function get_dual_workspace(sq::ElectrostaticsSolver{T}, sd::DispersionSolver{T}) where {T}
    w = get!(_DUAL_WS_CACHE, sq) do
        DualMeshWorkspace(T, sq.ws.mesh)
    end
    if w.mesh != sq.ws.mesh
        w = DualMeshWorkspace(T, sq.ws.mesh)
        _DUAL_WS_CACHE[sq] = w
    end
    return w::DualMeshWorkspace{T}
end

function _spread_dual_lines_kernel!(grid_q, grid_d, rx, ry, rz, q, s6, order,
                                    box, K1::Int32, K2::Int32, K3::Int32, stencil::S) where {S}
    T = eltype(rx)
    P = _stencil_order(stencil)
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    active = t <= length(order)
    @inbounds begin
        i = order[active ? t : Int32(1)]
        qi = active ? q[i] : zero(T)
        s6i = active ? s6[i] : zero(T)
        xi = rx[i]; yi = ry[i]; zi = rz[i]
        u1 = _grid_coordinate(xi, Lx, K1); u2 = _grid_coordinate(yi, Ly, K2); u3 = _grid_coordinate(zi, Lz, K3)
        wl, _, o = _lane_weights(stencil, u1, u2, u3, lane)
        k1 = _window_base(stencil, u1); k2 = _window_base(stencil, u2); k3 = _window_base(stencil, u3)
        w3 = ntuple(k -> _lane_shfl(wl[3], o[3] + Int32(k)), Val(P))
        stride = K1 * K2
        for r in 0:cld(P * P, MESH_LANES) - 1
            l = lane + Int32(r * MESH_LANES)
            j1 = l % Int32(P); j2 = min(l ÷ Int32(P), Int32(P - 1))
            wx = _lane_shfl(wl[1], o[1] + j1 + Int32(1)); wy = _lane_shfl(wl[2], o[2] + j2 + Int32(1))
            if active && l < Int32(P * P)
                n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
                base = n1 + K1 * n2 + Int32(1)
                wxy = wx * wy
                qi != zero(T) && _zspread!(grid_q, base, stride, k3, K3, qi * wxy, w3)
                s6i != zero(T) && _zspread!(grid_d, base, stride, k3, K3, s6i * wxy, w3)
            end
        end
    end
    return
end

function _solve_dual_kernel!(spec_q, spec_d, G_q, Vc_q, G_d, Vc_d, acc::CuDeviceVector{Float64},
                             K1::Int32, K1h::Int32, nmodes::Int32, ::Val{ENERGY}) where {ENERGY}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    e = 0.0; v1 = 0.0; v2 = 0.0; v3 = 0.0; v4 = 0.0; v5 = 0.0; v6 = 0.0
    if idx <= nmodes
        @inbounds begin
            zq = spec_q[idx]; Gmq = G_q[idx]
            zd = spec_d[idx]; Gmd = G_d[idx]
            if ENERGY
                m1 = (idx - Int32(1)) % K1h
                w = (m1 == Int32(0)) | (iseven(K1) & (m1 == K1 ÷ Int32(2))) ? 0.5 : 1.0
                aq = w * Float64(abs2(zq))
                ad = w * Float64(abs2(zd))
                e = aq * Float64(Gmq) + ad * Float64(Gmd)
                v1 = aq * Float64(Vc_q[1, idx]) + ad * Float64(Vc_d[1, idx])
                v2 = aq * Float64(Vc_q[2, idx]) + ad * Float64(Vc_d[2, idx])
                v3 = aq * Float64(Vc_q[3, idx]) + ad * Float64(Vc_d[3, idx])
                v4 = aq * Float64(Vc_q[4, idx]) + ad * Float64(Vc_d[4, idx])
                v5 = aq * Float64(Vc_q[5, idx]) + ad * Float64(Vc_d[5, idx])
                v6 = aq * Float64(Vc_q[6, idx]) + ad * Float64(Vc_d[6, idx])
            end
            spec_q[idx] = zq * Gmq
            spec_d[idx] = zd * Gmd
        end
    end
    ENERGY && _block_accumulate7!(acc, (e, v1, v2, v3, v4, v5, v6))
    return
end

function _gather_ad_dual_lines_kernel!(fx, fy, fz, Epot, rx, ry, rz, q, s6, order,
                                       grid_q, grid_d, box, K1::Int32, K2::Int32, K3::Int32,
                                       stencil::S, bgq_v::Float64, bgd_v::Float64,
                                       ::Val{ENERGY}, ::Val{SET}=Val(false)) where {S,ENERGY,SET}
    T = eltype(rx)
    P = _stencil_order(stencil)
    Lx, Ly, Lz = Definitions.box_lengths(box, Val(3))
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    t = tid ÷ Int32(MESH_LANES) + Int32(1); lane = tid % Int32(MESH_LANES)
    active = t <= length(order)
    phiq = zero(T); gqx = zero(T); gqy = zero(T); gqz = zero(T)
    phid = zero(T); gdx = zero(T); gdy = zero(T); gdz = zero(T)
    @inbounds begin
        i = order[active ? t : Int32(1)]
        qi = active ? q[i] : zero(T)
        s6i = active ? s6[i] : zero(T)
        u1 = _grid_coordinate(rx[i], Lx, K1); u2 = _grid_coordinate(ry[i], Ly, K2); u3 = _grid_coordinate(rz[i], Lz, K3)
        wl, dl, o = _lane_weights(stencil, u1, u2, u3, lane)
        k1 = _window_base(stencil, u1); k2 = _window_base(stencil, u2); k3 = _window_base(stencil, u3)
        w3 = ntuple(k -> _lane_shfl(wl[3], o[3] + Int32(k)), Val(P))
        d3 = ntuple(k -> _lane_shfl(dl[3], o[3] + Int32(k)), Val(P))
        stride = K1 * K2
        has_any = active && ((qi != zero(T)) | (s6i != zero(T)))
        for r in 0:cld(P * P, MESH_LANES) - 1
            l = lane + Int32(r * MESH_LANES)
            j1 = l % Int32(P); j2 = min(l ÷ Int32(P), Int32(P - 1))
            wx = _lane_shfl(wl[1], o[1] + j1 + Int32(1)); dx = _lane_shfl(dl[1], o[1] + j1 + Int32(1))
            wy = _lane_shfl(wl[2], o[2] + j2 + Int32(1)); dy = _lane_shfl(dl[2], o[2] + j2 + Int32(1))
            if has_any && l < Int32(P * P)
                n1 = _wrap_node(k1 + j1, K1); n2 = _wrap_node(k2 + j2, K2)
                base = n1 + K1 * n2 + Int32(1)
                w12 = wx * wy; dxwy = dx * wy; wxdy = wx * dy
                if qi != zero(T)
                    sq0, sqz = _zgather(grid_q, base, stride, k3, K3, w3, d3)
                    phiq = muladd(w12, sq0, phiq); gqz = muladd(w12, sqz, gqz)
                    gqx = muladd(dxwy, sq0, gqx); gqy = muladd(wxdy, sq0, gqy)
                end
                if s6i != zero(T)
                    sd0, sdz = _zgather(grid_d, base, stride, k3, K3, w3, d3)
                    phid = muladd(w12, sd0, phid); gdz = muladd(w12, sdz, gdz)
                    gdx = muladd(dxwy, sd0, gdx); gdy = muladd(wxdy, sd0, gdy)
                end
            end
        end
        gqx = _sum16(gqx); gqy = _sum16(gqy); gqz = _sum16(gqz)
        gdx = _sum16(gdx); gdy = _sum16(gdy); gdz = _sum16(gdz)
        if ENERGY
            phiq = _sum16(phiq); phid = _sum16(phid)
        end
        if lane == Int32(0) && active
            fmx = -(qi * gqx + s6i * gdx) * (T(K1) / Lx)
            fmy = -(qi * gqy + s6i * gdy) * (T(K2) / Ly)
            fmz = -(qi * gqz + s6i * gdz) * (T(K3) / Lz)
            if SET
                fx[i] = fmx; fy[i] = fmy; fz[i] = fmz
            elseif (qi != zero(T)) | (s6i != zero(T))
                fx[i] += fmx; fy[i] += fmy; fz[i] += fmz
            end
            if ENERGY
                invV = 1.0 / (Float64(Lx) * Float64(Ly) * Float64(Lz))
                em = T(0.5) * (qi * phiq + s6i * phid) + T(bgq_v * invV) * qi + T(bgd_v * invV) * s6i
                Epot[i] = SET ? em : (Epot[i] + em)
            end
        end
    end
    return
end

function _store_dual_global_virial_kernel!(Vg, acc, geom, ubg_v::Float64)
    i = threadIdx().x
    if i <= 6
        L1, L2, L3 = map(Float64, Definitions.box_lengths(geom, Val(3)))
        ubg = ubg_v / (L1 * L2 * L3)
        @inbounds Vg[i] = eltype(Vg)(acc[i + 1] + (i <= 3 ? ubg : 0.0))
    end
    return
end

function dual_mesh_transform!(dw::DualMeshWorkspace{T}, sq::ElectrostaticsSolver{T}, sd::DispersionSolver{T},
                              rx, ry, rz, q, s6, compute_energy::Bool, order; geom=nothing) where {T}
    K1, K2, K3 = Int32.(dw.mesh)
    Lb = geom === nothing ? T.(sq.box) : geom
    N = length(rx)
    fill!(dw.grid2, zero(T))
    @cuda threads=128 blocks=cld(N * MESH_LANES, 128) _spread_dual_lines_kernel!(
        dw.grid_q_flat, dw.grid_d_flat, rx, ry, rz, q, s6, order, Lb, K1, K2, K3, sq.stencil)
    mul!(dw.spec2, dw.plan_fwd, dw.grid2)
    nmodes = Int32(length(sq.ws.G))
    compute_energy && fill!(dw.acc, 0.0)
    mthreads, mblocks = _launch1d(nmodes, 256)
    @cuda threads=mthreads blocks=mblocks _solve_dual_kernel!(
        dw.spec_q_flat, dw.spec_d_flat, sq.ws.G, sq.ws.Vc, sd.ws.G, sd.ws.Vc,
        dw.acc, K1, K1 ÷ Int32(2) + Int32(1), nmodes, Val(compute_energy))
    _c2r!(dw.grid2, dw.plan_bwd, dw.spec2)
    return nothing
end

function _store_dual_global_virial!(Vglobal, dw::DualMeshWorkspace, sq::ElectrostaticsSolver,
                                    sd::DispersionSolver, geom)
    Lb = geom === nothing ? sq.box : geom
    ubg_q_v = sq.spec.net_charge === :background ? (-Float64(sq.C) * sq.bgcoef * (sq.qtotal^2)) : 0.0
    ubg_d_v = -sd.zcoef * (sd.s6total^2)
    @cuda threads=6 blocks=1 _store_dual_global_virial_kernel!(Vglobal, dw.acc, Lb, ubg_q_v + ubg_d_v)
    return nothing
end

function dual_mesh_gather!(dw::DualMeshWorkspace{T}, sq::ElectrostaticsSolver{T}, sd::DispersionSolver{T},
                           rx, ry, rz, q, s6, fx, fy, fz, Epot, Vglobal,
                           compute_energy::Bool, order; geom=nothing, set::Bool=false) where {T}
    K1, K2, K3 = Int32.(dw.mesh)
    Lb = geom === nothing ? T.(sq.box) : geom
    bgq_v = sq.spec.net_charge === :background ? (-Float64(sq.C) * sq.bgcoef * sq.qtotal) : 0.0
    bgd_v = -sd.zcoef * sd.s6total
    @cuda threads=128 blocks=cld(length(rx) * MESH_LANES, 128) _gather_ad_dual_lines_kernel!(
        fx, fy, fz, Epot, rx, ry, rz, q, s6, order, dw.grid_q_flat, dw.grid_d_flat,
        Lb, K1, K2, K3, sq.stencil, bgq_v, bgd_v, Val(compute_energy), Val(set))
    compute_energy && _store_dual_global_virial!(Vglobal, dw, sq, sd, geom)
    return nothing
end
