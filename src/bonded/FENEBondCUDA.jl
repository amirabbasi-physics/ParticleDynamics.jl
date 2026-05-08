# ------------------------------------------------------------------
# FENE kernels
# ------------------------------------------------------------------

function _fene2_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    accx = zero(T)
    accy = zero(T)
    eacc = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += T(0.5) * (-T(0.5) * k * (R0 * R0) * log(denom))
        end
    end
    fx[i] += accx
    fy[i] += accy
    E[i] += eacc
    return
end

function _fene2_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    accx = zero(T)
    accy = zero(T)
    eacc = zero(T)
    vxx = zero(T)
    vyy = zero(T)
    vxy = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            accx += fxij
            accy += fyij
            eacc += T(0.5) * (-T(0.5) * k * (R0 * R0) * log(denom))
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx
            vyy += dvyy
            vxy += dvxy
        end
    end
    fx[i] += accx
    fy[i] += accy
    E[i] += eacc
    V[i, 1] = vxx
    V[i, 2] = vyy
    V[i, 3] = vxy
    return
end

function _fene3_E!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    zi = rz[i]
    accx = zero(T)
    accy = zero(T)
    accz = zero(T)
    eacc = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += T(0.5) * (-T(0.5) * k * (R0 * R0) * log(denom))
        end
    end
    fx[i] += accx
    fy[i] += accy
    fz[i] += accz
    E[i] += eacc
    return
end

function _fene3_EV!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, E::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    zi = rz[i]
    accx = zero(T)
    accy = zero(T)
    accz = zero(T)
    eacc = zero(T)
    vxx = zero(T)
    vyy = zero(T)
    vzz = zero(T)
    vxy = zero(T)
    vxz = zero(T)
    vyz = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            fxij = f_over_r * dx
            fyij = f_over_r * dy
            fzij = f_over_r * dz
            accx += fxij
            accy += fyij
            accz += fzij
            eacc += T(0.5) * (-T(0.5) * k * (R0 * R0) * log(denom))
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx
            vyy += dvyy
            vzz += dvzz
            vxy += dvxy
            vxz += dvxz
            vyz += dvyz
        end
    end
    fx[i] += accx
    fy[i] += accy
    fz[i] += accz
    E[i] += eacc
    V[i, 1] = vxx
    V[i, 2] = vyy
    V[i, 3] = vzz
    V[i, 4] = vxy
    V[i, 5] = vxz
    V[i, 6] = vyz
    return
end

function _fene2_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    accx = zero(T)
    accy = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy * dy)
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] += accx
    fy[i] += accy
    return
end

function _fene3_noE!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    index::CuDeviceVector{Int32}, flat::CuDeviceVector{Int32}, counts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    k::T, R0::T) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx)
    if i > N
        return
    end
    xi = rx[i]
    yi = ry[i]
    zi = rz[i]
    accx = zero(T)
    accy = zero(T)
    accz = zero(T)
    base = index[i]
    nb = counts[i]
    invR02 = one(T) / (R0 * R0)
    @inbounds for t in 0:Int(nb - 1)
        j = flat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
        if r2 > zero(T)
            denom = one(T) - r2 * invR02
            denom = max(denom, T(1e-6))
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] += accx
    fy[i] += accy
    fz[i] += accz
    return
end

# ------------------------------------------------------------------
# Public wrappers
# ------------------------------------------------------------------

"""
    fene_forces_soa!(rx, ry[, rz], fx, fy[, fz], E, bonds, box, params)

Finite extensible nonlinear elastic bonds. Matches the `fene_bond(k=300, r0=1.5)`
configuration used in `examples/2D_polymer_bonded_BP.jl`.
"""
function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _fene2_E!(
        rx, ry, fx, fy, E,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.k, params.R0)
    k(rx, ry, fx, fy, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _fene2_EV!(
        rx, ry, fx, fy, E, V,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.k, params.R0)
    k(rx, ry, fx, fy, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

"""
    fene_forces_soa_noE!(rx, ry[, rz], fx, fy[, fz], bonds, box, params)

FENE bonds without per-particle energy accumulation.
"""
function fene_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1},
    bonds::BondList, box::Definitions.Box2{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    k = CUDA.@cuda launch=false _fene2_noE!(
        rx, ry, fx, fy,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy,
        params.k, params.R0)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _fene3_E!(
        rx, ry, rz, fx, fy, fz, E,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz, E,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, E::CuArray{T,1}, V::CuArray{T,2},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _fene3_EV!(
        rx, ry, rz, fx, fy, fz, E, V,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz, E, V,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end

function fene_forces_soa_noE!(
    rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
    fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
    bonds::BondList, box::Definitions.Box3{T},
    params::Definitions.FENEParams{T}) where {T<:AbstractFloat}
    N = length(rx)
    threads = min(_bond_threads(N), N)
    blocks = cld(N, threads)
    Lx, Ly, Lz = box
    halfLx = T(0.5) * Lx
    halfLy = T(0.5) * Ly
    halfLz = T(0.5) * Lz
    k = CUDA.@cuda launch=false _fene3_noE!(
        rx, ry, rz, fx, fy, fz,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz,
        params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz,
      params.k, params.R0; threads, blocks)
    return nothing
end
