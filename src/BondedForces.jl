module BondedForces

using CUDA
using ..Definitions

export BondList, build_bondlist,
       harmonic_forces_soa!, harmonic_forces_soa_noE!,
        fene_forces_soa!, fene_forces_soa_noE!

# Periodic minimum image convention for centered coords [-L/2, L/2)
@inline function mic_fast(dx::Float32, halfL::Float32, L::Float32)
    dx -= (dx >  halfL) * L
    dx += (dx < -halfL) * L
    return dx
end

function _fene2_E!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    E::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    k::Float32, R0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    invR02 = 1f0 / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > 0f0
            denom = 1f0 - r2*invR02
            if denom < 1f-6
                denom = 1f-6
            end
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += 0.5f0 * (-0.5f0 * k * (R0*R0) * log(denom))
        end
    end
    fx[i] += accx; fy[i] += accy
    E[i] += eacc
    return
end

function _fene3_E!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    E::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    k::Float32, R0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    invR02 = 1f0 / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > 0f0
            denom = 1f0 - r2*invR02
            if denom < 1f-6
                denom = 1f-6
            end
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += 0.5f0 * (-0.5f0 * k * (R0*R0) * log(denom))
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    E[i] += eacc
    return
end
function _harmonic2_noE!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    k::Float32, r0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > 0f0
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] += accx; fy[i] += accy
    return
end

function _harmonic3_noE!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    k::Float32, r0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > 0f0
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    return
end
"""
CSR-style bond adjacency list on device.
- `index[i]` is the 0-based starting offset into `flat`
- `counts[i]` is the number of bonded neighbors for particle i
- `flat` stores 1-based particle ids of bonded neighbors
"""
struct BondList
    index::CuArray{Int32,1}
    flat::CuArray{Int32,1}
    counts::CuArray{Int32,1}
end

"""
    build_bondlist(N::Integer, bonds::Vector{<:Tuple}) -> BondList

Build a symmetric CSR-style adjacency for bonds. Each tuple (i,j) is 1-based.
"""
function build_bondlist(N::Integer, bonds)
    N = Int(N)
    deg = zeros(Int32, N)
    for (i,j) in bonds
        @assert 1 <= i <= N && 1 <= j <= N "bond index out of range"
        deg[Int(i)] += 1
        deg[Int(j)] += 1
    end
    # prefix sums for index
    index = similar(deg)
    offs = Int32(0)
    for i in 1:N
        index[i] = offs
        offs += deg[i]
    end
    total = Int(offs)
    flat = Vector{Int32}(undef, total)
    counts = zeros(Int32, N)
    # fill
    for (i,j) in bonds
        ii = Int(i); jj = Int(j)
        bi = index[ii] + counts[ii]
        flat[Int(bi)+1] = Int32(jj)
        counts[ii] += 1
        bj = index[jj] + counts[jj]
        flat[Int(bj)+1] = Int32(ii)
        counts[jj] += 1
    end
    return BondList(CUDA.CuArray(index), CUDA.CuArray(flat), CUDA.CuArray(counts))
end

# ---------------- Harmonic bond kernels ----------------

function _harmonic2_E!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    E::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    k::Float32, r0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; eacc = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > 0f0
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
            eacc += 0.5f0 * (0.5f0 * k * (r - r0) * (r - r0))
        end
    end
    fx[i] += accx; fy[i] += accy
    E[i] += eacc
    return
end

function _harmonic3_E!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    E::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    k::Float32, r0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; eacc = 0f0
    base = bindex[i]
    nb   = bcounts[i]
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > 0f0
            r = sqrt(r2)
            f_over_r = (-k + k * (r0 / r))
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
            eacc += 0.5f0 * (0.5f0 * k * (r - r0) * (r - r0))
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    E[i] += eacc
    return
end

function harmonic_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              bonds::BondList, box::Definitions.Box2,
                              params::Definitions.HarmonicBondParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _harmonic2_E!(rx, ry, fx, fy, Epot,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.k, params.r0)
    k(rx, ry, fx, fy, Epot,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.k, params.r0; threads=t, blocks=b)
    return nothing
end

function harmonic_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                                  bonds::BondList, box::Definitions.Box2,
                                  params::Definitions.HarmonicBondParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _harmonic2_noE!(rx, ry, fx, fy,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.k, params.r0)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.k, params.r0; threads=t, blocks=b)
    return nothing
end

function harmonic_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                              bonds::BondList, box::Definitions.Box3,
                              params::Definitions.HarmonicBondParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _harmonic3_E!(rx, ry, rz, fx, fy, fz, Epot,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.r0)
    k(rx, ry, rz, fx, fy, fz, Epot,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.r0; threads=t, blocks=b)
    return nothing
end

function harmonic_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                                  fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                                  bonds::BondList, box::Definitions.Box3,
                                  params::Definitions.HarmonicBondParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _harmonic3_noE!(rx, ry, rz, fx, fy, fz,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.r0)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.r0; threads=t, blocks=b)
    return nothing
end

# ---------------- FENE bond kernels ----------------

function _fene2_noE!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, halfLx::Float32, halfLy::Float32,
    k::Float32, R0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = 0f0; accy = 0f0; 
    base = bindex[i]
    nb   = bcounts[i]
    invR02 = 1f0 / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if r2 > 0f0
            denom = 1f0 - r2*invR02
            if denom < 1f-6
                denom = 1f-6
            end
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] += accx; fy[i] += accy
    return
end

function _fene3_noE!(
    rx::CuDeviceVector{Float32}, ry::CuDeviceVector{Float32}, rz::CuDeviceVector{Float32},
    fx::CuDeviceVector{Float32}, fy::CuDeviceVector{Float32}, fz::CuDeviceVector{Float32},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::Float32, Ly::Float32, Lz::Float32, halfLx::Float32, halfLy::Float32, halfLz::Float32,
    k::Float32, R0::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = 0f0; accy = 0f0; accz = 0f0; 
    base = bindex[i]
    nb   = bcounts[i]
    invR02 = 1f0 / (R0*R0)
    @inbounds for t in 0:Int(nb-1)
        j = bflat[base + t + 1]
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if r2 > 0f0
            denom = 1f0 - r2*invR02
            if denom < 1f-6
                denom = 1f-6
            end
            f_over_r = -k / denom
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] += accx; fy[i] += accy; fz[i] += accz
    return
end

function fene_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                          fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                          bonds::BondList, box::Definitions.Box2,
                          params::Definitions.FENEParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _fene2_E!(rx, ry, fx, fy, Epot,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.k, params.R0)
    k(rx, ry, fx, fy, Epot,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.k, params.R0; threads=t, blocks=b)
    return nothing
end

function fene_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1},
                              bonds::BondList, box::Definitions.Box2,
                              params::Definitions.FENEParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly
    k = CUDA.@cuda launch=false _fene2_noE!(rx, ry, fx, fy,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, halfLx, halfLy, params.k, params.R0)
    k(rx, ry, fx, fy,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, halfLx, halfLy, params.k, params.R0; threads=t, blocks=b)
    return nothing
end

function fene_forces_soa!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                          fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1}, Epot::CuArray{Float32,1},
                          bonds::BondList, box::Definitions.Box3,
                          params::Definitions.FENEParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _fene3_E!(rx, ry, rz, fx, fy, fz, Epot,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz, Epot,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.R0; threads=t, blocks=b)
    return nothing
end

function fene_forces_soa_noE!(rx::CuArray{Float32,1}, ry::CuArray{Float32,1}, rz::CuArray{Float32,1},
                              fx::CuArray{Float32,1}, fy::CuArray{Float32,1}, fz::CuArray{Float32,1},
                              bonds::BondList, box::Definitions.Box3,
                              params::Definitions.FENEParams{Float32})
    N = length(rx); t = (N < 100_000) ? 128 : 256; b = cld(N, t)
    Lx = Float32(box[1]); Ly = Float32(box[2]); Lz = Float32(box[3])
    halfLx = 0.5f0*Lx; halfLy = 0.5f0*Ly; halfLz = 0.5f0*Lz
    k = CUDA.@cuda launch=false _fene3_noE!(rx, ry, rz, fx, fy, fz,
        bonds.index, bonds.flat, bonds.counts,
        Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.R0)
    k(rx, ry, rz, fx, fy, fz,
      bonds.index, bonds.flat, bonds.counts,
      Lx, Ly, Lz, halfLx, halfLy, halfLz, params.k, params.R0; threads=t, blocks=b)
    return nothing
end

end # module
