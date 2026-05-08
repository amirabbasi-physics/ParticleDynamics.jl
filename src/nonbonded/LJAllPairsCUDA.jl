# ───────────────────────────────────────────────────────────────────────────────
# All-pairs (no neighbor list) kernels
# ───────────────────────────────────────────────────────────────────────────────

function _lj2_allpairs_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj2_allpairs_kernel_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _lj3_allpairs_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj3_allpairs_kernel_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx; vyy += dvyy; vzz += dvzz
            vxy += dvxy; vxz += dvxz; vyz += dvyz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _lj2_allpairs_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_allpairs_noE_kernel!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    @inbounds for j in 1:N
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

function _lj2_allpairs_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    return
end

function _lj2_allpairs_kernel_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vxy = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, ep = lj_pair_2d(dx, dy, r2, ϵ, σ)
            accx += fxij; accy += fyij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvxy = _half_virial2(dx, dy, fxij, fyij)
            vxx += dvxx; vyy += dvyy; vxy += dvxy
        end
    end
    fx[i] = accx; fy[i] = accy; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vxy
    return
end

function _lj3_allpairs_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    return
end

function _lj3_allpairs_kernel_excl_virial!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T}, Epot::CuDeviceVector{T}, V::CuDeviceMatrix{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T); eacc = zero(T)
    vxx = zero(T); vyy = zero(T); vzz = zero(T); vxy = zero(T); vxz = zero(T); vyz = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            fxij, fyij, fzij, ep = lj_pair_3d(dx, dy, dz, r2, ϵ, σ)
            accx += fxij; accy += fyij; accz += fzij
            eacc += T(0.5) * ep
            dvxx, dvyy, dvzz, dvxy, dvxz, dvyz = _half_virial3(dx, dy, dz, fxij, fyij, fzij)
            vxx += dvxx; vyy += dvyy; vzz += dvzz
            vxy += dvxy; vxz += dvxz; vyz += dvyz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz; Epot[i] = eacc
    V[i, 1] = vxx; V[i, 2] = vyy; V[i, 3] = vzz
    V[i, 4] = vxy; V[i, 5] = vxz; V[i, 6] = vyz
    return
end

function _lj2_allpairs_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, halfLx::T, halfLy::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    accx = zero(T); accy = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        r2 = muladd(dx, dx, dy*dy)
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
        end
    end
    fx[i] = accx; fy[i] = accy
    return
end

function _lj3_allpairs_noE_kernel_excl!(
    rx::CuDeviceVector{T}, ry::CuDeviceVector{T}, rz::CuDeviceVector{T},
    fx::CuDeviceVector{T}, fy::CuDeviceVector{T}, fz::CuDeviceVector{T},
    bindex::CuDeviceVector{Int32}, bflat::CuDeviceVector{Int32}, bcounts::CuDeviceVector{Int32},
    Lx::T, Ly::T, Lz::T, halfLx::T, halfLy::T, halfLz::T,
    ϵ::T, σ::T, cutoff2::T
) where {T<:AbstractFloat}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    accx = zero(T); accy = zero(T); accz = zero(T)
    @inbounds for j in 1:N
        if _is_bonded(Int32(i), Int32(j), bindex, bflat, bcounts); continue; end
        dx = mic_fast(xi - rx[j], halfLx, Lx)
        dy = mic_fast(yi - ry[j], halfLy, Ly)
        dz = mic_fast(zi - rz[j], halfLz, Lz)
        r2 = muladd(dx, dx, muladd(dy, dy, dz*dz))
        if (r2 > zero(T)) & (r2 < cutoff2)
            invr2 = one(T) / r2
            s2    = (σ*σ) * invr2
            s6    = s2*s2*s2
            s12   = s6*s6
            f_over_r = T(24)*ϵ*(T(2)*s12 - s6)*invr2
            accx += f_over_r * dx
            accy += f_over_r * dy
            accz += f_over_r * dz
        end
    end
    fx[i] = accx; fy[i] = accy; fz[i] = accz
    return
end

# Host wrappers for AllPairsNeighborMatrix (LJ)
"""
    lj_forces_soa!(rx, ry[, rz], fx, fy[, fz], Epot, nbh, box, params)

Accumulate Lennard-Jones forces and per-particle potential energies into the
structure-of-arrays buffers. Dispatches on the neighbor matrix:

- `NeighborMatrix` / `StencilNeighborMatrix`: iterate CSR rows built with
  `build_neighbors_dense!` or `build_neighbors_stencil!`.
- `AllPairsNeighborMatrix`: evaluate every pair (used in
  `examples/2D_allpairs_quicktest.jl` when validating kernels).

`Epot[i]` stores half the pair energy so that summing the array yields the
total potential energy without double counting.

# Examples
The compact 3D example (`examples/3D_quicktest.jl`) uses the same parameter
relationships; the snippet below scales `N` down to 4096 for a quick check:

```julia
st = build_simulation(D=3, N=4096, box=(250f0, 250f0, 250f0),
                      cutoff=Float32(2^(1/6)), skin=0.4f0, cap=Int32(100),
                      neigh_interval=1,
                      epsilon=10f0, sigma=1f0,
                      gamma=10f0, temperature=1f0, dt=5f-5)
zero_forces!(st)
lj_forces_soa!(st.rx, st.ry, st.rz,
               st.fx, st.fy, st.fz,
               st.Epot, st.nbh,
               st.box3::Box3{Float32}, st.pair_lj)
```
"""
function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                        ::NeighborLists.AllPairsNeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_kernel!(rx, ry, fx, fy, Epot, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, Epot, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                        ::NeighborLists.AllPairsNeighborMatrix{T},
                        box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_kernel_virial!(rx, ry, fx, fy, Epot, V, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, Epot, V, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                        ::NeighborLists.AllPairsNeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_kernel!(rx, ry, rz, fx, fy, fz, Epot, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, Epot, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                        fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                        ::NeighborLists.AllPairsNeighborMatrix{T},
                        box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_kernel_virial!(rx, ry, rz, fx, fy, fz, Epot, V, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, Epot, V, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

"""
    lj_forces_soa_noE!(rx, ry[, rz], fx, fy[, fz], nbh, box, params)

Lennard-Jones force accumulation without touching `Epot`. Used when the caller
does not require instantaneous energies (e.g. the inner `step!` loops that
only sample `Epot` every `log_interval` steps in `examples/TwoT_2D_LD_VV.jl`).
"""
function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1},
                            ::NeighborLists.AllPairsNeighborMatrix{T},
                            box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_noE_kernel!(rx, ry, fx, fy, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                            fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                            ::NeighborLists.AllPairsNeighborMatrix{T},
                            box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_noE_kernel!(rx, ry, rz, fx, fy, fz, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1},
                             ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_kernel_excl!(rx, ry, fx, fy, Epot, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, Epot, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                             ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                             box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_kernel_excl_virial!(rx, ry, fx, fy, Epot, V, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, Epot, V, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1},
                             ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_kernel_excl!(rx, ry, rz, fx, fy, fz, Epot, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, Epot, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                             fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1}, Epot::CuArray{T,1}, V::CuArray{T,2},
                             ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                             box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 100_000) ? 128 : 256; blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_kernel_excl_virial!(rx, ry, rz, fx, fy, fz, Epot, V, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, Epot, V, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1},
                                 ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                                 box::Definitions.Box2{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly
    k = CUDA.@cuda launch=false _lj2_allpairs_noE_kernel_excl!(rx, ry, fx, fy, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2)
    k(rx, ry, fx, fy, bonds.index, bonds.flat, bonds.counts, Lx, Ly, halfLx, halfLy, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end

function lj_forces_soa_noE_excl!(rx::CuArray{T,1}, ry::CuArray{T,1}, rz::CuArray{T,1},
                                 fx::CuArray{T,1}, fy::CuArray{T,1}, fz::CuArray{T,1},
                                 ::NeighborLists.AllPairsNeighborMatrix{T}, bonds::BondedForces.BondList,
                                 box::Definitions.Box3{T}, params::Definitions.LJParams{T}) where {T<:AbstractFloat}
    N = length(rx); threads = (N < 50_000) ? 64 : ((N < 200_000) ? 128 : 256); blocks = cld(N, threads)
    cutoff2 = params.rcut * params.rcut
    Lx = box[1]; Ly = box[2]; Lz = box[3]
    halfLx = T(0.5)*Lx; halfLy = T(0.5)*Ly; halfLz = T(0.5)*Lz
    k = CUDA.@cuda launch=false _lj3_allpairs_noE_kernel_excl!(rx, ry, rz, fx, fy, fz, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2)
    k(rx, ry, rz, fx, fy, fz, bonds.index, bonds.flat, bonds.counts, Lx, Ly, Lz, halfLx, halfLy, halfLz, params.ϵ, params.σ, cutoff2; threads, blocks)
    return nothing
end
