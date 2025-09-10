module NeighborLists

using CUDA
using ..Definitions

export NeighborMatrix, build_neighbors_dense!, update_neighbors_inplace!, update_needed!

mutable struct NeighborMatrix
    neighbors::CuArray{Int32,2}
    cap::Int32
    cutoff::Definitions.FloatX
    skin::Definitions.FloatX
    last_build_step::Int
end

function _build_kernel2!(
    rx::CuDeviceVector{Definitions.FloatX}, ry::CuDeviceVector{Definitions.FloatX},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Definitions.FloatX, Ly::Definitions.FloatX,
    cutoff2::Definitions.FloatX
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]
    cnt = Int32(0)
    @inbounds for j in 1:N
        if j == i; continue; end
        dx = xi - rx[j]; dy = yi - ry[j]
        dx = (2abs(dx) > Lx) ? dx - sign(dx)*Lx : dx
        dy = (2abs(dy) > Ly) ? dy - sign(dy)*Ly : dy
        r2 = dx*dx + dy*dy
        if r2 < cutoff2
            cnt += 1
            id = cnt <= cap ? cnt : cap
            if id <= cap
                nbr[i,id] = Int32(j)
            end
        end
    end
    if cnt < cap
        for k in (cnt+1):cap
            nbr[i,k] = 0
        end
    end
    return
end

function _build_kernel3!(
    rx::CuDeviceVector{Definitions.FloatX}, ry::CuDeviceVector{Definitions.FloatX}, rz::CuDeviceVector{Definitions.FloatX},
    nbr::CuDeviceMatrix{Int32}, cap::Int32,
    Lx::Definitions.FloatX, Ly::Definitions.FloatX, Lz::Definitions.FloatX,
    cutoff2::Definitions.FloatX
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    N = length(rx); if i > N; return; end
    xi = rx[i]; yi = ry[i]; zi = rz[i]
    cnt = Int32(0)
    @inbounds for j in 1:N
        if j == i; continue; end
        dx = xi - rx[j]; dy = yi - ry[j]; dz = zi - rz[j]
        dx = (2abs(dx) > Lx) ? dx - sign(dx)*Lx : dx
        dy = (2abs(dy) > Ly) ? dy - sign(dy)*Ly : dy
        dz = (2abs(dz) > Lz) ? dz - sign(dz)*Lz : dz
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < cutoff2
            cnt += 1
            id = cnt <= cap ? cnt : cap
            if id <= cap
                nbr[i,id] = Int32(j)
            end
        end
    end
    if cnt < cap
        for k in (cnt+1):cap
            nbr[i,k] = 0
        end
    end
    return
end

function build_neighbors_dense!(rx::CuArray{Definitions.FloatX,1},
                                ry::CuArray{Definitions.FloatX,1};
                                box::Definitions.Box2,
                                cutoff::Definitions.FloatX,
                                cap::Int32=Int32(96),
                                skin::Definitions.FloatX=0.4f0)
    N = length(rx)
    neighbors = CUDA.fill(Int32(0), N, cap)
    threads = min(256,N); blocks = cld(N,threads)
    cutoff2 = cutoff*cutoff
    k = CUDA.@cuda launch=false _build_kernel2!(rx, ry, neighbors, cap, box[1], box[2], cutoff2)
    CUDA.@sync k(rx, ry, neighbors, cap, box[1], box[2], cutoff2; threads, blocks)
    return NeighborMatrix(neighbors, cap, cutoff, skin, 0)
end

function build_neighbors_dense!(rx::CuArray{Definitions.FloatX,1},
                                ry::CuArray{Definitions.FloatX,1},
                                rz::CuArray{Definitions.FloatX,1};
                                box::Definitions.Box3,
                                cutoff::Definitions.FloatX,
                                cap::Int32=Int32(96),
                                skin::Definitions.FloatX=0.4f0)
    N = length(rx)
    neighbors = CUDA.fill(Int32(0), N, cap)
    threads = min(256,N); blocks = cld(N,threads)
    cutoff2 = cutoff*cutoff
    k = CUDA.@cuda launch=false _build_kernel3!(rx, ry, rz, neighbors, cap, box[1], box[2], box[3], cutoff2)
    CUDA.@sync k(rx, ry, rz, neighbors, cap, box[1], box[2], box[3], cutoff2; threads, blocks)
    return NeighborMatrix(neighbors, cap, cutoff, skin, 0)
end

# In-place neighbor list update (no allocation!)
function update_neighbors_inplace!(nbh::NeighborMatrix,
                                   rx::CuArray{Definitions.FloatX,1},
                                   ry::CuArray{Definitions.FloatX,1};
                                   box::Definitions.Box2)
    N = length(rx)
    # Reuse existing neighbor matrix - just fill with zeros and rebuild
    fill!(nbh.neighbors, Int32(0))
    threads = min(256,N); blocks = cld(N,threads)
    cutoff2 = nbh.cutoff * nbh.cutoff
    k = CUDA.@cuda launch=false _build_kernel2!(rx, ry, nbh.neighbors, nbh.cap, box[1], box[2], cutoff2)
    CUDA.@sync k(rx, ry, nbh.neighbors, nbh.cap, box[1], box[2], cutoff2; threads, blocks)
    nbh.last_build_step = 0  # Reset counter
    return nothing
end

function update_neighbors_inplace!(nbh::NeighborMatrix,
                                   rx::CuArray{Definitions.FloatX,1},
                                   ry::CuArray{Definitions.FloatX,1},
                                   rz::CuArray{Definitions.FloatX,1};
                                   box::Definitions.Box3)
    N = length(rx)
    # Reuse existing neighbor matrix - just fill with zeros and rebuild
    fill!(nbh.neighbors, Int32(0))
    threads = min(256,N); blocks = cld(N,threads)
    cutoff2 = nbh.cutoff * nbh.cutoff
    k = CUDA.@cuda launch=false _build_kernel3!(rx, ry, rz, nbh.neighbors, nbh.cap, box[1], box[2], box[3], cutoff2)
    CUDA.@sync k(rx, ry, rz, nbh.neighbors, nbh.cap, box[1], box[2], box[3], cutoff2; threads, blocks)
    nbh.last_build_step = 0  # Reset counter
    return nothing
end

update_needed!(nbl::NeighborMatrix, step::Int, interval::Int) = (step == 0) || (interval > 0 && (step % interval == 0))

end # module