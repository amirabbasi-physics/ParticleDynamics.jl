module Definitions

using CUDA
using StaticArrays

export FloatX, IntX, Dim2, Dim3, Box2, Box3,
       LJParams, wrap_pbc2!, wrap_pbc3!, clamp_cap

const FloatX = Float32
const IntX   = Int32

const Dim2 = 2
const Dim3 = 3

const Box2 = NTuple{2,FloatX}
const Box3 = NTuple{3,FloatX}

struct LJParams{T}
    ϵ::T
    σ::T
    rcut::T
end

@inline clamp_cap(idx::IntX, cap::IntX) = ifelse(idx <= cap, idx, IntX(0))

# ---------------- PBC wrappers (SoA) ----------------

@inline function _wrap(x::FloatX, L::FloatX)::FloatX
    y = x + L*0.5f0
    y -= floor(y / L) * L
    return y - L*0.5f0
end

function wrap_pbc2!(rx::CuArray{FloatX,1}, ry::CuArray{FloatX,1}, box::Box2)
    function kern(rx, ry, Lx, Ly)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(rx); if i > N; return; end
        @inbounds begin
            rx[i] = _wrap(rx[i], Lx)
            ry[i] = _wrap(ry[i], Ly)
        end
        return
    end
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    k = CUDA.@cuda launch=false kern(rx, ry, box[1], box[2])
    CUDA.@sync k(rx, ry, box[1], box[2]; threads, blocks)
    return nothing
end

function wrap_pbc3!(rx::CuArray{FloatX,1}, ry::CuArray{FloatX,1}, rz::CuArray{FloatX,1}, box::Box3)
    function kern(rx, ry, rz, Lx, Ly, Lz)
        i = (blockIdx().x-1)*blockDim().x + threadIdx().x
        N = length(rx); if i > N; return; end
        @inbounds begin
            rx[i] = _wrap(rx[i], Lx)
            ry[i] = _wrap(ry[i], Ly)
            rz[i] = _wrap(rz[i], Lz)
        end
        return
    end
    N = length(rx); threads = min(256,N); blocks = cld(N,threads)
    k = CUDA.@cuda launch=false kern(rx, ry, rz, box[1], box[2], box[3])
    CUDA.@sync k(rx, ry, rz, box[1], box[2], box[3]; threads, blocks)
    return nothing
end

end # module