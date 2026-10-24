module DeviceEnginePhiloxGPUChecks
# counter-based Philox4x32-10 noise. Known-answer
# vectors (Random123), distribution moments and tails, independence across
# particles/steps/streams/components, and replay of a captured graph that reads
# its step from device memory (each replay must draw new numbers).
using Test, CUDA, Statistics, ParticleDynamics
const DE=ParticleDynamics.DeviceEngine

# Random123 kat_vectors, philox4x32 with 10 rounds: counter, key, expected output.
const KAT=(((0x00000000,0x00000000,0x00000000,0x00000000),(0x00000000,0x00000000),(0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8)),
           ((0xffffffff,0xffffffff,0xffffffff,0xffffffff),(0xffffffff,0xffffffff),(0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd)),
           ((0x243f6a88,0x85a308d3,0x13198a2e,0x03707344),(0xa4093822,0x299f31d0),(0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1)))

function _kat_kernel!(out,c,k)
    i=threadIdx().x
    if i<=size(c,2)
        w=DE.philox4x32(c[1,i],c[2,i],c[3,i],c[4,i],k[1,i],k[2,i])
        for q in 1:4; out[q,i]=w[q]; end
    end
    return nothing
end

function _normals_kernel!(out,::Type{T},::Val{D},seed,step,stream) where {T,D}
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=size(out,2)
        z=DE.philox_normals(T,Val(D),seed,step,i,stream)
        for q in 1:D; out[q,i]=z[q]; end
    end
    return nothing
end

function normals(::Type{T},D,N;seed=0x0123456789abcdef,step=UInt64(0),stream=UInt32(0)) where {T}
    out=CuArray{T}(undef,D,N)
    @cuda threads=256 blocks=cld(N,256) _normals_kernel!(out,T,Val(D),seed,step,stream)
    return Array(out)
end

@testset "Philox4x32-10 known-answer vectors (host and device)" begin
    for (c,k,expected) in KAT
        @test DE.philox4x32(c...,k...)==expected
    end
    c=CuArray(UInt32[KAT[j][1][q] for q in 1:4, j in 1:3]); k=CuArray(UInt32[KAT[j][2][q] for q in 1:2, j in 1:3])
    out=CUDA.zeros(UInt32,4,3)
    @cuda threads=3 _kat_kernel!(out,c,k)
    @test Array(out)==UInt32[KAT[j][3][q] for q in 1:4, j in 1:3]
end

@testset "Normal draws: moments, tails, independence ($T)" for T in (Float32,Float64)
    N=1<<20; z=normals(T,3,N)
    x=vec(Float64.(z)); n=length(x); se=1/sqrt(n)
    @test all(isfinite,x)
    @test abs(mean(x))<5se
    @test abs(var(x)-1)<5*sqrt(2)*se
    @test abs(mean(x.^3))<5*sqrt(15)*se                       # skewness
    @test abs(mean(x.^4)-3)<5*sqrt(96)*se                     # kurtosis
    for (t,p) in ((1.0,0.31731050786291415),(2.0,0.04550026389635842),(3.0,0.0026997960632601866))
        f=count(v->abs(v)>t,x)/n
        @test abs(f-p)<5*sqrt(p*(1-p)/n)
    end
    corr(a,b)=cor(Float64.(vec(a)),Float64.(vec(b)))
    lim=5/sqrt(N)
    @test abs(corr(z[1,:],z[2,:]))<lim && abs(corr(z[1,:],z[3,:]))<lim && abs(corr(z[2,:],z[3,:]))<lim
    @test abs(corr(z[1,1:end-1],z[1,2:end]))<lim             # neighbouring particles
    z1=normals(T,3,N;step=UInt64(1)); z2=normals(T,3,N;stream=UInt32(1)); z3=normals(T,3,N;seed=0x0123456789abcdee)
    @test abs(corr(z,z1))<lim && abs(corr(z,z2))<lim && abs(corr(z,z3))<lim
    big=normals(T,3,1024;step=UInt64(1)<<40)                   # high counter word is used
    @test big!=normals(T,3,1024;step=UInt64(0))
    @test normals(T,3,N)==z                                    # reproducible
end

# A captured graph reads its step from device memory and advances it, as the
# engine's step counter does: each replay must draw different numbers, equal to
# a direct draw at that step.
function _step_kernel!(out,step,seed)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=size(out,2)
        z=DE.philox_normals(Float32,Val(3),seed,step[1],i,UInt32(0))
        for q in 1:3; out[q,i]=z[q]; end
    end
    return nothing
end
_advance_kernel!(step)=(step[1]+=UInt64(1); nothing)

@testset "Captured graph replays draw new noise per step" begin
    N=4096; seed=0x00000000deadbeef
    out=CUDA.zeros(Float32,3,N); step=CuArray(UInt64[7])
    # Compile outside capture (module loading invalidates a capture), then reset.
    @cuda threads=256 blocks=cld(N,256) _step_kernel!(out,step,seed)
    @cuda threads=1 _advance_kernel!(step)
    copyto!(step,UInt64[7]); CUDA.synchronize()
    graph=ParticleDynamics.NeighborLists.capture_graph() do
        @cuda threads=256 blocks=cld(N,256) _step_kernel!(out,step,seed)
        @cuda threads=1 _advance_kernel!(step)
    end
    exec=CUDA.instantiate(graph)
    draws=map(1:3) do _
        CUDA.launch(exec); Array(out)
    end
    @test draws[1]!=draws[2] && draws[2]!=draws[3]
    for (s,d) in zip(7:9,draws)
        @test d==normals(Float32,3,N;seed,step=UInt64(s))
    end
    @test Array(step)==[10]
end
end
