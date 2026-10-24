# Counter-based Philox4x32-10 (Salmon et al., SC'11; Random123) for graph-safe
# noise. CUDA.jl's device `randn` seeds from a hidden launch argument, which a
# captured graph freezes, so every replay would draw the same numbers. Here the
# counter is (particle, stream, step low, step high) with the step read from
# device memory, and the key is the user seed: draws are independent of the
# launch configuration and reproducible for a given seed.

@inline function _mulhilo32(a::UInt32,b::UInt32)
    p=UInt64(a)*UInt64(b)
    return (p>>32)%UInt32,p%UInt32
end

@inline function _philox_round(c0::UInt32,c1::UInt32,c2::UInt32,c3::UInt32,k0::UInt32,k1::UInt32)
    hi0,lo0=_mulhilo32(0xD2511F53,c0)
    hi1,lo1=_mulhilo32(0xCD9E8D57,c2)
    return hi1⊻c1⊻k0,lo1,hi0⊻c3⊻k1,lo0
end

"""Philox4x32 with 10 rounds: four 32-bit words for counter (c0..c3) and key (k0,k1)."""
@inline function philox4x32(c0::UInt32,c1::UInt32,c2::UInt32,c3::UInt32,k0::UInt32,k1::UInt32)
    c0,c1,c2,c3=_philox_round(c0,c1,c2,c3,k0,k1)
    for _ in 2:10
        k0+=0x9E3779B9; k1+=0xBB67AE85
        c0,c1,c2,c3=_philox_round(c0,c1,c2,c3,k0,k1)
    end
    return c0,c1,c2,c3
end

# Uniforms in the open interval (0,1): Float32 from 24 bits, Float64 from 53 bits.
@inline _open01(::Type{Float32},u::UInt32)=(Float32(u>>8)+0.5f0)*Float32(2.0^-24)
@inline _open01(::Type{Float64},hi::UInt32,lo::UInt32)=
    (Float64((UInt64(hi)<<21)⊻(UInt64(lo)>>11))+0.5)*2.0^-53

@inline function _box_muller(u1::T,u2::T) where {T}
    r=sqrt(-T(2)*log(u1)); s,c=sincospi(T(2)*u2)
    return r*c,r*s
end

@inline function _philox_words(seed::UInt64,step::UInt64,i::Integer,stream::UInt32)
    return philox4x32(i%UInt32,stream,step%UInt32,(step>>32)%UInt32,seed%UInt32,(seed>>32)%UInt32)
end

"""Four standard normals (Float32) for particle `i`, `stream` and `step`."""
@inline function philox_normals4(::Type{Float32},seed::UInt64,step::UInt64,i::Integer,stream::UInt32)
    w0,w1,w2,w3=_philox_words(seed,step,i,stream)
    a,b=_box_muller(_open01(Float32,w0),_open01(Float32,w1))
    c,d=_box_muller(_open01(Float32,w2),_open01(Float32,w3))
    return a,b,c,d
end

"""Two standard normals (Float64) for particle `i`, `stream` and `step`."""
@inline function philox_normals2(::Type{Float64},seed::UInt64,step::UInt64,i::Integer,stream::UInt32)
    w0,w1,w2,w3=_philox_words(seed,step,i,stream)
    return _box_muller(_open01(Float64,w0,w1),_open01(Float64,w2,w3))
end

"""`D` standard normals (D ≤ 4) of type `T`. Float64 uses streams 2s and 2s+1."""
@inline function philox_normals(::Type{Float32},::Val{D},seed::UInt64,step::UInt64,i::Integer,stream::UInt32) where {D}
    a,b,c,d=philox_normals4(Float32,seed,step,i,stream)
    return D==1 ? (a,) : D==2 ? (a,b) : D==3 ? (a,b,c) : (a,b,c,d)
end
@inline function philox_normals(::Type{Float64},::Val{D},seed::UInt64,step::UInt64,i::Integer,stream::UInt32) where {D}
    a,b=philox_normals2(Float64,seed,step,i,UInt32(2)*stream)
    D<=2 && return D==1 ? (a,) : (a,b)
    c,d=philox_normals2(Float64,seed,step,i,UInt32(2)*stream+UInt32(1))
    return D==3 ? (a,b,c) : (a,b,c,d)
end
