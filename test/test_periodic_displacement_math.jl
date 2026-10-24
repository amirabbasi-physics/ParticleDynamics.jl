using Test, Random
# Same scalar device functions, tested without loading CUDA or the package.
module PeriodicMathUnderTest
include(joinpath(@__DIR__,"../src/geometry/PeriodicDisplacement.jl"))
end
using .PeriodicMathUnderTest: periodic_delta, force_coordinate

@testset "Periodic displacement precision (CPU)" begin
    rng = MersenneTwister(20261006)
    for T in (Float32,Float64), length0 in (4.3,149.38015821857212,1024.1)
        L = T(length0); h = L/T(2)
        for k in 1:300
            # Ordinary pairs and pairs straddling each side of the box.
            x,y = k <= 150 ? (T(rand(rng)-0.5)*L,T(rand(rng)-0.5)*L) :
                            (-h+T(rand(rng)*0.7),h-T(rand(rng)*0.7))
            d = x-y
            exact = BigFloat(x)-BigFloat(y)
            abs(d)>h && (exact -= copysign(BigFloat(L),BigFloat(d)))
            @test periodic_delta(x,y,L) == T(exact)
            @test periodic_delta(x,y,L) == -periodic_delta(y,x,L)
        end
        @test periodic_delta(zero(T),h,L) == -h
        @test periodic_delta(h,zero(T),L) == h
        @test isnan(periodic_delta(T(NaN),zero(T),L))
        for x in T.((-0.49,-0.23,0.0,0.31,0.49)).*L
            @test force_coordinate(x,L) === x
        end
        # Shifting a positive [0,L] coordinate to the centered image is exact.
        for x in T.((0.51,0.74,0.99)).*L
            @test BigFloat(force_coordinate(x,L)) == BigFloat(x)-BigFloat(L)
        end
    end
    # An explicit cancellation case that the old round-then-wrap loses.
    L=149.38016f0; x=-73.96331f0; y=74.49271f0
    old = (x-y)+L
    ref = Float32(Float64(x)-Float64(y)+Float64(L))
    @test old != ref
    @test periodic_delta(x,y,L) == ref
end
