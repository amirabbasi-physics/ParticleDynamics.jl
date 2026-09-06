@testset "Independent host neighbor oracle" begin
    @test reference_neighbor_rows(([0.0, 9.5, 5.0], zeros(3)), (10.0, 10.0), fill(1.0, 3)) == [[2], [1], Int[]]
    @test reference_neighbor_rows(([0.0, 1.0], zeros(2), zeros(2)), (10.0, 10.0, 10.0), [0.5, 1.5]) == [Int[], [1]]
    @test reference_neighbor_rows(([0.0], [0.0]), (10.0, 10.0), [1.0]) == [Int[]]
    @test_throws DimensionMismatch reference_neighbor_rows(([0.0], zeros(2)), (10.0, 10.0), [1.0])
end
