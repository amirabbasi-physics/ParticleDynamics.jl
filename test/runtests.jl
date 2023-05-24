using NonEqSimGPU
using CUDA
using StaticArrays
using Test
using LinearAlgebra

@testset "NonEqSimGPU.jl" begin
    
    function test_update_positions_lf!()
        Npart = 40000
        dt = 0.1f0
        box = SVector{3,Float32}(10.0, 10.0, 10.0)
        r = CuVector([SVector{3,Float32}(rand(Float32, 3) .* 10.0f0 .- 5.0f0) for _ in 1:Npart])
        v = CuVector([SVector{3,Float32}(rand(Float32, 3) .* 2.0f0 .- 1.0f0) for _ in 1:Npart])
        r_old = Array(r)  # Convert to array for easy manipulation
        v_old = Array(v)
    
        # Call the function to be tested
        update_positions_lf!(r, v, dt, box)
    
        # Calculate expected new positions
        expected_positions = [mod.((r_old[i] .+ (dt/2) .* v_old[i]) .+ box ./ 2, box) .- box ./ 2 for i in 1:Npart]
    
        # Convert GPU array to CPU array for comparison
        r_new = Array(r)
    
        # Compare actual positions with expected ones
        @test all(isapprox.(r_new, expected_positions))
    end
    
    test_update_positions_lf!()


    """
    N = 2
    T = Float32
    I = Int32

    function cpu_neighbor_list(r::Vector{SVector{N,T}}, neigh_cut_off::T, box::SVector{N,T})
        Npart = length(r)
        NNeigh = Npart - 1
        neighbors = zeros(I, Npart, NNeigh)
        ncut_off² = neigh_cut_off^2

        for i = 1:Npart
            pos₁ = r[i]
            idx = 0

            for j = 1:Npart
                if j != i
                    pos₂ = r[j]
                    dx = pos₁[1] - pos₂[1]
                    dy = pos₁[2] - pos₂[2]

                    dx = (2abs(dx) > box[1]) ? dx - sign(dx) * box[1] : dx
                    dy = (2abs(dy) > box[2]) ? dy - sign(dy) * box[2] : dy

                    dr² = dx*dx + dy*dy

                    if 0 < dr² < ncut_off²
                        idx += 1
                        neighbors[i, idx] = j
                    end
                end
            end
        end

        return neighbors
    end

    function test_neighbor_list()
        Npart = 40000
        box = @SVector [100.0f0, 100.0f0]
        r_host = [10*rand(SVector{N,T}) - box/2 for _ in 1:Npart]
        r = CuArray(r_host)
        neigh_cut_off = 5.0f0

        neighbors_host = cpu_neighbor_list(r_host, neigh_cut_off, box)
        Neighbors = CUDA.zeros(I, Npart, Npart-1)

        neighbor_list!(r, Neighbors, neigh_cut_off, box)

        @test Array(Neighbors) == neighbors_host
    end

    test_neighbor_list()
    """


    
    
end
