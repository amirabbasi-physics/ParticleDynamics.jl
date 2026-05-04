using ParticleDynamics
using ParticleDynamics.Thermostats
using CUDA

@testset "Thermostats: Interface" begin
    # Test that thermostat types are defined
    @test !isnothing(NoseHooverChainThermostat)
    @test !isnothing(CSVRThermostat)
    @test !isnothing(AbstractThermostat)
    @test !isnothing(ThermostatState)
end

@testset "Thermostats: NHC Parameters" begin
    T = Float32
    mass = 1.0f0
    temp = [1.0f0]
    tau = [1.0f0]
    
    # Create NHC params
    params = Thermostats.NHCThermostatParams{T}(
        mass, temp, tau, 5, 4, ones(Float32, 4, 1), :gromacs
    )
    
    @test params.mass ≈ 1.0f0
    @test params.target_temperature == [1.0f0]
    @test params.tau == [1.0f0]
    @test params.substeps == 5
    @test params.chain_length == 4
    @test params.propagator == :gromacs
end

@testset "Thermostats: CSVR Parameters" begin
    T = Float32
    mass = 1.0f0
    temp = [1.0f0, 2.0f0]
    tau = [1.0f0, 1.5f0]
    
    # Create CSVR params
    params = Thermostats.CSVRThermostatParams{T}(mass, temp, tau)
    
    @test params.mass ≈ 1.0f0
    @test length(params.target_temperature) == 2
    @test length(params.tau) == 2
end

println("Thermostats interface tests passed.")
