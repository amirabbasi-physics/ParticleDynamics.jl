abstract type Thermostat end

@kwdef struct CSVR <: Thermostat
    kT
    tau
end

@kwdef struct NoseHooverChain <: Thermostat
    kT
    tau
    chain_length::Int = 5
    substeps::Int = 4
end
