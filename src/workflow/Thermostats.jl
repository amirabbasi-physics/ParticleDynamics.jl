"""
    Thermostat

Abstract workflow thermostat descriptor.
"""
abstract type Thermostat end

"""
    CSVR(; kT, tau)

Canonical stochastic velocity-rescaling thermostat.
"""
@kwdef struct CSVR <: Thermostat
    kT
    tau
end

"""
    NoseHooverChain(; kT, tau, chain_length=5, substeps=4)

Nose-Hoover chain thermostat descriptor for the workflow API.
"""
@kwdef struct NoseHooverChain <: Thermostat
    kT
    tau
    chain_length::Int = 5
    substeps::Int = 4
end
