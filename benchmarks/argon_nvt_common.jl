using ParticleDynamics
using ParticleDynamics: nosehooverchain
using CUDA
using Random

const ARGON_SIGMA_SI = 3.405e-10          # m
const ARGON_EPS_K_SI = 119.8              # K
const ARGON_MASS_AMU = 39.948             # g/mol equivalent atomic mass unit
const AMU_TO_KG = 1.66053906660e-27       # kg
const KB_SI = 1.380649e-23                # J/K

const ARGON_MASS_SI = ARGON_MASS_AMU * AMU_TO_KG
const ARGON_EPSILON_SI = KB_SI * ARGON_EPS_K_SI

"""
    parse_bool_env(key, default)

Parse a boolean environment variable with permissive tokens.
"""
function parse_bool_env(key::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, key, "")))
    isempty(raw) && return default
    if raw in ("1", "true", "yes", "on", "y")
        return true
    elseif raw in ("0", "false", "no", "off", "n")
        return false
    end
    return default
end

"""
    argon_reduced_temperature(T_kelvin)

Convert physical temperature (K) to reduced LJ temperature T* for Argon.
"""
argon_reduced_temperature(T_kelvin::Real) = float(T_kelvin) / ARGON_EPS_K_SI

"""
    argon_reduced_density_from_mass_density(rho_kg_m3)

Convert physical mass density (kg/m^3) to reduced LJ number density rho*.
"""
function argon_reduced_density_from_mass_density(rho_kg_m3::Real)
    n_number = float(rho_kg_m3) / ARGON_MASS_SI
    return n_number * ARGON_SIGMA_SI^3
end

"""
    argon_mass_density_ideal(P_pascal, T_kelvin)

Ideal-gas mass density for Argon at (P,T) in SI units.
"""
function argon_mass_density_ideal(P_pascal::Real, T_kelvin::Real)
    n_number = float(P_pascal) / (KB_SI * float(T_kelvin))
    return n_number * ARGON_MASS_SI
end

"""
    argon_box_length_reduced(N, rho_star)

Cubic box edge length in reduced units for N particles at reduced density rho*.
"""
argon_box_length_reduced(N::Integer, rho_star::Real) = (float(N) / float(rho_star))^(1 / 3)

"""
    reduced_pressure_to_pascal(P_star)

Convert reduced LJ pressure to SI pressure (Pa) for Argon mapping.
"""
reduced_pressure_to_pascal(P_star::Real) = float(P_star) * ARGON_EPSILON_SI / ARGON_SIGMA_SI^3

"""
    initialize_simple_cubic_lattice!(st, box; jitter_frac=0)

Place particles on a simple-cubic lattice centered in the periodic box. Optional
small random jitter can be added to avoid perfect crystalline symmetry.
"""
function initialize_simple_cubic_lattice!(st, box::NTuple{3,T}; jitter_frac::T=zero(T)) where {T<:AbstractFloat}
    N = length(st.rx)
    n_side = ceil(Int, cbrt(Float64(N)))
    spacing_x = box[1] / n_side
    spacing_y = box[2] / n_side
    spacing_z = box[3] / n_side

    rx_host = Vector{T}(undef, N)
    ry_host = Vector{T}(undef, N)
    rz_host = Vector{T}(undef, N)

    n_side_sq = n_side^2
    for i in 1:N
        linear = i - 1
        ix = linear % n_side
        iy = (linear ÷ n_side) % n_side
        iz = linear ÷ n_side_sq

        rx_host[i] = (ix + T(0.5)) * spacing_x - box[1] / T(2)
        ry_host[i] = (iy + T(0.5)) * spacing_y - box[2] / T(2)
        rz_host[i] = (iz + T(0.5)) * spacing_z - box[3] / T(2)
    end

    if jitter_frac > zero(T)
        jx = rand(T, N)
        jy = rand(T, N)
        jz = rand(T, N)
        amp_x = jitter_frac * spacing_x
        amp_y = jitter_frac * spacing_y
        amp_z = jitter_frac * spacing_z
        for i in 1:N
            rx_host[i] += (jx[i] - T(0.5)) * amp_x
            ry_host[i] += (jy[i] - T(0.5)) * amp_y
            rz_host[i] += (jz[i] - T(0.5)) * amp_z
        end
    end

    copyto!(st.rx, rx_host)
    copyto!(st.ry, ry_host)
    copyto!(st.rz, rz_host)
    sync_unwrapped!(st)
    return st
end

"""
    select_nvt_integrator(st; temperature_reduced, dt, nhc_tau_reduced,
                          nhc_chain_length, nhc_substeps, nhc_chain_masses=nothing)

Construct a Nose-Hoover Chain (NHC) integrator for NVT simulations.
"""
function select_nvt_integrator(st;
                               temperature_reduced::Real,
                               dt::Real,
                               nhc_tau_reduced::Real,
                               nhc_chain_length::Integer,
                               nhc_substeps::Integer,
                               nhc_chain_masses::Union{Nothing,AbstractVector{<:Real}}=nothing)
    spec = nosehooverchain(st;
                           temperature=temperature_reduced,
                           tau=nhc_tau_reduced,
                           chain_length=nhc_chain_length,
                           substeps=nhc_substeps,
                           chain_masses=nhc_chain_masses)
    return spec, "Nose-Hoover Chain (NHC)", true
end

"""
    instantaneous_reduced_temperature(st)

Compute instantaneous kinetic temperature in reduced units from Ekin.
"""
function instantaneous_reduced_temperature(st)
    D = st.rz === nothing ? 2 : 3
    dof = D * length(st.rx)
    ekin_total = sum(Array(st.Ekin))
    return (2 * ekin_total) / dof
end

"""
    instantaneous_compressibility_factor(st, rho_star)

Compute instantaneous reduced pressure and compressibility factor from the
kinetic temperature and configurational virial trace.
"""
function instantaneous_compressibility_factor(st, rho_star::Real)
    D = st.rz === nothing ? 2 : 3
    N = length(st.rx)
    V_star = N / float(rho_star)
    T_star = instantaneous_reduced_temperature(st)
    W = sum(Array(st.virial))

    P_star = float(rho_star) * T_star + W / (D * V_star)
    Z = P_star / (float(rho_star) * T_star)
    return T_star, P_star, Z
end

"""
    lj_tail_energy_per_particle(rho_star, rcut_star)

Standard LJ long-range correction (tail) to potential energy per particle:
`u_tail* = (8πρ*/3)[(1/3)rc^-9 - rc^-3]`.
"""
function lj_tail_energy_per_particle(rho_star::Real, rcut_star::Real)
    ρ = float(rho_star)
    rc = float(rcut_star)
    inv3 = rc^(-3)
    inv9 = rc^(-9)
    return (8 * π * ρ / 3) * ((inv9 / 3) - inv3)
end

"""
    lj_tail_pressure(rho_star, rcut_star)

Standard LJ long-range correction (tail) to pressure:
`p_tail* = (16πρ*²/3)[(2/3)rc^-9 - rc^-3]`.
"""
function lj_tail_pressure(rho_star::Real, rcut_star::Real)
    ρ = float(rho_star)
    rc = float(rcut_star)
    inv3 = rc^(-3)
    inv9 = rc^(-9)
    return (16 * π * ρ^2 / 3) * ((2 * inv9 / 3) - inv3)
end

"""
    nist_lj_mc_reference(key)

Return an NVT Monte Carlo benchmark state point from the NIST LJ benchmark
table (`mc.htm`, truncation `rc=3σ` + standard LRC).
"""
function nist_lj_mc_reference(key::AbstractString)
    refs = Dict(
        "T0.90_RHO0.820" => (
            T_star = 0.90,
            rho_star = 0.820,
            U_ref = -5.7456,
            P_ref = 0.82386,
            U_sigma = 7.51e-4,
            P_sigma = 2.85e-3,
            source = "NIST LJ MC NVT benchmark (mc.htm, rc=3σ + sLRC)"
        ),
        "T0.90_RHO0.005" => (
            T_star = 0.90,
            rho_star = 0.005,
            U_ref = -4.9771e-2,
            P_ref = 4.3569e-3,
            U_sigma = 3.80e-5,
            P_sigma = 2.19e-7,
            source = "NIST LJ MC NVT benchmark (mc.htm, rc=3σ + sLRC)"
        ),
        "T0.85_RHO0.776" => (
            T_star = 0.85,
            rho_star = 0.776,
            U_ref = -5.5121,
            P_ref = 6.7714e-3,
            U_sigma = 4.55e-4,
            P_sigma = 1.77e-3,
            source = "NIST LJ MC NVT benchmark (mc.htm, rc=3σ + sLRC)"
        ),
    )
    haskey(refs, key) || error("Unknown NIST reference key: $(key). Available keys: $(join(sort(collect(keys(refs))), ", "))")
    return refs[key]
end
