"""
    OUSpectrum{T}

Container for the correlated active-noise process used by the stochastic
integrators.

Interpretation:

- each column is one active particle
- each row is one OU mode assigned to that particle
- `tau[k, j]` is the correlation time of mode `k` on particle `j`
- `scale[k, j]` is the stationary standard deviation of that OU mode

For one mode, the internal OU state `η` is updated exactly as

`η_{n+1} = a η_n + c ξ_n`

with

- `a = exp(-dt / τ)`
- `c = scale * sqrt(1 - a^2)`
- `ξ_n ~ Normal(0, 1)`

This choice guarantees that, at stationarity,

- `mean(η) = 0`
- `std(η) = scale`

So `scale` is the long-time RMS size of the OU state itself. 

The total active contribution for one particle coordinate is the sum of that
particle's active modes. The stochastic integrator then converts that active
signal into an actual displacement or velocity update. For example, on the
Brownian Euler-Maruyama path the position update uses `Δx_active = η_x / gamma`,
so the typical active step size is set by `scale / gamma`.

When an `OUSpectrum` is attached explicitly through
`ActiveOrnsteinUhlenbeck` or `Filters.set_ou_spectrum!`, it is kept separate
from the thermal white-noise `noise_scale` buffer. To model a thermal AOUP,
combine the explicit OU spectrum with a separate Brownian or Langevin thermal
term.

The `τ <= 0` limit is treated as white noise by setting `a = 0` and `c = scale`,
so each step draws an independent `η = scale * ξ`.
"""
mutable struct OUSpectrum{T<:AbstractFloat}
    dt::T
    active_idx::CuArray{Int32,1}
    tau::CuArray{T,2}
    scale::CuArray{T,2}
    coeff_a::CuArray{T,2}
    coeff_c::CuArray{T,2}
end
