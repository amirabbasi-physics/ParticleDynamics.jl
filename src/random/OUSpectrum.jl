"""
    OUSpectrum{T}

Generalized Ornstein-Uhlenbeck spectrum used by the stochastic integrators.
Each column corresponds to one active particle and each row to one OU mode.
The precomputed coefficients implement the exact discrete update

`x_{n+1} = a x_n + c ξ`

with `a = exp(-dt / τ)` and `c = scale * sqrt(1 - a^2)`. The `τ <= 0` limit is
encoded as `a = 0`, `c = scale`, which reproduces the package's legacy white
noise fallback. Here `scale` is the stationary standard deviation of the OU
state `x`, not the per-step innovation amplitude `c` and not a direct spatial
step length. Physical position increments depend on the integrator and any
mobility factors such as `1 / gamma`.
"""
mutable struct OUSpectrum{T<:AbstractFloat}
    dt::T
    active_idx::CuArray{Int32,1}
    tau::CuArray{T,2}
    scale::CuArray{T,2}
    coeff_a::CuArray{T,2}
    coeff_c::CuArray{T,2}
end
