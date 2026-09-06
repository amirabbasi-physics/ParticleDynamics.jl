"""
    set_noise_scale!(spec, value)
    set_noise_scale!(spec, st, value; filter=All())
    set_noise_scale!(st, spec, value; filter=All())

Integrator-parameter control for the stochastic noise amplitude.
"""
function set_noise_scale!(spec::IntegratorSpec, value::Real)
    fill!(_noise_scale_view(spec), eltype(_noise_scale_view(spec))(value))
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, value::Real; filter::Filter=All())
    idx = resolve_gpu(filter, st)
    assign_scalar!(_noise_scale_view(spec), idx, value)
    _rebuild_single_mode_ou!(spec)
    return idx
end

function set_noise_scale!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_noise_scale!(spec, st, value; filter=filter)
end

set_noise_scale!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_noise_scale!(spec, st, mapping)
set_noise_scale!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_noise_scale!(spec, st, pairs...)

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_noise_scale!(spec, st, val; filter=f)
    end
    return spec
end

function set_noise_scale!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_noise_scale!(spec, st, val; filter=f)
    end
    return spec
end

"""
    set_friction!(spec, γ)
    set_friction!(spec, st, γ; filter=All())
    set_friction!(st, spec, γ; filter=All())

Integrator-parameter control for per-particle friction coefficients.
"""
function set_friction!(spec::IntegratorSpec, value::Real)
    fill!(_gamma_view(spec), eltype(_gamma_view(spec))(value))
    return spec
end

function set_friction!(spec::IntegratorSpec, st::SimulationState, value::Real; filter::Filter=All())
    idx = resolve_gpu(filter, st)
    assign_scalar!(_gamma_view(spec), idx, value)
    return idx
end

function set_friction!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_friction!(spec, st, value; filter=filter)
end

set_friction!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_friction!(spec, st, mapping)
set_friction!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_friction!(spec, st, pairs...)

function set_friction!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_friction!(spec, st, val; filter=f)
    end
    return spec
end

function set_friction!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_friction!(spec, st, val; filter=f)
    end
    return spec
end

"""
    set_temperature!(spec, dt, T)
    set_temperature!(spec, st, dt, T; filter=All())
    set_temperature!(st, spec, dt, T; filter=All())

Set the effective thermostat temperature through the integrator-owned
`noise_scale` and `gamma` buffers. `dt` must match the spec's cached timestep.
"""
function set_temperature!(spec::IntegratorSpec{T}, dt::Real, temperature::Real) where {T<:AbstractFloat}
    _noise_scale_view(spec) # Preserve the unsupported-integrator diagnostic.
    SimulationCore._require_stochastic_dt!(spec.params, dt)
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    noise = _noise_scale_view(spec)
    gamma = _gamma_view(spec)
    @. noise = sqrt(T(2) * gamma * Tval * Δt)
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_temperature!(spec::IntegratorSpec{T}, st::SimulationState, dt::Real, temperature::Real; filter::Filter=All()) where {T<:AbstractFloat}
    _noise_scale_view(spec) # Preserve the unsupported-integrator diagnostic.
    SimulationCore._require_stochastic_dt!(spec.params, dt)
    idx = resolve_gpu(filter, st)
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(_noise_scale_view(spec), _gamma_view(spec), idx, Δt, Tval)
    _rebuild_single_mode_ou!(spec)
    return idx
end

function set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, temperature::Real; filter::Filter=All())
    return set_temperature!(spec, st, dt, temperature; filter=filter)
end

set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, mapping::AbstractDict{<:Filter,<:Real}) =
    set_temperature!(spec, st, dt, mapping)
set_temperature!(st::SimulationState, spec::IntegratorSpec, dt::Real, pairs::Pair{<:Filter,<:Real}...) =
    set_temperature!(spec, st, dt, pairs...)

function set_temperature!(spec::IntegratorSpec, st::SimulationState, dt::Real, mapping::AbstractDict{<:Filter,<:Real})
    for (f, temp) in mapping
        set_temperature!(spec, st, dt, temp; filter=f)
    end
    return spec
end

function set_temperature!(spec::IntegratorSpec, st::SimulationState, dt::Real, pairs::Pair{<:Filter,<:Real}...)
    for (f, temp) in pairs
        set_temperature!(spec, st, dt, temp; filter=f)
    end
    return spec
end

"""
    set_corr_time!(spec, τ)
    set_corr_time!(spec, st, τ; filter=All())
    set_corr_time!(st, spec, τ; filter=All())

Set per-particle correlation times for OU noise processes on the integrator
spec.
"""
function set_corr_time!(spec::IntegratorSpec{T}, value::Real) where {T<:AbstractFloat}
    corr = _ensure_corr_time_array(spec)
    fill!(corr, T(value))
    _rebuild_single_mode_ou!(spec)
    return spec
end

function set_corr_time!(spec::IntegratorSpec{T}, st::SimulationState, value::Real; filter::Filter=All()) where {T<:AbstractFloat}
    corr = _ensure_corr_time_array(spec)
    assign_scalar!(corr, st; filter=filter, value=value)
    _rebuild_single_mode_ou!(spec)
    return corr
end

function set_corr_time!(st::SimulationState, spec::IntegratorSpec, value::Real; filter::Filter=All())
    return set_corr_time!(spec, st, value; filter=filter)
end

set_corr_time!(st::SimulationState, spec::IntegratorSpec, mapping::AbstractDict{<:Filter,<:Real}) =
    set_corr_time!(spec, st, mapping)
set_corr_time!(st::SimulationState, spec::IntegratorSpec, pairs::Pair{<:Filter,<:Real}...) =
    set_corr_time!(spec, st, pairs...)

function set_corr_time!(spec::IntegratorSpec, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real})
    for (f, val) in mapping
        set_corr_time!(spec, st, val; filter=f)
    end
    return spec
end

function set_corr_time!(spec::IntegratorSpec, st::SimulationState, pairs::Pair{<:Filter,<:Real}...)
    for (f, val) in pairs
        set_corr_time!(spec, st, val; filter=f)
    end
    return spec
end

"""
    set_ou_spectrum!(spec, st, taus, scales; filter=All(), dt=spec.params.dt)
    set_ou_spectrum!(st, spec, taus, scales; filter=All(), dt=spec.params.dt)

Configure a generalized OU spectrum on the selected particles. This explicit OU
path stores the active spectrum separately from the white-noise
`noise_scale` buffer, so thermal Gaussian noise and active OU forcing can be
combined on the same particles. Scalars are treated as one-mode spectra.
"""
function set_ou_spectrum!(spec::IntegratorSpec{T},
                          st::SimulationState,
                          taus::Union{AbstractVector{<:Real},Real},
                          scales::Union{AbstractVector{<:Real},Real};
                          filter::Filter=All(),
                          dt::Real=_dt_view(spec)) where {T<:AbstractFloat}
    _noise_scale_view(spec) # Preserve the unsupported-integrator diagnostic.
    SimulationCore._require_stochastic_dt!(spec.params, dt)
    sel = selection(st, filter)
    backend = Backends.storage_backend(st)
    dtT = convert(T, dt)
    tau_vals, scale_vals = SimulationCore._canonical_mode_vectors(T, taus, scales)
    ou = SimulationCore._build_mode_ou(backend, T, sel.device, tau_vals, scale_vals, dtT)
    _set_corr_time_view!(spec, nothing)
    _set_ou_view!(spec, ou)
    return spec
end

function set_ou_spectrum!(st::SimulationState,
                          spec::IntegratorSpec,
                          taus::Union{AbstractVector{<:Real},Real},
                          scales::Union{AbstractVector{<:Real},Real};
                          filter::Filter=All(),
                          dt::Real=_dt_view(spec))
    return set_ou_spectrum!(spec, st, taus, scales; filter=filter, dt=dt)
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, sel::Selection, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.noise_scale, sel.device, value)
    return sel
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, idx::CuArray{Int32,1}, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.noise_scale, idx, value)
    return idx
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, filter::Filter, value::Real) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_noise_scale!(bp, sel, value)
    return sel
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T<:AbstractFloat}
    return set_noise_scale!(bp, st, filter, value)
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, val) in mapping
        set_noise_scale!(bp, st, f, val)
    end
    return bp
end

function set_noise_scale!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, val) in pairs
        set_noise_scale!(bp, st, f, val)
    end
    return bp
end

function set_corr_time!(bp::BrownianIntegrators.BrownianParams{T}, value::Real) where {T<:AbstractFloat}
    bp2 = _ensure_corr_time_array(bp)
    fill!(bp2.corr_time, T(value))
    ou = SimulationCore._build_single_mode_ou(Backends.CUDABackend(), T, bp2.noise_scale, bp2.corr_time, bp2.dt)
    return BrownianIntegrators.BrownianParams{T}(bp2.gamma, bp2.dt, bp2.noise_scale, bp2.corr_time, ou)
end

function set_corr_time!(em::BrownianIntegrators.EMParams{T}, value::Real) where {T<:AbstractFloat}
    em2 = _ensure_corr_time_array(em)
    fill!(em2.corr_time, T(value))
    ou = SimulationCore._build_single_mode_ou(Backends.CUDABackend(), T, em2.noise_scale, em2.corr_time, em2.dt)
    return BrownianIntegrators.EMParams{T}(em2.gamma, em2.dt, em2.noise_scale, em2.corr_time, ou)
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, sel::Selection, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.gamma, sel.device, value)
    return sel
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, idx::CuArray{Int32,1}, value::Real) where {T<:AbstractFloat}
    assign_scalar!(bp.gamma, idx, value)
    return idx
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, filter::Filter, value::Real) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_friction!(bp, sel, value)
    return sel
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState; filter::Filter=All(), value::Real) where {T<:AbstractFloat}
    return set_friction!(bp, st, filter, value)
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, val) in mapping
        set_friction!(bp, st, f, val)
    end
    return bp
end

function set_friction!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, val) in pairs
        set_friction!(bp, st, f, val)
    end
    return bp
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real; filter::Filter=All()) where {T<:AbstractFloat}
    sel = selection(st, filter)
    set_temperature!(bp, st, dt, temperature, sel)
    return sel
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real, sel::Selection) where {T<:AbstractFloat}
    SimulationCore._require_stochastic_dt!(bp, dt)
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(bp.noise_scale, bp.gamma, sel.device, Δt, Tval)
    return sel
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, temperature::Real, idx::CuArray{Int32,1}) where {T<:AbstractFloat}
    SimulationCore._require_stochastic_dt!(bp, dt)
    Δt = convert(T, dt)
    Tval = convert(T, temperature)
    _set_noise_from_gamma!(bp.noise_scale, bp.gamma, idx, Δt, Tval)
    return idx
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, mapping::AbstractDict{<:Filter,<:Real}) where {T<:AbstractFloat}
    for (f, temp) in mapping
        set_temperature!(bp, st, dt, temp; filter=f)
    end
    return bp
end

function set_temperature!(bp::BrownianIntegrators.BrownianParams{T}, st::SimulationState, dt::Real, pairs::Pair{<:Filter,<:Real}...) where {T<:AbstractFloat}
    for (f, temp) in pairs
        set_temperature!(bp, st, dt, temp; filter=f)
    end
    return bp
end
