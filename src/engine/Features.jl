# Engine features (user-approved refactor, 10 Oct 2026): optional per-particle
# behavior attached to a captured run. Each feature lives in its own file and
# implements only the hooks it needs; everything else defaults to "no effect".
# Adding a feature = one file + one entry in `ENGINE_FEATURES` (end of the
# engine includes). Absent features are not in the bundle, so their hooks
# compile away: no cost for runs without them.
#
# Host side, a feature type `F` provides
#   make_feature(::Type{F}, st, spec)  -> instance or nothing (feature inactive)
#   feature_signature(::Type{F}, st, spec) -> host-only value for the fingerprint
#   before_step!(::Type{F}, st), after_step!(::Type{F}, st)   (host, every step)
# and an instance `f` provides
#   device_part(f)       -> NamedTuple of device arrays passed to kernels, or nothing
#   sorted_real(f), sorted_int(f) -> per-particle arrays permuted by spatial sorting
#   fusable(f)           -> may the fused NVE path run with this feature
#   generic_integration(f), generic_kick(f) -> needs the generic (non-tuned) NVE
#                          kick-drift / final-kick kernels
#   generic_pairs(f)     -> needs the generic cooperative pair kernel (feature hooks
#                          such as `excluded` are not in the tuned uniform-LJ kernels)
#   host_forces(f)       -> the feature supplies every force from the host: each step
#                          is split into graph segments at its force evaluations, and
#                          host_force!(f, st, energy) runs between them
#   needs_neighbors(f)   -> the feature reads the engine's neighbor list; a run with
#                          host forces keeps the list only when some feature does
#   after_position!(run, f; midpoint, launch), after_force!(run, f; energy, launch),
#   at_boundary!(f), after_rebuild!(run, f; launch), position_sample!(run, f; launch)
#                        -> extra launches / host work at those points;
#   samples_half_drift(f) -> BAOAB-type schemes sample positions after A1 too;
# Device side, kernels receive the bundle `fx` (tuple of device parts) and call
# fold hooks defined by the features (`displace!`, `noise_add`, `counted`, mass
# coefficients …). Device parts are NamedTuples whose first field name is the
# feature's key, found with `_find(fx, Val(key))`.

abstract type EngineFeature end

feature_signature(::Type,st,spec)=nothing
before_step!(::Type,st)=nothing
after_step!(::Type,st)=nothing
device_part(::EngineFeature)=nothing
sorted_real(::EngineFeature)=()
sorted_int(::EngineFeature)=()
fusable(::EngineFeature)=false
generic_integration(::EngineFeature)=false   # needs the generic kick-drift kernel
generic_kick(::EngineFeature)=false          # needs the generic final-kick kernel
generic_pairs(::EngineFeature)=false         # needs the generic cooperative pair kernel
host_forces(::EngineFeature)=false           # supplies all forces from the host (split step)
host_force!(::EngineFeature,st,energy::Bool)=nothing
needs_neighbors(::EngineFeature)=false       # reads the engine's neighbor list
after_position!(run,::EngineFeature;midpoint::Bool=false,launch::Bool=true)=nothing
after_force!(run,::EngineFeature;energy::Bool,launch::Bool=true)=nothing
after_kick!(run,::EngineFeature;energy::Bool,launch::Bool=true)=nothing   # after the final kick (RATTLE)
constrained_dof(::EngineFeature)=0           # holonomic constraints removed from the kinetic dof
constrains_velocities(::EngineFeature)=false  # velocity constraints that thermostats must apply before measuring K
constrain_velocities!(run,::EngineFeature;launch::Bool=true)=nothing
at_boundary!(::EngineFeature)=nothing
after_rebuild!(run,::EngineFeature;launch::Bool=true)=nothing   # after each neighbor rebuild (and sort)
position_sample!(run,::EngineFeature;launch::Bool=true)=nothing # at each physical position sample
samples_half_drift(::EngineFeature)=false   # BAOAB/BAOA/GSM also sample after A1 (split kernel)
feature_active(::EngineFeature)=true          # e.g. an expired freeze is attached but inactive

"""Is an active feature of type `F` attached to `run`?"""
has_feature(run,::Type{F}) where {F}=any(f->f isa F && feature_active(f),run.features)

"""Instances of the registered features active for `st`/`spec` (host)."""
make_features(st,spec)=_active(map(F->make_feature(F,st,spec),ENGINE_FEATURES))
_active(fs::Tuple{})=()
_active(fs::Tuple)=first(fs)===nothing ? _active(Base.tail(fs)) : (first(fs),_active(Base.tail(fs))...)

"""Device bundle: the non-nothing device parts of the active features."""
device_bundle(features::Tuple)=_active(map(device_part,features))

feature_signatures(st,spec)=map(F->(F,feature_signature(F,st,spec)),ENGINE_FEATURES)
all_before_step!(st)=foreach(F->before_step!(F,st),ENGINE_FEATURES)
all_after_step!(st)=foreach(F->after_step!(F,st),ENGINE_FEATURES)
features_after_position!(run;midpoint::Bool=false,launch::Bool=true)=
    foreach(f->after_position!(run,f;midpoint,launch),run.features)
features_after_force!(run;energy::Bool,launch::Bool=true)=foreach(f->after_force!(run,f;energy,launch),run.features)
features_after_kick!(run;energy::Bool,launch::Bool=true)=foreach(f->after_kick!(run,f;energy,launch),run.features)
features_constrained_dof(run)=sum(constrained_dof,run.features;init=0)
features_at_boundary!(run)=foreach(at_boundary!,run.features)
features_after_rebuild!(run;launch::Bool=true)=foreach(f->after_rebuild!(run,f;launch),run.features)
features_position_sample!(run;launch::Bool=true)=foreach(f->position_sample!(run,f;launch),run.features)
any_half_drift_sample(run)=any(samples_half_drift,run.features)
features_sorted_real(features)=_cat(map(sorted_real,features))
features_sorted_int(features)=_cat(map(sorted_int,features))
_cat(t::Tuple{})=()
_cat(t::Tuple)=(first(t)...,_cat(Base.tail(t))...)

# Device: find the part with key K (first field name), compile-time unrolled.
@inline _find(::Tuple{},::Val)=nothing
@inline _find(fx::Tuple,k::Val)=_pick(first(fx),Base.tail(fx),k)
@inline _pick(p::NamedTuple{N},rest,k::Val{K}) where {N,K}=first(N)===K ? p : _find(rest,k)
