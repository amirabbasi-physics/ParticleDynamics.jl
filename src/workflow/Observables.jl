abstract type Observable end

@kwdef struct ThermodynamicObservable <: Observable
    group
    name::Symbol = :thermo
end

@kwdef struct BathExchangeObservable <: Observable
    name::Symbol = :bath
end

@kwdef struct VirialObservable <: Observable
    group
    name::Symbol = :virial
end

@kwdef struct CollisionObservable <: Observable
    name::Symbol = :collisions
end

@kwdef struct MSDObservable <: Observable
    group
    name::Symbol = :msd
    reference = :start
end

@kwdef struct VACFObservable <: Observable
    group
    name::Symbol = :vacf
    reference = :start
end
