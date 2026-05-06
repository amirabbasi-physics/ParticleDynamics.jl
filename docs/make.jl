using Documenter
using ParticleDynamics

DocMeta.setdocmeta!(ParticleDynamics, :DocTestSetup, :(using ParticleDynamics); recursive=true)

makedocs(
    modules=[ParticleDynamics],
    sitename="ParticleDynamics.jl",
    format=Documenter.HTML(prettyurls=false, edit_link=nothing),
    checkdocs=:none,
    doctest=false,
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "manual/getting_started.md",
            "Simulation State" => "manual/simulation_state.md",
            "Integrators" => "manual/integrators.md",
            "Forces" => "manual/forces.md",
            "Groups and Filters" => "manual/groups_filters.md",
            "Thermostats" => "manual/thermostats.md",
            "Observables" => "manual/observables.md",
            "I/O" => "manual/io.md",
            "Restarts" => "manual/restarts.md",
        ],
        "Legacy Notes" => [
            "Collision Rate Notes" => "legacy/collision_rate.md",
        ],
    ],
)
