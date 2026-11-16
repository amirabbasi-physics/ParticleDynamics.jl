using Documenter
using ParticleDynamics

DocMeta.setdocmeta!(ParticleDynamics, :DocTestSetup, :(using ParticleDynamics); recursive=true)

makedocs(
    modules=[ParticleDynamics],
    sitename="ParticleDynamics",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true", edit_link=nothing,
                           repolink="https://github.com/amirabbasi-physics/ParticleDynamics.jl",
                           assets=["assets/site.css"], collapselevel=1),
    checkdocs=:none,
    doctest=false,
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Start here" => [
            "Your first simulation" => "quickstart.md",
            "Examples" => "examples.md",
        ],
        "Build your simulation" => [
            "Initial configurations" => "manual/initialization.md",
            "Interactions" => "manual/forces.md",
            "Dynamics" => "manual/integrators.md",
            "Temperature control" => "manual/thermostats.md",
            "Particle groups" => "manual/groups_filters.md",
            "Electrostatics" => "manual/electrostatics.md",
            "Bonds, minimization & pressure" => "manual/bonded_minimization_npt.md",
        ],
        "Measure and save" => [
            "Observables" => "manual/observables.md",
            "Trajectories and tables" => "manual/io.md",
            "Continue a simulation" => "manual/restarts.md",
        ],
        "Reference" => [
            "Workflow API" => "manual/workflow.md",
        ],
    ],
)

# Local documentation validation only builds. CI retains the existing
# Documenter deployment policy (including its own branch/PR checks).
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo="github.com/amirabbasi-physics/ParticleDynamics.jl",
        devbranch="master",
    )
end
