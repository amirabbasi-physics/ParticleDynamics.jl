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
        ],
        "Legacy Notes" => [
            "Collision Rate Notes" => "legacy/collision_rate.md",
        ],
    ],
)
