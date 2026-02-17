using Documenter
using NonEqSimGPU

DocMeta.setdocmeta!(NonEqSimGPU, :DocTestSetup, :(using NonEqSimGPU); recursive=true)

makedocs(
    modules=[NonEqSimGPU],
    sitename="NonEqSimGPU.jl",
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
