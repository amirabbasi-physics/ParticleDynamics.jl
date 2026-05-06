module Workflow

using Base: @kwdef
using ..SimulationCore: SimulationState

include("workflow/Selections.jl")
include("workflow/System.jl")
include("workflow/Forces.jl")
include("workflow/Integrator.jl")
include("workflow/Thermostats.jl")
include("workflow/Observables.jl")
include("workflow/Writers.jl")
include("workflow/Schedules.jl")
include("workflow/Stages.jl")
include("workflow/SimulationFacade.jl")
include("workflow/RunLoop.jl")

end # module Workflow
