module Writers

using CUDA
using Printf
using StaticArrays
using GSDFiles
using ..SimulationCore
using ..Electrostatics

export InMemoryLogger, CSVWriter, XYZWriter
export write_xyz!, write_observables_csv!
export gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!, read_gsd_log, GSDFrameData, GSDTopology

include("io/WriterInterface.jl")
include("io/CSVXYZWriters.jl")
include("io/GSDIO.jl")
include("io/RestartIO.jl")

end # module
