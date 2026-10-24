"""CPU-only input generation and result checks for the GROMACS comparison."""
module Protocol

using Printf, SHA, TOML

export read_argon, prepare_argon, mdrun_command, performance_ms, require_gpu_offload,
       source_fingerprint, fresh_directory, run_logged, EngineProcessError, compile_input

struct EngineProcessError <: Exception
    executable::String
    logfile::String
    exitcode::Int
    signal::Int
end

function Base.showerror(io::IO, e::EngineProcessError)
    reason = e.signal == 0 ? "exit $(e.exitcode)" : "signal $(e.signal)"
    print(io, basename(e.executable), " failed (", reason, "); details: ", e.logfile)
end

"""Run an external engine without printing its inherited environment on failure."""
function run_logged(cmd::Cmd, logfile::AbstractString)
    proc = run(pipeline(ignorestatus(cmd); stdout=logfile, stderr=logfile))
    success(proc) || throw(EngineProcessError(first(cmd), abspath(logfile),
                                            Int(proc.exitcode), Int(proc.termsignal)))
    return nothing
end

function compile_input(gmx, dir, name)
    tpr = joinpath(dir, "$name.tpr")
    log = joinpath(dir, "$name.grompp.log")
    run_logged(`$gmx grompp -f $(joinpath(dir, "$name.mdp")) -c $(joinpath(dir, "input.gro")) -p $(joinpath(dir, "argon.top")) -o $tpr -po $(joinpath(dir, "$name.expanded.mdp")) -maxwarn 1`, log)
    return tpr
end

const SIGMA_NM = 0.3405
const EPSILON_KJMOL = 119.8 * 0.00831446261815324
const MASS_AMU = 39.948
const TAU_PS = SIGMA_NM * sqrt(MASS_AMU / EPSILON_KJMOL)

function read_argon(dir)
    m = TOML.parsefile(joinpath(dir, "metadata.toml"))
    N = Int(m["N"])
    N > 0 || error("Empty Argon fixture")
    function triplets(key, fallback)
        p = joinpath(dir, get(m, key, fallback))
        filesize(p) == 3N * sizeof(Float32) || error("Unexpected triplet file size: $p")
        return reshape(open(io -> read!(io, Vector{Float32}(undef, 3N)), p), 3, N)
    end
    R = triplets("position_file", "positions_float32.bin")
    # These fixtures call the m=1 velocities 'momenta'. Reject other masses
    # rather than silently guessing which convention their producer used.
    m["argon_mass"] == 1 || error("This fixture adapter requires reduced mass = 1")
    V = triplets("momenta_file", "momenta_float32.bin")
    L, rc, dt = Float64.((m["box_length"], m["cutoff_radius"], m["timestep"]))
    all(isfinite, R) && all(isfinite, V) || error("Nonfinite fixture")
    0 < rc < L / 2 && dt > 0 || error("Invalid cutoff/box/timestep")
    paths = ["metadata.toml", get(m,"position_file","positions_float32.bin"), get(m,"momenta_file","momenta_float32.bin")]
    input_hashes = Dict(name => bytes2hex(sha256(read(joinpath(dir,name)))) for name in paths)
    return (; N, R, V, L, rc, dt, sigma=Float64(m["lj_sigma"]),
            epsilon=Float64(m["lj_epsilon"]), metadata=m, input_hashes)
end

"""Make a new result directory; never reuse timings or overwrite a prior run."""
function fresh_directory(path)
    path = abspath(path)
    ispath(path) && error("Result directory already exists: $path. Choose a new directory.")
    mkpath(path)
    return path
end

function source_fingerprint(root)
    files = String[]
    for sub in ("src", "benchmark"), (dir, _, names) in walkdir(joinpath(root, sub))
        append!(files, [joinpath(dir, n) for n in names if endswith(n, ".jl")])
    end
    append!(files, filter(isfile, [joinpath(root, n) for n in ("Project.toml", "Manifest.toml")]))
    io = IOBuffer()
    for f in sort!(files)
        println(io, relpath(f, root), " ", bytes2hex(sha256(read(f))))
    end
    return bytes2hex(sha256(take!(io)))
end

function prepare_argon(c, dir; warmup=1000, steps=1000, nstlist=20, energy_every=100)
    warmup > 0 && steps > 0 && nstlist > 0 && energy_every > 0 || error("Intervals must be positive")
    warmup % energy_every == 0 && steps % energy_every == 0 ||
        error("warmup and steps must be multiples of energy_every")
    mkpath(dir)
    gro, top = joinpath(dir, "input.gro"), joinpath(dir, "argon.top")
    open(gro, "w") do io
        println(io, "Shared reduced Argon fixture mapped to nm, ps, kJ/mol")
        println(io, c.N)
        for i in 1:c.N
            @printf(io, "%5d%-5s%5s%5d", mod(i, 100000), "AR", "AR", mod(i, 100000))
            for d in 1:3; @printf(io, "%13.8f", c.R[d,i] * SIGMA_NM); end
            for d in 1:3; @printf(io, "%13.9f", c.V[d,i] * SIGMA_NM / TAU_PS); end
            println(io)
        end
        @printf(io, "%15.9f%15.9f%15.9f\n", ntuple(_ -> c.L * SIGMA_NM, 3)...)
    end
    write(top, """
    [ defaults ]
    1 2 no 1.0 1.0
    [ atomtypes ]
    AR 18 $MASS_AMU 0.0 A $(c.sigma * SIGMA_NM) $(c.epsilon * EPSILON_KJMOL)
    [ moleculetype ]
    AR 0
    [ atoms ]
    1 AR 1 AR AR 1 0.0 $MASS_AMU
    [ system ]
    Argon NVE GPU comparison
    [ molecules ]
    AR $(c.N)
    """)
    common = """
    integrator = md
    dt = $(c.dt * TAU_PS)
    constraints = none
    tcoupl = no
    pcoupl = no
    comm-mode = None
    pbc = xyz
    gen-vel = no
    cutoff-scheme = Verlet
    nstlist = $nstlist
    verlet-buffer-tolerance = 0.005
    coulombtype = Cut-off
    coulomb-modifier = None
    rcoulomb = $(c.rc * SIGMA_NM)
    vdwtype = Cut-off
    vdw-modifier = None
    rvdw = $(c.rc * SIGMA_NM)
    DispCorr = no
    nstxout = 0
    nstvout = 0
    nstxout-compressed = 0
    nstlog = 0
    """
    write(joinpath(dir, "production.mdp"), common * """
    nsteps = $(warmup + steps)
    nstcalcenergy = $energy_every
    nstenergy = 0
    nstfout = 0
    """)
    # An independent CPU double-precision force/energy reference, not a timer.
    write(joinpath(dir, "reference.mdp"), common * """
    continuation = yes
    nsteps = 0
    nstcalcenergy = 1
    nstenergy = 1
    nstfout = 1
    """)
    return (; gro, top, dt_ps=c.dt * TAU_PS)
end

function mdrun_command(gmx, tpr, prefix; ntomp=8, warmup=1000, pme=false)
    ntomp > 0 && warmup > 0 || throw(ArgumentError("ntomp and warmup must be positive"))
    mesh = pme ? `-pme gpu -notunepme` : ``
    return `$gmx mdrun -s $tpr -deffnm $prefix -ntmpi 1 -ntomp $ntomp -pin on -gpu_id 0 -nb gpu -update gpu $mesh -resetstep $warmup -noconfout`
end

function require_gpu_offload(log; pme=false)
    occursin(r"PP tasks will do .*short-ranged interactions on the GPU", log) ||
        error("GROMACS log does not confirm GPU nonbonded execution")
    occursin("PP task will update and constrain coordinates on the GPU", log) ||
        error("GROMACS log does not confirm GPU coordinate update")
    !pme || occursin("PME tasks will do all aspects on the GPU", log) ||
        error("GROMACS log does not confirm full GPU PME execution")
    return true
end

function performance_ms(log, dt_ps)
    dt_ps > 0 && isfinite(dt_ps) || throw(ArgumentError("Invalid timestep"))
    rows = collect(eachmatch(r"Performance:\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)", log))
    isempty(rows) && error("No GROMACS performance measurement")
    rate = parse(Float64, last(rows)[1])
    isfinite(rate) && rate > 0 || error("Invalid GROMACS ns/day")
    return dt_ps * 86400 / rate  # ps → ns and seconds → ms cancel
end

end
