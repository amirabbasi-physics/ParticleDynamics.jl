"""
Drivers for the reference engines (GROMACS, LAMMPS) used by the cross-engine
electrostatics comparison. Everything here is host-side tooling: it writes
input files into a work directory, runs the external binaries, and parses
their outputs. Binaries are taken from the environment:

- `PD_GMX_DOUBLE` (default `gmx_d`): double-precision CPU GROMACS for references
- `PD_GMX_GPU`: GPU GROMACS for timings
- `PD_LMP_KOKKOS`: LAMMPS built with KOKKOS/CUDA
"""
module Engines

using Printf
using ..Fixtures

export read_trr, gmx_tpr, gmx_rerun, gmx_single_point, tpr_settings, gromacs_coulomb, gromacs_step_time, gromacs_cycle_accounting, gromacs_tuned_pme, openmm_water,
       zero_charge_topology, equilibrate,
       lammps_data, lammps_run

# -------------------------
# TRR (XDR, big-endian)
# -------------------------
_rd(io, ::Type{T}) where {T} = ntoh(read(io, T))

"""
    read_trr(path) -> Vector{NamedTuple}

Frames of a GROMACS `.trr` file: `(step, natoms, box, x, v, f)` with 3×N
arrays (`nothing` when absent); single or double precision.
"""
function read_trr(path::AbstractString)
    frames = []
    open(path) do io
        while !eof(io)
            magic = _rd(io, Int32)
            magic == 1993 || error("Not a TRR file: $(path)")
            _rd(io, Int32)                              # version string length + 1
            slen = _rd(io, Int32)
            skip(io, 4 * cld(slen, 4))
            ir, e, box_size, vir, pres, top, sym, x_size, v_size, f_size, natoms, step, nre =
                ntuple(_ -> _rd(io, Int32), 13)
            dbl = (box_size > 0 ? box_size : x_size) ÷ (box_size > 0 ? 9 : 3 * natoms) == 8
            R = dbl ? Float64 : Float32
            _rd(io, R); _rd(io, R)                      # t, lambda
            readvec(n) = Float64.([_rd(io, R) for _ in 1:n])
            box = box_size > 0 ? reshape(readvec(9), 3, 3) : nothing
            vir > 0 && readvec(9)
            pres > 0 && readvec(9)
            x = x_size > 0 ? reshape(readvec(3 * natoms), 3, natoms) : nothing
            v = v_size > 0 ? reshape(readvec(3 * natoms), 3, natoms) : nothing
            f = f_size > 0 ? reshape(readvec(3 * natoms), 3, natoms) : nothing
            push!(frames, (step=Int(step), natoms=Int(natoms), box=box, x=x, v=v, f=f))
        end
    end
    return frames
end

# -------------------------
# GROMACS
# -------------------------
const _GMX_D = get(ENV, "PD_GMX_DOUBLE", "gmx_d")
const _GMX_GPU = get(ENV, "PD_GMX_GPU", "gmx")

"""
    mdp_text(settings) -> String

MDP for electrostatics runs. `settings` keys: `rc`, `spacing`, `order`, `rtol`,
`modifier` ("None"/"Potential-shift"), `epsilon_r`, `rvdw`, `nsteps`, `nstfout`,
`charges_off` (cut-off electrostatics baseline), `nstlist`.
"""
function mdp_text(s::Dict)
    rc = s[:rc]
    lines = [
        "integrator = md", "dt = 0.002", "nsteps = $(get(s, :nsteps, 0))",
        "cutoff-scheme = Verlet", "nstlist = $(get(s, :nstlist, 20))",
        (get(s, :auto_buffer, false) ? "verlet-buffer-tolerance = 0.005" :
         "verlet-buffer-tolerance = -1\nrlist = $(get(s, :rlist, rc))"),
        "rvdw = $(get(s, :rvdw, rc))", "vdw-modifier = Potential-shift",
        "constraints = $(get(s, :constraints, "h-bonds"))", "tcoupl = no", "pcoupl = no",
        "continuation = $(get(s, :continuation, "no"))", "define = $(get(s, :define, ""))",
        "gen-vel = $(get(s, :gen_vel, "no"))", "gen-temp = 300",
        "nstcalcenergy = $(get(s, :nstcalcenergy, 1))", "nstenergy = $(get(s, :nstenergy, 1))",
        "nstfout = $(get(s, :nstfout, 0))", "nstxout = 0", "nstvout = 0", "nstlog = $(get(s, :nstlog, 0))",
        "epsilon-r = $(get(s, :epsilon_r, 1))",
    ]
    if get(s, :charges_off, false)
        push!(lines, "coulombtype = Cut-off", "rcoulomb = $(rc)")
    else
        push!(lines, "coulombtype = PME", "rcoulomb = $(rc)",
              "coulomb-modifier = $(get(s, :modifier, "None"))",
              "fourierspacing = $(s[:spacing])", "pme-order = $(s[:order])",
              "ewald-rtol = $(s[:rtol])", "ewald-geometry = 3d", "epsilon-surface = 0")
    end
    return join(lines, "\n") * "\n"
end

"""
    gmx_tpr(workdir, tag, gro, top, settings; gmx=gmx_d) -> path

Compile a run input in `workdir` (the topology directory is used for includes).
"""
function gmx_tpr(workdir, tag, gro, top, settings::Dict; gmx::AbstractString=_GMX_D)
    mkpath(workdir)
    mdp = joinpath(workdir, "$(tag).mdp")
    write(mdp, mdp_text(settings))
    tpr = joinpath(workdir, "$(tag).tpr")
    log = joinpath(workdir, "$(tag).grompp.log")
    cd(dirname(abspath(top))) do
        run(pipeline(`$gmx grompp -f $mdp -c $(abspath(gro)) -p $(abspath(top)) -o $tpr -po $(joinpath(workdir, "$(tag).mdout.mdp")) -maxwarn 30`;
                     stdout=log, stderr=log))
    end
    return tpr
end

"""
    tpr_settings(tpr; gmx) -> NamedTuple

FFT grid, order, cutoff and Ewald tolerance actually stored in a run input.
"""
function tpr_settings(tpr; gmx::AbstractString=_GMX_D)
    text = read(pipeline(`$gmx dump -s $tpr`; stderr=devnull), String)
    get1(key) = (m = match(Regex("\\n\\s+$(key)\\s+=\\s+(\\S+)"), text)) === nothing ? nothing : m[1]
    return (grid=(parse(Int, get1("fourier-nx")), parse(Int, get1("fourier-ny")), parse(Int, get1("fourier-nz"))),
            order=parse(Int, get1("pme-order")), rc=parse(Float64, get1("rcoulomb")),
            rtol=parse(Float64, get1("ewald-rtol")), modifier=get1("coulomb-modifier"))
end

"""
    gmx_rerun(workdir, tag, tpr, gro; gmx, gpu=false, ntomp=8)
        -> (F, energies::Dict{String,Float64}, log)

Single-frame rerun writing forces; energies read with `gmx energy -dp`.
"""
function gmx_rerun(workdir, tag, tpr, gro; gmx::AbstractString=_GMX_D, gpu::Bool=false, ntomp::Int=8,
                   terms=("Coulomb-(SR)", "Coul.-recip.", "Coulomb-14", "Potential",
                          "Vir-XX", "Vir-YY", "Vir-ZZ", "Vir-XY", "Vir-XZ", "Vir-YZ"))
    deffnm = joinpath(workdir, tag)
    devices = gpu ? `-nb gpu -pme gpu` : `-nb cpu`
    env = ("OMP_NUM_THREADS" => string(ntomp),)
    run(pipeline(setenv(`$gmx mdrun -s $tpr -rerun $(abspath(gro)) -deffnm $deffnm -ntmpi 1 -ntomp $ntomp $devices -notunepme`,
                        merge(copy(ENV), Dict(env))); stdout="$deffnm.mdrun.out", stderr="$deffnm.mdrun.out"))
    F = isfile("$deffnm.trr") ? read_trr("$deffnm.trr")[1].f : nothing   # inputs with nstfout = 0 write no forces
    energies = Dict{String,Float64}()
    for term in terms
        # terms absent from the system (e.g. Coulomb-14 in pure water) make gmx energy fail
        run(pipeline(ignorestatus(`$gmx energy -f $deffnm.edr -o $deffnm.$(replace(term, r"[^A-Za-z0-9]" => "_")).xvg -dp -xvg none`);
                     stdin=IOBuffer(term * "\n"), stdout=devnull, stderr=devnull))
        xvg = "$deffnm.$(replace(term, r"[^A-Za-z0-9]" => "_")).xvg"
        isfile(xvg) || continue
        rows = filter(l -> !isempty(strip(l)), readlines(xvg))
        isempty(rows) && continue
        vals = parse.(Float64, split(rows[1]))
        length(vals) >= 2 && (energies[term] = vals[2])
    end
    return F, energies, read("$deffnm.log", String)
end

"""
    gmx_single_point(workdir, tag, tpr; gmx, gpu=false, ntomp=8)
        -> (F, energies::Dict{String,Float64})

Step-0 forces and energy terms (including the configurational virial) of a
`nsteps = 0` run whose input was compiled with `continuation = yes` and no
constraints, so the coordinates are used exactly as given.
"""
function gmx_single_point(workdir, tag, tpr; gmx::AbstractString=_GMX_D, gpu::Bool=false, ntomp::Int=8,
                          terms=("Coulomb-(SR)", "Coul.-recip.", "Coulomb-14", "Potential",
                                 "Vir-XX", "Vir-YY", "Vir-ZZ", "Vir-XY", "Vir-XZ", "Vir-YZ"))
    deffnm = joinpath(workdir, tag)
    devices = gpu ? `-nb gpu -pme gpu` : `-nb cpu`
    run(pipeline(setenv(`$gmx mdrun -s $tpr -deffnm $deffnm -ntmpi 1 -ntomp $ntomp $devices -notunepme`,
                        merge(copy(ENV), Dict("OMP_NUM_THREADS" => string(ntomp))));
                 stdout="$deffnm.mdrun.out", stderr="$deffnm.mdrun.out"))
    F = read_trr("$deffnm.trr")[1].f
    energies = Dict{String,Float64}()
    for term in terms
        xvg = "$deffnm.$(replace(term, r"[^A-Za-z0-9]" => "_")).xvg"
        run(pipeline(ignorestatus(`$gmx energy -f $deffnm.edr -o $xvg -dp -xvg none`);
                     stdin=IOBuffer(term * "\n"), stdout=devnull, stderr=devnull))
        isfile(xvg) || continue
        rows = filter(l -> !isempty(strip(l)), readlines(xvg))
        isempty(rows) && continue
        vals = parse.(Float64, split(rows[1]))
        length(vals) >= 2 && (energies[term] = vals[2])
    end
    return F, energies
end

"""
    gromacs_coulomb(workdir, gro, top, settings; gmx, gpu=false)
        -> (F, E, W)

Electrostatic forces, energy and virial isolated exactly from two
single-point evaluations at `epsilon-r = 1` and `2`: every Coulomb term
scales with 1/εᵣ while LJ, bonded and restraint terms are identical, so
`X_coul = 2 (X₁ − X₂)`. Water is made flexible (`-DFLEXIBLE`) and constraints
are off so that the configurational virial is reported; this does not change
the electrostatic terms. GROMACS virial terms use Ξ = −½ Σ r⊗F; they are
converted to the package convention W = Σ r⊗F.
"""
function gromacs_coulomb(workdir, gro, top, settings::Dict; gmx::AbstractString=_GMX_D, gpu::Bool=false,
                         tag::AbstractString="coul")
    base = Dict{Symbol,Any}(:nstfout => 1, :constraints => "none", :continuation => "yes", :define => "-DFLEXIBLE")
    s1 = merge(settings, base, Dict(:epsilon_r => 1))
    s2 = merge(settings, base, Dict(:epsilon_r => 2))
    t1 = gmx_tpr(workdir, "$(tag)_eps1", gro, top, s1; gmx=gmx)
    t2 = gmx_tpr(workdir, "$(tag)_eps2", gro, top, s2; gmx=gmx)
    F1, e1 = gmx_single_point(workdir, "$(tag)_eps1", t1; gmx=gmx, gpu=gpu)
    F2, e2 = gmx_single_point(workdir, "$(tag)_eps2", t2; gmx=gmx, gpu=gpu)
    F = 2 .* (F1 .- F2)
    E = 2 * (e1["Potential"] - e2["Potential"])
    vir(e, a) = get(e, a, NaN)
    Ξ = [2 * (vir(e1, "Vir-$(a)") - vir(e2, "Vir-$(a)")) for a in ("XX", "YY", "ZZ", "XY", "XZ", "YZ")]
    W = -2 .* Ξ
    return F, E, W
end

"""
    equilibrate(workdir, gro, top; gmx, nsteps=10_000, extra=1_000) -> (frame_a, frame_b)

NVT molecular dynamics on the GPU (v-rescale, 300 K, dt = 2 fs, SETTLE/h-bond
constraints, PME) from `gro`, returning two configurations: after `nsteps`
steps and after `extra` further steps. The paper's water inputs are perfect
lattices (replicated single-molecule cells), whose Bragg peaks alias onto
low mesh modes; these derived liquid frames are used instead (frame A for
tuning, frame B for held-out evaluation).
"""
function equilibrate(workdir, gro, top; gmx::AbstractString=_GMX_GPU, nsteps::Int=10_000, extra::Int=1_000,
                     ntomp::Int=8)
    mkpath(workdir)
    a = joinpath(workdir, "frame_a.gro")
    b = joinpath(workdir, "frame_b.gro")
    isfile(a) && isfile(b) && return a, b
    function md(tag, start, n, genvel, out)
        mdp = joinpath(workdir, "$(tag).mdp")
        write(mdp, join(["integrator = md", "dt = 0.002", "nsteps = $(n)", "cutoff-scheme = Verlet",
                         "coulombtype = PME", "rcoulomb = 1.0", "rvdw = 1.0", "fourierspacing = 0.12",
                         "constraints = h-bonds", "tcoupl = v-rescale", "tc-grps = System", "tau-t = 0.1",
                         "ref-t = 300", "pcoupl = no", "gen-vel = $(genvel)", "gen-temp = 300",
                         "continuation = $(genvel == "yes" ? "no" : "yes")",
                         "nstxout = 0", "nstvout = 0", "nstenergy = 0", "nstlog = 0"], "\n") * "\n")
        tpr = joinpath(workdir, "$(tag).tpr")
        cd(dirname(abspath(top))) do
            run(pipeline(`$gmx grompp -f $mdp -c $start -p $(abspath(top)) -o $tpr -maxwarn 30`; stdout=devnull, stderr=devnull))
        end
        run(pipeline(setenv(`$gmx mdrun -s $tpr -deffnm $(joinpath(workdir, tag)) -c $out -ntmpi 1 -ntomp $ntomp -pin on -nb gpu -pme gpu -update gpu`,
                            merge(copy(ENV), Dict("OMP_NUM_THREADS" => string(ntomp))));
                     stdout=joinpath(workdir, "$(tag).out"), stderr=joinpath(workdir, "$(tag).out")))
    end
    md("eq", abspath(gro), nsteps, "yes", a)
    md("eq_b", a, extra, "no", b)
    return a, b
end

"""
    zero_charge_topology(top, workdir; gmx) -> path

Fully preprocessed copy of `top` (`grompp -pp`) with every charge in the
`[ atoms ]` sections set to zero, for LJ-only baseline runs.
"""
function zero_charge_topology(gro, top, workdir; gmx::AbstractString=_GMX_GPU)
    mkpath(workdir)
    pp = joinpath(workdir, "processed.top")
    mdp = joinpath(workdir, "pp.mdp")
    write(mdp, "integrator = md\ncutoff-scheme = Verlet\n")
    cd(dirname(abspath(top))) do
        run(pipeline(`$gmx grompp -f $mdp -c $(abspath(gro)) -p $(abspath(top)) -pp $pp -o $(joinpath(workdir, "pp.tpr")) -maxwarn 30`;
                     stdout=devnull, stderr=devnull))
    end
    out = joinpath(workdir, "zero_charge.top")
    section = ""
    open(out, "w") do io
        for l in eachline(pp)
            m = match(r"^\s*\[\s*(\w+)\s*\]", l)
            m !== nothing && (section = m[1])
            if section == "atoms" && !startswith(strip(l), ";") && !startswith(strip(l), "[") && !isempty(strip(l))
                cols = split(l)
                if length(cols) >= 7
                    cols[7] = "0.0"
                    l = join(cols, " ")
                end
            end
            println(io, l)
        end
    end
    return out
end

"""
    gromacs_step_time(workdir, tag, gro, top, settings; gmx=gmx_gpu, nsteps=4000)
        -> (ms_per_step, log)

GPU-resident MD (`-nb gpu -pme gpu -update gpu`, PME tuning off) with the
given settings; returns the wall time per step of the second half of the run.
"""
function gromacs_step_time(workdir, tag, gro, top, settings::Dict; gmx::AbstractString=_GMX_GPU,
                           nsteps::Int=4000, ntomp::Int=8, tunepme::Bool=false)
    s = merge(Dict{Symbol,Any}(:nstlog => 0), settings,
              Dict(:nsteps => nsteps, :nstcalcenergy => 1000, :nstenergy => 0, :gen_vel => "yes"))
    tpr = gmx_tpr(workdir, tag, gro, top, s; gmx=gmx)
    deffnm = joinpath(workdir, tag)
    pme = get(settings, :charges_off, false) ? `-nb gpu` : `-nb gpu -pme gpu`
    # PME load balancing must finish before the counters are reset: tuned runs
    # reset at 3/4 of the run instead of half-way
    tune = tunepme ? `-tunepme -resetstep $(3 * nsteps ÷ 4)` : `-notunepme -resethway`
    run(pipeline(setenv(`$gmx mdrun -s $tpr -deffnm $deffnm -ntmpi 1 -ntomp $ntomp -pin on $pme -update gpu $tune -noconfout`,
                        merge(copy(ENV), Dict("OMP_NUM_THREADS" => string(ntomp))));
                 stdout="$deffnm.mdrun.out", stderr="$deffnm.mdrun.out"))
    log = read("$deffnm.log", String)
    m = match(r"Performance:\s+([0-9.]+)\s+([0-9.]+)", log)
    m === nothing && error("No performance line in $(deffnm).log")
    ns_per_day = parse(Float64, m[1])
    ms = 2e-6 / ns_per_day * 86400 * 1e3           # dt = 2 fs = 2e-6 ns
    return ms, log
end

"""
    gromacs_tuned_pme(log) -> (grid, rcoulomb) or nothing

Final PME grid and Coulomb cutoff chosen by GROMACS's PME load balancing
(the "final" row of the md.log load-balancing table; needs `nstlog > 0`).
"""
function gromacs_tuned_pme(log::AbstractString)
    m = match(r"^\s+final\s+([0-9.]+) nm\s+([0-9.]+) nm\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+) nm"m, log)
    m === nothing && return nothing
    return (grid=(parse(Int, m[3]), parse(Int, m[4]), parse(Int, m[5])), rcoulomb=parse(Float64, m[1]),
            rlist=parse(Float64, m[2]), spacing=parse(Float64, m[6]))
end

"""
    gromacs_cycle_accounting(log) -> NamedTuple

Per-step times (ms) from the md.log cycle accounting of a timed run (counters
reset half-way): the whole step, the CPU pair search amortised over steps,
and `gpu_estimate = step − search`, the GPU-resident part (nonbonded, PME,
update, constraints). The GPU waits for the CPU search, so the two do not
overlap.
"""
function gromacs_cycle_accounting(log::AbstractString)
    row(name) = match(Regex("^ " * name * "\\s+\\d+\\s+\\d+\\s+(\\d+)\\s+([0-9.]+)", "m"), log)
    f = row("Force")
    tot = match(r"^ Total\s+([0-9.]+)\s+[0-9.]+\s+100\.0"m, log)
    ns = row("Neighbor search")
    (f === nothing || tot === nothing) && return nothing
    steps = parse(Int, f[1])
    step = 1e3 * parse(Float64, tot[1]) / steps
    search = ns === nothing ? 0.0 : 1e3 * parse(Float64, ns[2]) / steps
    return (steps=steps, step=step, search=search, gpu_estimate=step - search)
end

# -------------------------
# OpenMM (CUDA, mixed precision) through openmm_water.py
# -------------------------
const _OPENMM_PY = get(ENV, "PD_OPENMM_PYTHON", "/net/storage/abbaa90/openmm-gpu-env/bin/python")
const _GMX_TOPDIR = get(ENV, "PD_GMX_TOPDIR", "/net/storage/abbaa90/gromacs-gpu/share/gromacs/top")

"""
    openmm_water(gro, top, mode, tol; workdir, nsteps=2000) -> NamedTuple

Run `openmm_water.py` (`mode` = "forces", "w1" or "w2"; `tol` = OpenMM
`ewaldErrorTolerance`). Returns `(alpha, grid, ms_per_step, F)`; `F` is the
3×N Coulomb force matrix (kJ/mol/nm, .gro order) for "forces", else `nothing`.
"""
function openmm_water(gro, top, mode::AbstractString, tol::Real; workdir::AbstractString, nsteps::Int=2000)
    out = joinpath(mkpath(workdir), "openmm_forces_$(tol).bin")
    script = joinpath(@__DIR__, "openmm_water.py")
    line = readchomp(`$(_OPENMM_PY) $script $gro $top $(_GMX_TOPDIR) $mode $tol $out $nsteps`)
    m = match(r"alpha ([0-9.eE+-]+) grid (\d+) (\d+) (\d+) ms_per_step ([0-9.eE+-]+)", split(line, "\n")[end])
    m === nothing && error("unexpected OpenMM output: $(line)")
    F = nothing
    if mode == "forces"
        v = Vector{Float64}(undef, filesize(out) ÷ 8)
        read!(out, v)
        F = reshape(v, 3, :)
    end
    return (alpha=parse(Float64, m[1]), grid=(parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4])),
            ms_per_step=parse(Float64, m[5]), F=F)
end

# -------------------------
# LAMMPS (units real)
# -------------------------
const _LMP = get(ENV, "PD_LMP_KOKKOS", "lmp")

"""
    lammps_data(path, f::Fixture)

Write an `atom_style full` data file with one bond per exception (bond and
angle types only define exclusions; their styles are `zero`). Atom order
follows `f.ids`.
"""
function lammps_data(path::AbstractString, f::Fixture; types=nothing, length_scale::Real=1.0)
    order = sortperm(f.ids)
    newid = similar(order); newid[order] = 1:length(order)
    typ = types === nothing ? ones(Int, length(f)) : types
    ntypes = maximum(typ)
    open(path, "w") do io
        println(io, "electrostatics fixture $(f.name)\n")
        @printf(io, "%d atoms\n%d atom types\n%d bonds\n1 bond types\n\n", length(f), ntypes, length(f.exceptions))
        λ = length_scale
        for (d, ax) in enumerate(("x", "y", "z"))
            @printf(io, "0.0 %.12f %slo %shi\n", λ * f.box[d], ax, ax)
        end
        println(io, "\nMasses\n")
        for t in 1:ntypes
            println(io, "$(t) 1.0")
        end
        println(io, "\nAtoms # full\n")
        for (k, r) in enumerate(order)
            @printf(io, "%d 1 %d %.10f %.12f %.12f %.12f\n", k, typ[r], f.q[r],
                    λ * mod(f.R[1, r], f.box[1]), λ * mod(f.R[2, r], f.box[2]), λ * mod(f.R[3, r], f.box[3]))
        end
        println(io, "\nBonds\n")
        for (b, (i, j, s)) in enumerate(f.exceptions)
            s == 0.0 || error("lammps_data encodes only full exclusions (scale 0); got $(s).")
            @printf(io, "%d 1 %d %d\n", b, newid[i], newid[j])
        end
    end
    return path
end

"""
    lammps_run(workdir, tag, data; rc, accuracy, order=5, nsteps=200, kokkos=true,
               dump_forces=true, mesh=nothing) -> NamedTuple

Frozen-coordinate PPPM force evaluations: no time integration fix, so every
step evaluates `coul/long` + `pppm` forces on the same coordinates. Reports
the synchronized per-step Pair and Kspace times (`timer sync full`), the
chosen G-ewald/grid, and optionally the step-0 forces in data-file order.
"""
function lammps_run(workdir, tag, data; rc::Real, accuracy::Real, order=5, nsteps::Int=200,
                    kokkos::Bool=true, dump_forces::Bool=true, mesh=nothing, lmp::AbstractString=_LMP,
                    pair::AbstractString="coul/long $(rc)", kspace::AbstractString="pppm $(accuracy)",
                    pair_coeffs=["* *"])
    mkpath(workdir)
    inp = joinpath(workdir, "$(tag).in")
    fdump = joinpath(workdir, "$(tag).forces")
    open(inp, "w") do io
        println(io, "units real\natom_style full\nboundary p p p")
        kokkos && println(io, "package kokkos neigh half newton on comm device")
        println(io, "read_data $(abspath(data))")
        println(io, "pair_style $(pair)")
        for c in pair_coeffs
            println(io, "pair_coeff $(c)")
        end
        # exclusions are encoded as bonds: only the listed (1-2) pairs are excluded;
        # 1-3/1-4 neighbours derived from that graph keep full interactions
        println(io, "bond_style zero\nbond_coeff *\nspecial_bonds lj/coul 0.0 1.0 1.0")
        println(io, "kspace_style $(kspace)")
        order === nothing || println(io, "kspace_modify order $(order)")
        mesh === nothing || println(io, "kspace_modify mesh $(mesh[1]) $(mesh[2]) $(mesh[3])")
        println(io, "neighbor 1.0 bin\nneigh_modify every 1000 delay 1000 check no")
        println(io, "thermo_style custom step pe ecoul elong\nthermo_modify format float %.12f\nthermo $(nsteps)")
        println(io, "timer sync full")
        if dump_forces
            println(io, "dump d all custom $(nsteps + 1) $(fdump) id fx fy fz\ndump_modify d format float %.17g sort id")
        end
        println(io, "run $(nsteps)")
    end
    log = joinpath(workdir, "$(tag).log")
    cmd = kokkos ? `$lmp -k on g 1 -sf kk -in $inp -log $log -screen none` : `$lmp -in $inp -log $log -screen none`
    run(pipeline(cmd; stdout=devnull, stderr=joinpath(workdir, "$(tag).err")))
    text = read(log, String)
    timing(sec) = (m = match(Regex("\\n$(sec)\\s+\\|\\s+[0-9.e+-]+\\s+\\|\\s+([0-9.e+-]+)"), text)) === nothing ?
                  NaN : parse(Float64, m[1])
    loop = match(r"Loop time of ([0-9.e+-]+) on", text)
    thermo = match(r"\n\s+0\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)", text)
    gewald = match(r"G vector \(1/distance\) = ([0-9.e+-]+)", text)
    grid = match(r"grid = (\d+) (\d+) (\d+)", text)
    esp_c = match(r"Splitting parameter c = ([0-9.e+-]+)", text)
    esp_c1 = match(r"Spreading parameter c = ([0-9.e+-]+)", text)
    stencil = match(r"stencil order = (\d+)", text)
    F = nothing
    if dump_forces && isfile(fdump)
        rows = readlines(fdump)
        k = findfirst(l -> startswith(l, "ITEM: ATOMS"), rows)
        vals = [parse.(Float64, split(l)) for l in rows[(k + 1):end]]
        F = Matrix{Float64}(undef, 3, length(vals))
        for v in vals
            F[:, Int(v[1])] .= v[2:4]
        end
    end
    return (pair_ms=1e3 * timing("Pair") / nsteps, kspace_ms=1e3 * timing("Kspace") / nsteps,
            loop_ms=loop === nothing ? NaN : 1e3 * parse(Float64, loop[1]) / nsteps,
            energy=thermo === nothing ? NaN : parse(Float64, thermo[1]),
            gewald=gewald === nothing ? NaN : parse(Float64, gewald[1]),
            grid=grid === nothing ? nothing : parse.(Int, (grid[1], grid[2], grid[3])),
            order=stencil === nothing ? nothing : parse(Int, stencil[1]),
            esp_split_shape=esp_c === nothing ? NaN : parse(Float64, esp_c[1]),
            esp_window_shape=esp_c1 === nothing ? NaN : parse(Float64, esp_c1[1]), F=F, log=text)
end

end # module Engines
