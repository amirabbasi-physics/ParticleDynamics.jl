"""
Frozen-coordinate electrostatics fixtures for cross-engine benchmarks.

A `Fixture` holds exactly what the Coulomb Hamiltonian needs: positions,
an orthorhombic box, charges, the excluded/scaled pair list and the Coulomb
prefactor of the source engine's unit system, with source checksums and the
original atom order. Bonded, angular, dihedral and LJ terms are not imported;
only their effect on electrostatics (exclusions and 1–4 scaling) is.

Loaders:
- `gromacs_fixture(gro; top=…, tpr=…)`: coordinates from the `.gro` file; charges,
  exclusions and 1–4 pairs (`fudgeQQ`) from the compiled run input, read through
  `gmx dump`. Topologies are compiled with `gmx grompp` first.
- `lammps_fixture(data; special=(0, 0, 0.5))`: `atom_style full` data file, atoms
  sorted by ID; `special_bonds lj/coul` factors give the 1–2/1–3/1–4 scales.
"""
module Fixtures

using Printf

export Fixture, gromacs_fixture, lammps_fixture, replicate, spatially_sorted, write_gro,
       GROMACS_COULOMB, LAMMPS_REAL_COULOMB

# GROMACS ONE_4PI_EPS0 in nm·kJ/mol/e² (CODATA 2018 constants, as in GROMACS 2025/2026).
const GROMACS_COULOMB = 1.602176634e-19^2 * 6.02214076e23 / (4pi * 8.8541878128e-12) / (1e-9 * 1e3)
# LAMMPS `units real` qqr2e (Å·kcal/mol/e²).
const LAMMPS_REAL_COULOMB = 332.06371

struct Fixture
    name::String
    ids::Vector{Int}                                  # original 1-based atom index of each row
    R::Matrix{Float64}                                # 3×N positions (source length unit)
    box::NTuple{3,Float64}
    q::Vector{Float64}
    exceptions::Vector{Tuple{Int32,Int32,Float64}}    # (i, j, scale) in row order, i < j
    prefactor::Float64
    units::String
    source::Vector{Pair{String,String}}               # file => md5
end

Base.length(f::Fixture) = length(f.q)

_md5(path) = first(split(read(`md5sum $path`, String)))

# -------------------------
# GROMACS
# -------------------------

"""
    read_gro(path) -> (R, box)

Positions (3×N) and orthorhombic box of a `.gro` file. Variable coordinate
precision is supported; nonzero off-diagonal box entries are rejected.
"""
function read_gro(path::AbstractString)
    lines = readlines(path)
    N = parse(Int, strip(lines[2]))
    first_atom = lines[3]
    d1 = findnext('.', first_atom, 21)
    d2 = findnext('.', first_atom, d1 + 1)
    w = d2 - d1                                        # field width
    R = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        l = lines[2 + i]
        for d in 1:3
            R[d, i] = parse(Float64, l[(21 + (d - 1) * w):(20 + d * w)])
        end
    end
    b = parse.(Float64, split(lines[3 + N]))
    length(b) == 3 || all(abs.(b[4:end]) .== 0) ||
        throw(ArgumentError("$(path) has a triclinic box; only orthorhombic cells are supported."))
    return R, (b[1], b[2], b[3])
end

function _compile_tpr(gro, top, workdir; gmx)
    mdp = joinpath(workdir, "fixture.mdp")
    write(mdp, "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\ncoulombtype = PME\nrcoulomb = 0.9\nrvdw = 0.9\n")
    tpr = joinpath(workdir, "fixture.tpr")
    cd(dirname(top)) do
        run(pipeline(`$gmx grompp -f $mdp -c $gro -p $top -o $tpr -maxwarn 20`;
                     stdout=joinpath(workdir, "grompp.log"), stderr=joinpath(workdir, "grompp.log")))
    end
    return tpr
end

"""
    parse_tpr_dump(text) -> (moltypes, molblocks, fudgeQQ)

Charges, exclusion lists and LJ-14 pairs of every molecule type from
`gmx dump -s` output, plus the molecule block layout.
"""
function parse_tpr_dump(lines::Vector{String})
    moltypes = Dict{Int,Any}()
    molblocks = Tuple{Int,Int}[]
    fudge = 1.0
    i = 1
    n = length(lines)
    current = nothing
    section = :none
    while i <= n
        l = lines[i]
        if (m = match(r"^\s+moltype\s+=\s+(\d+)", l)) !== nothing
            mt = parse(Int, m[1])
            j = i + 1
            while !occursin("#molecules", lines[j]); j += 1; end
            push!(molblocks, (mt, parse(Int, split(lines[j], "=")[2])))
        elseif (m = match(r"^\s+fudgeQQ\s+=\s+(\S+)", l)) !== nothing
            fudge = parse(Float64, m[1])
        elseif (m = match(r"^\s{3}moltype \((\d+)\):", l)) !== nothing
            current = Dict(:q => Float64[], :excls => Vector{Int}[], :lj14 => Tuple{Int,Int}[])
            moltypes[parse(Int, m[1])] = current
            section = :none
        elseif current !== nothing
            if (m = match(r"^\s+atom\[\s*\d+\]=\{.*?\sq=\s*([-+0-9.eE]+)", l)) !== nothing
                push!(current[:q], parse(Float64, m[1]))
            elseif occursin(r"^\s+excls\[\d+\]\[num=\d+\]=\{", l)
                # long lists wrap over several lines until the closing brace
                body = l[(findfirst('{', l) + 1):end]
                while !occursin('}', body)
                    i += 1
                    body *= lines[i]
                end
                body = strip(body[1:(findfirst('}', body) - 1)])
                push!(current[:excls], isempty(body) ? Int[] : parse.(Int, filter(!isempty, strip.(split(body, ",")))))
            elseif occursin(r"^\s{6}LJ-14:", l)
                section = :lj14
            elseif occursin(r"^\s{6}\S.*:$", l)
                section = :none
            elseif section === :lj14 && (m = match(r"\(LJ14\)\s+(\d+)\s+(\d+)", l)) !== nothing
                push!(current[:lj14], (parse(Int, m[1]), parse(Int, m[2])))
            end
        end
        i += 1
    end
    return moltypes, molblocks, fudge
end

"""
    gromacs_fixture(gro; top=nothing, tpr=nothing, gmx="gmx_d", name=basename(gro)) -> Fixture

Charges come from the compiled topology. Every excluded pair (including the
GROMACS `nrexcl` neighbours) gets scale 0, and LJ-14 pairs get `fudgeQQ`
(pairs with scale 1 are not exceptions). Units: nm, kJ/mol, e.
"""
function gromacs_fixture(gro::AbstractString; top=nothing, tpr=nothing, gmx::AbstractString="gmx_d",
                         name::AbstractString=basename(gro), workdir::AbstractString=mktempdir())
    R, box = read_gro(gro)
    sources = [gro => _md5(gro)]
    if tpr === nothing
        top === nothing && throw(ArgumentError("gromacs_fixture needs `top` or `tpr`."))
        tpr = _compile_tpr(abspath(gro), abspath(top), workdir; gmx=gmx)
        push!(sources, top => _md5(top))
    else
        push!(sources, tpr => _md5(tpr))
    end
    dump = readlines(pipeline(`$gmx dump -s $tpr`; stderr=devnull))
    moltypes, molblocks, fudge = parse_tpr_dump(dump)
    q = Float64[]
    exc = Tuple{Int32,Int32,Float64}[]
    offset = 0
    for (mt, nmol) in molblocks
        t = moltypes[mt]
        na = length(t[:q])
        lj14 = Set((min(a, b), max(a, b)) for (a, b) in t[:lj14])
        local_pairs = Tuple{Int,Int,Float64}[]
        for a in 0:(na - 1), b in t[:excls][a + 1]
            b > a || continue
            s = (a, b) in lj14 ? fudge : 0.0
            s == 1.0 || push!(local_pairs, (a, b, s))
        end
        for (a, b) in lj14                       # 1–4 pairs not listed as exclusions
            any(p -> p[1] == a && p[2] == b, local_pairs) && continue
            b in t[:excls][a + 1] && continue
            fudge == 1.0 || push!(local_pairs, (a, b, fudge))
        end
        for m in 0:(nmol - 1)
            base = offset + m * na
            append!(q, t[:q])
            for (a, b, s) in local_pairs
                push!(exc, (Int32(base + a + 1), Int32(base + b + 1), s))
            end
        end
        offset += nmol * na
    end
    offset == size(R, 2) ||
        throw(ArgumentError("Topology has $(offset) atoms but $(gro) has $(size(R, 2))."))
    return Fixture(name, collect(1:offset), R, box, q, exc, GROMACS_COULOMB, "nm, kJ/mol, e", sources)
end

"""
    write_gro(path, fixture; precision=7)

Write the fixture's coordinates in source order (for reference reruns).
"""
function write_gro(path::AbstractString, f::Fixture; names=nothing, precision::Int=7)
    w = precision + 5
    order = sortperm(f.ids)
    open(path, "w") do io
        println(io, f.name)
        println(io, length(f))
        fmt = Printf.Format("%5d%-5s%5s%5d" * "%$(w).$(precision)f"^3 * "\n")
        for (k, r) in enumerate(order)
            nm = names === nothing ? "X" : names[f.ids[r]]
            Printf.format(io, fmt, k % 100000, "MOL", nm, k % 100000, f.R[1, r], f.R[2, r], f.R[3, r])
        end
        Printf.format(io, Printf.Format("%$(w).$(precision)f"^3 * "\n"), f.box...)
    end
    return path
end

# -------------------------
# LAMMPS
# -------------------------

"""
    lammps_fixture(data; special=(0.0, 0.0, 0.5), name=basename(data)) -> Fixture

`atom_style full` data file (`units real`: Å, kcal/mol, e). Atoms are ordered
by ID. Bonds give 1–2 pairs, angle end atoms 1–3 pairs and dihedral end atoms
1–4 pairs; `special` holds the `special_bonds coul` factors. Pairs are taken
at their minimum-image separation.
"""
function lammps_fixture(data::AbstractString; special=(0.0, 0.0, 0.5), name::AbstractString=basename(data))
    lines = readlines(data)
    N = 0
    lo = zeros(3); hi = zeros(3)
    for l in lines
        (m = match(r"^\s*(\d+)\s+atoms", l)) !== nothing && (N = parse(Int, m[1]))
        for (d, ax) in enumerate(("x", "y", "z"))
            if (m = match(Regex("^\\s*(\\S+)\\s+(\\S+)\\s+$(ax)lo\\s+$(ax)hi"), l)) !== nothing
                lo[d] = parse(Float64, m[1]); hi[d] = parse(Float64, m[2])
            end
        end
        occursin(r"xy\s+xz\s+yz", l) && throw(ArgumentError("Triclinic LAMMPS boxes are not supported."))
    end
    section(head) = findfirst(l -> startswith(strip(l), head), lines)
    function rows(head, ncol)
        s = section(head)
        s === nothing && return Vector{Vector{Float64}}()
        out = Vector{Vector{Float64}}()
        k = s + 1
        while k <= length(lines) && isempty(strip(lines[k])); k += 1; end
        while k <= length(lines) && !isempty(strip(lines[k]))
            push!(out, parse.(Float64, split(lines[k])[1:ncol]))
            k += 1
        end
        return out
    end
    atoms = rows("Atoms", 7)
    length(atoms) == N || throw(ArgumentError("Expected $(N) atom rows, found $(length(atoms))."))
    sort!(atoms; by=a -> a[1])
    R = Matrix{Float64}(undef, 3, N)
    q = Vector{Float64}(undef, N)
    for (k, a) in enumerate(atoms)
        Int(a[1]) == k || throw(ArgumentError("Atom IDs must be 1…N."))
        q[k] = a[4]
        R[:, k] .= a[5:7]
    end
    scales = Dict{Tuple{Int32,Int32},Float64}()
    key(i, j) = i < j ? (Int32(i), Int32(j)) : (Int32(j), Int32(i))
    for (rowsdata, s) in ((rows("Dihedrals", 6), special[3]), (rows("Angles", 5), special[2]),
                          (rows("Bonds", 4), special[1]))         # closer neighbours override
        for r in rowsdata
            i, j = Int(r[3]), Int(r[end])
            scales[key(i, j)] = s
        end
    end
    exc = Tuple{Int32,Int32,Float64}[(k[1], k[2], s) for (k, s) in scales if s != 1.0]
    sort!(exc)
    box = (hi[1] - lo[1], hi[2] - lo[2], hi[3] - lo[3])
    return Fixture(name, collect(1:N), R, box, q, exc, LAMMPS_REAL_COULOMB, "Å, kcal/mol, e",
                   [data => _md5(data)])
end

# -------------------------
# Derived fixtures
# -------------------------

"""
    replicate(f, nx, ny, nz) -> Fixture

Periodic replication (a derived fixture; atom IDs continue image by image).
"""
function replicate(f::Fixture, nx::Int, ny::Int, nz::Int)
    N = length(f)
    reps = nx * ny * nz
    R = Matrix{Float64}(undef, 3, N * reps)
    q = repeat(f.q, reps)
    exc = Tuple{Int32,Int32,Float64}[]
    k = 0
    for ix in 0:(nx - 1), iy in 0:(ny - 1), iz in 0:(nz - 1)
        shift = [ix * f.box[1], iy * f.box[2], iz * f.box[3]]
        R[:, (k * N + 1):((k + 1) * N)] .= f.R .+ shift
        for (i, j, s) in f.exceptions
            push!(exc, (Int32(i + k * N), Int32(j + k * N), s))
        end
        k += 1
    end
    box = (nx * f.box[1], ny * f.box[2], nz * f.box[3])
    return Fixture("$(f.name) ×$(nx)$(ny)$(nz)", collect(1:(N * reps)), R, box, q, exc, f.prefactor, f.units, f.source)
end

"""
    spatially_sorted(f; cell) -> Fixture

Rows reordered by spatial cell (x fastest), exceptions remapped; `ids` keeps
the original atom index of every row. Engines sort atoms internally too
(GROMACS clusters, LAMMPS bins); this gives ParticleDynamics the same
locality without the spatial-reorder machinery.
"""
function spatially_sorted(f::Fixture; cell::Real)
    nc = ntuple(d -> max(1, floor(Int, f.box[d] / cell)), 3)
    key = map(1:length(f)) do i
        c = ntuple(d -> clamp(floor(Int, mod(f.R[d, i], f.box[d]) / f.box[d] * nc[d]), 0, nc[d] - 1), 3)
        c[1] + nc[1] * (c[2] + nc[2] * c[3])
    end
    perm = sortperm(key)
    newpos = similar(perm); newpos[perm] = 1:length(perm)
    exc = Tuple{Int32,Int32,Float64}[]
    for (i, j, s) in f.exceptions
        a, b = newpos[i], newpos[j]
        push!(exc, a < b ? (Int32(a), Int32(b), s) : (Int32(b), Int32(a), s))
    end
    return Fixture(f.name, f.ids[perm], f.R[:, perm], f.box, f.q[perm], exc, f.prefactor, f.units, f.source)
end

function Base.show(io::IO, f::Fixture)
    @printf(io, "Fixture(%s: %d atoms, box %.4f×%.4f×%.4f, net charge %.2e, %d exceptions, %s)",
            f.name, length(f), f.box..., sum(f.q), length(f.exceptions), f.units)
end

end # module Fixtures
