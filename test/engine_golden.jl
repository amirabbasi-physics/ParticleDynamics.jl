module EngineGolden
isdefined(Main, :GoldenFixtures) || Base.include(Main, joinpath(@__DIR__, "golden_fixtures.jl"))
using Main.GoldenFixtures
# Golden references recorded from the previous staged engine.
# A device-engine test that compared against a live staged twin now compares
# against that twin's final state, recorded once from the staged engine while it
# still exists. `staged_reference(f, name)` runs `f()` (the staged twin's loop,
# returning the twin and optional extra values) only in capture mode
# (`PD_CAPTURE_STAGED=1`) and saves the record; otherwise it loads the record and
# checks its hash. Records hold per-particle data in particle-id order (`st.tag`),
# so they are independent of storage sorting.
using TOML, SHA, CUDA

const DIR=golden_directory(:engine)
capturing()=get(ENV,"PD_CAPTURE_STAGED","0")=="1"

# Per-particle fields recorded when present (vectors, and N×k matrices by row).
const FIELDS=(:rx,:ry,:rz,:vx,:vy,:vz,:fx,:fy,:fz,:f0x,:f0y,:f0z,:Epot,:Ekin,:dq,:dU,
    :virial,:virial_nonbonded,:virial_bonded,:virial_tensor,:rx_unwrap,:ry_unwrap,:rz_unwrap)

"""Host copy of a per-particle array in particle-id order (build order)."""
function by_id(st,x)
    a=Float64.(Array(x))
    st.tag===nothing && return a
    out=similar(a); tags=Array(st.tag)
    ndims(a)==1 ? (out[tags].=a) : (out[tags,:].=a)
    return out
end

"""Snapshot of a state's per-particle fields (id order), e.g. before observables
that refresh buffers are collected."""
golden_snapshot(st)=_record(st,(;))
function _record(st,extras)
    rec=Dict{String,Any}("step"=>st.step)
    for f in FIELDS
        hasproperty(st,f) || continue
        x=getfield(st,f)
        x isa CuArray || continue
        a=by_id(st,x)
        rec[String(f)]=ndims(a)==1 ? a : Dict("size"=>collect(size(a)),"data"=>vec(a))
    end
    _extras!(rec,extras)
    return rec
end
function _extras!(rec,extras)
    for (k,v) in pairs(extras)
        rec["extra_"*String(k)]=v isa NamedTuple ? Dict(String(n)=>_plain(w) for (n,w) in pairs(v) if _plain(w)!==nothing) : _plain(v)
    end
    return rec
end
_plain(v::Real)=v isa Bool ? v : Float64(v)
_plain(v::AbstractArray{<:Real})=Float64.(vec(collect(v)))   # matrices flattened (column-major)
_plain(v::Union{AbstractString,Symbol})=String(v)
_plain(v)=nothing

function _sums()
    file=joinpath(DIR,"SHA256SUMS")
    isfile(file) || return Dict{String,String}()
    return Dict(last(split(l))=>first(split(l)) for l in eachline(file) if !isempty(strip(l)))
end

"""Record of the staged twin `name`: captured from `staged()` in capture mode,
loaded (hash-checked) otherwise. `staged()` returns the twin state, or
`(state, extras::NamedTuple)` for further values (observables, counts), where the
state may also be a `golden_snapshot(state)` taken earlier."""
function staged_reference(staged::Function,name::AbstractString)
    file=joinpath(DIR,name*".toml")
    if capturing()
        out=staged()
        st,extras=out isa Tuple ? out : (out,(;))
        rec=st isa AbstractDict ? _extras!(st,extras) : _record(st,extras)
        mkpath(DIR)
        ispath(file) && error("Refusing to overwrite golden record $file")
        open(io->TOML.print(io,rec;sorted=true),file,"w")
        open(io->println(io,bytes2hex(sha256(read(file)))," ",name*".toml"),joinpath(DIR,"SHA256SUMS"),"a")
        return rec
    end
    isfile(file) || error("Missing golden record $file (capture from the staged engine with PD_CAPTURE_STAGED=1)")
    get(_sums(),name*".toml","")==bytes2hex(sha256(read(file))) || error("Golden record $name changed (hash mismatch)")
    return TOML.parsefile(file)
end

"""Recorded field `f` (vector, or N×k matrix)."""
function field(rec,f)
    v=rec[String(f)]
    v isa AbstractDict || return Float64.(v)
    return reshape(Float64.(v["data"]),Tuple(v["size"]))
end
has(rec,f)=haskey(rec,String(f))
extra(rec,k)=rec["extra_"*String(k)]

relerr(a,b)=maximum(abs.(a.-b))/max(maximum(abs.(b)),1e-30)
"""Largest relative difference of each field present in both the state and the record."""
function differences(st,rec;fields=FIELDS)
    d=Dict{Symbol,Float64}()
    for f in fields
        (hasproperty(st,f) && getfield(st,f) isa CuArray && has(rec,f)) || continue
        d[f]=relerr(by_id(st,getfield(st,f)),field(rec,f))
    end
    return d
end

export staged_reference, golden_snapshot, field, extra, differences, capturing
end
