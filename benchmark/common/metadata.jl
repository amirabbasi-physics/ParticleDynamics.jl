module BenchmarkMetadata
# Host orchestration/provenance only. No simulation/reference calculation here.
using SHA, TOML
include(joinpath(@__DIR__,"phase_metadata.jl"))

digest(path) = bytes2hex(sha256(read(path)))
aggregate(hashes) = bytes2hex(sha256(join((k * "=" * hashes[k] for k in sort!(collect(keys(hashes)))), "\n")))

function source_hashes(root=normpath(joinpath(@__DIR__,"../..")))
    paths = String[]
    for subdir in ("src","benchmark")
        for (base,_,names) in walkdir(joinpath(root,subdir)), name in names
            any(ext->endswith(name,ext),(".jl",".py",".sh")) && push!(paths,joinpath(base,name))
        end
    end
    # Freeze project/fallback manifests and actual mutable path dependency source.
    for project in Base.load_path()
        isfile(project) || continue
        push!(paths,project)
        manifest=joinpath(dirname(project),"Manifest.toml")
        isfile(manifest) || continue
        push!(paths,manifest)
        for (_,records) in get(TOML.parsefile(manifest),"deps",Dict()), record in records
            haskey(record,"path") || continue
            dep=normpath(joinpath(dirname(project),record["path"]))
            for name in ("Project.toml","Manifest.toml")
                isfile(joinpath(dep,name)) && push!(paths,joinpath(dep,name))
            end
            isdir(joinpath(dep,"src")) || continue
            for (base,_,names) in walkdir(joinpath(dep,"src")), name in names
                push!(paths,joinpath(base,name))
            end
        end
    end
    return Dict(p=>digest(p) for p in unique(paths))
end

function begin_case(inputs)
    return (source=source_hashes(),inputs=Dict(abspath(p)=>digest(p) for p in inputs))
end

function array_bytes(a)
    isbitstype(eltype(a)) || error("model buffers must have isbits elements")
    order=ENDIAN_BOM == 0x04030201 ? "little" : "big"
    header=string(eltype(a),";",join(size(a),","),";",order,"\n")
    bytes=IOBuffer(); write(bytes,header); write(bytes,reinterpret(UInt8,vec(a)))
    return take!(bytes)
end

function array_descriptor(values)
    # Hash actual values, type, shape and byte order; no host physics calculation.
    a=Array(values)
    order=ENDIAN_BOM == 0x04030201 ? "little" : "big"
    result=Dict{String,Any}("sha256"=>bytes2hex(sha256(array_bytes(a))),
        "eltype"=>string(eltype(a)),"shape"=>collect(size(a)),"byte_order"=>order)
    # Keep compact coefficient tables readable; large buffers retain full hashes.
    length(a)<=16 && (result["values_column_major"]=collect(vec(a)))
    if eltype(a)<:Real
        counts=Dict{eltype(a),Int}()
        for v in a
            counts[v]=get(counts,v,0)+1
            length(counts)>16 && break
        end
        length(counts)<=16 && (result["value_counts"]=[Dict("value"=>v,"count"=>counts[v]) for v in sort!(collect(keys(counts)))])
    end
    return result
end

function window_frame(st,directory)
    # Diagnostic only, after the caller synchronizes and outside its timing range.
    # Original-ID ordering prevents a storage permutation looking like divergence.
    n=length(st.rx); T=eltype(st.rx)
    tags=st.tag===nothing ? Int32.(1:n) : Array(st.tag)
    sort(tags)==collect(1:n) || error("window frame requires bijective original particle tags")
    permutation=sortperm(tags)
    arrays=Dict(String(k)=>Array(getproperty(st,k))[permutation]
        for k in (:rx,:ry,:rz,:vx,:vy,:vz,:typeid))
    arrays["box"]=collect(st.box3)
    arrays["mass"]=st.mass_particle===nothing ? fill(st.mass,n) : Array(st.mass_particle)[permutation]
    arrays["charge"]=st.charge===nothing ? zeros(T,n) : Array(st.charge)[permutation]
    arrays["storage_tags"]=tags
    mkdir(directory) # refuse to overwrite a previous/partial diagnostic frame
    buffers=Dict{String,Any}()
    for (key,values) in arrays
        descriptor=array_descriptor(values)
        descriptor["file"]=key*".bin"
        write(joinpath(directory,descriptor["file"]),array_bytes(values))
        buffers[key]=descriptor
    end
    th=st.engine.run.thermostat; pc=st.pressure_control
    result=Dict{String,Any}("step"=>st.step,"directory"=>abspath(directory),
        "encoding"=>"typed header plus raw column-major bytes; original particle ID order",
        "restart"=>"reference frame only; not a full integrator/restart checkpoint",
        "state_sha256"=>aggregate(Dict(k=>buffers[k]["sha256"] for k in ("rx","ry","rz","vx","vy","vz","box"))),
        "particle_model_sha256"=>aggregate(Dict(k=>buffers[k]["sha256"] for k in ("mass","charge","typeid"))),
        "buffers"=>buffers,"pressure_control_present"=>pc!==nothing,
        "barostat_events"=>pc===nothing ? 0 : pc.control.events,
        "rng_seeds"=>Dict("device_thermostat"=>th===nothing || !hasproperty(th,:seed) ? "none" : string(th.seed),
            "barostat"=>pc===nothing ? "none" : string(pc.control.seed)))
    open(joinpath(directory,"frame.toml"),"w") do io
        TOML.print(io,result)
    end
    return result
end

function solver_settings(s)
    s===nothing && return Dict{String,Any}("method"=>"none")
    spec=s.spec
    result=Dict{String,Any}("method"=>spec.influence===:spme ? "pme" : String(spec.influence),
        "real_cutoff"=>spec.real_cutoff,"order"=>spec.window.order,"mesh"=>collect(s.ws.mesh),
        "grid_policy"=>"explicit mesh chosen for initial box; geometry-dependent influence refreshed",
        "coulomb_prefactor_requested"=>spec.coulomb_prefactor,
        "relative_permittivity"=>spec.relative_permittivity,
        "net_charge_policy"=>String(spec.net_charge),"real_space_modifier"=>String(spec.real_space_modifier))
    if hasproperty(spec.split,:alpha)
        result["alpha"]=spec.split.alpha
    else
        result["split_shape"]=spec.split.shape
        result["window_shape"]=spec.window.shape
    end
    return result
end

function initial_model(st; mixing, kinetic_dof)
    # Capture before the first force/list call can reorder particles. For water
    # this is after the builder's existing velocity constraint projection.
    st.step==0 || error("initial model must be captured before stepping")
    n=length(st.rx); T=eltype(st.rx)
    buffers=Dict("rx"=>array_descriptor(st.rx),"ry"=>array_descriptor(st.ry),
        "rz"=>array_descriptor(st.rz),"vx"=>array_descriptor(st.vx),
        "vy"=>array_descriptor(st.vy),"vz"=>array_descriptor(st.vz),
        "box"=>array_descriptor(collect(st.box3)))
    masses=array_descriptor(st.mass_particle===nothing ? fill(st.mass,n) : st.mass_particle)
    charges=array_descriptor(st.charge===nothing ? zeros(T,n) : st.charge)
    tables=Dict("types"=>array_descriptor(st.typeid),
        "sigma"=>array_descriptor(st.sigma_pair===nothing ? reshape([st.pair_lj.σ],1,1) : st.sigma_pair),
        "epsilon"=>array_descriptor(st.epsilon_pair===nothing ? reshape([st.pair_lj.ϵ],1,1) : st.epsilon_pair),
        "cutoff"=>array_descriptor(st.rcut_pair===nothing ? reshape([st.pair_lj.rcut],1,1) : st.rcut_pair))
    constraints=Dict{String,Any}("kind"=>"none")
    bound_molecules=0
    if st.rigid_water!==nothing
        ws=st.rigid_water.workspace
        bound_molecules=length(ws.atoms)
        constraints=Dict("kind"=>"rigid-water-triples","atoms"=>array_descriptor(ws.atoms),
            "lengths_squared"=>collect(ws.lengths2),"tolerance"=>ws.tolerance,"maxiter"=>ws.maxiter,
            "initial_velocities"=>"builder projects velocities before this snapshot")
    end
    s=st.electrostatics
    exclusions=Dict("coulomb"=>s===nothing ? [] : s.exceptions,
        "lj"=>"no explicit exclusions in these builders; zero pair coefficients retained in lj_table")
    topology=Dict("masses"=>masses["sha256"],"charges"=>charges["sha256"],
        (k=>v["sha256"] for (k,v) in tables)...)
    # Constraint/exclusion details remain explicit alongside this buffer digest;
    # do not use a partial buffer hash as the full topology identity.
    return Dict{String,Any}("state_sha256"=>aggregate(Dict(k=>v["sha256"] for (k,v) in buffers)),
        "topology_sha256"=>nothing,
        "topology_hash_unavailable_reason"=>"full cross-engine canonical topology encoding still required",
        "initial_state_buffers"=>buffers,"model_buffer_sha256"=>aggregate(topology),
        "molecular_groups"=>Dict("rigid_water_triples"=>bound_molecules,
            "free_monatomic_sites"=>n-3bound_molecules,
            "total_molecules"=>n-2bound_molecules),
        "state_encoding"=>"typed component buffers including byte order; after builder constraints, before first forces",
        "masses"=>masses,"charges"=>charges,"lj_table"=>tables,"mixing"=>mixing,
        "constraints"=>constraints,"exclusions"=>exclusions,
        "units"=>Dict("length"=>"nm","time"=>"ps","energy"=>"kJ/mol","mass"=>"u","charge"=>"e"),
        "coulomb_constant"=>s===nothing ? 0.0 : Float64(s.C),
        "boundary"=>s===nothing ? "periodic" : "periodic/"*String(s.spec.boundary),
        "net_charge"=>s===nothing ? 0.0 : s.qtotal,
        "lj_cutoff"=>Float64(st.pair_lj.rcut),
        "lj_modifier"=>replace(String(st.vdw_modifier),"_"=>"-"),
        "dt"=>Float64(st.dt),"kinetic_dof"=>kinetic_dof,
        "conversion_policy"=>"actual stored values retained without rounding to nominal input values")
end

function process_affinity(status="/proc/self/status")
    isfile(status) || return Dict("unavailable_reason"=>"Linux process status is unavailable")
    fields=Dict(split(line,':';limit=2)[1]=>strip(split(line,':';limit=2)[2])
        for line in readlines(status) if occursin(':',line))
    haskey(fields,"Cpus_allowed_list") || return Dict("unavailable_reason"=>"Cpus_allowed_list is absent")
    return Dict("cpus_allowed_list"=>fields["Cpus_allowed_list"],
        "mems_allowed_list"=>get(fields,"Mems_allowed_list","unknown"),
        "scope"=>"thread-group leader (/proc/self/status), not all worker threads; JULIA_EXCLUSIVE is a request")
end

function emit_observation!(io,step,o,lx)
    println(io,step,",",o.Epot_total,",",o.Ekin_total,",",o.virial_total,",",lx)
    flush(io)
end

function warm_output!(path,o,lx)
    # Use the same IOStream dispatch as production; preserve this setup-only record.
    open(path*".warmup.csv","w") do io
        println(io,"setup_pilot_step,Epot,Ekin,virial,Lx")
        emit_observation!(io,2,o,lx)
    end
end

function run_observations!(advance!, observe, length_x, finish!, path, warm, steps)
    # Shared by normal and profiled runs, including transfers, flushed output and
    # the final synchronization. Call with zero steps before timing to compile
    # this host path without advancing the simulation.
    steps >= 0 || error("measured steps must be nonnegative")
    rows = 0
    open(path,"w") do io
        println(io,"step,Epot,Ekin,virial,Lx")
        for i in 1:steps
            sampled = (warm+i)%2000 == 0
            advance!(sampled)
            if sampled
                emit_observation!(io,i,observe(),length_x())
                rows += 1
            end
        end
    end
    finish!()
    return (advanced_steps=steps, observation_collections=rows, csv_rows=rows)
end

# Opt-in diagnostics: read existing counters only at synchronized host boundaries.
# These reads stay outside measured/CUPTI windows; never reset device state.
function work_snapshot(st, pd)
    e=st.engine
    e isa pd.DeviceEngine.StateEngine || error("work audit requires device engine")
    status=pd.NeighborLists.device_neighbor_status(e.run.neighbors)
    result=Dict{String,Int64}("device_steps"=>status.step,"neighbor_checks"=>status.checks,
        "neighbor_builds"=>status.builds,"capacity_recoveries"=>e.recoveries)
    if st.cluster!==nothing
        cp=pd.ClusterPairs; list=st.cluster.list; counts=Array(list.counts)
        counts[cp.CL_ERR]==0 || error("work audit: cluster list error")
        result["cluster_builds"]=list.nbuild+Int64(counts[cp.CL_NBUILD])
        result["cluster_prunes"]=list.nprune+Int64(counts[cp.CL_NPRUNE])
    end
    if st.pressure_control!==nothing
        result["barostat_events"]=st.pressure_control.control.events
    end
    all(x->x>=0,values(result)) || error("work audit: negative counter")
    return result
end

function work_delta(before,after,steps)
    steps>=0 || error("work audit: negative measured steps")
    keys(before)==keys(after) || error("work audit: counter schema changed")
    all(x->x isa Integer && !(x isa Bool) && x>=0,values(before)) || error("work audit: invalid before counters")
    all(x->x isa Integer && !(x isa Bool) && x>=0,values(after)) || error("work audit: invalid after counters")
    delta=Dict(k=>after[k]-before[k] for k in keys(before))
    all(x->x>=0,values(delta)) || error("work audit: nonmonotonic counter")
    delta["device_steps"]==steps || error("work audit: device step delta mismatch")
    return Dict("before"=>before,"after"=>after,"delta"=>delta,
        "scope"=>"existing device/list/barostat counters bounded outside measured window",
        "unavailable"=>"actual thermostat/K/virial/transfer phases require further counters or trace evidence; launches alone are insufficient")
end

function attach!(record, before, settings, cuda; warm, steps, profiled=false, requested_events=0, requested_backend, observed_work=nothing, state=nothing, window_audit=nothing, device_work=nothing, phase_work=nothing)
    after=source_hashes(); input_after=Dict(p=>digest(p) for p in keys(before.inputs))
    phase_work===nothing || (settings["cadence_unavailable_reason"]=phase_work["unavailable"])
    record["resolved_settings"]=settings
    executable=joinpath(Sys.BINDIR,Base.julia_exename())
    record["runtime"]=Dict("host"=>gethostname(),"gpu_uuid"=>string(cuda.uuid(cuda.device())),
        "gpu_model"=>cuda.name(cuda.device()),"driver"=>string(cuda.driver_version()),
        "julia_or_gromacs"=>string(VERSION),"executable_sha256"=>digest(executable),
        "threads"=>Threads.nthreads(),"affinity"=>process_affinity(),
        "depot"=>DEPOT_PATH,"load_path"=>Base.load_path(),"cuda_package"=>string(pkgversion(cuda)))
    record["provenance"]=Dict("source_before"=>aggregate(before.source),"source_after"=>aggregate(after),
        "inputs_before"=>aggregate(before.inputs),"inputs_after"=>aggregate(input_after),
        "project_sha256"=>digest(Base.active_project()),
        "dependencies_sha256"=>digest(joinpath(dirname(Base.active_project()),"Manifest.toml")),
        "source_files_before"=>before.source,"source_files_after"=>after,
        "input_files_before"=>before.inputs,"input_files_after"=>input_after)
    record["protocol"]=Dict("warm_steps"=>warm,"measured_steps"=>steps,"profiled"=>profiled,
        "requested_observable_events"=>requested_events,"thermal_stabilization"=>nothing,
        "observed_host_work"=>observed_work,
        "window_audit"=>window_audit,
        "observed_device_work"=>device_work,
        "observed_phase_work"=>phase_work,
        "phase_audit_enabled"=>get(ENV,"PD_PHASE_AUDIT","0")=="1",
        "stabilization_unavailable_reason"=>"requires retained GPU telemetry; warm-up alone is not stabilization",
        "event_counts_unavailable_reason"=>phase_work!==nothing ? phase_work["unavailable"] : device_work===nothing ? "work_audit=1 required for bounded existing device counters; coupling/K counters unavailable" : device_work["unavailable"])
    if state!==nothing
        th=state.engine.run.thermostat
        pc=state.pressure_control
        record["protocol"]["actual_rng_seeds"]=Dict(
            "device_thermostat"=>th===nothing || !hasproperty(th,:seed) ? nothing : string(th.seed),
            "barostat"=>pc===nothing ? nothing : string(pc.control.seed))
    end
    record["benchmark_manifest"]=Dict("schema_version"=>1,"card"=>get(ENV,"PD_CARD",""),
        "attempt"=>get(ENV,"PD_ATTEMPT","unknown"),"case_id"=>get(ENV,"PD_CASE_ID","unknown"),
        "engine"=>"ParticleDynamics","track"=>get(ENV,"PD_TRACK","unknown"),
        "runtime"=>record["runtime"],"provenance"=>record["provenance"],
        "requested"=>Dict("energy_interval"=>2000,"log_interval"=>2000,"backend"=>requested_backend),
        "completed"=>Dict("kind"=>"simulation (not independent numerical validation)","cases"=>1),
        "gates"=>Dict())
    aggregate(before.source)==aggregate(after) || error("benchmark source/dependencies changed during run")
    before.inputs==input_after || error("benchmark inputs changed during run")
    return record
end
end
