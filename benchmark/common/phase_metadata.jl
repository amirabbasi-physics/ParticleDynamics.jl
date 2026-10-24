# Included by BenchmarkMetadata. Diagnostic reads only; never reference physics.
const PHASE_CHANNELS=["half_K_separate","half_K_fused","bath_coupled","bath_stochastic_update",
    "full_K_report_separate","full_K_report_fused","global_observable_finish",
    "barostat_virial_full","barostat_virial_trace","barostat_event","RNG_precompute"]

function phase_snapshot(st,pd,path)
    get(ENV,"PD_PHASE_AUDIT","0")=="1" || return nothing
    nl=pd.NeighborLists; de=pd.DeviceEngine
    isdefined(nl,:DN_PHASE_ENABLED) && nl.DN_PHASE_ENABLED || error("phase audit requires generated snapshot hooks")
    st.engine isa de.StateEngine || error("phase audit requires device engine")
    th=st.engine.run.thermostat
    th isa de.CSVRData && de.nbaths(th)==1 || error("phase audit supports one CSVR bath only")
    nl.DN_PHASE_CHANNELS==length(PHASE_CHANNELS) || error("phase channel schema mismatch")
    status=nl.device_neighbor_status(st.engine.run.neighbors)
    whole=Array(st.engine.run.neighbors.state)
    raw=whole[nl.DN_SECTION+1:end]
    length(raw)==length(PHASE_CHANNELS)*(nl.DN_PHASE_CAP+1) || error("phase buffer length mismatch")
    all(x->x>=0,raw) || error("negative phase counter")
    values=reshape(raw,length(PHASE_CHANNELS),nl.DN_PHASE_CAP+1)
    descriptor=array_descriptor(values);descriptor["file"]=abspath(path)
    ispath(path) && error("phase snapshot already exists")
    write(path,array_bytes(values))
    stamp=Array(st.engine.run.observables.stamp)
    route=de.rw_fast_path(st.engine.run) ? "rigid_water_fast" : st.engine.run.fusion!==nothing ? "fused" : "generic"
    return (;cursor=Int(status.step),samples=Int(stamp[2]),raw,descriptor,route)
end

function phase_delta(before,after,steps)
    before===nothing && after===nothing && return nothing
    before===nothing || after===nothing ? error("missing phase boundary") : nothing
    after.cursor-before.cursor==steps || error("phase cursor delta mismatch")
    length(before.raw)==length(after.raw) && length(before.raw)%length(PHASE_CHANNELS)==0 || error("phase schema changed")
    all(x->x isa Integer && !(x isa Bool) && x>=0,before.raw) || error("invalid before phase counters")
    all(x->x isa Integer && !(x isa Bool) && x>=0,after.raw) || error("invalid after phase counters")
    get(before,:route,"unknown")==get(after,:route,"unknown") || error("phase route changed")
    delta=after.raw.-before.raw
    all(x->x>=0,delta) || error("phase counter reset")
    rows=reshape(delta,length(PHASE_CHANNELS),:)
    events=Dict(name=>[[j-1,Int(rows[i,j])] for j in axes(rows,2) if rows[i,j]!=0]
        for (i,name) in enumerate(PHASE_CHANNELS))
    totals=Dict(name=>sum(Int(x[2]) for x in events[name];init=0) for name in PHASE_CHANNELS)
    totals["global_observable_finish"]==after.samples-before.samples || error("phase/stamp sample mismatch")
    boundary(s)=Dict("cursor"=>s.cursor,"observable_samples"=>s.samples,"buffer"=>s.descriptor,"route"=>get(s,:route,"unknown"))
    return Dict("schema"=>1,"channels"=>PHASE_CHANNELS,"before"=>boundary(before),"after"=>boundary(after),
        "events"=>events,"totals"=>totals,
        "cursor_conventions"=>"raw device cursor at each execution; RNG_precompute uses target cursor; no phase renumbering",
        "unavailable"=>"transfer bytes/phase timing and uninstrumented internal reduction stages unknown; counters are diagnostic work only")
end
