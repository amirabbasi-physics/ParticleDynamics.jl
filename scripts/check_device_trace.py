#!/usr/bin/env python3
"""Audit only the Nsight cudaProfilerApi region; never run simulation math."""
import json
import sqlite3
import sys
import tomllib
from pathlib import Path


def audit(database, metadata):
    with open(metadata, "rb") as stream:
        record = tomllib.load(stream)
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    api = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            rows = db.execute(f"SELECT s.value, count(*) FROM {table} a JOIN StringIds s "
                              "ON s.id=a.nameId GROUP BY s.value")
            for name, count in rows:
                api[name] = api.get(name, 0) + count
    transfers = {}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        transfers = dict(db.execute("SELECT e.label,count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
                                    "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.label"))
    kernels = {}
    eager_kernels = {}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        kernels = dict(db.execute("SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                                 "JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value"))
        eager_kernels = dict(db.execute("SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                                        "JOIN StringIds s ON s.id=k.shortName WHERE k.graphNodeId IS NULL "
                                        "GROUP BY s.value"))
    db.close()
    graph_launches = sum(n for name, n in api.items() if "GraphLaunch" in name)
    allocations = {name: n for name, n in api.items()
                   if name.startswith(("cuMemAlloc", "cudaMalloc", "cuMemCreate"))}
    host_transfers = {name: n for name, n in transfers.items() if "Host" in name}
    commits = sum(n for name, n in kernels.items() if "device_commit_kernel" in name)
    step_kernels = sum(n for name, n in kernels.items() if "device_neighbor_advance_kernel" in name)
    fused = record.get("fused", False)
    fused_prefix = sum(n for name, n in kernels.items() if "_fused_prefix_kernel" in name)
    fused_force = sum(n for name, n in kernels.items() if "_kernel_kick" in name)
    separate = {part: sum(n for name, n in kernels.items() if part in name)
                for part in ("_device_displacement_kernel", "_device_decide_kernel", "_displacement_decide_kernel",
                             "_device_set_rebuild_condition_kernel", "_nve_kick3_kernel", "_nve_kick_drift3_kernel")}
    if fused:
        step_kernels = fused_force
    observable_forces = sum(n for name, n in kernels.items() if "_lj3_ell_kernel_virial" in name)
    regular_forces = sum(n for name, n in kernels.items() if "_lj3_ell_noE_kernel" in name)
    # Generic pair kernels and the earlier LJ-specific names.
    cooperative_regular = sum(n for name, n in kernels.items()
                              if "_lj_cooperative_noE_kernel" in name or "_pair_cooperative_noE_kernel" in name)
    cooperative_observable = sum(n for name, n in kernels.items()
                                 if "_lj_cooperative_observable_kernel" in name or "_pair_cooperative_observable_kernel" in name)
    observable_forces += cooperative_observable
    regular_forces += cooperative_regular
    errors = []
    if record.get("status") != "GPU checks passed; trace audit pending":
        errors.append("GPU trajectory/reference checks did not pass")
    if graph_launches < record["steps"] or step_kernels != record["steps"]:
        errors.append("Missing graph launches or device step kernels")
    # Split steps (host forces): one launch per graph segment of every step.
    segments = record.get("graph_segments")
    if segments is not None and graph_launches != segments * record["steps"]:
        errors.append(f"Graph launches {graph_launches} differ from {segments} segments per step")
    # Barostat events: each event may evaluate eagerly on the engine (displacement
    # check, rebuild without sorting, force with energy), read a few scalars to the host
    # and allocate small reduction buffers; bounded per event. Steps between events stay
    # graph replays (graph launches and step kernels are checked as usual).
    events = record.get("barostat_events", 0)
    eager_total = sum(eager_kernels.values())
    if events:
        per_event = {"eager_kernels": eager_total / events,
                     "host_transfers": sum(host_transfers.values()) / events,
                     "allocations": sum(allocations.values()) / events}
        limits = {"eager_kernels": 80, "host_transfers": 16, "allocations": 8}
        for key, limit in limits.items():
            if per_event[key] > limit:
                errors.append(f"Barostat events: {per_event[key]:.1f} {key} per event (limit {limit})")
    else:
        per_event = {}
        # With host forces every kernel outside the graphs is the provider's (host call).
        if eager_total and not record.get("host_forces", False):
            errors.append(f"Kernels outside graphs: {eager_total}")
        if host_transfers:
            errors.append("Host/device transfers occurred inside the replay region")
        if allocations:
            errors.append("GPU allocation APIs occurred inside the replay region")
    # Cluster-pair runs: the cluster list and the mesh replace the ELL list and
    # pair kernels; their own launch counts are checked here, the ELL checks below skipped.
    cluster = record.get("pair_backend") == "cluster"
    cluster_counts = {}
    if cluster:
        def count(part):
            return sum(n for name, n in kernels.items() if part in name)
        steps = record["steps"]; builds = record["rebuilds_in_trace"]; prunes = record["prunes_in_trace"]
        # The cluster decision kernel only: `_displacement_decide_kernel` (ELL contact rows,
        # collision counting) and `_device_decide_kernel` also contain "_decide_kernel".
        cluster_decide = sum(n for name, n in kernels.items() if name.startswith("_decide_kernel"))
        cluster_counts = {"decide": cluster_decide, "entry_offsets": count("_entry_offsets_kernel"),
                          "compact": count("_compact_warp_kernel"), "pairs": count("_cluster_pair_kernel"),
                          # the force scatter only: `_bin_scatter_kernel` belongs to the list build
                          "scatter": count("_scatter_kernel") - count("_bin_scatter_kernel"), "spread": count("_spread"),
                          "gather": count("_gather_ad_kernel") + count("_gather_ik_kernel")}
        if cluster_counts["decide"] != steps:
            errors.append("Cluster decision kernels differ from the step count")
        # Collision counting keeps ELL contact rows: one displacement decision per step.
        contact_decide = count("_displacement_decide_kernel")
        cluster_counts["contact_decide"] = contact_decide
        if contact_decide != (steps if record.get("collisions", False) else 0):
            errors.append("Contact-row decision kernels differ from the collision setting")
        if cluster_counts["entry_offsets"] != builds or not 0 < builds < steps:
            errors.append("Cluster rebuild kernels differ from the device rebuild counter")
        if cluster_counts["compact"] != 2 * (builds + prunes) or prunes < builds:
            errors.append("Cluster compaction kernels differ from rebuilds and prunes")
        if cluster_counts["pairs"] != steps + prunes or cluster_counts["scatter"] != steps:
            errors.append("Cluster pair/scatter kernels differ from steps and pruning passes")
        if record.get("electrostatics", False) and (cluster_counts["spread"] != steps or cluster_counts["gather"] != steps):
            errors.append("Mesh spread/gather kernels differ from the step count")
    # Rigid water: SHAKE after every drift, RATTLE and the reference save after every
    # final kick, the constraint virial on sampled steps; all inside the graphs.
    rigid_counts = {}
    if record.get("rigid_water", False):
        def count2(part):
            return sum(n for name, n in kernels.items() if part in name)
        rigid_counts = {"shake": count2("_shake_water_kernel"), "rattle": count2("_rattle_water_kernel"),
                        "save": count2("_save_water_kernel"), "virial": count2("_constraint_virial_rows_kernel")}
        every_r = record.get("energy_every", 0)
        sampled = ((record["warmup"] + record["steps"]) // every_r - record["warmup"] // every_r) if every_r > 0 else 0
        if (rigid_counts["shake"] != record["steps"] or rigid_counts["rattle"] != record["steps"]
                or rigid_counts["save"] != record["steps"] or rigid_counts["virial"] != sampled):
            errors.append("Rigid-water constraint kernels differ from the step and sample counts")
    # LJ bond exclusions on the cluster backend: the LJ restore of unbonded Coulomb
    # exceptions every step, the Coulomb shift of bonded-only pairs on sampled steps, the
    # harmonic bonds every step; all inside the graphs.
    exclusion_counts = {}
    if record.get("bond_exclusions", False):
        def count3(part):
            return sum(n for name, n in kernels.items() if part in name)
        every_b = record.get("energy_every", 0)
        sampled_b = ((record["warmup"] + record["steps"]) // every_b - record["warmup"] // every_b) if every_b > 0 else 0
        exclusion_counts = {"lj_restore": count3("_lj_restore_kernel"), "coulomb_shift": count3("_coulomb_shift_kernel"),
                            "bonds": count3("_harmonic3_")}
        # The Coulomb shift kernel runs only with a potential-shifted real space
        # (`coulomb_shift` in the record); records before that key used unshifted Ewald.
        expected_restore = record["steps"] if record.get("lj_restore", True) else 0
        expected_shift = sampled_b if record.get("coulomb_shift", False) else 0
        if (exclusion_counts["lj_restore"] != expected_restore or exclusion_counts["coulomb_shift"] != expected_shift
                or exclusion_counts["bonds"] != record["steps"]):
            errors.append("Bond-exclusion correction or bond kernels differ from the step and sample counts")
    # Collision counting on the cluster backend: one event pass per position sample
    # (one per step here), flag initialisation after every contact-row and cluster rebuild.
    if cluster and record.get("collisions", False):
        def count4(part):
            return sum(n for name, n in kernels.items() if part in name)
        events = count4("_events3")
        inits = count4("_init_prev3")
        expected_inits = record.get("contact_rebuilds_in_trace", -1) + record["rebuilds_in_trace"]
        exclusion_counts["collision_events"] = events; exclusion_counts["collision_inits"] = inits
        if events != record["steps"] or inits != expected_inits:
            errors.append("Collision event/initialisation kernels differ from samples and rebuilds")
    if cluster:
        pass
    elif record.get("neighbor_list", True):
        if commits != record["rebuilds_in_trace"] or not 0 < commits < record["steps"]:
            errors.append("Rebuild branch executions do not match the device counters")
    elif commits != 0 or record["rebuilds_in_trace"] != 0:
        errors.append("A run without a kept neighbor list rebuilt it")
    diagnostics = record.get("neighbors_in_trace", {})
    if record.get("neighbor_diagnostics", False) and not cluster:
        causes = [diagnostics.get(k, -1) for k in
                  ("invalid_builds", "displacement_builds", "age_builds")]
        if (any(n < 0 for n in causes) or sum(causes) != commits
                or diagnostics.get("attempts") != commits or diagnostics.get("builds") != commits):
            errors.append("Rebuild cause/attempt counters differ from rebuild kernel launches")
        if diagnostics.get("sorting_builds") != record.get("sorts_in_trace", 0):
            errors.append("Sorting subset does not match sorting counters")
        candidates = diagnostics.get("total_candidates", -1)
        accepted = diagnostics.get("total_accepted", -1)
        if not 0 <= accepted <= candidates:
            errors.append("Invalid candidate/accepted totals")
    every = record.get("energy_every", 0)
    expected_observables = ((record["warmup"] + record["steps"]) // every - record["warmup"] // every
                            if every > 0 else 0)
    # Euler–Heun evaluates forces twice per step (midpoint without energy, final with).
    forces_per_step = record.get("forces_per_step", 1)
    host_force_kernels = {}
    if record.get("host_forces", False):
        # Forces come from the host provider between segments: no pair kernels at all,
        # and the provider's own kernel once per host force call.
        if regular_forces + observable_forces != 0:
            errors.append("Pair force kernels ran although forces come from the host")
        for part, per_step in record.get("host_force_kernels", {}).items():
            host_force_kernels[part] = sum(n for name, n in kernels.items() if part in name)
            if host_force_kernels[part] != per_step * record["steps"]:
                errors.append(f"Host force kernel {part}: {host_force_kernels[part]} launches, expected "
                              f"{per_step} per step")
    elif cluster:
        pass
    elif (observable_forces != expected_observables + record.get("eager_energy_evaluations", 0)
            or regular_forces + observable_forces
               != forces_per_step * record["steps"] + record.get("eager_energy_evaluations", 0)):
        errors.append("Force/observable kernel counts differ from the recorded cadence")
    lanes = record.get("force_lanes", 1)
    layout = record.get("force_layout", "ell")
    packing = {part: sum(n for name, n in kernels.items() if part in name)
               for part in ("_pack_tiled_neighbors_kernel", "_packing_commit_kernel")}
    expected_packs = commits if layout == "tiled" else 0
    if "force_layout" in record and not cluster:
        if any(n != expected_packs for n in packing.values()) or record.get("packs_in_trace") != expected_packs:
            errors.append("Packing kernels/counts differ from successful rebuilds")
        # Fused steps always use the cooperative kernel (with the kick), even at G=1;
        # so do non-LJ pair models and 2D, which have no serial kernel.
        # Feature hooks in the pair kernel (bonded exclusions) also select it.
        generic = (record.get("pair_model", "UniformLJ") != "UniformLJ" or record.get("dims", 3) != 3
                   or record.get("generic_pairs", False))
        expected_cooperative = ((forces_per_step * record["steps"] + record.get("eager_energy_evaluations", 0))
                                if lanes != 1 or layout != "ell" or fused or generic else 0)
        if cooperative_regular + cooperative_observable != expected_cooperative:
            errors.append("Cooperative force kernels do not match selected traversal")
    dims = record.get("dims", 3)
    serial_builder = f"_kernel_neighbors{dims}"
    builders = {part: sum(n for name, n in kernels.items() if part in name)
                for part in (serial_builder, "_cooperative_rows3_kernel", "_kernel_neighbors_allpairs")}
    # All-pairs rows (small boxes) replace the cell-list builders. Records written
    # before the "allpairs" key existed identify the mode by the smallbox case.
    allpairs = record.get("allpairs", record.get("case") == "smallbox_lj")
    if "build_lanes" in record and not cluster:
        if allpairs:
            expected = (0, 0, commits)
        else:
            expected = ((0, commits, 0) if record["build_lanes"] != 1 else (commits, 0, 0))
        if (builders[serial_builder], builders["_cooperative_rows3_kernel"],
                builders["_kernel_neighbors_allpairs"]) != expected:
            errors.append("Row-builder kernels do not match the selected builder and rebuild count")
    if record.get("rebuild_policy") == "displacement" and diagnostics.get("age_builds", 0) != 0:
        errors.append("Age-triggered builds occurred under the displacement-only policy")
    if fused and (fused_prefix != record["steps"] or fused_force != record["steps"] or any(separate.values())
                  or sum(n for name, n in kernels.items() if "device_neighbor_advance_kernel" in name)):
        errors.append("Fused control kernels missing or separate control/integration kernels present")
    partial = "_observable_partial_kernel" if dims == 3 else "_observable_partial2_kernel"
    observable_reductions = {part: sum(n for name, n in kernels.items() if part in name)
                             for part in (partial, "_observable_finish_kernel")}
    if record.get("global_observables", False):
        if (any(n != expected_observables for n in observable_reductions.values())
                or record.get("observable_samples_in_trace") != expected_observables):
            errors.append("Global reduction kernels/sample stamps differ from the recorded cadence")
    # Bonded terms: each force evaluation launches these kernels inside the graph.
    bonded_kernels = {part: sum(n for name, n in kernels.items() if part in name)
                      for part in record.get("bonded_kernels", {})}
    for part, per_force in record.get("bonded_kernels", {}).items():
        if bonded_kernels[part] != per_force * forces_per_step * record["steps"]:
            errors.append(f"Bonded kernel {part}: {bonded_kernels[part]} launches, expected "
                          f"{per_force} per force evaluation")
    sorting_kernels = {part: sum(n for name, n in kernels.items() if part in name)
                       for part in ("_ordering_gather_kernel", "_ordering_copyback_kernel", "_ordering_commit_kernel")}
    if record.get("spatial_reorder", False):
        if record.get("sorts_in_trace", 0) <= 0 or any(n != commits for n in sorting_kernels.values()):
            errors.append("Missing sorting activity or sorting kernels in the rebuild branch")
    return {"status": "failed" if errors else "passed", "N": record["N"],
            "graph_launches": graph_launches, "step_kernels": step_kernels,
            "rebuild_kernel_launches": commits, "host_transfers": host_transfers,
            "regular_force_kernels": regular_forces, "observable_force_kernels": observable_forces,
            "observable_reduction_kernels": observable_reductions,
            "packing_kernels": packing, "force_lanes": lanes, "force_layout": layout,
            "row_builder_kernels": builders, "build_lanes": record.get("build_lanes", 1),
            "skin": record.get("skin", 0.55), "rebuild_policy": record.get("rebuild_policy", "adaptive"),
            "fused": fused, "fused_prefix_kernels": fused_prefix, "fused_force_kick_kernels": fused_force,
            "separate_control_kernels": separate,
            "sorts_in_trace": record.get("sorts_in_trace", 0), "sorting_kernels": sorting_kernels,
            "bonded_kernels": bonded_kernels, "host_force_kernels": host_force_kernels,
            "graph_segments": segments, "eager_kernels": eager_kernels,
            "barostat_events": events, "per_event": per_event,
            "graph_refreshes": record.get("graph_refreshes"),
            "neighbor_diagnostics": diagnostics, "cluster_kernels": cluster_counts, "rigid_kernels": rigid_counts,
            "bond_exclusion_kernels": exclusion_counts,
            "allocation_apis": allocations, "errors": errors}


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: check_device_trace.py TRACE.sqlite PROFILE.toml NEW_AUDIT.json")
    result = audit(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], "x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    return result["status"] != "passed"


if __name__ == "__main__":
    sys.exit(main())
