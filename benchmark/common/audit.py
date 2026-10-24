#!/usr/bin/env python3
"""Fail-closed benchmark artifact adapter and comparability gates (no simulations).

import: adapt a legacy record/log, retaining unknowns and rejecting missing evidence.
compare: compare two imported bundles, reporting every mismatch by field.
Timing acceptance additionally needs paired stabilized samples and independent science gates.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile

MODEL_FIELDS = '''state_sha256 topology_sha256 n_atoms n_molecules masses charges lj_table
mixing exclusions constraints units coulomb_constant boundary net_charge lj_cutoff lj_modifier
dispersion_correction dt temperature pressure kinetic_dof velocity_convention ensemble'''.split()
WORK_FIELDS = '''thermostat thermostat_tau thermostat_interval thermostat_phase barostat barostat_tau
barostat_interval barostat_phase compressibility com_removal energy_interval virial_interval
kinetic_interval reduction_interval host_transfer_interval log_interval trajectory_interval
checkpoint_interval output_fields'''.split()
SOLVER_FIELDS = 'method real_cutoff alpha order mesh grid_policy'.split()
RUNTIME_FIELDS = 'host gpu_uuid gpu_model driver julia_or_gromacs executable_sha256 threads affinity'.split()
PIN_FIELDS = 'source_before source_after inputs_before inputs_after project_sha256 dependencies_sha256'.split()
ACTIVATION_FIELDS = 'backend offload graphs list_policy sort_policy'.split()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unknown(value):
    return value is None or value == '' or value == 'unknown' or (
        isinstance(value, dict) and value.get('unavailable_reason') is not None)


def number(value):
    try:
        x = float(value)
        if math.isfinite(x):
            return int(x) if x.is_integer() else x
    except (ValueError, TypeError):
        pass
    return value.strip() if isinstance(value, str) else value


def parse_gromacs(settings, log):
    # md.log's inputrec + gmx dump are actual settings. Runtime tuning overrides inputrec.
    raw = {}
    for key, value in re.findall(r'^\s*([\w-]+)\s*=\s*([^\n]+)', log + '\n' + settings, re.M):
        raw[key] = value.strip()
    names = {'dt': 'dt', 'rvdw': 'lj_cutoff', 'vdw-modifier': 'lj_modifier', 'DispCorr': 'dispersion_correction',
             'rcoulomb': 'real_cutoff', 'tcoupl': 'thermostat', 'pcoupl': 'barostat',
             'nsttcouple': 'thermostat_interval', 'nstpcouple': 'barostat_interval',
             'nstcalcenergy': 'energy_interval', 'nstenergy': 'log_interval', 'comm-mode': 'com_removal',
             'pme-order': 'order', 'coulombtype': 'method', 'rlist': 'rlist', 'nstlist': 'nstlist'}
    out = {dest: number(raw[src]) for src, dest in names.items() if src in raw}
    for key in ('method', 'lj_modifier', 'dispersion_correction', 'com_removal', 'thermostat', 'barostat'):
        if key in out and isinstance(out[key], str):
            out[key] = out[key].lower()
    if out.get('thermostat') == 'v-rescale':
        out['thermostat'] = 'csvr'
    mesh = [number(raw.get('fourier-n' + axis)) for axis in 'xyz']
    if all(isinstance(x, int) and x > 0 for x in mesh):
        out['mesh'] = mesh
    changes = re.findall(r'Changing nstlist from \d+ to (\d+), rlist from [\d.]+ to ([\d.]+)', log)
    if changes:
        out['nstlist'], out['rlist'] = int(changes[-1][0]), float(changes[-1][1])
    out['list_history'] = re.findall(r'(?:outer|inner) list: updated every [^\n]+', log)
    beta = re.search(r'Gaussian width \(1/beta\) of ([\d.eE+-]+)', log)
    if beta:
        out['alpha'] = 1 / float(beta[1])
        out['alpha_parse_precision'] = 'rounded width in log; use full-precision upstream metadata for locked alpha'
    # Never infer graph activation from an input flag or a generic GPU-support line.
    if 'PP tasks will do non-perturbed short-ranged interactions on the GPU' in log:
        out['offload_nonbonded'] = 'gpu'
    if 'PME tasks will do all aspects on the GPU' in log:
        out['offload_pme'] = 'gpu'
    if 'PP task will update and constrain coordinates on the GPU' in log:
        out['offload_update'] = 'gpu'
    reset = re.search(r'step (\d+): resetting all time and cycle counters', log)
    if reset:
        out['warm_steps'] = int(reset[1])
    # Cycle rows establish their named host regions only. Keep missing or
    # unsupported runtime evidence explicit; it cannot populate device cadences.
    from gromacs_work import parse as parse_runtime_work
    try:
        out['gromacs_runtime_accounting'] = parse_runtime_work(log)
    except ValueError as exc:
        out['gromacs_runtime_accounting'] = dict(unavailable_reason=str(exc))
    if re.search(r'^topology:', settings, re.M):
        # Counts are exact integers; printed atom/parameter numbers are not a
        # substitute for a complete, precision-preserving canonical model.
        from gromacs_topology import parse as parse_topology
        try:
            topology = parse_topology(settings)
            out.update(n_atoms=topology['n_atoms'], n_molecules=topology['n_molecules'],
                       topology_sha256=None, topology_hash_unavailable_reason=topology['topology_hash_unavailable_reason'],
                       gromacs_topology_metadata=topology)
        except (ValueError, KeyError) as exc:
            out['gromacs_topology_import_error'] = str(exc)
    return out


def parse_pd(record):
    out = dict(record.get('resolved_settings', {}))
    for key in MODEL_FIELDS + WORK_FIELDS + SOLVER_FIELDS + ACTIVATION_FIELDS + ['warm_steps', 'timed_steps', 'loop_s']:
        if key in record:
            out.setdefault(key, record[key])
    if 'beta' in record and record.get('method', 'pme') == 'pme':
        out.setdefault('alpha', record['beta'])
    if 'n_ions' in record:
        out.setdefault('n_atoms', record['n_ions'])
    if 'n_waters' in record:
        out.setdefault('n_molecules', record['n_waters'])
    return out


def gate_evidence(manifest, name):
    """A boolean alone is not an accuracy/list gate; require retained, hash-checked evidence."""
    gate = manifest.get('gates', {}).get(name, {})
    evidence = gate.get('evidence', [])
    if gate.get('passed') is not True or not evidence:
        return False
    return all(Path(e.get('path', '')).is_file() and sha(e['path']) == e.get('sha256') for e in evidence)


def validate(manifest, resolved, exit_code):
    errors = []
    def fail(field, reason):
        errors.append(dict(field=field, reason=reason))
    if exit_code != 0:
        fail('exit_code', f'upstream failure or missing status: {exit_code!r}')
    if manifest.get('schema_version') != 1:
        fail('schema_version', 'expected 1')
    for key in ['card', 'attempt', 'case_id', 'engine', 'track']:
        if unknown(manifest.get(key)):
            fail(key, 'missing metadata')
    for group, required in [('runtime', RUNTIME_FIELDS), ('provenance', PIN_FIELDS)]:
        for key in required:
            if unknown(manifest.get(group, {}).get(key)):
                fail(group + '.' + key, 'missing metadata')
    pin = manifest.get('provenance', {})
    for name in ('source', 'inputs'):
        if pin.get(name + '_before') != pin.get(name + '_after'):
            fail('provenance.' + name, 'changed during execution')
    fields = MODEL_FIELDS + WORK_FIELDS + ACTIVATION_FIELDS
    method = str(resolved.get('method', '')).lower()
    if method == 'esp':
        fields += ['method', 'real_cutoff', 'split_shape', 'window_shape', 'order', 'mesh', 'grid_policy']
    elif method not in ('none', 'cut-off', 'lj'):
        fields += SOLVER_FIELDS
    for key in fields:
        if unknown(resolved.get(key)):
            fail('resolved.' + key, 'missing actual setting; no default inferred')
    for key, value in manifest.get('requested', {}).items():
        if unknown(resolved.get(key)) or resolved[key] != value:
            fail('requested.' + key, f'requested {value!r}, resolved {resolved.get(key)!r}')
    counts = manifest.get('completed', {})
    if not isinstance(counts.get('cases'), int) or counts['cases'] < 1:
        fail('completed.cases', 'zero/missing completed cases')
    if counts.get('kind') == 'tests' and (counts.get('passed', 0) < 1 or any(counts.get(k, 0) for k in ('failed', 'errored', 'unexpected_skipped'))):
        fail('completed.tests', 'no numerical pass established')
    return errors


def compare(a, b, track):
    errors = []
    for label, bundle in [('baseline', a), ('candidate', b)]:
        errors += [dict(field=label + '.' + e['field'], reason=e['reason']) for e in
                   validate(bundle['manifest'], bundle['resolved'], bundle['exit_code'])]
    for key in ('host', 'gpu_uuid', 'gpu_model'):
        if a['manifest'].get('runtime', {}).get(key) != b['manifest'].get('runtime', {}).get(key):
            errors.append(dict(field='runtime.' + key, reason='paired timing requires the same host/device'))
    fields = MODEL_FIELDS + WORK_FIELDS
    fields += SOLVER_FIELDS if track == 'locked-pme' else ['real_cutoff']
    for key in fields:
        av, bv = a['resolved'].get(key), b['resolved'].get(key)
        if av != bv:
            errors.append(dict(field=key, baseline=av, candidate=bv, reason='settings mismatch'))
    if track not in ('locked-pme', 'common-accuracy'):
        errors.append(dict(field='track', reason='unknown comparison track'))
    if track == 'locked-pme' and any(str(x['resolved'].get('method', '')).lower() != 'pme' for x in (a, b)):
        errors.append(dict(field='method', reason='locked-PME requires PME in both engines'))
    if any(a['resolved'].get(k) != b['resolved'].get(k) for k in ('list_policy', 'sort_policy')):
        if not all(gate_evidence(x['manifest'], 'list') for x in (a, b)):
            errors.append(dict(field='list_policy', reason='different lists require passed list-error evidence'))
    if track == 'common-accuracy' and not all(gate_evidence(x['manifest'], 'accuracy') for x in (a, b)):
        errors.append(dict(field='accuracy', reason='common-accuracy requires independent force/pressure/drift evidence'))
    return dict(comparable=not errors, errors=errors, track=track)


def paired_summary(samples):
    """Predeclared Student-t CI on paired log ratios; retain invalid rows, reject incomplete pairs."""
    errors = []
    groups = {}
    for key in ('host', 'gpu_uuid'):
        if len({s.get(key) for s in samples}) > 1:
            errors.append(f'{key}: cannot pool pairs across machines/devices')
    for index, s in enumerate(samples):
        if 'pair' not in s:
            errors.append(f'row {index}: missing pair')
        pair = s.get('pair', f'missing-{index}')
        groups.setdefault(pair, []).append(s)
        for key in ('valid', 'stabilized', 'unprofiled'):
            if s.get(key) is not True:
                errors.append(f'pair {pair}: {key} is not true')
        for key, lower in [('warm_steps', 5000), ('seconds', 5), ('rebuilds', 20)]:
            if not isinstance(s.get(key), (int, float)) or not math.isfinite(s[key]) or s[key] < lower:
                errors.append(f'pair {pair}: {key} below {lower} or invalid')
        if s.get('ensemble') == 'npt' and s.get('pressure_events', 0) < 20:
            errors.append(f'pair {pair}: fewer than 20 pressure events')
    ratios = []; base = []; cand = []
    for pair, rows in groups.items():
        r = {s.get('role'): s for s in rows}
        if len(rows) != 2 or set(r) != {'baseline', 'candidate'}:
            errors.append(f'pair {pair}: incomplete/duplicate pair'); continue
        a, b = r['baseline'], r['candidate']
        for key in ('host', 'gpu_uuid', 'steps', 'dt', 'ensemble'):
            if unknown(a.get(key)) or a.get(key) != b.get(key):
                errors.append(f'pair {pair}: {key} mismatch/missing')
        if any(not isinstance(s.get('steps'), int) or s['steps'] <= 0 for s in (a, b)):
            errors.append(f'pair {pair}: no measured steps'); continue
        if any(not isinstance(s.get('seconds'), (int, float)) or not math.isfinite(s['seconds']) or s['seconds'] <= 0 for s in (a, b)):
            errors.append(f'pair {pair}: invalid/missing measured time'); continue
        if any(s['steps'] % 2000 for s in (a, b)):
            errors.append(f'pair {pair}: incomplete production cadence cycles')
        av = a['seconds'] / a['steps']; bv = b['seconds'] / b['steps']
        if av <= 0 or bv <= 0:
            errors.append(f'pair {pair}: invalid time'); continue
        base.append(av); cand.append(bv); ratios.append(math.log(av / bv))
    if len(ratios) not in (5, 10):
        errors.append('exactly five pairs, or one predeclared ten-pair extension, required')
    if errors:
        return dict(accepted=False, errors=errors, pair_count=len(ratios))
    mean = statistics.mean(ratios)
    t = 2.7764451052 if len(ratios) == 5 else 2.2621571629
    half = t * statistics.stdev(ratios) / math.sqrt(len(ratios))
    lo, hi = math.exp(mean-half), math.exp(mean+half)
    return dict(accepted=True, pair_count=len(ratios), median_baseline_ms=1000*statistics.median(base),
                median_candidate_ms=1000*statistics.median(cand), speed_ratio=math.exp(mean), ci95=[lo, hi],
                estimator='Student-t on paired log ratios; independent pairs, approximately normal log ratios',
                outcome='improvement' if lo > 1 else 'slower' if hi < 1 else 'inconclusive')


def import_bundle(args):
    output = args.output.resolve()
    output.mkdir()  # unique outputs, never PID-only scratch or overwrite
    record = json.loads(args.record.read_text()) if args.record else {}
    settings = args.settings.read_text() if args.settings else ''
    log = args.log.read_text() if args.log else ''
    manifest = json.loads(args.manifest.read_text()) if args.manifest else record.get('benchmark_manifest', {})
    resolved = parse_pd(record) if args.engine == 'pd' else parse_gromacs(settings, log)
    errors = validate(manifest, resolved, args.exit_code)
    bundle = dict(manifest=manifest, resolved=resolved, exit_code=args.exit_code)
    originals = output / 'originals'; originals.mkdir()
    fingerprints = {}
    for name in ('record', 'settings', 'log', 'manifest'):
        path = getattr(args, name)
        if path:
            (originals / (name + path.suffix)).write_bytes(path.read_bytes())
            fingerprints[str(path.resolve())] = sha(path)
    write(output / 'bundle.json', bundle)
    write(output / 'manifest.json', manifest or {'unavailable_reason': 'legacy artifact has no complete case manifest'})
    write(output / 'resolved-settings.json', resolved)
    write(output / 'comparability.json', dict(accepted=not errors, errors=errors))
    write(output / 'metrics.json', dict(metadata_gate=not errors, physics_gate=None, performance_gate=None,
                                      unavailable_reason='adapter performs no physics or throughput measurement'))
    write(output / 'status.json', dict(state='completed adapter', exit_code=0 if not errors else 2,
                                     upstream_exit_code=args.exit_code, scientific_acceptance='pending'))
    (output / 'profiles').mkdir()
    (output / 'source.sha256').write_text(''.join(f'{h}  {p}\n' for p, h in fingerprints.items()))
    (output / 'inputs.sha256').write_text('# unavailable: adapter inputs are indexed in source.sha256; simulation input hashes require manifest\n')
    (output / 'commands.txt').write_text(repr(sys.argv) + '\n')
    (output / 'run.log').write_text(json.dumps(dict(errors=errors)) + '\n')
    with (output / 'samples.csv').open('w') as f:
        csv.writer(f).writerow(['pair', 'role', 'steps', 'seconds', 'valid', 'exclusion_reason'])
    (output / 'decision.md').write_text('Metadata adapter: ' + ('pass' if not errors else 'rejected/incomplete') +
        '. No physics or performance acceptance. Preserve originals; supply missing actual metadata before comparison.\n')
    write(output / 'artifact-index.json', dict(files=sorted(p.name for p in output.iterdir()) + ['artifact-index.json'],
        unavailable={'profiles': 'legacy adapter does not profile', 'timing_samples': 'not supplied; no inferred timings'}))
    return 0 if not errors else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import', help='adapt saved actual records/logs, reject unknown evidence')
    imp.add_argument('--engine', choices=['pd', 'gromacs'], required=True)
    for key in ('record', 'settings', 'log', 'manifest'):
        imp.add_argument('--' + key, type=Path)
    imp.add_argument('--exit-code', type=int)
    imp.add_argument('--output', type=Path, required=True)
    comp = sub.add_parser('compare')
    comp.add_argument('baseline', type=Path); comp.add_argument('candidate', type=Path)
    comp.add_argument('--track', choices=['locked-pme', 'common-accuracy'], required=True)
    comp.add_argument('--output', type=Path, required=True)
    mesh = sub.add_parser('mesh', help='require successful upstream run and three positive resolved mesh dimensions')
    mesh.add_argument('--settings', type=Path, required=True)
    mesh.add_argument('--status', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'import':
        return import_bundle(args)
    if args.command == 'mesh':
        try:
            if args.status.read_text().strip() != '0':
                raise ValueError('exit_code: upstream result did not succeed')
            value = parse_gromacs(args.settings.read_text(), '').get('mesh')
            if value is None:
                raise ValueError('mesh: three positive resolved dimensions required')
            print(','.join(map(str, value)))
            return 0
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    result = compare(json.loads((args.baseline/'bundle.json').read_text()),
                     json.loads((args.candidate/'bundle.json').read_text()), args.track)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    return 0 if result['comparable'] else 2


if __name__ == '__main__':
    sys.exit(main())
