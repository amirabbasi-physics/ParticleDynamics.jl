#!/usr/bin/env python3
"""Import bounded, actual GROMACS dump topology metadata; never accept exact equivalence.

Supports the benchmark's point-particle and rigid SETTLE models. Text numeric
tokens are retained: atom masses/charges are printed at limited precision, so
this importer deliberately cannot produce a canonical cross-engine topology hash.
No forces, reference physics, integration or parameter mixing are calculated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def one(pattern, text, label):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise ValueError(f'{label}: expected one record, found {len(matches)}')
    return matches[0]


def scalar(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError('nonfinite numeric token: '+token)
    return value


def indexed(rows, count, label):
    ids = [int(row[0]) for row in rows]
    if ids != list(range(count)):
        raise ValueError(f'{label}: missing, duplicate or unordered indices')


def parse(text):
    # DOTALL is intentionally local to this bounded topology block.
    topology = one(r'^topology:\n((?s:.*?))(?=^grp\[)', text, 'complete topology')
    n = int(one(r'^   #atoms\s*=\s*(\d+)\s*$', topology, 'atom count'))
    header_n = int(one(r'^   natoms\s*=\s*(\d+)\s*$', text, 'TPR header atom count'))
    if n != header_n or n <= 0:
        raise ValueError('TPR header/topology atom count mismatch')
    block_count = int(one(r'^   #molblock\s*=\s*(\d+)\s*$', topology, 'molecule block count'))
    if one(r'^   bIntermolecularInteractions\s*=\s*(\S+)\s*$', topology, 'intermolecular flag') != 'false':
        raise ValueError('unsupported intermolecular interactions')
    blocks = re.findall(r'^   molblock \((\d+)\):\n((?s:.*?))(?=^   \S)', topology, re.M)
    indexed(blocks, block_count, 'molecule blocks')
    molecules = []
    for _, body in blocks:
        typ, name = one(r'^      moltype\s*=\s*(\d+) "([^"]+)"\s*$', body, 'molecule type')
        count = int(one(r'^      #molecules\s*=\s*(\d+)\s*$', body, 'molecule count'))
        for state in ('A', 'B'):
            if int(one(r'^      #posres_x'+state+r'\s*=\s*(\d+)\s*$', body, 'position restraints')):
                raise ValueError('unsupported position restraints')
        if count <= 0:
            raise ValueError('empty molecule block')
        molecules.append(dict(type=int(typ), name=name, count=count))
    atnr = int(one(r'^      atnr=(\d+)\s*$', topology, 'atom type count'))
    ntypes = int(one(r'^      ntypes=(\d+)\s*$', topology, 'function type count'))
    functions = re.findall(r'^         functype\[(\d+)\]=(\w+), (.+)$', topology, re.M)
    indexed(functions, ntypes, 'function types')
    if atnr <= 0 or ntypes < atnr*atnr:
        raise ValueError('missing nonbonded pair table')
    parameters = []
    for idx, kind, body in functions:
        expected = {'c6', 'c12'} if kind == 'LJ_SR' else {'doh', 'dhh'} if kind == 'SETTLE' else None
        tokens = re.findall(r'(\w+)=\s*([^,\s]+)', body)
        if expected is None or len(tokens) != len(expected) or {k for k, _ in tokens} != expected:
            raise ValueError('unsupported or incomplete function '+kind)
        if int(idx) < atnr*atnr and kind != 'LJ_SR':
            raise ValueError('nonbonded table contains non-LJ function')
        if any(scalar(v) < 0 for _, v in tokens):
            raise ValueError('negative nonbonded/constraint parameter')
        parameters.append(dict(kind=kind, printed_parameters=dict(tokens)))
    reppow = scalar(one(r'^      reppow\s*=\s*(\S+)\s*$', topology, 'repulsion power'))
    if reppow != 12:
        raise ValueError('unsupported repulsion power')
    fudge = one(r'^      fudgeQQ\s*=\s*(\S+)\s*$', topology, 'Coulomb 1-4 scale')
    scalar(fudge)
    sections = re.findall(r'^   moltype \((\d+)\):\n((?s:.*?))(?=^   moltype \(|\Z)', topology, re.M)
    indexed(sections, len(sections), 'molecule templates')
    templates = []
    for _, body in sections:
        name = one(r'^      name="([^"]+)"\s*$', body, 'template name')
        count = int(one(r'^      atoms:\n         atom \((\d+)\):', body, 'template atom count'))
        atoms = re.findall(r'^            atom\[\s*(\d+)\]=\{type=([^\n]+)\}$', body, re.M)
        indexed(atoms, count, 'template atoms')
        parsed_atoms = []
        for _, row in atoms:
            fields = dict([('type', row.split(',', 1)[0].strip())] + re.findall(r',\s*(\w+)=\s*([^,}]+)', row))
            if not {'type', 'typeB', 'ptype', 'm', 'q', 'mB', 'qB'} <= fields.keys():
                raise ValueError('incomplete atom record')
            for key in ('m', 'q', 'mB', 'qB'):
                scalar(fields[key])
            if fields['ptype'].strip() != 'Atom' or scalar(fields['m']) <= 0:
                raise ValueError('unsupported particle type or nonpositive mass')
            if any(scalar(fields[a]) != scalar(fields[b]) for a, b in [('m', 'mB'), ('q', 'qB'), ('type', 'typeB')]):
                raise ValueError('unsupported A/B-state perturbation')
            typ = int(fields['type'])
            if not 0 <= typ < atnr:
                raise ValueError('atom type outside pair table')
            parsed_atoms.append(dict(type=typ, mass_printed=fields['m'].strip(), charge_printed=fields['q'].strip()))
        lists = int(one(r'^         numLists=(\d+)\s*$', body, 'exclusion list count'))
        elements = int(one(r'^         numElements=(\d+)\s*$', body, 'exclusion element count'))
        ex = re.findall(r'^         excls\[(\d+)\]\[num=(\d+)\]=\{([^}]*)\}\s*$', body, re.M)
        indexed(ex, lists, 'exclusions')
        excluded = [[int(x.strip()) for x in row.split(',') if x.strip()] for _, _, row in ex]
        if lists != count or sum(map(len, excluded)) != elements:
            raise ValueError('exclusion counts inconsistent')
        for i, ((_, expected, _), values) in enumerate(zip(ex, excluded)):
            if len(values) != int(expected) or len(set(values)) != len(values) or i not in values or any(j < 0 or j >= count or i not in excluded[j] for j in values):
                raise ValueError('invalid exclusion membership')
        interactions = re.findall(r'^      ([^\n:]+):\n((?s:.*?))(?=^      \S|\Z)', body, re.M)
        names = [name for name, _ in interactions if name != 'atoms']
        if len(names) != len(set(names)) or not {'Bond', 'Constraint', 'Constr. No Co.', 'Settle', 'Virtual site N', 'dVtemp/dl'} <= set(names):
            raise ValueError('missing or duplicate interaction sections')
        settles = []
        for label, section in interactions:
            if label in ('atoms', 'excls'):
                continue
            nr = int(one(r'^         nr: (\d+)\s*$', section, label+' count'))
            if label != 'Settle' and nr:
                raise ValueError('unsupported nonzero interaction '+label)
            if label == 'Settle':
                records = re.findall(r'^            (\d+) type=(\d+) \(SETTLE\)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', section, re.M)
                if nr != 4*len(records):
                    raise ValueError('SETTLE count mismatch')
                for offset, p, *ids in records:
                    ids = list(map(int, ids)); p = int(p)
                    if int(offset) != 4*len(settles) or p >= len(parameters) or parameters[p]['kind'] != 'SETTLE' or len(set(ids)) != 3 or any(i >= count for i in ids):
                        raise ValueError('invalid SETTLE assignment')
                    settles.append(dict(parameter=p, atoms=ids))
        templates.append(dict(name=name, atoms=parsed_atoms, exclusions_including_self=excluded, settle=settles))
    expanded = 0; masses = Counter(); charges = Counter(); constraints = 0
    for block in molecules:
        if block['type'] >= len(templates):
            raise ValueError('missing referenced molecule template')
        template = templates[block['type']]
        if template['name'] != block['name']:
            raise ValueError('molecule name mismatch')
        expanded += len(template['atoms'])*block['count']
        constraints += 3*len(template['settle'])*block['count']
        for atom in template['atoms']:
            masses[atom['mass_printed']] += block['count']; charges[atom['charge_printed']] += block['count']
    if expanded != n:
        raise ValueError('expanded molecule/header atom count mismatch')
    return dict(n_atoms=n, n_molecules=sum(b['count'] for b in molecules), atom_type_count=atnr,
                molecule_blocks=molecules, molecule_templates=templates, function_types=parameters,
                constraints_count=constraints, masses_printed_counts=dict(masses), charges_printed_counts=dict(charges),
                fudgeQQ_printed=fudge, topology_sha256=None,
                topology_hash_unavailable_reason='gmx dump prints rounded atom/state numbers; exact native values and complete cross-engine conversion are not established',
                numerical_acceptance='not measured: topology inspection only', canonical_equivalence=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = parse(a.dump.read_text())
    result['provenance'] = dict(dump=str(a.dump.resolve()), dump_sha256=hashlib.sha256(a.dump.read_bytes()).hexdigest(),
                                importer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
