#!/usr/bin/env python3
"""Import printed GROMACS cycle counters; no device-phase or speed acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

TITLE='R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G'
VALUE=r'(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
ROW=re.compile(r'^\s*(.*?)\s{2,}(\d+)\s+(\d+)\s+(\d+)\s+('+VALUE+r')\s+('+VALUE+r')\s+('+VALUE+r')\s*$')
SUMMARY=re.compile(r'^\s*(Rest|Total)\s+('+VALUE+r')\s+('+VALUE+r')\s+('+VALUE+r')\s*$')


def parse(text, exit_code=None):
    if exit_code is not None and (type(exit_code)!=int or exit_code!=0):
        raise ValueError('failed GROMACS execution')
    if text.count(TITLE)!=1:raise ValueError('missing/duplicate cycle accounting table')
    if len(re.findall(r'^Finished mdrun on rank 0 .+$',text,re.M))!=1:
        raise ValueError('missing/duplicate final mdrun marker')
    tail=text.split(TITLE,1)[1]
    if tail.count('Core t (s)')!=1:raise ValueError('truncated cycle accounting footer')
    body=tail.split('Core t (s)',1)[0]
    layout=re.findall(r'^On (\d+) MPI ranks?, each using (\d+) OpenMP threads$',body,re.M)
    if len(layout)!=1 or any(int(x)<=0 for x in layout[0]):raise ValueError('unsupported cycle rank/thread layout')
    ranks,threads=map(int,layout[0]);groups={'main':{},'pme_breakdown':{}};group='main';summary={};total_seen=False;rows_started=False
    for line in body.splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith(('On ','Activity:','Ranks Threads')) or re.fullmatch('-+',stripped):continue
        if stripped=='Breakdown of PME mesh activities':
            if group!='main' or not total_seen:raise ValueError('duplicate/misplaced PME breakdown')
            group='pme_breakdown';continue
        match=ROW.fullmatch(line)
        if match:
            name,nrank,nthread,calls,wall,cycles,percent=match.groups()
            if total_seen and group=='main' or not name or name in groups[group]:raise ValueError('duplicate/misplaced cycle row')
            if (int(nrank),int(nthread))!=(ranks,threads):raise ValueError('unsupported per-row rank/thread layout')
            groups[group][name]=dict(calls=int(calls),ranks=int(nrank),threads=int(nthread),
                                    wall_seconds_printed=wall,gigacycles_printed=cycles,percent_printed=percent)
            rows_started=True;continue
        match=SUMMARY.fullmatch(line)
        if match:
            name,wall,cycles,percent=match.groups()
            if group!='main' or name in summary or not rows_started or total_seen:raise ValueError('duplicate/misplaced cycle summary')
            summary[name]=dict(wall_seconds_printed=wall,gigacycles_printed=cycles,percent_printed=percent)
            total_seen=name=='Total';continue
        raise ValueError('unsupported/truncated cycle row: '+stripped)
    if not total_seen or not groups['main'] or not {'Rest','Total'}<=summary.keys():raise ValueError('incomplete cycle table')
    if group=='pme_breakdown' and not groups['pme_breakdown']:raise ValueError('empty PME breakdown')
    reset=[int(x) for x in re.findall(r'^step (\d+): resetting all time and cycle counters\s*$',text,re.M)]
    if len(reset)>1:raise ValueError('unsupported multiple cycle reset windows')
    messages={k:line for k,line in {
        'nonbonded':'PP tasks will do non-perturbed short-ranged interactions on the GPU',
        'pme':'PME tasks will do all aspects on the GPU',
        'update':'PP task will update and constrain coordinates on the GPU'}.items() if line in text}
    return dict(schema=1,kind='printed GROMACS cycle accounting',groups=groups,summary=summary,
                reset_step=reset[0] if reset else None,
                reset_scope='counter reset reported at this step' if reset else 'no reset marker; entire run scope, not an assumed warm window',
                offload_messages=messages,execution_exit_code=exit_code,printed_finish=True,
                execution_scope='exit status supplied by caller' if exit_code==0 else 'historical process exit unknown; finish marker alone does not establish job success',
                counter_scope='calls to named host wallcycle regions; not device kernel launches or logical physical-event counts; boundaries may include initial/final evaluation',
                timing_scope='rounded printed wall/cycle regions may overlap; PME breakdown is contained in main regions; no sum or critical-path attribution',
                unavailable=dict(graph_replays='no measured graph replay evidence imported',
                    half_full_K='Kinetic energy region does not distinguish half/full K or internal reductions',
                    thermostat_updates='stored coupling interval and wallcycle calls do not establish executed stochastic updates',
                    virial_events='no actual per-step virial execution evidence imported',
                    transfer_bytes='state-copy region counts are not transfer counts/bytes'),
                scientific_acceptance='not measured',performance_acceptance='not measured; no current equivalent-physics paired controls')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',required=True,type=Path)
    p.add_argument('--exit-code',type=int,help='actual recorded upstream exit, never inferred from the printed footer')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    raw=a.log.read_bytes();result=parse(raw.decode(),a.exit_code)
    result['input_sha256']={str(a.log.resolve()):hashlib.sha256(raw).hexdigest()}
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
