#!/usr/bin/env python3
"""Opt-in MDP neighbor-list controls; defaults leave the input bytes unchanged."""
import math
import os
from pathlib import Path
import re
import sys


def configure(text,nstlist=None,rlist=None,tolerance=None):
    changes={}
    if tolerance not in (None,'') and rlist in (None,''):
        value=float(tolerance)
        if not math.isfinite(value) or value<=0:raise ValueError('VERLET_BUFFER_TOLERANCE must be positive')
        changes['verlet-buffer-tolerance']=repr(value)  # shortest exact round-trip
    if nstlist not in (None,''):
        if not re.fullmatch(r'[1-9][0-9]*',str(nstlist)) or int(nstlist)>2147483647:
            raise ValueError('NSTLIST must be a positive Int32')
        changes['nstlist']=str(int(nstlist))
    if rlist not in (None,''):
        radius=float(rlist)
        cutoffs=[float(x) for x in re.findall(r'^\s*(?:rcoulomb|rvdw)\s*=\s*([^;\s]+)',text,re.M)]
        if not cutoffs or not math.isfinite(radius) or radius<max(cutoffs):
            raise ValueError('RLIST must be finite and at least the largest interaction cutoff')
        changes.update({'rlist':format(radius,'.17g'),'verlet-buffer-tolerance':'-1'})
    for key,value in changes.items():
        pattern=r'^\s*'+key+r'\s*=.*$';count=len(re.findall(pattern,text,re.M))
        if count>1:raise ValueError('Duplicate MDP field: '+key)
        if count:text=re.sub(pattern,key+' = '+value,text,flags=re.M)
        else:text+=key+' = '+value+'\n'
    return text


if __name__=='__main__':
    p=Path(sys.argv[1]);original=p.read_text();candidate=configure(original,os.getenv('NSTLIST'),os.getenv('RLIST'),os.getenv('VERLET_BUFFER_TOLERANCE'))
    if candidate!=original:p.write_text(candidate)
