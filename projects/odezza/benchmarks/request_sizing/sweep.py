# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Matched native JSON sizing grid. No derivative fitting or target leakage.

Run with CUDA_MODULE_LOADING=EAGER PYTHONPATH=python. Every mode enumerates
the same finite 4,096 structures and bank addresses. Store raw reports outside
Git; summarize only after exact winner/count comparison and CPU replay.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time
from odezza.grammar.native_service import NativeService


def balanced(terms):
    if len(terms) == 1:
        return terms[0]
    m = len(terms)//2
    return '(' + balanced(terms[:m]) + '+' + balanced(terms[m:]) + ')'


def request(states, slots, long_ast, steps):
    names = [f'x{i}' for i in range(states)]
    ts = [0, .01, .02] if steps == 4 else [i*.04 for i in range(9)]
    problem = dict(states=names, known_rhs={x:'-'+x for x in names[1:]},
        trajectories=[dict(initial=[.1]*states, times=ts,
            values=[[.1*math.exp(-t)]*states for t in ts])])
    a = [f'sin(x{i%states}+{(i+1)/1000:.3f})' for i in range(64)]
    b = [f'cos(x{i%states}+{(i+1)/100:.2f})' for i in range(64)]
    rhs = '-x0+.01*(A*B)'
    if long_ast:
        rhs += '+.01*'+balanced([f'sin(x{i%states}*x{(i+1)%states})' for i in range(16)])
    if slots:
        rhs += '+'+balanced([f'rng.c{i}*x{i%states}' for i in range(slots)])
    g = dict(states=names, integration=dict(method='rk4',dt=.01),
        rules=dict(A=a,B=b), rhs=dict(x0=rhs),
        expansion=dict(strategy='enumerate',max_nodes=255,max_depth=32),
        limits=dict(max_skeletons=4096,max_variants=4096,max_configurations=4096*(256 if slots else 1)),
        retain={'global':{'k':8,'unit':'evaluation_row'},'per_family':{'k':8,'unit':'evaluation_row'}})
    if slots:
        g['rng_banks'] = {'u':dict(base='uniform01',count=256,seed=23865)}
        g['rng'] = {f'c{i}':dict(bank='u',axis='trial',stream=f'c{i}',
            transform=dict(kind='uniform',low=-.01,high=.01)) for i in range(slots)}
    return dict(problem=problem,grammar=g,execution=dict(max_seconds=120))


def signature(r):
    return [(i,r['candidates'][i]['mse'],r['candidates'][i]['values'],
             r['candidates'][i]['resolved_programs']) for i in r['leaderboards']['global']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--devices',default='0,1');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--modes',default='');p.add_argument('--long-only',action='store_true')
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    cases=[(2,0,False,4),(2,1,False,4),(6,8,False,4),(6,8,True,4),
           (12,24,False,4),(12,24,True,4),(6,8,False,32),(12,24,True,32)]
    modes={'previous_defaults':{'batch_variants':256,'module_systems':64},
           'automatic':{},'page1024_module64':{'batch_variants':1024,'module_systems':64},
           'page256_module32':{'batch_variants':256,'module_systems':32},
           'page1024_module16':{'batch_variants':1024,'module_systems':16}}
    if a.modes:modes={k:v for k,v in modes.items() if k in a.modes.split(',')}
    if a.long_only:cases=[c for c in cases if c[-1]==32]
    result=dict(devices=a.devices,runs=[],summary=[])
    service=NativeService('/tmp/odz-sizing-grid-cache',devices=list(map(int,a.devices.split(','))))
    try:
        for case in cases:
            expected=None
            for mode,options in modes.items():
                times=[]
                for repeat in range(a.repeats+1):
                    q=request(*case);q['execution'].update(options)
                    # Extra profiled run: excluded from the performance median.
                    profile=repeat==a.repeats;q['execution']['profile_timing']=profile
                    t=time.monotonic();h=service.submit(**q)
                    r=service.jobs[h['job_id']].result(timeout=150)
                    assert r['status']=='complete' and r['retention_complete'],r
                    assert sum(f['generated_asts'] for f in r['families'])==4096,r['families']
                    assert r['counts']['completed_configurations']==q['grammar']['limits']['max_configurations']
                    sig=signature(r)
                    if expected is None:expected=sig
                    assert sig==expected,(case,mode,'changed results')
                    for d in r['execution']['devices']:
                        b=d['pipeline_breakdown']
                        if profile:
                            assert b['profiled_modules']==b['modules']
                            assert b['gpu_active_seconds']<=b['gpu_span_seconds']+1e-6
                            assert b['gpu_active_seconds']<=b['gpu_sum_seconds']+1e-6
                        else:assert b['gpu_active_seconds'] is None
                    if repeat==0:
                        ident=r['leaderboards']['global'][0]
                        replay=service.replay(h['job_id'],ident,cpu=True)
                        cpu=replay['cpu_reference']['mse'];gpu=r['candidates'][ident]['mse']
                        assert abs(cpu-gpu)<1e-8+3e-5*abs(cpu),(cpu,gpu)
                    if repeat and not profile:times.append(r['timing']['total_seconds'])
                    result['runs'].append(dict(case=case,mode=mode,repeat=repeat,profile=profile,
                        elapsed=time.monotonic()-t,report=r))
                    out.write_text(json.dumps(result,indent=2)+'\n')
                    service.release(job_id=h['job_id'])
                row=dict(case=case,mode=mode,warm_median=statistics.median(times),actual_rk4_steps=r['integration']['steps_per_configuration'],actual_points=sum(t['points'] for t in r['trajectories']))
                result['summary'].append(row);out.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(row),flush=True)
    finally:service.close()


if __name__=='__main__':main()
