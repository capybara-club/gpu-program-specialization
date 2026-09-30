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
"""Matched old/new runtime coverage and sizing, including long integrations.

Uses frozen analytic observations and identical grammar/RNG addresses. No search
controller and no private cap override. Raw reports belong outside Git.
"""
import argparse
from copy import deepcopy
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from memory_checks import Runtime, P


def signature(r):
    return hashlib.sha256(json.dumps(dict(candidates=r['candidates'], leaderboards=r['leaderboards'],
        counts={k:r['counts'][k] for k in ('completed_configurations','valid','invalid')}),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def request(states=3, slots=1, rows=524288, steps=5120, structures=1, toggles=0, normal=False):
    names=[f'x{i}' for i in range(states)]
    intervals,substeps=(20,256) if steps==5120 else (16,4096) if steps==65536 else (1,steps)
    times=[i/16 for i in range(intervals+1)]
    initial=[.1+.01*i for i in range(states)]
    problem=dict(states=names,known_rhs={x:'-.1*'+x for x in names[1:]},
        trajectories=[dict(initial=initial,times=times,
            values=[[v*math.exp(-.1*t) for v in initial] for t in times])])
    terms=[f'rng.c{i}*x{i%states}' for i in range(slots)]
    expr='-.1*x0'+('+'+'+'.join(terms) if terms else '')
    if toggles:expr+='+.001*leaf.s'
    g=dict(version=1,states=names,rhs={'x0':'R'},
        rules={'R':[expr+f'+{i/10000:.4f}' for i in range(structures)]},
        integration=dict(method='rk4',dt=1/(16*substeps)),
        expansion=dict(max_nodes=255,max_depth=64),
        limits=dict(max_skeletons=structures,max_variants=structures,
            max_configurations=structures*(rows if slots else 1)*(toggles or 1)),
        retain={'global':{'k':8,'unit':'evaluation_row'},'per_family':{'k':8,'unit':'evaluation_row'}})
    if slots:
        g['rng_banks']={'u':dict(base='normal01' if normal else 'uniform01',count=rows,seed=12857)}
        g['rng']={f'c{i}':dict(bank='u',axis='row',stream=f'c{i}',
            transform=dict(kind='affine',scale=.02,shift=-.01)) for i in range(slots)}
    if toggles:g['leaves']={'s':dict(states=names[:toggles],arity=toggles,coverage='explicit',groups=[names[:toggles]])}
    return dict(problem=problem,grammar=g,execution=dict(max_seconds=120,profile_timing=True,dedup_bytes_per_family=1<<20))


def cancelled(runtime,q):
    raw=json.dumps(q).encode();job=P();l=runtime.lib
    assert not l.odz_job_create(raw,len(raw),C.byref(job))
    l.odz_job_cancel.argtypes=[P]
    cancel_times=[]
    def cancel():
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            size=C.c_size_t();out=C.create_string_buffer(8192)
            assert not l.odz_job_status(job,out,len(out),C.byref(size))
            status=json.loads(out.value)
            if status['counts']['completed_configurations']:
                cancel_times.append(time.monotonic());l.odz_job_cancel(job);return
            time.sleep(.002)
        cancel_times.append(time.monotonic());l.odz_job_cancel(job)
    thread=threading.Thread(target=cancel);thread.start()
    try:
        l.odz_job_run(runtime.handle,job);finished=time.monotonic();thread.join()
        size=C.c_size_t();assert not l.odz_job_report(job,None,0,C.byref(size))
        out=C.create_string_buffer(size.value);assert not l.odz_job_report(job,out,len(out),C.byref(size))
        r=json.loads(out.value)
        assert r['status']=='cancelled' and not r['runtime_quarantined'],r.get('error')
        assert 0<r['counts']['completed_configurations']<q['grammar']['limits']['max_configurations']
        assert finished-cancel_times[0]<5,'cooperative cancellation regression'
        return dict(cancel_seconds=finished-cancel_times[0],report=r)
    finally:thread.join();l.odz_job_destroy(job)


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);p.add_argument('--devices',default='0,1');a=p.parse_args()
    os.environ['CUDA_MODULE_LOADING']='EAGER';devices=list(map(int,a.devices.split(',')))
    out=Path(a.output);assert not out.exists();result=dict(passed=False,devices=devices,records=[])
    def save(name,**fields):
        result['records'].append(dict(name=name,**fields));out.write_text(json.dumps(result,indent=2)+'\n')
        print(name,fields.get('seconds',''),flush=True)
    cases=[('long_single',request()),('long_tail',request(rows=524291)),
        ('long_toggle2',request(rows=131073,toggles=2)),
        ('long_toggle4_normal',request(states=6,slots=8,rows=32769,toggles=4,normal=True)),
        ('maximum_steps',request(rows=8193,steps=65536)),
        ('many_structures_few_rows',request(rows=8,structures=1024,steps=4)),
        ('literal_structures',request(slots=0,structures=1024,steps=40)),
        ('twelve_states_24_slots',request(states=12,slots=24,rows=32769,steps=40)),
        ('scarce_work',request(rows=7,steps=5120))]
    explicit=request();explicit['execution']['max_chunk_configurations']=32768;cases.append(('explicit_small_limit',explicit))
    memory=request(rows=65539,slots=8);memory['execution']['max_device_bytes']=128<<10;cases.append(('memory_limited',memory))
    for name,q in cases:
        reports=[]
        for label,library in [('baseline',a.baseline),('production',a.library)]:
            runtime=Runtime(devices=devices,library=library)
            try:
                record=runtime.run(q);r=record['report']
                save(name+'_'+label,seconds=r['timing']['total_seconds'],report=r)
                assert r['status']=='complete' and r['retention_complete'],(name,label,r.get('error'))
                assert r['counts']['completed_configurations']==q['grammar']['limits']['max_configurations'],r['counts']
                if reports:assert signature(r)==signature(reports[0]),(name,'results/coverage changed')
                reports.append(r)
            finally:runtime.close()
        r=reports[1]
        for d in r['execution']['devices']:
            if not d['completed_configurations']:continue
            s=d['sizing'];assert d['largest_tile_configurations']<=s['work_ceiling_configurations']
            assert s['p95_scoring_call_seconds_upper_bound']<=max(.000001,2*s['maximum_scoring_call_seconds'])
        if name in ('long_single','long_tail'):
            assert r['counts']['completed_chunks']<=3
        if name=='scarce_work':assert sum(d['sizing']['underfilled_tiles'] for d in r['execution']['devices'])>0
        if name=='memory_limited':assert sum(d['sizing']['memory_limited_tiles'] for d in r['execution']['devices'])>0
        if name=='explicit_small_limit':
            assert max(d['largest_tile_configurations'] for d in r['execution']['devices'])<=32768
    runtime=Runtime(devices=devices,library=a.library)
    try:
        q=request(rows=262145,steps=65536);record=runtime.run(q);save('maximum_steps_large_production_only',**record)
        assert record['report']['status']=='complete' and record['report']['counts']['completed_configurations']==262145
        q=request(rows=524288,structures=64)
        save('cancel_during_long_rollout',**cancelled(runtime,q))
        q=request(rows=262145,steps=65536,structures=16)
        save('cancel_during_maximum_steps',**cancelled(runtime,q))
        small=request(rows=7);record=runtime.run(small);assert record['report']['status']=='complete'
        save('reuse_after_cancel',**record)
        # Same runtime, tiny job then a substantial bank: no poisoned rate.
        record=runtime.run(request());assert record['report']['status']=='complete'
        assert record['report']['counts']['completed_chunks']==2
        save('large_after_tiny_job',**record)
    finally:runtime.close()
    result['passed']=True;out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
