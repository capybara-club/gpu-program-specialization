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
"""Native JSON execution contracts; requires built runtime and a CUDA device."""
import argparse
from copy import deepcopy
import ctypes as C
import json
import math
from pathlib import Path
import struct
import time
from unittest.mock import patch
from odezza.grammar.native_service import NativeService
from odezza.grammar.mcp import Transport
ROOT = Path(__file__).resolve().parents[2]


def problem():
    return dict(states=['x0', 'x1'], trajectories=[dict(times=[0., .03, .1, .23], initial=[1, 2],
        values=[[math.exp(-t), None] for t in [0., .03, .1, .23]])])


def grammar():
    return dict(version=1, states=['x0', 'x1'], integration=dict(method='rk4', dt=.01),
        rhs=dict(x0='-const.k*leaf.s', x1='0'), constants=dict(k=dict(values=[.5,1.,2.])),
        leaves=dict(s=dict(states=['x0','x1'], arity=2)),
        retain={'global': {'k':4, 'unit':'numeric_candidate'}, 'per_family':3})


def signature(report):
    return [(report['candidates'][i]['resolved_programs'], report['candidates'][i]['mse'])
            for i in report['leaderboards']['global']]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();service=NativeService(args.root, max_jobs=64);reports=[]
    def run(name,g,execution=None,spec=None,expected=None):
        submitted=service.submit(problem=spec or problem(),grammar=g,execution=execution or {})
        report=service.jobs[submitted['job_id']].result(timeout=120)
        assert report['status']=='complete',(name,report)
        if expected is not None: assert report['counts']['completed_configurations']==expected,(name,report['counts'])
        assert report['counts']['valid']+report['counts']['invalid']==report['counts']['completed_configurations']
        for identifier,candidate in report['candidates'].items():
            replay=service.replay(submitted['job_id'],identifier,cpu=True)
            cpu=replay['cpu_reference']['mse']
            assert abs(cpu-candidate['mse']) < 3e-5*max(1,abs(cpu)),(name,cpu,candidate)
            byindex=service.replay(submitted['job_id'],address=candidate['origin'])
            assert byindex['candidate']==replay['candidate']
        reports.append(dict(name=name,**report)); print(name,report['counts'],report['timing'],flush=True)
        return report
    try:
        # SQLite may be installed for the legacy reference, but the native path must never open it.
        with patch('sqlite3.connect', side_effect=AssertionError('native service opened a database')):
            basic=run('masked_irregular_toggle',grammar(),expected=6)
            assert min(x['mse'] for x in basic['candidates'].values())<1e-10
            tiled=run('toggle_chunk_invariance',grammar(),{'max_chunk_configurations':2},expected=6)
            assert signature(basic)==signature(tiled)
            product=json.loads((ROOT/'examples/grammar/constant_rng_product.json').read_text())
            a=run('mixed_rng_product',product,expected=384)
            b=run('rng_chunk_invariance',product,{'max_chunk_configurations':17},expected=384)
            assert signature(a)==signature(b)
            # Uniform values reconstructed using the separately tested core CPU Philox.
            core=C.CDLL(str(ROOT/'build/libodezza.so'))
            core.odezza_rng_uniform.argtypes=[C.c_uint64]*3+[C.POINTER(C.c_float)]
            f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
            for candidate in a['candidates'].values():
                for slot,value in zip(candidate['slots'],candidate['values']):
                    if slot['kind']!=2:continue
                    raw=C.c_float();assert core.odezza_rng_uniform(int(slot['seed']),int(slot['stream']),int(slot['axis_index']),C.byref(raw))==0
                    if slot['transform']==1: expected=f32(f32(f32(slot['scale'])*raw.value)+f32(slot['shift']))
                    elif slot['transform']==2: expected=f32(f32(f32(1-raw.value)*f32(slot['scale']))+f32(raw.value*f32(slot['shift'])))
                    else:expected=raw.value
                    assert value==f32(value) or abs(value-f32(value))<1e-8
                    assert abs(value-expected)<1e-7,(slot,value,expected)
            # Multiple unique numeric winners can lie beyond reducer top-16 duplicate rows.
            g=grammar();g.pop('leaves');g['rhs']['x0']='-const.k*x0';g['constants']['k']['values']=[1]*48+[.5]*20+[2]*20
            g['retain']={'global':{'k':3,'unit':'numeric_candidate'}}
            d=run('duplicate_rows_past_top16',g,expected=88)
            assert len(d['candidates'])==3
            # A late duplicate attaches its tag to an earlier accepted AST after its page was recycled.
            g=grammar();g.pop('leaves');g.pop('constants');g['rhs']['x0']='R'
            g['rules']={'R':['-x0','-2*x0',{'expr':'-x0','tags':['late']}]}
            g['retain']={'global':1,'by_tag':{'late':{'k':1,'unit':'numeric_candidate'}}}
            d=run('late_duplicate_tag',g,{'batch_variants':1},expected=2)
            assert d['leaderboards']['tags']['late']
            assert d['families'][0]['duplicates']==1
            # Metadata tags survive even when nobody requested a tag leaderboard.
            g['retain']={'global':2};g['families']=[{'id':'metadata','tags':['family-metadata']}]
            d=run('unrequested_tag_metadata',g,{'batch_variants':1},expected=2)
            best=d['candidates'][d['leaderboards']['global'][0]]
            assert {'late','family-metadata'} <= set(best['tags']),best
            # Round robin with reserved allocations and partial bank rows.
            g=grammar();g.pop('leaves');g['rhs']['x0']='-const.k*x0';g['constants']['k']['values']=list(range(1,41))
            g['families']=[{'id':'a','limits':{'max_skeletons':1,'max_variants':1,'max_configurations':13}},
                           {'id':'b','limits':{'max_skeletons':1,'max_variants':1,'max_configurations':17}}]
            d=run('family_allocations',g,{'max_chunk_configurations':4},expected=30)
            assert [f['completed_configurations'] for f in d['families']]==[13,17]
            combined=json.loads((ROOT/'examples/grammar/submit.json').read_text())
            d=run('fixed_prespecialization_growth',combined['grammar'],{'patch_capacity':2},combined['problem'],3)
            assert d['cache']['capacity_growths']>=1
            run('fixed_cached',combined['grammar'],{'patch_capacity':2},combined['problem'],3)
            # Failures are explicit and leave the persistent context usable for the next job.
            invalid=deepcopy(product);invalid['constants']['scale']={'values':[-1,1]}
            job=service.submit(problem=problem(),grammar=invalid)
            failure=service.jobs[job['job_id']].result();assert failure['status']=='failed' and 'std' in failure['error'],failure
            reports.append(dict(name='negative_std_rejected',**failure))
            job=service.submit(problem=problem(),grammar=grammar(),execution={'max_host_bytes':1024})
            failure=service.jobs[job['job_id']].result();assert failure['status']=='failed',failure
            reports.append(dict(name='memory_ceiling',**failure))
            # Cancellation of a queued job returns a terminal state without parsing it.
            gate=__import__('threading').Event();blocker=service.executor.submit(gate.wait)
            job=service.submit(problem=problem(),grammar=grammar());service.cancel(job['job_id']);gate.set();blocker.result()
            cancelled=service.jobs[job['job_id']].result();assert cancelled['status']=='cancelled',cancelled
            reports.append(dict(name='queued_cancel',**cancelled))
            # A huge duplicate bank must be interruptible during uniqueness reduction.
            g=grammar();g.pop('leaves');g['rhs']['x0']='-const.k*x0';g['constants']['k']['values']=[1]*8192
            g['retain']={'global':{'k':2,'unit':'numeric_candidate'}}
            job=service.submit(problem=problem(),grammar=g,execution={'max_seconds':.03,'dedup_bytes_per_family':1048576})
            timed=service.jobs[job['job_id']].result(timeout=10)
            assert timed['status']=='timeout',timed['status']
            reports.append(dict(name='bounded_retention_timeout',**timed))
            run('context_reuse_after_errors',grammar(),expected=6)
            transport=Transport(service)
            transport.dispatch(dict(jsonrpc='2.0',id=1,method='initialize'))
            transport.dispatch(dict(jsonrpc='2.0',method='notifications/initialized'))
            listed=transport.dispatch(dict(jsonrpc='2.0',id=2,method='tools/list'))
            assert not any(t['name']=='odezza_fit' for t in listed['result']['tools'])
            answer=transport.dispatch(dict(jsonrpc='2.0',id=3,method='tools/call',params=dict(name='odezza_submit',arguments=combined)))
            assert not answer['result']['isError'],answer
            job=answer['result']['structuredContent'];assert service.jobs[job['job_id']].result()['status']=='complete'
    finally:
        service.close();Path(args.output).write_text(json.dumps(reports,indent=2)+'\n')
    print('PASS',len(reports),'native runtime checks',flush=True)

if __name__=='__main__':main()
