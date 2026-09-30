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
"""Matched static-population execution benchmark; synthetic data, not recovery."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import struct
from odezza.grammar.native_service import NativeService
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--legacy',action='store_true');a=p.parse_args()
    # 8^3 = 512 different full-vector ASTs; 2048 values each => 1,048,576 configs.
    g=dict(version=1,states=['x0','x1','x2'],integration=dict(method='rk4',dt=.01),
           constants={'k':{'values':[.25+i/2048 for i in range(2048)]}},
           rules={'R(q)':['q','sin(q)','tanh(q)','q*q','cos(q)','q/(1+q*q)','sin(q*q)','tanh(q*q)']},
           rhs={f'x{i}':f'-const.k*hole(R,x{i})' for i in range(3)},
           expansion={'strategy':'enumerate'},limits={'max_skeletons':512,'max_variants':512,'max_configurations':1048576},
           retain={'global':{'k':20,'unit':'resolved_structure'},'per_family':{'k':8,'unit':'resolved_structure'}})
    problem={'states':g['states'],'trajectories':[dict(times=[0,.02,.04,.07,.1],initial=[.2,.25,.3],
             values=[[x*math.exp(-.5*t) for x in [.2,.25,.3]] for t in [0,.02,.04,.07,.1]])]}
    out=[];service=NativeService(a.root)
    try:
        for batch,chunk in [(256,1048576),(256,1048576),(256,1048576),(512,1048576),(128,1048576)]:
            start=time.perf_counter();job=service.submit(problem=problem,grammar=g,execution={'batch_variants':batch,'max_chunk_configurations':chunk,'module_systems':32,'patch_capacity':384})
            acknowledgement=time.perf_counter()-start
            report=service.jobs[job['job_id']].result(timeout=180)
            assert report['status']=='complete',(report['status'],report['error'],report['counts'])
            assert report['counts']['completed_configurations']==1048576,report['counts']
            out.append(dict(case='grid_1m',batch=batch,acknowledgement_seconds=acknowledgement,end_to_end_seconds=time.perf_counter()-start,report=report))
            print(json.dumps({k:v for k,v in out[-1].items() if k!='report'}),report['timing'],flush=True)
        # Same grammar, more coefficient rows: reuse all 512 structures, count full work.
        larger=deepcopy(g);larger['constants']['k']['values']=[.25+i/16384 for i in range(16384)];larger['limits']['max_configurations']=8388608
        start=time.perf_counter();job=service.submit(problem=problem,grammar=larger,execution={'batch_variants':512,'module_systems':32})
        report=service.jobs[job['job_id']].result(timeout=180);assert report['status']=='complete',(report['status'],report['error'],report['counts'])
        out.append(dict(case='grid_8m',end_to_end_seconds=time.perf_counter()-start,report=report));print('grid_8m',report['timing'],flush=True)
        # Existing handoff population exercises 8192 ASTs, six Philox streams and all its tags.
        import importlib.util
        spec=importlib.util.spec_from_file_location('fixtures',ROOT/'tests/grammar/gpu_examples.py');fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
        supplied=json.loads((ROOT/'examples/grammar/adaptation_search.json').read_text());data=fixtures.fixture(supplied)
        start=time.perf_counter();job=service.submit(problem=data,grammar=supplied,execution={'module_systems':32,'patch_capacity':1024})
        report=service.jobs[job['job_id']].result(timeout=180);assert report['status']=='complete',(report['status'],report['error'],report['counts'])
        out.append(dict(case='supplied_adaptation_rng',end_to_end_seconds=time.perf_counter()-start,report=report));print('adaptation',report['counts'],report['timing'],flush=True)
        for entry in out:
            report=entry['report'];identifier=report['leaderboards']['global'][0]
            replay=service.replay(report['job_id'],identifier,cpu=True);cpu=replay['cpu_reference']['mse']
            assert abs(cpu-report['candidates'][identifier]['mse'])<3e-5
            entry['best_cpu_mse']=cpu
    finally:service.close();Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    if a.legacy:
        from odezza.grammar.service import Service
        legacy=Service(a.root+'-legacy',library=str(ROOT/'build/libodezza.so'))
        try:
            for repetition in range(3):
                start=time.perf_counter();job=legacy.submit(problem=problem,grammar=g,execution={'module_systems':32,'patch_capacity':384})
                report=legacy.jobs[job['job_id']].result(timeout=300)
                assert report['status']=='complete',(report['status'],report['error'],report['counts'])
                native=out[0]['report'];np=native['candidates'][native['leaderboards']['global'][0]];lp=report['winners']['global'][0]
                assert np['resolved_programs']==lp['resolved_programs'] and struct.pack('f',np['mse'])==struct.pack('f',lp['mse']),(np,lp)
                nm={tuple(v['resolved_programs']):struct.pack('f',v['mse']) for v in native['candidates'].values()}
                lm={tuple(v['resolved_programs']):struct.pack('f',v['mse']) for v in report['winners']['global']}
                assert nm==lm, 'Matched-grid leaderboards differ'
                out.append(dict(case='legacy_matched_grid_1m',repetition=repetition,end_to_end_seconds=time.perf_counter()-start,report=report))
                print('legacy',out[-1]['end_to_end_seconds'],report['timing'],flush=True)
        finally:legacy.close();Path(a.output).write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
