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
"""Matched complete JSON requests; run from the repository build being measured."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from odezza.grammar.native_service import NativeService

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--label',required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--automatic',action='store_true')
    p.add_argument('--devices',default='0,1');p.add_argument('--raw',action='store_true');p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
    root=Path.cwd();g=json.loads((root/'examples/grammar/million.json').read_text())
    g['constants']={'c':{'values':[.25+i/2048 for i in range(2048)]}}
    g['rhs']['x0']='const.c*('+g['rhs']['x0']+')'
    g['limits']['max_configurations']=2048000000;g['limits']['max_seconds']=120
    if a.raw:g['retain']={'global':{'k':16,'unit':'evaluation_row'},'per_family':{'k':16,'unit':'evaluation_row'}}
    problem={'states':g['states'],'trajectories':[{'initial':[.1]*6,'times':[0,.01,.03],
               'values':[[.1*math.exp(t)]*6 for t in [0,.01,.03]]}]}
    execution={'batch_variants':1024,'module_systems':32,'dedup_bytes_per_family':134217728,
               'max_chunk_configurations':2097152,'max_seconds':120}
    if a.profile:execution['profile_timing']=True
    if a.automatic:
        for key in ('batch_variants','module_systems','max_chunk_configurations'):execution.pop(key)
    result={'label':a.label,'request':{'problem':problem,'grammar':g,'execution':execution},'devices':a.devices,
            'binaries':{x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in ['build/libodezza.so','build/runtime/libodezza_runtime.so']},'runs':[]}
    s=NativeService('/tmp/odezza-cub-request-bench-cache',devices=list(map(int,a.devices.split(','))))
    try:
        for i in range(a.repeats):
            t=time.perf_counter();j=s.submit(problem=problem,grammar=g,execution=execution);r=s.jobs[j['job_id']].result(timeout=150);elapsed=time.perf_counter()-t
            assert r['status']=='complete' and r['retention_complete'],r
            assert r['counts']['completed_configurations']==2048000000
            assert sum(f['generated_asts'] for f in r['families'])==1000000
            assert r['cache']['retry_requested_configurations']==0
            replays=[]
            for identifier in r['leaderboards']['global'][:3]:
                replay=s.replay(j['job_id'],identifier,cpu=True);cpu=replay['cpu_reference']['mse'];gpu=r['candidates'][identifier]['mse']
                assert abs(cpu-gpu)<3e-10+3e-5*abs(cpu),(cpu,gpu)
                replays.append(replay)
            result['runs'].append({'warm':i>0,'end_to_end_seconds':elapsed,'report':r,'replays':replays})
            Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
            print(a.label,i,'wall',elapsed,'native',r['timing']['total_seconds'],'reduction',r['timing']['reduction_gather_seconds'],flush=True)
            s.release(job_id=j['job_id'])
    finally:s.close()
if __name__=='__main__':main()
