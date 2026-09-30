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
"""Real-service work bounds including initial-only trajectory reset overhead."""
import argparse
import copy
import json
from pathlib import Path
import time
from client import Client
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=15223);p.add_argument('--output',type=Path,default=ROOT/'service_trial/validation/work-gpu-final.json');a=p.parse_args()
c=Client(port=a.port);records=[];h=None
try:
    for initial_only in (False,True):
        bank=524288
        trajectories=[dict(initial=[0],times=[0,1],values=[[0],[0]])]
        if initial_only:trajectories=[dict(initial=[0],times=[0],values=[[0]]) for _ in range(8190)]+trajectories
        request=dict(problem=dict(states=['x'],trajectories=trajectories),grammar=dict(version=1,states=['x'],rhs={'x':'-rng.c*x'},rng_banks={'u':{'base':'uniform01','count':bank,'seed':713}},rng={'c':{'bank':'u'}},integration=dict(method='rk4',dt=1 if initial_only else 1/4096),retain={'global':{'k':1,'unit':'numeric_candidate'}}),execution=dict(max_chunk_configurations=4294967296,max_seconds=120))
        start=time.monotonic();h=c.submit(json.dumps(request).encode())['handle']
        while c.status(h)['state']!='terminal':
            assert time.monotonic()-start<90;time.sleep(.02)
        r=c.result(h);assert r['status']=='complete',r.get('error')
        assert r['counts']['completed_configurations']==bank,r['counts']
        units=max(r['integration']['steps_per_configuration'],sum(x['points'] for x in r['trajectories']))
        for d in r['execution']['devices']:
            assert d['largest_tile_configurations']<=d['sizing']['work_ceiling_configurations'],d
        assert r['counts']['completed_chunks']>=min((bank+d['sizing']['work_ceiling_configurations']-1)//d['sizing']['work_ceiling_configurations'] for d in r['execution']['devices'])
        assert next(iter(r['candidates'].values()))['mse']==0
        records.append(dict(check='point_visits' if initial_only else 'rk4_steps',seconds=time.monotonic()-start,report=r))
        c.rpc('release.'+h);h=None
    a.output.write_text(json.dumps(dict(passed=True,records=records),indent=2)+'\n');print(json.dumps(dict(passed=True,seconds=[x['seconds'] for x in records])))
finally:
    if h:
        try:c.rpc('cancel.'+h)
        except Exception:pass
    c.close()
