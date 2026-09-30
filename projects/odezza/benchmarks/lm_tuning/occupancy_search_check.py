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
"""Paired public-case search replays to validate population-control integration.

Retrospective diagnostics only; never report these as fresh blind recoveries.
"""
import argparse
from pathlib import Path
import time
from .common import load,save,append,REPO,digest
from .run_trial import Engine,search


def run(plan,output,device):
    output.mkdir(parents=True,exist_ok=False)
    plan=load(plan);save(output/'plan.json',plan)
    engine=Engine(REPO/'scratch/fitting_batch_trial/bin/odezza-fit-core-run',[device])
    state=dict(status='running',started_at=time.time(),completed=0)
    try:
        for index,case in enumerate(plan['cases']):
            for mode in (plan['order'] if index%2==0 else list(reversed(plan['order']))):
                state['current']=case['id']+'-'+mode;save(output/'status.json',state)
                opts=dict(case['settings'],seconds=plan['seconds'],lanes_per_fit=1,shape_fallback=True)
                if mode=='population':opts.update(plan['population'])
                start=time.monotonic()
                result=search(case['public'],opts,'native_lm',output/state['current'],engine)
                result.update(case_id=case['id'],mode=mode,total_seconds=time.monotonic()-start,
                    public_sha256=digest(case['public']),options=opts)
                append(output/'results.jsonl',result)
                state['completed']+=1
                print(state['current'],result['status'],round(result['total_seconds'],3),flush=True)
                if result['status']=='error':raise RuntimeError(result.get('error','Search failed'))
        state['status']='complete'
    except Exception as error:
        state.update(status='error',error=str(error));raise
    finally:
        state['finished_at']=time.time();save(output/'status.json',state);engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();run(a.plan,a.output,a.device)
