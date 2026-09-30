#!/usr/bin/env python3
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
"""Long-budget toggle GP with in-generation GPU LM; no CPU polishing/fallback."""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from run_srbench_toggle import DEFAULTS, atomic_json, execute, load_manifest, sha256

def run(a):
    manifest=load_manifest(a.manifest)
    cases=[j for i,j in enumerate(manifest['jobs']) if i%a.shards==a.shard and (not a.problems or j['problem'] in a.problems)]
    if not cases:
        raise ValueError('worker has no cases')
    for key in list(os.environ):
        if key.startswith('SECANT_BENCH_'): os.environ.pop(key)
    config=dict(DEFAULTS,population=8192,generations=100000,seconds=a.seconds,refine_rounds=0,
                lm_iterations=a.iterations,lm_budget=a.budget,lm_bindings=a.bindings,lm_starts=a.starts,
                lm_parameters=8,lm_interval=1,lm_threads=64,lm_scale=1,finalists=16)
    identity=dict(manifest_sha256=sha256(a.manifest),executable_sha256=sha256(a.executable),
                  runner_sha256=sha256(Path(__file__)),adapter_sha256=sha256(Path(__file__).resolve().parents[2]/'python/run_srbench_toggle.py'),
                  config=config,repeats=a.repeats,shard=a.shard,shards=a.shards,gpu=a.gpu,
                  problems=a.problems,python=platform.python_version())
    plan_path=a.output/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if not a.resume or plan['identity']!=identity: raise ValueError('resume requires identical code, data and configuration')
    else:
        if any(p.name!='.lock' for p in a.output.iterdir()): raise ValueError('nonempty output without plan')
        plan=dict(identity=identity,host=platform.node(),started_unix=time.time(),planned=len(cases)*a.repeats,
                  hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True),
                  limitations=[
                    'Diagnostic subset of SRBench, not a full-suite success estimate',
                    'Register-only CUDA evaluator computes LM Jacobians/statistics and damped solves; no SASS LM specialization',
                    'All fitting uses full training rows only; holdout assessed after selection',
                    'At most 32 used-bit bindings per genome by default, always includes incumbent',
                    'Up to eight active parameters jointly; wider bindings rotate blocks, with counts reported',
                    'Every proposal rescored with ordinary native-SASS scorer before population acceptance',
                    'LM row/Jacobian evaluations are separate from broad-search configurations',
                    'Wall-time limit is cooperative, includes setup, checked between LM iterations and generations',
                    'Incumbent AST checkpoints support later budget analysis; no holdout-driven early stopping',
                    'Two repeats use identical search seeds; time-limited generation counts can differ',
                    'Numerical success is held-out R2>.999; no symbolic equivalence claim'])
    records=[]
    done=set()
    for path in sorted(a.output.glob('rep*/trials/*/result.json')):
        r=json.loads(path.read_text())
        if r['status'] not in {'completed','no_finite_model'}: raise RuntimeError('retained failure needs investigation; no automatic retry')
        r['arm']=path.parents[2].name
        records.append(r);done.add((r['arm'],r['id']))
    plan['status']='running'
    def save():
        atomic_json(plan_path,plan)
        atomic_json(a.output/'results.json',records)
        groups=collections.defaultdict(list)
        for r in records: groups['gpu-lm'].append(r)
        atomic_json(a.output/'progress.json',dict(status=plan['status'],finished=len(records),planned=plan['planned'],updated_unix=time.time(),
            arms={key:dict(finished=len(rs),successes=sum(r.get('accuracy_solution')==1 for r in rs),
                errors=sum(r['status'] not in {'completed','no_finite_model'} for r in rs),
                seconds=sum(r.get('process_wall_seconds',0) for r in rs),
                configurations=str(sum(int(r.get('total_configurations',0)) for r in rs)),
                lm_row_evaluations=str(sum(int(r.get('lm',{}).get('row_evaluations',0)) for r in rs)),
                lm_seconds=sum(r.get('lm',{}).get('seconds',0) for r in rs)) for key,rs in groups.items()}))
    save()
    try:
        # Alternate repeats by problem; each case receives its full declared budget.
        for job in cases:
            for rep in range(a.repeats):
                arm=f'rep{rep}-gpu-lm'
                if (arm,job['id']) in done: continue
                plan['active']=dict(id=job['id'],arm=arm,started_unix=time.time());save()
                print(f"start {arm} {job['id']}",flush=True)
                r=execute(job,a.manifest.parent,a.executable,config,'cuda',a.gpu,a.output/arm,a.seconds+180)
                r['arm']=arm;r['budget_overrun_seconds']=max(0,r.get('process_wall_seconds',0)-a.seconds)
                atomic_json(a.output/arm/'trials'/job['id']/'result.json',r)
                records.append(r);save()
                print(f"finish {len(records)}/{plan['planned']} {job['id']} status={r['status']} r2={r.get('validation_r2')} seconds={r.get('process_wall_seconds')} LM={r.get('lm',{}).get('seconds')}",flush=True)
                if r['status'] not in {'completed','no_finite_model'}: raise RuntimeError(r.get('error','fit failed'))
        plan['status']='complete'
    except BaseException:
        plan['status']='failed'
        raise
    finally:
        plan.pop('active',None);plan['finished_unix']=time.time();save()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','executable','output']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=2);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seconds',type=float,default=900)
    p.add_argument('--iterations',type=int,default=4);p.add_argument('--budget',type=int,default=512)
    p.add_argument('--bindings',type=int,default=32);p.add_argument('--starts',type=int,default=4)
    p.add_argument('--problems',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 0<=a.shard<a.shards or not 1<=a.repeats<=10 or not 0<a.seconds<=3600 or not 1<=a.iterations<=1000 or not 1<=a.budget<=8192 or not 1<=a.bindings<=64 or not 1<=a.starts<=128:
        p.error('invalid campaign budget or worker')
    for key in ['manifest','executable','output']: setattr(a,key,getattr(a,key).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)

if __name__=='__main__': main()
