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
"""Matched GP/refinement pilot plus native-pipeline controls; one worker per GPU."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from run_srbench_toggle import DEFAULTS, atomic_json, execute, load_manifest, sha256

ARMS = [(engine, fit, 8 if engine == 'native' else 2)
        for engine in ['settings', 'toggles', 'native'] for fit in [0, 1]]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ['manifest', 'executable', 'legacy', 'output']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=3)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.repeats < 1 or not 0 < a.seconds <= 600:
        p.error('invalid worker/budget')
    for field in ['manifest', 'executable', 'legacy', 'output']:
        setattr(a, field, getattr(a, field).resolve())
    manifest = load_manifest(a.manifest)
    cases = [(i,j) for i,j in enumerate(manifest['jobs']) if i % a.shards == a.shard]
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_campaign(a, manifest, cases)

def run_campaign(a, manifest, cases):
    config = dict(DEFAULTS, population=8192, seconds=a.seconds, generations=10000,
                  refine_budget=128, kernels=16)
    identity = dict(manifest=sha256(a.manifest), executable=sha256(a.executable),
                    legacy=sha256(a.legacy), runner=sha256(Path(__file__)),
                    adapter=sha256(Path(__file__).resolve().parents[2]/'python/run_srbench_toggle.py'),
                    arms=ARMS, config=config, repeats=a.repeats, shard=a.shard,
                    shards=a.shards, gpu=a.gpu)
    # JSON normalization makes tuple/list identity identical on resume.
    identity = json.loads(json.dumps(identity))
    plan_path = a.output/'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if not a.resume or plan['identity'] != identity:
            raise ValueError('resume requires identical binaries, sources and configuration')
    else:
        if any(p.name != '.lock' for p in a.output.iterdir()):
            raise ValueError('nonempty output without plan')
        plan = dict(identity=identity, started_unix=time.time(), host=platform.node(),
                    status='running', planned=len(cases)*len(ARMS)*a.repeats,
                    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True),
                    limitations=[
                        'Same current GP and shared Philox proposal/acceptance policy; not a port of old GP or old random optimizer',
                        'Common scheduler processes serial modules, with independent CUDA graph kernels inside each module',
                        'Common pack 2; legacy slot templates adapt among 4,8,16,32 without changing candidates',
                        'Native pack 8 arms separately measure the production pipeline; do not conflate scheduler/packing with selectors',
                        '64 banks x 64 permutations; 4 coefficient slots; 128 selected fitting starts; 1 refinement round when enabled',
                        'Parameter-capacity skips retained; no claim every coefficient-bearing genome is fitted',
                        'Two official data/GP seeds and two identical-seed repeats; not full official SRBench',
                        'Floating-point GPU accumulation can change GP selection; matching seeds do not guarantee identical search prefixes',
                        'Held-out R2 > .999 is numerical recovery, not symbolic equivalence',
                        'Warmups retained separately; per-fit setup and generation-boundary overruns included'])
        atomic_json(plan_path, plan)
    plan['status'] = 'running'
    records = []
    done = set()
    for path in sorted(a.output.glob('rep*/trials/*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] not in ['completed','no_finite_model']:
            raise RuntimeError('retained failure requires investigation; no automatic retries')
        name = path.parents[2].name
        r['arm'] = name
        records.append(r)
        done.add((r['id'],name))
    def save():
        atomic_json(a.output/'results.json',records)
        groups=collections.defaultdict(list)
        for r in records:
            groups[r['arm'].split('-',1)[1]].append(r)
        atomic_json(a.output/'progress.json',dict(status=plan['status'],finished=len(records),planned=plan['planned'],
            updated_unix=time.time(),arms={k:dict(finished=len(rs),successes=sum(r.get('accuracy_solution')==1 for r in rs),
                errors=sum(r['status'] not in ['completed','no_finite_model'] for r in rs),
                seconds=sum(r.get('process_wall_seconds',0) for r in rs),
                configurations=str(sum(int(r.get('total_configurations',0)) for r in rs))) for k,rs in groups.items()}))
    def run(job, engine, fit, pack, folder, warm=False):
        os.environ['SECANT_BENCH_MODE'] = engine
        os.environ['SECANT_BENCH_LEGACY'] = str(a.legacy)
        for name in ['SECANT_BENCH_CAPTURE','SECANT_BENCH_GRID','SECANT_BENCH_SAMPLES']:
            os.environ.pop(name,None)
        dest=folder/'trials'/job['id']
        dest.mkdir(parents=True,exist_ok=True)
        os.environ['SECANT_BENCH_STATS'] = str(dest/'engine-stats.jsonl')
        cfg=dict(config,pack=pack,refine_rounds=fit)
        if warm:
            cfg.update(population=256,generations=1,seconds=180)
        r=execute(job,a.manifest.parent,a.executable,cfg,'cuda',a.gpu,folder,300)
        r.update(comparison_engine=engine,comparison_pack=pack,comparison_refine_rounds=fit)
        atomic_json(dest/'result.json',r)
        return r
    save()
    try:
        for index, job in cases:
            for engine in ['settings','toggles','native']:
                dest=a.output/'warmups'/engine
                path=dest/'trials'/job['id']/'result.json'
                if not path.exists():
                    r=run(job,engine,1,8 if engine=='native' else 2,dest,True)
                else:
                    r=json.loads(path.read_text())
                if r['status'] not in ['completed','no_finite_model']:
                    raise RuntimeError('warmup failed; inspect retained result')
            for repeat in range(a.repeats):
                order=ARMS.copy()
                shift=(index+repeat*3)%len(order)
                order=order[shift:]+order[:shift]
                for engine,fit,pack in order:
                    arm=f'rep{repeat}-{engine}-fit{fit}'
                    if (job['id'],arm) in done:
                        continue
                    print(f"start {job['id']} {arm}",flush=True)
                    r=run(job,engine,fit,pack,a.output/arm)
                    r['arm']=arm
                    records.append(r)
                    done.add((job['id'],arm))
                    save()
                    print(f"finish {len(records)}/{plan['planned']} {job['id']} {arm} {r['status']} R2={r.get('validation_r2')} time={r.get('process_wall_seconds')}",flush=True)
                    if r['status'] not in ['completed','no_finite_model']:
                        raise RuntimeError('fit failed; shard stopped with evidence retained')
        plan['status']='complete'
    except BaseException:
        plan['status']='failed'
        raise
    finally:
        plan['finished_unix']=time.time()
        atomic_json(plan_path,plan)
        save()

if __name__=='__main__':
    main()
