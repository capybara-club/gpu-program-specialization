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
"""Matched fresh-seed GP versus budgeted final-polishing experiment."""
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
import numpy as np
import scipy
from final_polish import polish

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from run_srbench_toggle import DEFAULTS, atomic_json, execute, load_manifest, sha256

ARMS = {
    'baseline': dict(power_mutation_probability=0, polish=False),
    'power': dict(power_mutation_probability=.25, polish=False),
    'polish': dict(power_mutation_probability=0, polish=True),
    'power-polish': dict(power_mutation_probability=.25, polish=True),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['manifest', 'executable', 'replay', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.repeats < 1 or not 0 < a.seconds <= 600:
        p.error('invalid worker/budget')
    for key in ['manifest', 'executable', 'replay', 'output']:
        setattr(a, key, getattr(a, key).resolve())
    manifest = load_manifest(a.manifest)
    cases = [(i, j) for i, j in enumerate(manifest['jobs']) if i % a.shards == a.shard]
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_campaign(a, cases)


def run_campaign(a, cases):
    config = dict(DEFAULTS, population=8192, seconds=a.seconds, generations=10000,
                  refine_rounds=0, kernels=16, pack=8, finalists=16)
    identity = dict(manifest=sha256(a.manifest), executable=sha256(a.executable),
                    runner=sha256(Path(__file__)), replay=sha256(a.replay),
                    polish=sha256(Path(__file__).with_name('final_polish.py')),
                    optimizer=sha256(Path(__file__).resolve().parents[1]/'polish/polish.py'),
                    numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
                    adapter=sha256(Path(__file__).resolve().parents[2]/'python/run_srbench_toggle.py'),
                    arms=ARMS, config=config, repeats=a.repeats, shard=a.shard,
                    shards=a.shards, gpu=a.gpu)
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
                        '21 diagnostic problems selected previously; two fresh official seeds, not a full-suite estimate',
                        'All arms retain the same 16-entry passive archive and use the same CUDA pipeline',
                        '60-second total cooperative budget: search-only 60s; polish arms search 50s plus at most 10s within remaining time',
                        'CPU SciPy LM polishes tied adjustable coefficients; literals fixed, no feedback into GP',
                        'Archive diversity uses ordered structure, state binding and parameter-sharing identity, not algebraic equivalence',
                        'Selection and LM use training rows only; the final holdout is reported after selection',
                        'Two identical-seed repeats measure nondeterministic search outcomes; no common search-prefix claim',
                        'GPU configurations count search only; CPU Jacobian evaluations are separate work',
                        'Numerical success is held-out R2 > .999, not symbolic equivalence',
                        'Measured elapsed includes search, archive emission, CPU polishing and final native replay; overruns reported'])
        atomic_json(plan_path, plan)
    plan['status'] = 'running'
    records, done = [], set()
    for path in sorted(a.output.glob('rep*/trials/*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] not in ['completed', 'no_finite_model'] or not r.get('pipeline_completed'):
            raise RuntimeError('retained failure requires investigation; no automatic retries')
        r['arm'] = path.parents[2].name
        records.append(r)
        done.add((r['id'], r['arm']))

    def save():
        atomic_json(a.output/'results.json', records)
        groups = collections.defaultdict(list)
        for r in records:
            groups[r['arm'].split('-', 1)[1]].append(r)
        atomic_json(a.output/'progress.json', dict(status=plan['status'], finished=len(records), planned=plan['planned'],
            updated_unix=time.time(), arms={k:dict(finished=len(rs), successes=sum(r.get('accuracy_solution')==1 for r in rs),
                errors=sum(r['status'] not in ['completed','no_finite_model'] for r in rs),
                seconds=sum(r.get('process_wall_seconds',0) for r in rs),
                configurations=str(sum(int(r.get('total_configurations',0)) for r in rs))) for k,rs in groups.items()}))

    def run(job, name, folder, warm=False):
        # Avoid accidentally activating any previously used instrumentation.
        for key in list(os.environ):
            if key.startswith('SECANT_BENCH_'):
                os.environ.pop(key)
        dest = folder/'trials'/job['id']
        dest.mkdir(parents=True, exist_ok=True)
        policy = dict(ARMS[name]); do_polish = policy.pop('polish')
        cfg = dict(config, **policy)
        if do_polish:
            cfg['seconds'] = max(.1, a.seconds-10)
        if warm:
            cfg.update(population=256, generations=2, seconds=1)
        started=time.monotonic()
        deadline=started+(5 if warm else a.seconds)
        r=execute(job,a.manifest.parent,a.executable,cfg,'cuda',a.gpu,folder,300)
        try:
            if r['status'] not in ['completed','no_finite_model']:
                raise RuntimeError(r.get('error','search failed'))
            search_wall=r['process_wall_seconds']
            if do_polish and r['status']=='completed':
                polish_deadline=min(deadline,time.monotonic()+10)
                r=polish(r,job,a.manifest.parent/job['path'],a.replay,polish_deadline)
            else:
                r['final_polish']=dict(seconds=0,skipped='disabled',accepted=False)
                if r['status']=='completed':
                    r['strict_success']=r['native_result']['solved']
                    r['final_model']=dict(r['native_result'],stage='search')
                    r['resolved_ast_hex']=r['native_result']['resolved_ast_hex']
            r['search_process_wall_seconds']=search_wall
            r['process_wall_seconds']=time.monotonic()-started
            r['elapsed_seconds']=r['process_wall_seconds']
            r['budget_overrun_seconds']=max(0,time.monotonic()-deadline)
            r['pipeline_completed']=True
        except Exception as error:
            r['status']='failed';r['pipeline_completed']=False;r['error']=f'{type(error).__name__}: {error}'
        r.update(comparison_policy=name)
        atomic_json(dest/'result.json', r)
        return r

    save()
    try:
        for index, job in cases:
            path = a.output/'warmups'/'trials'/job['id']/'result.json'
            r = json.loads(path.read_text()) if path.exists() else run(job, 'power-polish', a.output/'warmups', True)
            if r['status'] not in ['completed', 'no_finite_model'] or not r.get('pipeline_completed'):
                raise RuntimeError('warmup failed; evidence retained')
            for repeat in range(a.repeats):
                order = list(ARMS)
                shift = (index+repeat*2) % len(order)
                for name in order[shift:]+order[:shift]:
                    arm = f'rep{repeat}-{name}'
                    if (job['id'], arm) in done:
                        continue
                    print(f'start {job["id"]} {arm}', flush=True)
                    r = run(job, name, a.output/arm)
                    r['arm'] = arm
                    records.append(r)
                    done.add((job['id'], arm))
                    save()
                    print(f'finish {len(records)}/{plan["planned"]} {job["id"]} {arm} {r["status"]} R2={r.get("validation_r2")} time={r.get("process_wall_seconds")}', flush=True)
                    if r['status'] not in ['completed', 'no_finite_model']:
                        raise RuntimeError('fit failed; shard stopped with evidence retained')
        plan['status'] = 'complete'
    except BaseException:
        plan['status'] = 'failed'
        raise
    finally:
        plan['finished_unix'] = time.time()
        atomic_json(plan_path, plan)
        save()


if __name__ == '__main__':
    main()
