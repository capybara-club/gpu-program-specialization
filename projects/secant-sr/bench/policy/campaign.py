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
"""Paired native-pipeline search-policy trial, with frozen inputs and fail-fast audits."""
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

ARMS = {
    'baseline': dict(refine_rounds=0, refine_parameters='all', power_mutation_probability=0),
    'all-fit': dict(refine_rounds=1, refine_parameters='all', power_mutation_probability=0),
    'block-fit': dict(refine_rounds=1, refine_parameters='active-block', power_mutation_probability=0),
    'power': dict(refine_rounds=0, refine_parameters='all', power_mutation_probability=.25),
    'block-power': dict(refine_rounds=1, refine_parameters='active-block', power_mutation_probability=.25),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['manifest', 'executable', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=3)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 0 <= a.shard < a.shards or a.repeats < 1 or not 0 < a.seconds <= 600:
        p.error('invalid worker/budget')
    for key in ['manifest', 'executable', 'output']:
        setattr(a, key, getattr(a, key).resolve())
    manifest = load_manifest(a.manifest)
    cases = [(i, j) for i, j in enumerate(manifest['jobs']) if i % a.shards == a.shard]
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_campaign(a, cases)


def run_campaign(a, cases):
    config = dict(DEFAULTS, population=8192, seconds=a.seconds, generations=10000,
                  refine_budget=128, kernels=16, pack=8)
    identity = dict(manifest=sha256(a.manifest), executable=sha256(a.executable),
                    runner=sha256(Path(__file__)),
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
                        'Outcome-selected diagnostic panel, not an unbiased full SRBench estimate',
                        'All arms use the same new binary and native pack-8 scorer; kernel sources unchanged',
                        'Active-block fits only a bounded subset of the incumbent active parameters each round',
                        'Fitting still selects up to 128 top-scoring eligible genomes, not a diversity archive',
                        'Fixed literal vocabulary remains fixed; no LM or CPU polishing in these search arms',
                        'Power mutations copy existing subtrees, retain their toggle bits and parameters, and use ordinary multiplication',
                        'Two official seeds and two repeats; matching seeds do not imply identical search prefixes',
                        'Numerical success is held-out R2 > .999, not symbolic equivalence',
                        '60-second cooperative budget includes setup; generation-boundary overruns and teardown are recorded'])
        atomic_json(plan_path, plan)
    plan['status'] = 'running'
    records, done = [], set()
    for path in sorted(a.output.glob('rep*/trials/*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] not in ['completed', 'no_finite_model']:
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
        cfg = dict(config, **ARMS[name])
        if warm:
            cfg.update(population=256, generations=1, seconds=180)
        r = execute(job, a.manifest.parent, a.executable, cfg, 'cuda', a.gpu, folder, 300)
        r.update(comparison_policy=name)
        atomic_json(dest/'result.json', r)
        return r

    save()
    try:
        for index, job in cases:
            path = a.output/'warmups'/'trials'/job['id']/'result.json'
            r = json.loads(path.read_text()) if path.exists() else run(job, 'block-power', a.output/'warmups', True)
            if r['status'] not in ['completed', 'no_finite_model']:
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
