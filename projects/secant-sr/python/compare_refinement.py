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
"""Paired native-toggle refinement trials on prepared SRBench data.

Each worker receives whole dataset/seed groups: no cross-GPU policy comparison.
Truth formulas are never passed to the search. Uses existing validated scoring,
replay and subprocess deadline handling from the SRBench adapter.
"""
import argparse
import fcntl
import json
import platform
from pathlib import Path
import subprocess
import time

from run_srbench_toggle import DEFAULTS, atomic_json, execute, load_manifest, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=3)
    p.add_argument('--seconds', type=int, default=20)
    p.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 4])
    a = p.parse_args()
    if a.seconds <= 0 or not 0 <= a.shard < a.shards or a.gpu < 0:
        p.error('invalid budget, shard or GPU')
    if len(set(a.rounds)) != len(a.rounds) or any(n < 0 or n > 1000 for n in a.rounds):
        p.error('round counts must be distinct and in [0, 1000]')
    manifest = load_manifest(a.manifest)
    groups = [(i, j) for i, j in enumerate(manifest['jobs']) if i % a.shards == a.shard]
    if not groups:
        p.error('empty shard')
    config = dict(DEFAULTS, generations=10000, seconds=a.seconds,
                  refine_budget=128, refine_scale=1)
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = dict(started_unix=time.time(), host=platform.node(), gpu=a.gpu,
                    shard=a.shard, shards=a.shards, config=config, rounds=a.rounds,
                    jobs=[j for _, j in groups], planned=len(groups)*len(a.rounds),
                    executable_sha256=sha256(a.executable), manifest_sha256=sha256(a.manifest),
                    runner_sha256=sha256(Path(__file__)),
                    adapter_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                                    ['run_srbench_toggle.py','secant_sr_ast.py','srbench_v2.py']},
                    limitations=['Numerical held-out R2 > 0.999; symbolic equivalence unassessed',
                        'One-generation warmup immediately before each group is excluded and retained',
                        f'{a.seconds}-second budget includes scorer setup; cooperative budget can overrun',
                        'Host Philox; one start per selected AST; 4 coefficient slots',
                        'Over-capacity fitting candidates are skipped and counted, still broadly scored',
                        'Parallel reduction ties may change GP paths; no bitwise repeatability claim',
                        'Different seeds and datasets are sharded; paired policies stay on the same GPU'])
        plan['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version',
                                                    '--format=csv,noheader'], text=True)
        atomic_json(a.output/'plan.json', plan)
        records = []
        def save():
            atomic_json(a.output/'results.json', records)
            summary = {}
            for rounds in a.rounds:
                rows = [r for r in records if r['rounds'] == rounds]
                summary[str(rounds)] = dict(finished=len(rows), planned=len(groups),
                    errors=sum(r['status'] not in ('completed', 'no_finite_model') for r in rows),
                    numerical_successes=sum(r.get('accuracy_solution') == 1 for r in rows),
                    strict_successes=sum(r.get('native_result', {}).get('solved') is True for r in rows),
                    process_seconds=sum(r['process_wall_seconds'] for r in rows),
                    configurations=str(sum(int(r.get('configurations', 0)) for r in rows)),
                    refinement_configurations=str(sum(int(r.get('native_result', {}).get('refinement', {}).get('configurations', 0)) for r in rows)))
            atomic_json(a.output/'progress.json', dict(arms=summary, updated_unix=time.time(),
                                                     finished=len(records), planned=plan['planned']))
        save()
        for index, job in groups:
            print(f"warmup {job['id']} gpu={a.gpu}", flush=True)
            warm = execute(job, a.manifest.parent, a.executable,
                dict(config, generations=1, seconds=120, refine_rounds=0), 'cuda', a.gpu,
                a.output/'warmups', 180)
            if warm['status'] not in ('completed', 'no_finite_model'):
                atomic_json(a.output/'failure.json', dict(stage='warmup', result=warm))
                raise RuntimeError('warmup failed; shard paused with logs retained')
            # Rotate within each shard as well as across the complete panel.
            order = a.rounds[:]
            shift = (index + index//a.shards) % len(order)
            order = order[shift:] + order[:shift]
            for rounds in order:
                print(f"start {job['id']} rounds={rounds}", flush=True)
                result = execute(job, a.manifest.parent, a.executable,
                    dict(config, refine_rounds=rounds), 'cuda', a.gpu,
                    a.output/f'rounds-{rounds}', max(180, 2*a.seconds+120))
                result['rounds'] = rounds
                records.append(result)
                save()
                print(f"finish {len(records)}/{plan['planned']} {job['id']} rounds={rounds} "
                      f"status={result['status']} R2={result.get('validation_r2')} "
                      f"seconds={result['process_wall_seconds']:.3f}", flush=True)
                if result['status'] not in ('completed', 'no_finite_model'):
                    atomic_json(a.output/'failure.json', dict(stage='search', result=result))
                    raise RuntimeError('trial failed; shard paused without discarding failure')
        plan['finished_unix'] = time.time()
        atomic_json(a.output/'plan.json', plan)


if __name__ == '__main__':
    main()
