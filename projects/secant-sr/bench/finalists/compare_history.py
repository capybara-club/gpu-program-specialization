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
"""Compare retained finalist trials with hash-verified historical SRBench CSVs.

No searches are run. Historical timing scopes and hardware remain distinct.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from secant_sr_ast import Expression


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, audit):
    manifest = read(root / 'manifest.json')
    jobs = {(j['problem'], j['seed']): j for j in manifest['jobs']}
    if len(jobs) != 42:
        raise ValueError('expected 42 distinct diagnostic cases')
    archive = {Path(r['name']).name: r for r in read(audit / 'rohini-manifest.json')['records']}
    historical = {}
    specs = [
        ('lm100', 'srbench_v2_lm32b4s_interval4_full1160_rohini_sm120_20260814.csv', '100 generations; LM every fourth generation'),
        ('lm600', 'srbench_v2_lm32b4s_i4_pop8192_gen600_full1160_rohini_sm120_20260814.csv', '600 generations; LM every fourth generation'),
        ('lm60', 'srbench_v2_lm32b4s_i1_it4_leaf8192_scientific_pop8192_t60_full1160_rohini_sm120_20260815.csv', '60 seconds; LM every generation'),
    ]
    for key, name, budget in specs:
        path = audit / 'raw/rohini' / name
        if digest(path) != archive[name]['sha256']:
            raise ValueError('historical archive hash mismatch: ' + name)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        selected = {(r['problem'], int(r['seed'])): r for r in rows if (r['problem'], int(r['seed'])) in jobs}
        if len(rows) != 1160 or set(selected) != set(jobs):
            raise ValueError('historical case coverage mismatch')
        for pair, r in selected.items():
            j = jobs[pair]
            if (r['source_sha256'] != j['source_sha256'] or r['protocol'] != j['protocol']
                    or int(r['train_rows']) != j['num_train_rows']
                    or int(r['validation_rows']) != j['num_validation_rows']
                    or float(r['target_noise']) != j['target_noise']
                    or r['scale_x'] != '0' or r['scale_y'] != '0'
                    or r['stop_metric'] != 'train' or r['validation_mode'] != 'final'):
                raise ValueError('historical data or stopping protocol differs')
        successes = [int(float(r['validation_r2']) > .999) for r in rows]
        grouped = collections.defaultdict(list)
        for r, success in zip(rows, successes):
            grouped[r['problem']].append(success)
        historical[key] = dict(file=str(path), sha256=digest(path), budget=budget,
            fits=42, successes=sum(float(r['validation_r2']) > .999 for r in selected.values()),
            mean_seconds=statistics.mean(float(r['elapsed_seconds']) for r in selected.values()),
            median_seconds=statistics.median(float(r['elapsed_seconds']) for r in selected.values()),
            full_fits=len(rows), full_successes=sum(successes),
            full_mean_seconds=statistics.mean(float(r['elapsed_seconds']) for r in rows),
            full_dataset_median_success=statistics.mean(statistics.median(v) for v in grouped.values()),
            termination=dict(collections.Counter(r['termination_reason'] for r in selected.values())),
            cases=[dict(id=jobs[pair]['id'],success=int(float(r['validation_r2'])>.999),
                        validation_r2=float(r['validation_r2']),seconds=float(r['elapsed_seconds']))
                   for pair,r in sorted(selected.items())])
    old = {r['id']: r for r in historical['lm60']['cases']}
    launch = read(root / 'launch.json')
    records, hashes, plans = [], {}, []
    for worker in launch['workers']:
        folder = root / ('shard' + str(worker['shard']))
        if any((folder/n).read_text().strip() != '0' for n in ['worker-exit.txt','copy-exit.txt']):
            raise ValueError('worker or result copy failed')
        plan, progress, rows = [read(folder/n) for n in ['plan.json','progress.json','results.json']]
        if plan['status'] != 'complete' or progress['status'] != 'complete' or len(rows) != plan['planned']:
            raise ValueError('incomplete shard')
        identity = {k:v for k,v in plan['identity'].items() if k not in ['shard','gpu']}
        if plans and identity != plans[0]:
            raise ValueError('different policy/binary identities')
        if identity['manifest'] != digest(root/'manifest.json'):
            raise ValueError('manifest changed')
        plans.append(identity)
        records.extend(rows)
        hashes[str(folder/'results.json')] = digest(folder/'results.json')
    expected = {(a,j['id'],f'rep{n}') for a in plans[0]['arms'] for j in jobs.values() for n in range(2)}
    index = {}
    for r in records:
        key = r['comparison_policy'],r['id'],r['arm'].split('-',1)[0]
        if key in index or key not in expected or r['status'] != 'completed' or not r['pipeline_completed']:
            raise ValueError('invalid or duplicate current result')
        if r['prepared_sha256'] != jobs[r['problem'],r['seed']]['prepared_sha256']:
            raise ValueError('current prepared data changed')
        native = r.get('search_result',r)['native_result']
        if not native['score_audit']['accepted']:
            raise ValueError('search numerical audit failed')
        genome = Expression.decode(bytes.fromhex(native['genotype_hex']))
        if genome.resolve(native['coefficients'],native['permutation']).encode().hex() != native['resolved_ast_hex']:
            raise ValueError('search winner reconstruction failed')
        if r['accuracy_solution'] != int(r['validation_r2'] > .999):
            raise ValueError('success flag mismatch')
        if r['final_polish']['accepted'] and not r['final_polish']['native_audit']['accepted']:
            raise ValueError('polished native audit failed')
        if r['final_model']['resolved_ast_hex'] != r['resolved_ast_hex']:
            raise ValueError('final-model identity mismatch')
        index[key] = r
    if set(index) != expected or len(records) != launch['fits']:
        raise ValueError('incomplete current matrix')
    arms = {}
    for a in plans[0]['arms']:
        rs = [r for r in records if r['comparison_policy'] == a]
        arms[a] = dict(fits=len(rs),successes=sum(r['accuracy_solution'] for r in rs),
            strict_successes=sum(r['strict_success'] for r in rs),
            mean_seconds=statistics.mean(r['process_wall_seconds'] for r in rs),
            mean_polish_seconds=statistics.mean(r['final_polish']['seconds'] for r in rs),
            polishing_rescues=sum(r['accuracy_solution'] and not r.get('search_result',r)['accuracy_solution'] for r in rs),
            repeat_disagreements=sum(index[a,i,'rep0']['accuracy_solution'] != index[a,i,'rep1']['accuracy_solution'] for i in old),
            repeats={})
        for rep in ['rep0','rep1']:
            rr = [index[a,i,rep] for i in old]
            arms[a]['repeats'][rep] = dict(successes=sum(r['accuracy_solution'] for r in rr),
                mean_seconds=statistics.mean(r['process_wall_seconds'] for r in rr),
                gains_vs_old=[r['id'] for r in rr if r['accuracy_solution'] and not old[r['id']]['success']],
                losses_vs_old=[r['id'] for r in rr if not r['accuracy_solution'] and old[r['id']]['success']])
    return dict(historical=historical,current=arms,current_result_hashes=hashes,
        current_identity=plans[0],validated_current_fits=len(records),
        limitations=[
            'Same 21 problems and seeds 860/5390; same source hashes, protocol, noise and row counts. Historical prepared-byte hashes unavailable.',
            'Historical runs on Rohini RTX 5090; current fits on rack1 RTX 5080, one GPU per fit. No hardware normalization or clean speedup claim.',
            'Historical elapsed uses request-level timing and amortized persistent setup; current elapsed includes process dispatch/teardown.',
            'Historical has one observation per case; current has two repeats. Historical results are not duplicated into 84 independent observations.',
            'Outcome-selected diagnostic panel, not a full-suite performance estimate. Success is numerical held-out R2 > .999.',
            'Final-polish arms reserve 10s but do not recycle unused fitting time into GP search. They are maximum-budget, not equal-time-utilization arms.',
        ])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=analyze(a.run,a.audit)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for k,r in report['historical'].items():
        print(k,r['successes'],r['fits'],r['mean_seconds'])
    for k,r in report['current'].items():
        print(k,r['successes'],r['fits'],r['mean_seconds'])


if __name__=='__main__':
    main()
