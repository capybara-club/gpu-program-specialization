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
"""Validate a completed matched policy campaign and report paired outcomes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from secant_sr_ast import Expression


def read(path):
    return json.loads(path.read_text())


def analyze(root, old_ids):
    launch = read(root/'launch.json')
    manifest = read(root/'manifest.json')
    ids = {j['id'] for j in manifest['jobs']}
    if len(ids) != len(manifest['jobs']) or not old_ids <= ids:
        raise ValueError('duplicate or missing cases')
    records, plans, sources = [], [], {}
    for worker in launch['workers']:
        folder = root/f'shard{worker["shard"]}'
        if any((folder/name).read_text().strip() != '0' for name in ['worker-exit.txt', 'copy-exit.txt']):
            raise ValueError('worker or copy failed')
        plan, progress, rows = [read(folder/name) for name in ['plan.json', 'progress.json', 'results.json']]
        if plan['status'] != 'complete' or progress['status'] != 'complete' or len(rows) != plan['planned']:
            raise ValueError('incomplete shard')
        identity = dict(plan['identity'])
        for key in ['shard', 'gpu']:
            if identity.pop(key) != worker[key]:
                raise ValueError('worker identity mismatch')
        if plans:
            previous = {k:v for k,v in plans[0]['identity'].items() if k not in ['shard', 'gpu']}
            if identity != previous:
                raise ValueError('unmatched experiment identities')
        for r in rows:
            r['host'], r['shard'] = worker['host'], worker['shard']
        records.extend(rows)
        plans.append(plan)
        path = folder/'results.json'
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = plans[0]['identity']
    if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() != identity['manifest']:
        raise ValueError('manifest identity mismatch')
    arms, repeats = list(identity['arms']), identity['repeats']
    expected = {(a, i, f'rep{n}') for a in arms for i in ids for n in range(repeats)}
    index = {}
    for r in records:
        key = (r['comparison_policy'], r['id'], r['arm'].split('-', 1)[0])
        if key in index or key not in expected:
            raise ValueError('duplicate or unexpected result')
        if r['status'] != 'completed' or r['exit_code'] != 0:
            raise ValueError('invalid fit status')
        native = r['native_result']
        if native['backend'] != 'cuda' or native['score_audit']['accepted'] is not True:
            raise ValueError('backend or numerical audit mismatch')
        if Expression.decode(bytes.fromhex(native['genotype_hex'])).resolve(native['coefficients'], native['permutation']) != Expression.decode(bytes.fromhex(native['resolved_ast_hex'])):
            raise ValueError('model does not replay')
        if r['accuracy_solution'] != int(r['validation_r2'] > .999):
            raise ValueError('success classification mismatch')
        if int(r['total_configurations']) != int(r['configurations'])+int(r['refinement_configurations']):
            raise ValueError('work counter mismatch')
        index[key] = r
    if set(index) != expected or len(records) != launch['fits']:
        raise ValueError('incomplete matched matrix')
    for i in ids:
        if len({(r['host'], r['gpu'], r['prepared_sha256']) for r in records if r['id']==i}) != 1:
            raise ValueError('case moved device or data between policies')
    pairs = [(i, f'rep{n}') for i in sorted(ids) for n in range(repeats)]
    output = dict(status='complete', fits=len(records), cases=len(ids), repeats=repeats,
        elapsed_seconds=max(p['finished_unix'] for p in plans)-min(p['started_unix'] for p in plans),
        started_unix=min(p['started_unix'] for p in plans), finished_unix=max(p['finished_unix'] for p in plans),
        total_configurations=str(sum(int(r['total_configurations']) for r in records)),
        ast_occurrences=sum(int(r['ast_occurrences']) for r in records),
        errors=0, replay_and_score_audits_passed=len(records), input_hashes=sources,
        identity=identity, limitations=plans[0]['limitations'], arms={}, comparisons={}, cases_detail=[])
    for a in arms:
        rs = [index[(a,*p)] for p in pairs]
        metrics = {k:sum(float(r[k]) for r in rs) for k in [
            'process_wall_seconds','scoring_seconds','refinement_seconds','generation_seconds',
            'setup_seconds','nvrtc_seconds','pipeline_seconds','device_seconds','module_load_seconds',
            'reduction_seconds','transfer_seconds','selection_seconds','validation_seconds','teardown_seconds']}
        counts = {k:sum(int(r[k]) for r in rs) for k in [
            'refinement_models','refinement_skipped_parameter_capacity','refinement_skipped_inactive',
            'power_mutations','ast_occurrences']}
        output['arms'][a] = dict(fits=len(rs), numerical_successes=sum(r['accuracy_solution'] for r in rs),
            strict_successes=sum(r['native_result']['solved'] for r in rs),
            mean_seconds=statistics.mean(r['process_wall_seconds'] for r in rs),
            mean_generations=statistics.mean(r['ast_occurrences']/identity['config']['population'] for r in rs),
            total_configurations=str(sum(int(r['total_configurations']) for r in rs)),
            old_only_successes=sum(r['accuracy_solution'] for r in rs if r['id'] in old_ids),
            old_only_attempts=len(old_ids)*repeats,
            repeat_disagreements=[i for i in sorted(ids) if len({index[(a,i,f'rep{n}')]['accuracy_solution'] for n in range(repeats)})>1],
            stop_reasons=dict(collections.Counter(r['stop_reason'] for r in rs)),
            timing_sums=metrics, counts=counts)
    for a,b in [('baseline',a) for a in arms if a!='baseline']+[('all-fit','block-fit'),('power','block-power')]:
        wins=[p for p in pairs if not index[(a,*p)]['accuracy_solution'] and index[(b,*p)]['accuracy_solution']]
        losses=[p for p in pairs if index[(a,*p)]['accuracy_solution'] and not index[(b,*p)]['accuracy_solution']]
        output['comparisons'][a+' -> '+b] = dict(gains=wins,losses=losses,net=len(wins)-len(losses))
    for i in sorted(ids):
        output['cases_detail'].append(dict(id=i,old_only=i in old_ids,host=index[(arms[0],i,'rep0')]['host'],
            any_success=any(index[(a,i,f'rep{n}')]['accuracy_solution'] for a in arms for n in range(repeats)),
            policies={a:[{k:index[(a,i,f'rep{n}')][k] for k in ['validation_r2','train_r2','accuracy_solution','process_wall_seconds','total_configurations']} for n in range(repeats)] for a in arms}))
    output['cases_never_successful']=[r['id'] for r in output['cases_detail'] if not r['any_success']]
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--old-misses',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=analyze(a.run,{j['job']['id'] for j in read(a.old_misses)['jobs']})
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','fits','elapsed_seconds','total_configurations','replay_and_score_audits_passed']}))
    for arm,r in report['arms'].items():
        print(arm,r['numerical_successes'],r['strict_successes'],r['mean_seconds'],r['mean_generations'],r['old_only_successes'])
    print('comparisons',json.dumps(report['comparisons']))


if __name__=='__main__':
    main()
